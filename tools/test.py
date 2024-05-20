from pathlib import Path
from typing import Any, Optional

import torch
from ema_pytorch import EMA
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader
from torch.nn import functional as F

from engine import builder, transform
from engine.category import Category
from engine.dataloader import ImgAnnDataset, RCSConfig, RCSImgAnnDataset
from engine.inferencer import Inferencer
from engine.logger import Logger
from engine.metric import Metrics
from engine.misc import NanException, set_seed
from engine.models.segformer import Segformer
from engine.optimizer import AdamW, Optimizer
from engine.visualizer import IdMapVisualizer, ImgSaver

# from engine.models.discriminator import LogitsDiscriminator
from engine.models.segformer import SegformerDiscriminator
import os


class Validator:

    def __init__(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        model: torch.nn.Module,
        inferencer: Inferencer,
        metrics: Metrics,
        categories: list[Category],
        device: str,
        img_saver: ImgSaver,
    ) -> None:
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        # self.another_target_dataloader = another_target_dataloader
        self.model = model
        self.inferencer = inferencer
        self.metrics = metrics
        self.categories = categories
        self.device = device
        self.img_saver = img_saver

    def validate(
        self,
        progress: Progress,
    ):
        self.model.eval()

        # source validation
        source_val_task = progress.add_task(
            "Source Validating", total=len(self.source_dataloader)
        )
        for data in self.source_dataloader:
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)
            ann_path = data["ann_path"][0]

            with torch.no_grad():
                pred = self.inferencer.inference(self.model, img)

            self.img_saver.save_pred(pred, f"clear/{Path(ann_path).name}")

            self.metrics.compute_and_accum(pred.argmax(1), ann)

            progress.update(source_val_task, advance=1)

        source_iou = self.metrics.get_and_reset()["IoU"]

        progress.remove_task(source_val_task)

        # target validation
        target_val_task = progress.add_task(
            "Target Validating", total=len(self.target_dataloader)
        )
        for data in self.target_dataloader:
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)
            ann_path = data["ann_path"][0]

            with torch.no_grad():
                pred = self.inferencer.inference(self.model, img)

            self.img_saver.save_pred(pred, f"night/{Path(ann_path).name}")

            self.metrics.compute_and_accum(pred.argmax(1), ann)

            progress.update(target_val_task, advance=1)

        target_iou = self.metrics.get_and_reset()["IoU"]

        progress.remove_task(target_val_task)

        # iou table
        table = Table()
        table.add_column("Id", justify="right")
        table.add_column("Name")
        table.add_column("Source IoU")
        table.add_column("Target IoU")
        for cat, siou, tiou in zip(self.categories, source_iou, target_iou):
            table.add_row(
                str(cat.id),
                cat.name,
                "{:.5f}".format(siou),
                "{:.5f}".format(tiou),
            )
        table.add_row(
            "",
            "",
            "{:.5f}".format(source_iou.mean()),
            "{:.5f}".format(target_iou.mean()),
        )
        rprint(table)


path_dict = {
    "vpgnet": {
        "csv_path": "data/csv/vpgnet.csv",
        "train_data_root": "data/vpgnet/clear/train",
        "target_train_data_root": "data/vpgnet/night/train",
        "val_data_root": "data/vpgnet/clear/val",
        "target_val_data_root": "data/vpgnet/night/val",
        "img_prefix": "images",
        "ann_prefix": "labels",
        "img_suffix": ".png",
        "ann_suffix": ".png",
    },
    "ceymo": {
        "csv_path": "data/csv/ceymo.csv",
        "train_data_root": "data/ceymo/clear/train",
        "target_train_data_root": "data/ceymo/night/train",
        "val_data_root": "data/ceymo/clear/val",
        "target_val_data_root": "data/ceymo/night/val",
        "img_prefix": "images",
        "ann_prefix": "labels",
        "img_suffix": ".jpg",
        "ann_suffix": ".png",
    },
    "cityscapes_to_acdc_night": {
        "csv_path": "data/csv/cityscapes.csv",
        "train_data_root": "data/cityscapes/train",
        "target_train_data_root": "data/acdc/night/train",
        "val_data_root": "data/cityscapes/val",
        "target_val_data_root": "data/acdc/night/val",
        "img_prefix": "images",
        "ann_prefix": "labels",
        "img_suffix": ".png",
        "ann_suffix": ".png",
    },
}


def main():
    set_seed(0)

    device = "cuda"
    pin_memory = True
    num_workers = 16
    logdir = "logs/ceymo/stadv4to1_rcs_b8_iter80000_caadv_night"

    categories = Category.load("data/csv/ceymo.csv")
    num_categories = Category.get_num_categories(categories)

    model = builder.build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            # "pretrained": "peldrak/segformer-b0-cityscapes-512-512-finetuned-coastTrain",
            "num_classes": num_categories,
        }
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(logdir, "checkpoint_iter_80000.pth"))["model"]
    )

    crop_size = (512, 512)
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(categories),
        # transform.Resize((480, 640)),
        transform.Resize((1080, 1920)),
        transform.Normalize(),
    ]

    source_val_dataloader = ImgAnnDataset(
        root="data/mmseg_data/ceymo/clear/val",
        # root="../mmseg_data/rlmd_ac/clear/val",
        # root="data/vpgnet/clear/val",
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        # img_suffix=".png",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    target_val_dataloader = ImgAnnDataset(
        root="data/mmseg_data/ceymo/rainy/val",
        # root="../mmseg_data/rlmd_ac/rainy/val",
        # root="data/vpgnet/night/val",
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        # img_suffix=".png",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    metrics = Metrics(num_categories=num_categories, nan_to_num=0)

    inferencer = builder.build_inferencer(
        {
            # "mode": "basic",
            "mode": "slide",
            "crop_size": crop_size,
            "stride": (384, 384),
            "num_categories": num_categories,
        }
    )

    img_saver = ImgSaver(logdir, IdMapVisualizer(categories))

    validator = Validator(
        source_val_dataloader,
        target_val_dataloader,
        # another_target_val_dataloader,
        model,
        inferencer,
        metrics,
        categories,
        device,
        img_saver,
    )

    torch.compile(model)

    with Progress() as progress:
        validator.validate(progress)


if __name__ == "__main__":

    main()
