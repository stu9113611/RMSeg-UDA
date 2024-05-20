from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader

from engine.models.segformer import Segformer
from engine.builder import (
    build_criterion,
    build_ema_model,
    build_entropy_loss_computer,
    build_inferencer,
    build_model,
    build_soft_loss_computer,
)
from engine.category import Category
from engine.dataloader import ImgAnnDataset, RCSConfig, RCSImgAnnDataset
from engine.ema import (
    BasicEntropy,
    EMAModel,
    EntropyLossComputer,
    NoThreshold,
    SoftLossComputer,
)
from engine.inferencer import Inferencer
from engine.logger import Logger
from engine.metric import Metrics
from engine.misc import NanException, set_seed
from engine.optimizer import AdamW, Optimizer
from engine.transform import (
    ColorJitter,
    Identity,
    LoadAnn,
    LoadImg,
    Normalize,
    RandomGaussian,
    RandomResizeCrop,
    Resize,
    WeakAndStrong,
)
from engine.visualizer import IdMapVisualizer, ImgSaver

from ema_pytorch import EMA
import numpy as np


class Validator:
    def __init__(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        inferencer: Inferencer,
        metrics: Metrics,
        img_saver: ImgSaver,
        categories: list[Category],
        logger: Logger,
        device: str,
    ) -> None:
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.model = model
        self.criterion = criterion
        self.inferencer = inferencer
        self.metrics = metrics
        self.img_saver = img_saver
        self.categories = categories
        self.num_categories = Category.get_num_categories(categories)
        self.logger = logger
        self.device = device

    def validate(
        self,
        iteration: int,
        progress: Progress,
        use_bayesian: bool = False,
    ):
        if use_bayesian:
            self.model.train()
        else:
            self.model.eval()

        # source validation
        source_val_task = progress.add_task(
            "Source Validating", total=len(self.source_dataloader)
        )
        avg_loss = 0
        for data in self.source_dataloader:
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)

            with torch.no_grad():
                if use_bayesian:
                    pred = torch.zeros(
                        [ann.shape[0], self.num_categories, ann.shape[1], ann.shape[2]]
                    ).to(self.device)
                    for _ in range(4):
                        pred += self.inferencer.inference(self.model, img)
                    pred /= 4
                else:
                    pred = self.inferencer.inference(self.model, img)
                loss = self.criterion(pred, ann)

            self.metrics.compute_and_accum(pred.argmax(1), ann)
            avg_loss += loss.mean().item()

            progress.update(source_val_task, advance=1)

        avg_loss /= len(self.source_dataloader)

        source_iou = self.metrics.get_and_reset()["IoU"]

        progress.remove_task(source_val_task)

        self.logger.info("SourceVal", f"Iter: {iteration}, Loss: {avg_loss: .5f}")
        self.logger.tb_log("Source Val Loss", avg_loss, iteration)
        self.logger.tb_log("Source Val mIoU", source_iou.mean(), iteration)

        self.img_saver.save_img(img, f"val_source_{iteration}_img.jpg")
        self.img_saver.save_ann(ann, f"val_source_{iteration}_ann.jpg")
        self.img_saver.save_pred(pred, f"val_source_{iteration}_pred.jpg")

        # target validation
        target_val_task = progress.add_task(
            "Target Validating", total=len(self.target_dataloader)
        )
        avg_loss = 0
        for data in self.target_dataloader:
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)

            with torch.no_grad():
                pred = self.inferencer.inference(self.model, img)
                loss = self.criterion(pred, ann)

            self.metrics.compute_and_accum(pred.argmax(1), ann)
            avg_loss += loss.mean().item()

            progress.update(target_val_task, advance=1)

        avg_loss /= len(self.target_dataloader)

        target_iou = self.metrics.get_and_reset()["IoU"]

        progress.remove_task(target_val_task)

        self.logger.info("TargetVal", f"Iter: {iteration}, Loss: {avg_loss: .5f}")
        self.logger.tb_log("Target Val Loss", avg_loss, iteration)
        self.logger.tb_log("Target Val mIoU", target_iou.mean(), iteration)

        self.img_saver.save_img(img, f"val_target_{iteration}_img.jpg")
        self.img_saver.save_ann(ann, f"val_target_{iteration}_ann.jpg")
        self.img_saver.save_pred(pred, f"val_target_{iteration}_pred.jpg")

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
        "source_train_data_root": "data/ceymo/clear/train",
        "target_train_data_root": "data/ceymo/night/train",
        "source_val_data_root": "data/ceymo/clear/val",
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

rlmd_cat_prob = np.array(
    [
        0.9804834940,
        0.0022226280,
        0.0047020205,
        0.0004640183,
        0.0026288298,
        0.0010185393,
        0.0003065209,
        0.0006291530,
        0.0007521704,
        0.0020228435,
        0.0000262015,
        0.0000932766,
        0.0002470924,
        0.0000254667,
        0.0000901788,
        0.0001215752,
        0.0007469230,
        0.0006075721,
        0.0000775862,
        0.0001131051,
        0.0013196985,
        0.0002486688,
        0.0000117560,
        0.0000067976,
        0.0010338839,
    ]
)


def main():
    seed = 0
    set_seed(seed)

    batch_size = 8
    dataset_name = "ceymo"
    csv_path = path_dict[dataset_name]["csv_path"]
    train_max_length = None
    train_num_workers = batch_size
    val_max_length = None
    log_dir = Path("log/b0_rlmd_clear_to_night_rcs_ema4_80000_8")
    pin_memory = True
    img_scale = (1080, 1920)
    crop_size = (512, 512)
    stride = (384, 384)

    categories = Category.load(csv_path)

    cfg = dict(
        model_cfg={
            # ----- SegFormer -----
            #
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            "num_classes": len(categories),
            #
            # ----- DeepLabV3plus -----
            #
            # ...
            #
        },
        criterion_cfg={
            # ----- CrossEntropy -----
            #
            "name": "cross_entropy_loss",
            "ignore_index": 255,
            "reduction": "mean",
            "label_smoothing": 0,
            #
            # ----- Focal -----
            #
            # "name": "focal_loss",
            # "gamma": 2,
            # "normalized": False,
            #
            # ----- Dice -----
            #
            # "name": "dice_loss",
            #
        },
        inference_cfg={
            # ----- No slide -----
            #
            # "mode": "basic",
            #
            # ----- slide -----
            #
            "mode": "slide",
            "crop_size": crop_size,
            "stride": stride,
            "num_categories": len(categories),
            #
        },
    )

    model = build_model(cfg["model_cfg"]).cuda()

    source_dataloader = ImgAnnDataset(
        root=path_dict[dataset_name]["source_val_data_root"],
        img_prefix=path_dict[dataset_name]["img_prefix"],
        ann_prefix=path_dict[dataset_name]["ann_prefix"],
        img_suffix=path_dict[dataset_name]["img_suffix"],
        ann_suffix=path_dict[dataset_name]["ann_suffix"],
        transforms=[
            LoadImg(),
            LoadAnn(categories),
            Resize(img_scale),
            Normalize(),
        ],
        max_len=val_max_length,
        check_exist=False,
    ).get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
    )

    target_dataloader = ImgAnnDataset(
        root=path_dict[dataset_name]["target_val_data_root"],
        img_prefix=path_dict[dataset_name]["img_prefix"],
        ann_prefix=path_dict[dataset_name]["ann_prefix"],
        img_suffix=path_dict[dataset_name]["img_suffix"],
        ann_suffix=path_dict[dataset_name]["ann_suffix"],
        transforms=[
            LoadImg(),
            LoadAnn(categories),
            Resize(img_scale),
            Normalize(),
        ],
        max_len=val_max_length,
        check_exist=False,
    ).get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
    )

    model.load_state_dict(
        torch.load("log/b0_ceymo_clear_to_night_baseline_norcs/latest.pth")
    )

    inferencer = build_inferencer(cfg["inference_cfg"])
    img_saver = ImgSaver(
        root=log_dir / "inference", visualizer=IdMapVisualizer(categories)
    )

    validator = Validator(
        source_dataloader,
        target_dataloader,
        model,
        torch.nn.CrossEntropyLoss(),
        inferencer,
        Metrics(
            Category.get_num_categories(categories), metric_types=["IoU"], nan_to_num=0
        ),
        img_saver,
        categories,
        Logger(log_dir),
        "cuda",
    )

    with Progress() as prog:
        # 0.35173, 0.10989
        # validator.validate(0, prog, use_bayesian=True)
        task = prog.add_task("Inferencing", total=len(source_dataloader))
        model.eval()
        # model.train()
        for data in source_dataloader:
            img = data["img"].cuda()

            model: Segformer
            out = model.segformer(
                torch.rand((1, 3, 512, 512)).cuda(), output_hidden_states=True
            )
            for en in out.hidden_states:
                print(en.shape)
            input("")
            # filename = data["img_path"]

            # with torch.no_grad():
            #     pred = inferencer.inference(model, img)

            # img_saver.save_pred(pred, Path(filename[0]).name)

            prog.update(task, advance=1)

        prog.remove_task(task)


if __name__ == "__main__":

    main()
