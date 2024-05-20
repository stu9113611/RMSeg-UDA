from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader

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
import cv2


class Validator:
    def __init__(
        self,
        dataloader: DataLoader,
        target_dataloader: DataLoader,
        criterion: torch.nn.Module,
        inferencer: Inferencer,
        metrics: Metrics,
        img_saver: ImgSaver,
        categories: list[Category],
        logger: Logger,
    ) -> None:
        self.dataloader = dataloader
        self.target_dataloader = target_dataloader
        self.criterion = criterion
        self.inferencer = inferencer
        self.metrics = metrics
        self.img_saver = img_saver
        self.categories = categories
        self.logger = logger

    def validate(
        self,
        model: torch.nn.Module,
        iteration: int,
        prog: Progress,
    ):
        task = prog.add_task("Validating", total=len(self.dataloader))
        model.eval()
        avg_loss = 0
        for data in self.dataloader:
            img = data["img"].cuda()
            ann = data["ann"].cuda()

            with torch.cuda.amp.autocast(), torch.no_grad():
                pred = self.inferencer.inference(model, img)
                loss = self.criterion(pred, ann).mean()

            self.metrics.compute_and_accum(pred.argmax(1), ann)
            avg_loss += loss.item()
            prog.update(task, advance=1)

        prog.remove_task(task)

        self.img_saver.save_img(img, f"val_{iteration}_image.jpg")
        self.img_saver.save_ann(ann, f"val_{iteration}_ann.png")
        self.img_saver.save_pred(pred, f"val_{iteration}_pred.jpg")

        avg_loss /= len(self.dataloader)

        results = self.metrics.get_and_reset()
        mAcc = sum(results["Acc"]) / len(self.categories)
        mIoU = sum(results["IoU"]) / len(self.categories)
        mDice = sum(results["Dice"]) / len(self.categories)
        mFs = sum(results["Fscore"]) / len(self.categories)
        mPre = sum(results["Precision"]) / len(self.categories)
        mRec = sum(results["Recall"]) / len(self.categories)

        table = Table()
        table.add_column("Id", justify="right")
        table.add_column("Name")
        table.add_column("Acc.")
        table.add_column("IoU")
        table.add_column("Dice")
        table.add_column("Fscore")
        table.add_column("Precision")
        table.add_column("Recall")
        for cat, acc, iou, dice, fs, pre, rec in zip(
            self.categories,
            results["Acc"],
            results["IoU"],
            results["Dice"],
            results["Fscore"],
            results["Precision"],
            results["Recall"],
        ):
            table.add_row(
                str(cat.id),
                cat.name,
                "{:.3f}".format(acc),
                "{:.3f}".format(iou),
                "{:.3f}".format(dice),
                "{:.3f}".format(fs),
                "{:.3f}".format(pre),
                "{:.3f}".format(rec),
            )
        table.add_row(
            "",
            "",
            "{:.3f}".format(mAcc),
            "{:.3f}".format(mIoU),
            "{:.3f}".format(mDice),
            "{:.3f}".format(mFs),
            "{:.3f}".format(mPre),
            "{:.3f}".format(mRec),
        )
        rprint(table)

        self.logger.info("ValLoop", f"Iter: {iteration}, Loss: {avg_loss: .5f}")
        self.logger.tb_log("Val Loss", avg_loss, iteration)
        self.logger.tb_log("Val mAcc", mAcc, iteration)
        self.logger.tb_log("Val mIoU", mIoU, iteration)
        self.logger.tb_log("Val mDice", mDice, iteration)
        self.logger.tb_log("Val mFs", mFs, iteration)
        self.logger.tb_log("Val mPre", mPre, iteration)
        self.logger.tb_log("Val mRec", mRec, iteration)

        # Target validation
        task = prog.add_task("Target Validating", total=len(self.target_dataloader))
        model.eval()
        avg_loss = 0
        for data in self.target_dataloader:
            img = data["img"].cuda()
            ann = data["ann"].cuda()

            with torch.cuda.amp.autocast(), torch.no_grad():
                pred = self.inferencer.inference(model, img)
                loss = self.criterion(pred, ann).mean()

            self.metrics.compute_and_accum(pred.argmax(1), ann)
            avg_loss += loss.item()
            prog.update(task, advance=1)

        self.img_saver.save_img(img, f"target_val_{iteration}_image.jpg")
        self.img_saver.save_ann(ann, f"target_val_{iteration}_ann.png")
        self.img_saver.save_pred(pred, f"target_val_{iteration}_pred.jpg")

        prog.remove_task(task)

        avg_loss /= len(self.dataloader)

        results = self.metrics.get_and_reset()
        mAcc = sum(results["Acc"]) / len(self.categories)
        mIoU = sum(results["IoU"]) / len(self.categories)
        mDice = sum(results["Dice"]) / len(self.categories)
        mFs = sum(results["Fscore"]) / len(self.categories)
        mPre = sum(results["Precision"]) / len(self.categories)
        mRec = sum(results["Recall"]) / len(self.categories)

        table = Table()
        table.add_column("Id", justify="right")
        table.add_column("Name")
        table.add_column("Acc.")
        table.add_column("IoU")
        table.add_column("Dice")
        table.add_column("Fscore")
        table.add_column("Precision")
        table.add_column("Recall")
        for cat, acc, iou, dice, fs, pre, rec in zip(
            self.categories,
            results["Acc"],
            results["IoU"],
            results["Dice"],
            results["Fscore"],
            results["Precision"],
            results["Recall"],
        ):
            table.add_row(
                str(cat.id),
                cat.name,
                "{:.3f}".format(acc),
                "{:.3f}".format(iou),
                "{:.3f}".format(dice),
                "{:.3f}".format(fs),
                "{:.3f}".format(pre),
                "{:.3f}".format(rec),
            )
        table.add_row(
            "",
            "",
            "{:.3f}".format(mAcc),
            "{:.3f}".format(mIoU),
            "{:.3f}".format(mDice),
            "{:.3f}".format(mFs),
            "{:.3f}".format(mPre),
            "{:.3f}".format(mRec),
        )
        rprint(table)

        self.logger.info("TargetValLoop", f"Iter: {iteration}, Loss: {avg_loss: .5f}")
        self.logger.tb_log("Target Val Loss", avg_loss, iteration)
        self.logger.tb_log("Target Val mAcc", mAcc, iteration)
        self.logger.tb_log("Target Val mIoU", mIoU, iteration)
        self.logger.tb_log("Target Val mDice", mDice, iteration)
        self.logger.tb_log("Target Val mFs", mFs, iteration)
        self.logger.tb_log("Target Val mPre", mPre, iteration)
        self.logger.tb_log("Target Val mRec", mRec, iteration)


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
    "bdd100k": {
        "csv_path": "data/csv/bdd100k.csv",
        "train_data_root": "data/bdd100k/daytime/train",
        "target_train_data_root": "data/bdd100k/night/train",
        "val_data_root": "data/bdd100k/daytime/val",
        "target_val_data_root": "data/bdd100k/night/val",
        "img_prefix": "images",
        "ann_prefix": "labels",
        "img_suffix": ".jpg",
        "ann_suffix": ".png",
    },
    "rlmd": {
        "csv_path": "data/csv/rlmd.csv",
        "train_data_root": "data/rlmd/clear/train",
        "target_train_data_root": "data/rlmd/night/train",
        "val_data_root": "data/rlmd/clear/val",
        "target_val_data_root": "data/rlmd/night/val",
        "img_prefix": "images",
        "ann_prefix": "labels",
        "img_suffix": ".jpg",
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
    dataset_name = "rlmd"
    csv_path = path_dict[dataset_name]["csv_path"]
    train_max_length = None
    train_num_workers = batch_size
    val_max_length = None
    log_dir = Path("./logs/rlmd/stadv4to1_rcs_b16_iter80000_caadv")
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

    dataloader = ImgAnnDataset(
        # root=path_dict[dataset_name]["val_data_root"],
        root=path_dict[dataset_name]["train_data_root"],
        img_prefix=path_dict[dataset_name]["img_prefix"],
        ann_prefix=path_dict[dataset_name]["ann_prefix"],
        img_suffix=path_dict[dataset_name]["img_suffix"],
        ann_suffix=path_dict[dataset_name]["ann_suffix"],
        transforms=[
            LoadImg(),
            Resize(img_scale),
            Normalize(),
        ],
        max_len=val_max_length,
        check_exist=False,
    ).get_loader(
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=pin_memory,
    )

    model.load_state_dict(
        torch.load(log_dir / "checkpoint_iter_80000.pth")["ema_model"]
    )
    model.eval()

    inferencer = build_inferencer(cfg["inference_cfg"])
    img_saver = ImgSaver(
        root=log_dir / "inference", visualizer=IdMapVisualizer(categories)
    )

    # file = open("text.txt", "w")
    with Progress() as prog:
        for i in range(1):
            preds = torch.Tensor([]).cuda()
            task = prog.add_task("Inferencing", total=len(dataloader))
            model.eval()
            for data in dataloader:
                img = data["img"].cuda()
                filename = data["img_path"]
                # ann = data["ann"].cuda()

                with torch.cuda.amp.autocast(), torch.no_grad():
                    pred = inferencer.inference(model, img)

                img_saver.save_pred(pred, Path(filename[0]).name)

                # for p, f in zip(pred, filename):
                #     pr, id = p.max(0)
                #     preds = torch.cat((preds, pr[id == i].detach().flatten()), 0)
                #     img_saver.save_pred(p[None, :], Path(f).name)

                prog.update(task, advance=1)

            prog.remove_task(task)

            # preds = preds.sort()[0]
            # p = preds[-int(len(preds) * 0.4)]
            # print(i, "{:.10f}".format(p))
            # file.write("{:.10f}".format(p) + "\n")

    # file.close()


if __name__ == "__main__":

    main()
