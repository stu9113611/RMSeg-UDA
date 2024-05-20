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


class Validator:
    def __init__(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        # another_target_dataloader: DataLoader,
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
        # self.another_target_dataloader = another_target_dataloader
        self.model = model
        self.criterion = criterion
        self.inferencer = inferencer
        self.metrics = metrics
        self.img_saver = img_saver
        self.categories = categories
        self.logger = logger
        self.device = device

    def validate(
        self,
        iteration: int,
        progress: Progress,
    ):
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

        # # another target validation
        # another_target_val_task = progress.add_task(
        #     "Another Target Validating", total=len(self.another_target_dataloader)
        # )
        # avg_loss = 0
        # for data in self.another_target_dataloader:
        #     img = data["img"].to(self.device)
        #     ann = data["ann"].to(self.device)

        #     with torch.no_grad():
        #         pred = self.inferencer.inference(self.model, img)
        #         loss = self.criterion(pred, ann)

        #     self.metrics.compute_and_accum(pred.argmax(1), ann)
        #     avg_loss += loss.mean().item()

        #     progress.update(another_target_val_task, advance=1)

        # avg_loss /= len(self.another_target_dataloader)

        # another_target_iou = self.metrics.get_and_reset()["IoU"]

        # progress.remove_task(another_target_val_task)

        # self.logger.info(
        #     "AnotherTargetVal", f"Iter: {iteration}, Loss: {avg_loss: .5f}"
        # )
        # self.logger.tb_log("Another Target Val Loss", avg_loss, iteration)
        # self.logger.tb_log(
        #     "Another Target Val mIoU", another_target_iou.mean(), iteration
        # )

        # self.img_saver.save_img(img, f"val_another_target_{iteration}_img.jpg")
        # self.img_saver.save_ann(ann, f"val_another_target_{iteration}_ann.jpg")
        # self.img_saver.save_pred(pred, f"val_another_target_{iteration}_pred.jpg")

        # iou table
        table = Table()
        table.add_column("Id", justify="right")
        table.add_column("Name")
        table.add_column("Source IoU")
        table.add_column("Target IoU")
        table.add_column("Another Target IoU")
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

        # # iou table
        # table = Table()
        # table.add_column("Id", justify="right")
        # table.add_column("Name")
        # table.add_column("Source IoU")
        # table.add_column("Target IoU")
        # table.add_column("Another Target IoU")
        # for cat, siou, tiou, atiou in zip(
        #     self.categories, source_iou, target_iou, another_target_iou
        # ):
        #     table.add_row(
        #         str(cat.id),
        #         cat.name,
        #         "{:.5f}".format(siou),
        #         "{:.5f}".format(tiou),
        #         "{:.5f}".format(atiou),
        #     )
        # table.add_row(
        #     "",
        #     "",
        #     "{:.5f}".format(source_iou.mean()),
        #     "{:.5f}".format(target_iou.mean()),
        #     "{:.5f}".format(another_target_iou.mean()),
        # )
        # rprint(table)


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

    max_iters = 80000
    batch_size = 16
    device = "cuda"
    pin_memory = True
    num_workers = 16
    # logdir = "logs/b5_ceymo_clear_to_rainy_baseline_norcs"
    logdir = "logs/rlmd/stadv4to1_rcs_b16_iter80000_caadv_rainy_erase_10_10"
    log_interval = 60
    val_interval = 2000
    ema_warmup = -1
    # ema_warmup = float("inf")
    ema_update_interval = [4, 3, 2, 1]
    checkpoint_interval = 10000
    domain_loss_weight = 1.0
    domain_class_weight = torch.tensor(
        # [
        #     0.9999539852,
        #     0.8983560205,
        #     0.9999921322,
        #     0.9658663273,
        #     0.8474009037,
        #     0.8837525845,
        #     0.9777691364,
        #     0.5891840458,
        #     0.8989521265,
        #     0.9994749427,
        #     0.8210323453,
        #     0.9005804658,
        #     0.8215123415,
        #     0.3415269852,
        #     0.5229205489,
        #     0.9514445662,
        #     0.4865016043,
        #     0.5822640657,
        #     0.8033366203,
        #     0.6576340795,
        #     0.9980352521,
        #     0.5169944167,
        #     0.4158474505,
        #     0.2590734661,
        #     0.5173485875,
        # ]
        # # ceymo rainy
        # [
        #     0.9999992847,
        #     0.7751473784,
        #     0.0000000000 + 1e-10,
        #     0.9462216496,
        #     0.7918519378,
        #     0.4353685081,
        #     0.9956757426,
        #     0.5977853537,
        #     0.9715418220,
        #     0.9626251459,
        #     0.8135588765,
        #     0.4058961868,
        # ]
        # RLMD rainy
        [
            1.0000000000,
            0.9847636819,
            0.6744751930,
            0.6760249734,
            0.8032004833,
            0.6848960519,
            0.5430545807,
            0.3707607090,
            0.9762351513,
            0.9694154859,
            0.3504707813,
            0.8379714489,
            0.6728857756,
            0.3835082650,
            0.9062615633,
            0.5001960993,
            0.7516241074,
            0.6672520638,
            0.7447574735,
            0.7297241688,
            0.9927268028,
            0.4018235505,
            0.4442237616,
            0.3304268718,
            0.6232442260,
        ]
    )
    domain_class_weight = -domain_class_weight.log().to(device)

    if isinstance(ema_update_interval, list):
        current_ema_update_interval_id = 0
        ema_update_interval_update_interval = max_iters // len(ema_update_interval)

    # rcs_cfg = None
    rcs_cfg = RCSConfig(
        file_path="data/mmseg_data/rlmd_ac/clear/rcs.json",
        # file_path="data/rlmd_ac/clear/rcs.json",
        # file_path="data/vpgnet/clear/rcs.json",
        ignore_ids=[0],
        temperature=0.5,
    )

    categories = Category.load("data/csv/rlmd.csv")
    # categories = Category.load("data/csv/ceymo.csv")
    # categories = Category.load("data/csv/vpgnet.csv")
    num_categories = Category.get_num_categories(categories)

    model = builder.build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            # "pretrained": "peldrak/segformer-b0-cityscapes-512-512-finetuned-coastTrain",
            "num_classes": num_categories,
        }
    ).to(device)

    ema = EMA(
        model,
        beta=0.999,
        update_after_step=ema_warmup,
        update_every=ema_update_interval,
    ).to(device)
    # ema.ema_model.load_state_dict(
    #     torch.load(
    #         "log/b0_ceymo_clear_to_night_st_norcs_pixelthred_cbst_noisy_round2/latest.pth"
    #     )
    # )
    # ema.register_buffer("initted", torch.tensor(True))
    soft_loss_computer = builder.build_soft_loss_computer(
        {
            "name": "PixelThreshold",
            "threshold": 0.968,
        }
    )

    discriminator = SegformerDiscriminator.from_pretrained(
        pretrained_model_name_or_path="nvidia/mit-b0",
        # num_labels=2,
        num_labels=1,
    ).to(device)

    criterion = builder.build_criterion(
        {
            "name": "cross_entropy_loss",
            "ignore_index": 255,
            "reduction": "none",  # Should be none
            "label_smoothing": 0,
        }
    ).to(device)
    # domain_criterion = builder.build_criterion(
    #     {
    #         "name": "bce_with_logits_loss",
    #         "reduction": "none",  # Should be none
    #     }
    # ).to(device)

    optimizer = AdamW(
        [
            {"name": "backbone", "params": model.segformer.parameters(), "lr": 6e-5},
            {"name": "head", "params": model.decode_head.parameters(), "lr": 6e-4},
            {"name": "dicriminator", "params": discriminator.parameters(), "lr": 6e-4},
        ],
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer.torch(), 1e-4, 1, 1500
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer.torch(), max_iters, 1
    )

    crop_size = (512, 512)
    train_rand_crop = transform.RandomResizeCrop(
        # (480, 640),
        # (1, 2),
        # (480, 640),
        (1080, 1920),
        (0.5, 2),
        (512, 512),
    )

    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(categories),
        # transform.Resize((480, 640)),
        transform.Resize((1080, 1920)),
        transform.Normalize(),
    ]

    if rcs_cfg is not None:
        source_train_dataloader = RCSImgAnnDataset(
            # root="data/rlmd_ac/clear/train",
            root="data/mmseg_data/rlmd_ac/clear/train",
            # root="data/vpgnet/clear/train",
            transforms=[
                transform.LoadImg(),
                transform.LoadAnn(categories),
                train_rand_crop,
                transform.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                ),
                transform.RandomGaussian(kernel_size=5),
                transform.Normalize(),
            ],
            img_prefix="images",
            ann_prefix="labels",
            # img_suffix=".png",
            img_suffix=".jpg",
            ann_suffix=".png",
            categories=categories,
            rcs_cfg=rcs_cfg,
        ).get_loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            infinite=True,
        )
    else:
        source_train_dataloader = ImgAnnDataset(
            # root="data/rlmd_ac/clear/train",
            root="data/mmseg_data/rlmd_ac/clear/train",
            # root="data/vpgnet/clear/train",
            transforms=[
                transform.LoadImg(),
                transform.LoadAnn(categories),
                train_rand_crop,
                transform.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                ),
                transform.RandomGaussian(kernel_size=5),
                transform.Normalize(),
            ],
            img_prefix="images",
            ann_prefix="labels",
            # img_suffix=".png",
            img_suffix=".jpg",
            ann_suffix=".png",
        ).get_loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            infinite=True,
        )

    target_train_dataloader = ImgAnnDataset(
        # root="data/rlmd_ac/rainy/train",
        root="data/mmseg_data/rlmd_ac/rainy/train",
        # root="data/vpgnet/night/train",
        transforms=[
            transform.LoadImg(),
            train_rand_crop,
            *[transform.RandomErase(scale=(0.02, 0.04)) for _ in range(10)],
            transform.Normalize(),
        ],
        img_prefix="images",
        ann_prefix="labels",
        # img_suffix=".png",
        img_suffix=".jpg",
        ann_suffix=".png",
        check_exist=False,
    ).get_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        infinite=True,
    )

    # another_target_train_dataloader = ImgAnnDataset(
    #     root="data/ceymo/night/train",
    #     # root="data/vpgnet/night/train",
    #     transforms=[
    #         transform.LoadImg(),
    #         train_rand_crop,
    #         transform.Normalize(),
    #     ],
    #     img_prefix="images",
    #     ann_prefix="labels",
    #     # img_suffix=".png",
    #     img_suffix=".jpg",
    #     ann_suffix=".png",
    #     check_exist=False,
    # ).get_loader(
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     drop_last=True,
    #     pin_memory=pin_memory,
    #     infinite=True,
    # )

    source_val_dataloader = ImgAnnDataset(
        # root="data/rlmd_ac/clear/val",
        root="data/mmseg_data/rlmd_ac/clear/val",
        # root="data/vpgnet/clear/val",
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        # img_suffix=".png",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    target_val_dataloader = ImgAnnDataset(
        # root="data/rlmd_ac/rainy/val",
        root="data/mmseg_data/rlmd_ac/rainy/val",
        # root="data/vpgnet/night/val",
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        # img_suffix=".png",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    # another_target_val_dataloader = ImgAnnDataset(
    #     root="data/ceymo/night/val",
    #     # root="data/vpgnet/night/val",
    #     transforms=val_transforms,
    #     img_prefix="images",
    #     ann_prefix="labels",
    #     # img_suffix=".png",
    #     img_suffix=".jpg",
    #     ann_suffix=".png",
    # ).get_loader(
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     drop_last=False,
    #     pin_memory=pin_memory,
    # )

    logger = Logger(logdir)
    img_saver = ImgSaver(logdir, IdMapVisualizer(categories))
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

    validator = Validator(
        source_val_dataloader,
        target_val_dataloader,
        # another_target_val_dataloader,
        model,
        criterion,
        inferencer,
        metrics,
        img_saver,
        categories,
        logger,
        device,
    )

    torch.compile(model)
    torch.compile(ema.ema_model)
    torch.compile(discriminator)
    torch.compile(criterion)
    # torch.compile(domain_criterion)

    with Progress() as progress:
        train_task = progress.add_task("Training", total=max_iters)
        for it in range(1, max_iters + 1):

            if isinstance(ema_update_interval, list):
                ema.update_every = ema_update_interval[current_ema_update_interval_id]
                if it % ema_update_interval_update_interval == 0:
                    current_ema_update_interval_id += 1

            # logger.tb_log("EMA Update interval", ema.update_every, it)

            ema.update()

            model.train()
            discriminator.train()

            optimizer.zero_grad()

            # source train
            data = next(source_train_dataloader)
            source_img = data["img"].to(device)
            source_ann = data["ann"].to(device)

            source_pred = model(source_img)
            source_loss = criterion(source_pred, source_ann).mean()
            source_loss.backward()

            if it > ema_warmup and it % ema.update_every == 0:
                # target train
                data = next(target_train_dataloader)
                target_img = data["img"].to(device)
                erased_target_img = data["erased img"].to(device)

                with torch.no_grad():
                    target_ann = ema(target_img).softmax(1)

                target_pred = model(target_img)
                target_loss = soft_loss_computer.compute(
                    target_pred, target_ann, criterion
                )
                target_loss.backward()

                erased_target_pred = model(erased_target_img)
                erased_target_loss = soft_loss_computer.compute(
                    erased_target_pred, target_ann, criterion
                )
                erased_target_loss.backward()

                # # another target train
                # data = next(another_target_train_dataloader)
                # another_target_img = data["img"].to(device)

                # with torch.no_grad():
                #     another_target_ann = ema(another_target_img).softmax(1)

                # another_target_pred = model(another_target_img)
                # another_target_loss = soft_loss_computer.compute(
                #     another_target_pred, another_target_ann, criterion
                # )
                # another_target_loss.backward()

                # Latent Domain Discriminator
                source_latent = model.segformer(
                    source_img, output_hidden_states=True
                ).hidden_states
                source_dis_pred = discriminator(source_latent)
                source_dis_pred = F.interpolate(
                    source_dis_pred, crop_size, mode="bilinear"
                )
                dis_label = torch.zeros(source_dis_pred.shape).to(device)
                source_dis_loss = F.binary_cross_entropy_with_logits(
                    source_dis_pred,
                    dis_label,
                    domain_class_weight[source_ann][:, None, :],
                ).mean()
                # source_dis_loss = domain_criterion(source_dis_pred, dis_label).mean()
                (source_dis_loss * 0.5).backward()

                target_latent = model.segformer(
                    target_img, output_hidden_states=True
                ).hidden_states
                target_dis_pred = discriminator(target_latent)
                target_dis_pred = F.interpolate(
                    target_dis_pred, crop_size, mode="bilinear"
                )
                dis_label.fill_(1)
                target_dis_loss = F.binary_cross_entropy_with_logits(
                    target_dis_pred,
                    dis_label,
                    domain_class_weight[target_ann.argmax(1)][:, None, :],
                ).mean()
                # target_dis_loss = domain_criterion(target_dis_pred, dis_label).mean()
                (target_dis_loss * 0.5).backward()

                # another_target_latent = model.segformer(
                #     another_target_img, output_hidden_states=True
                # ).hidden_states
                # another_target_dis_pred = discriminator(another_target_latent)
                # another_target_dis_label = torch.zeros(
                #     another_target_dis_pred.shape
                # ).to(device)
                # another_target_dis_label[:, 2] = 1
                # another_target_dis_loss = domain_criterion(
                #     another_target_dis_pred, another_target_dis_label
                # ).mean()
                # another_target_dis_loss.backward()

            optimizer.step()

            # Logging
            if it % log_interval == 0:
                if it > ema_warmup and it % ema.update_every == 0:
                    logger.info(
                        "Train",
                        # f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}",
                        # f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Another Target Loss: {another_target_loss.item(): .5f}",
                        # f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                        f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Erased Target Loss: {erased_target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                        # f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Another Target Loss: {another_target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}, Discriminator Another Target Loss: {another_target_dis_loss.item(): .5f}",
                    )
                else:
                    logger.info(
                        "Train",
                        f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}",
                    )

                logger.tb_log("Source Train Loss", source_loss.item(), it)
                if it > ema_warmup and it % ema.update_every == 0:
                    logger.tb_log("Target Train Loss", target_loss.item(), it)
                    logger.tb_log(
                        "Erased Target Train Loss", erased_target_loss.item(), it
                    )
                    # logger.tb_log(
                    #     "Another Target Train Loss", another_target_loss.item(), it
                    # )
                    logger.tb_log(
                        "Source Train Discriminator Loss", source_dis_loss.item(), it
                    )
                    logger.tb_log(
                        "Target Train Discriminator Loss", target_dis_loss.item(), it
                    )
                    # logger.tb_log(
                    #     "Another Target Train Discriminator Loss",
                    #     another_target_dis_loss.item(),
                    #     it,
                    # )

                img_saver.save_img(source_img, f"train_source_{it}_img.jpg")
                img_saver.save_ann(source_ann, f"train_source_{it}_ann.jpg")
                img_saver.save_pred(source_pred, f"train_source_{it}_pred.jpg")

                if it > ema_warmup and it % ema.update_every == 0:
                    img_saver.save_img(target_img, f"train_target_{it}_img.jpg")
                    img_saver.save_pred(target_ann, f"train_target_{it}_ann.jpg")
                    img_saver.save_pred(target_pred, f"train_target_{it}_pred.jpg")
                    img_saver.save_img(
                        erased_target_img, f"train_target_{it}_erased_img.jpg"
                    )
                    img_saver.save_pred(
                        erased_target_pred, f"train_target_{it}_erased_pred.jpg"
                    )

                    img_saver.save_heatmap(
                        source_dis_pred.squeeze().detach().softmax(1),
                        f"train_source_{it}_dis_pred.jpg",
                    )
                    img_saver.save_heatmap(
                        target_dis_pred.squeeze().detach().softmax(1),
                        f"train_target_{it}_dis_pred.jpg",
                    )

                    # img_saver.save_img(
                    #     another_target_img, f"train_another_target_{it}_img.jpg"
                    # )
                    # img_saver.save_pred(
                    #     another_target_ann, f"train_another_target_{it}_ann.jpg"
                    # )
                    # img_saver.save_pred(
                    #     another_target_pred, f"train_another_target_{it}_pred.jpg"
                    # )

            warmup_scheduler.step()
            poly_scheduler.step()

            if it % val_interval == 0:
                validator.validate(it, progress)

            torch.save(
                {
                    "model": model.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer": optimizer.torch().state_dict(),
                },
                f"{logdir}/checkpoint_latest.pth",
            )
            if it % checkpoint_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.torch().state_dict(),
                    },
                    f"{logdir}/checkpoint_iter_{it}.pth",
                )

            progress.update(train_task, completed=it)

        progress.remove_task(train_task)


if __name__ == "__main__":

    main()
