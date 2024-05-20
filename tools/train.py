import torch
from ema_pytorch import EMA
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from torch.utils.data import DataLoader
from torch.nn import functional as F
from simple_parsing import Serializable
from dataclasses import dataclass, field

from engine import builder, transform
from engine.category import Category
from engine.dataloader import (
    ImgAnnDataset,
    RCSConfig,
    RCSImgAnnDataset,
    RareCategoryManager,
)
from engine.inferencer import Inferencer
from engine.logger import Logger
from engine.metric import Metrics
from engine.misc import set_seed
from engine.optimizer import AdamW
from engine.visualizer import IdMapVisualizer, ImgSaver
from engine.models.segformer import SegformerDiscriminator


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
    ) -> None:
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.model = model
        self.criterion = criterion
        self.inferencer = inferencer
        self.metrics = metrics
        self.img_saver = img_saver
        self.categories = categories
        self.logger = logger

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
            img = data["img"].cuda()
            ann = data["ann"].cuda()

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
            img = data["img"].cuda()
            ann = data["ann"].cuda()

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


@dataclass
class TrainingConfig(Serializable):
    category_csv: str
    rcs_path: str
    source_train_root: str
    source_val_root: str
    target_train_root: str
    target_val_root: str

    logdir: str
    log_interval: int
    checkpoint_interval: int

    max_iters: int
    train_batch_size: int
    val_batch_size: int
    val_interval: int

    seed: int = 0
    num_workers: int = 16
    pin_memory: bool = True

    # ema_update_interval = None
    # ema_update_interval = 4
    ema_update_interval: list[int] | int | None = field(
        default_factory=[4, 3, 2, 1].copy
    )
    rcs_temperature: float = 0.5
    rcs_ignore_ids: tuple[int] = field(default_factory=[0].copy)
    domain_loss_weight: float = 1.0
    num_masks: int = 20


def main(cfg: TrainingConfig):

    set_seed(cfg.seed)

    if isinstance(cfg.ema_update_interval, list):
        current_ema_update_interval_id = 0
        ema_update_interval_update_interval = cfg.max_iters // len(
            cfg.ema_update_interval
        )

    categories = Category.load(cfg.category_csv)
    num_categories = Category.get_num_categories(categories)

    rcs_cfg = RCSConfig(
        file_path=cfg.rcs_path,
        ignore_ids=cfg.rcs_ignore_ids,
        temperature=cfg.rcs_temperature,
    )
    rcm = RareCategoryManager(categories, rcs_cfg)

    source_domain_class_weight = torch.zeros((num_categories)).cuda()
    target_domain_class_weight = torch.zeros((num_categories)).cuda()

    model = builder.build_model(
        {
            "name": "segformer",
            "pretrained": "nvidia/mit-b0",
            "num_classes": num_categories,
        }
    ).cuda()

    ema = EMA(
        model,
        beta=0.999,
        update_after_step=-1,
        update_every=cfg.ema_update_interval,
    ).cuda()

    soft_loss_computer = builder.build_soft_loss_computer(
        {"name": "PixelThreshold", "threshold": 0.968}
    )

    discriminator = SegformerDiscriminator.from_pretrained(
        "nvidia/mit-b0", num_labels=1
    ).cuda()

    criterion = builder.build_criterion(
        {
            "name": "cross_entropy_loss",
            "ignore_index": 255,
            "reduction": "none",  # Should be none
            "label_smoothing": 0,
        }
    ).cuda()

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
        optimizer.torch(), cfg.max_iters, 1
    )

    crop_size = (512, 512)
    stride = (384, 384)
    image_scale = (1080, 1920)

    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(categories),
        transform.Resize(image_scale),
        transform.Normalize(),
    ]

    source_train_dataloader = RCSImgAnnDataset(
        root=cfg.source_train_root,
        transforms=[
            transform.LoadImg(),
            transform.LoadAnn(categories),
            transform.RandomResizeCrop(image_scale, (0.5, 2), crop_size),
            transform.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
            ),
            transform.RandomGaussian(kernel_size=5),
            transform.Normalize(),
        ],
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
        categories=categories,
        rcm=rcm,
    ).get_loader(
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory,
        infinite=True,
    )

    target_train_dataloader = ImgAnnDataset(
        root=cfg.target_train_root,
        transforms=[
            transform.LoadImg(),
            transform.RandomResizeCrop(image_scale, (0.5, 2), crop_size),
            *[transform.RandomErase(scale=(0.02, 0.04)) for _ in range(cfg.num_masks)],
            transform.Normalize(),
        ],
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
        check_exist=False,
    ).get_loader(
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory,
        infinite=True,
    )

    source_val_dataloader = ImgAnnDataset(
        root=cfg.source_val_root,
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory,
    )

    target_val_dataloader = ImgAnnDataset(
        root=cfg.target_val_root,
        transforms=val_transforms,
        img_prefix="images",
        ann_prefix="labels",
        img_suffix=".jpg",
        ann_suffix=".png",
    ).get_loader(
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory,
    )

    logger = Logger(cfg.logdir)
    img_saver = ImgSaver(cfg.logdir, IdMapVisualizer(categories))
    metrics = Metrics(num_categories=num_categories, nan_to_num=0)

    inferencer = builder.build_inferencer(
        {
            "mode": "slide",
            "crop_size": crop_size,
            "stride": stride,
            "num_categories": num_categories,
        }
    )

    validator = Validator(
        source_val_dataloader,
        target_val_dataloader,
        model,
        criterion,
        inferencer,
        metrics,
        img_saver,
        categories,
        logger,
    )

    torch.compile(model)
    torch.compile(ema.ema_model)
    torch.compile(discriminator)
    torch.compile(criterion)

    scaler = torch.cuda.amp.GradScaler()

    with Progress() as progress:
        train_task = progress.add_task("Training", total=cfg.max_iters)
        for it in range(1, cfg.max_iters + 1):

            if isinstance(cfg.ema_update_interval, list):
                ema.update_every = cfg.ema_update_interval[
                    current_ema_update_interval_id
                ]
                if it % ema_update_interval_update_interval == 0:
                    current_ema_update_interval_id += 1

            ema.update()

            model.train()
            discriminator.train()

            optimizer.zero_grad()

            # source train
            data = next(source_train_dataloader)
            source_img = data["img"].cuda()
            source_ann = data["ann"].cuda()

            with torch.cuda.amp.autocast():
                source_pred = model(source_img)
                source_loss = criterion(source_pred, source_ann).mean()
            scaler.scale(source_loss).backward()

            if it % ema.update_every == 0:
                # target train
                data = next(target_train_dataloader)
                target_img = data["img"].cuda()
                if cfg.num_masks > 0:
                    erased_target_img = data["erased img"].cuda()

                with torch.no_grad(), torch.cuda.amp.autocast():
                    pseudo_source_ann = ema(target_img).softmax(1)
                    target_ann = ema(target_img).softmax(1)
                    max_pseudo_source_ann = torch.max(pseudo_source_ann, 1)
                    max_pseudo_target_ann = torch.max(target_ann, 1)
                    for cat in categories:
                        source_confidences = max_pseudo_source_ann.values[
                            max_pseudo_source_ann.indices == cat.id
                        ].flatten()
                        source_confidences.sort()  # min to max
                        if len(source_confidences) > 0:
                            source_domain_class_weight[cat.id] = -source_confidences[
                                int(len(source_confidences) * 0.8)
                            ].log()

                        target_confidences = max_pseudo_target_ann.values[
                            max_pseudo_target_ann.indices == cat.id
                        ].flatten()
                        target_confidences.sort()  # min to max
                        if len(target_confidences) > 0:
                            target_domain_class_weight[cat.id] = -target_confidences[
                                int(len(target_confidences) * 0.8)
                            ].log()

                with torch.cuda.amp.autocast():
                    target_pred = model(target_img)
                    target_loss = soft_loss_computer.compute(
                        target_pred, target_ann, criterion
                    )
                scaler.scale(target_loss).backward()

                if cfg.num_masks > 0:
                    with torch.cuda.amp.autocast():
                        erased_target_pred = model(erased_target_img)
                        erased_target_loss = soft_loss_computer.compute(
                            erased_target_pred, target_ann, criterion
                        )
                    scaler.scale(erased_target_loss).backward()

                # Latent Domain Discriminator
                with torch.cuda.amp.autocast():
                    source_latent = model.segformer(
                        source_img, output_hidden_states=True
                    ).hidden_states
                    source_dis_pred = discriminator(source_latent)
                    source_dis_pred = F.interpolate(
                        source_dis_pred, crop_size, mode="bilinear"
                    )
                    dis_label = torch.zeros(source_dis_pred.shape).cuda()
                    source_dis_loss = F.binary_cross_entropy_with_logits(
                        source_dis_pred,
                        dis_label,
                        source_domain_class_weight[source_ann][:, None, :],
                    ).mean()
                scaler.scale(source_dis_loss * cfg.domain_loss_weight).backward()

                with torch.cuda.amp.autocast():
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
                        target_domain_class_weight[target_ann.argmax(1)][:, None, :],
                    ).mean()
                scaler.scale(target_dis_loss * cfg.domain_loss_weight).backward()

                source_domain_class_weight.fill_(0)
                target_domain_class_weight.fill_(0)

            optimizer.step(scaler)

            # Logging
            if it % cfg.log_interval == 0:
                if it % ema.update_every == 0:
                    if cfg.num_masks > 0:
                        logger.info(
                            "Train",
                            f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Erased Target Loss: {erased_target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                        )
                    else:
                        logger.info(
                            "Train",
                            f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}, Target Loss: {target_loss.item(): .5f}, Discriminator Source Loss: {source_dis_loss: .5f}, Discriminator Target Loss: {target_dis_loss: .5f}",
                        )
                else:
                    logger.info(
                        "Train",
                        f"Iteration: {it}, Source Loss: {source_loss.item(): .5f}",
                    )

                logger.tb_log("Source Train Loss", source_loss.item(), it)
                if it % ema.update_every == 0:
                    logger.tb_log("Target Train Loss", target_loss.item(), it)
                    if cfg.num_masks > 0:
                        logger.tb_log(
                            "Erased Target Train Loss", erased_target_loss.item(), it
                        )
                    logger.tb_log(
                        "Source Train Discriminator Loss", source_dis_loss.item(), it
                    )
                    logger.tb_log(
                        "Target Train Discriminator Loss", target_dis_loss.item(), it
                    )

                img_saver.save_img(source_img, f"train_source_{it}_img.jpg")
                img_saver.save_ann(source_ann, f"train_source_{it}_ann.jpg")
                img_saver.save_pred(source_pred, f"train_source_{it}_pred.jpg")

                if it % ema.update_every == 0:
                    img_saver.save_img(target_img, f"train_target_{it}_img.jpg")
                    img_saver.save_pred(target_ann, f"train_target_{it}_ann.jpg")
                    img_saver.save_pred(target_pred, f"train_target_{it}_pred.jpg")
                    if cfg.num_masks > 0:
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

            warmup_scheduler.step()
            poly_scheduler.step()

            if it % cfg.val_interval == 0:
                validator.validate(it, progress)

            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer": optimizer.torch().state_dict(),
                },
                f"{cfg.logdir}/checkpoint_latest.pth",
            )
            if it % cfg.checkpoint_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.torch().state_dict(),
                    },
                    f"{cfg.logdir}/checkpoint_iter_{it}.pth",
                )

            progress.update(train_task, completed=it)

        progress.remove_task(train_task)


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2
    cfg = TrainingConfig.load(sys.argv[1])
    main(cfg)
