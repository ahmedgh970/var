from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from var.datasets.image_dataset import build_image_datasets
from var.models.tokenizer.vqvae import VQVAE
from var.training.optim import build_optimizer
from var.training.schedulers import build_scheduler
from var.training.tokenizer_trainer import TokenizerTrainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: DictConfig):
    _, train_set, val_set = build_image_datasets(
        data_root=cfg.datasets.data_root,
        image_size=cfg.datasets.image_size,
        hflip=cfg.datasets.hflip,
        mid_reso=cfg.datasets.mid_reso,
        train_subdir=cfg.datasets.train_subdir,
        val_subdir=cfg.datasets.val_subdir,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.datasets.batch_size,
        shuffle=True,
        num_workers=cfg.datasets.num_workers,
        pin_memory=cfg.datasets.pin_memory,
        drop_last=cfg.datasets.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.datasets.eval_batch_size,
        shuffle=False,
        num_workers=cfg.datasets.num_workers,
        pin_memory=cfg.datasets.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(cfg: DictConfig):
    model_cfg = cfg.tokenizer.model
    quantizer_type = model_cfg.get("quantizer_type", "multi")

    model = VQVAE(
        vocab_size=model_cfg.vocab_size,
        z_channels=model_cfg.z_channels,
        ch=model_cfg.ch,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        dropout=model_cfg.dropout,
        beta=model_cfg.beta,
        using_znorm=model_cfg.using_znorm,
        patch_nums=tuple(model_cfg.patch_nums),
        quantizer_type=quantizer_type,
        quant_conv_ks=model_cfg.quant_conv_ks,
    )
    return model


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_tokenizer")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    train_cfg = cfg.train
    optimizer = build_optimizer(
        model,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=tuple(train_cfg.betas),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        name=cfg.scheduler.name,
        total_epochs=train_cfg.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        min_lr_ratio=cfg.scheduler.min_lr_ratio,
    )

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    trainer = TokenizerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp=train_cfg.amp,
        grad_clip=train_cfg.grad_clip,
        recon_weight=cfg.trainer.recon_weight,
        vq_weight=cfg.trainer.vq_weight,
        loss_type=cfg.trainer.loss_type,
        save_dir=str(ckpt_dir),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        eval_every=train_cfg.eval_every,
        save_every=train_cfg.save_every,
    )


if __name__ == "__main__":
    main()
