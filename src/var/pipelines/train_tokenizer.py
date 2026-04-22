from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from var.utils.seed import set_seed
from var.utils.distributed import init_distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from var.datasets.image_dataset import build_image_datasets
from var.models.tokenizer.vqvae import VQVAE
from var.training.optim import build_optimizer
from var.training.schedulers import build_scheduler
from var.training.tokenizer_trainer import TokenizerTrainer





def build_dataloaders(cfg: DictConfig, use_ddp: bool):
    train_set, val_set = build_image_datasets(
        data_root=cfg.datasets.data_root,
        image_size=cfg.datasets.image_size,
        hflip=cfg.train.hflip,
        mid_reso=cfg.datasets.mid_reso,
        train_subdir=cfg.datasets.train_subdir,
        val_subdir=cfg.datasets.val_subdir,
    )
    if len(train_set) == 0:
        raise ValueError(f"Empty train dataset: {Path(cfg.datasets.data_root) / cfg.datasets.train_subdir}")
    if len(val_set) == 0:
        raise ValueError(f"Empty val dataset: {Path(cfg.datasets.data_root) / cfg.datasets.val_subdir}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=cfg.train.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader



@hydra.main(version_base=None, config_path="../../../configs", config_name="train_tokenizer")
def main(cfg: DictConfig):
    requested_device = cfg.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    use_ddp, rank, local_rank = init_distributed(requested_device)
    is_main_process = rank == 0

    set_seed(int(cfg.seed) + rank)

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir)
    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = ckpt_dir / "train.log"

    train_loader, val_loader = build_dataloaders(cfg, use_ddp=use_ddp)
    model = VQVAE.from_config(cfg)

    train_cfg = cfg.train
    device = requested_device
    if device == "cuda":
        device = f"cuda:{local_rank}" if use_ddp else "cuda"
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if requested_device == "cuda" else None)

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
        log_file=str(log_file),
        is_main_process=is_main_process,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        eval_every=train_cfg.eval_every,
        save_every=train_cfg.save_every,
    )
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
