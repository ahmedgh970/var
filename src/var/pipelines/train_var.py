import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from var.datasets.token_dataset import build_token_datasets
from var.models.var.var_model import VARModel
from var.training.optim import build_optimizer
from var.training.schedulers import build_scheduler
from var.training.var_trainer import VARTrainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(device_type: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1

    if use_ddp:
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
    return use_ddp, rank, local_rank


def build_dataloaders(cfg: DictConfig, use_ddp: bool):
    datasets = build_token_datasets(
        token_root=cfg.tokens_root,
        include_train=True,
        include_val=True,
        include_test=False,
    )
    train_set = datasets["train"]
    val_set = datasets["val"]

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


def build_model(cfg: DictConfig):
    var_cfg = cfg.var
    model = VARModel(
        vocab_size=var_cfg.vocab_size,
        patch_nums=tuple(var_cfg.patch_nums),
        dim=var_cfg.dim,
        depth=var_cfg.depth,
        num_heads=var_cfg.num_heads,
        mlp_ratio=var_cfg.mlp_ratio,
        dropout=var_cfg.dropout,
    )
    return model


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_var")
def main(cfg: DictConfig):
    requested_device = cfg.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    use_ddp, rank, local_rank = init_distributed(requested_device)
    is_main_process = rank == 0

    set_seed(int(cfg.seed) + rank)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"

    train_loader, val_loader = build_dataloaders(cfg, use_ddp=use_ddp)
    model = build_model(cfg)

    device = requested_device
    if device == "cuda":
        device = f"cuda:{local_rank}" if use_ddp else "cuda"
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if requested_device == "cuda" else None)

    train_cfg = cfg.train
    optimizer = build_optimizer(
        model=model,
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

    trainer = VARTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp=train_cfg.amp,
        grad_clip=train_cfg.grad_clip,
        save_dir=str(run_dir),
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
