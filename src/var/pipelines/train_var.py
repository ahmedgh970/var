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
from var.models.tokenizer.vqvae import VQVAE
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
    datasets = build_token_datasets(token_root=cfg.tokens_root)
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
        cvae_dim=cfg.tokenizer.z_channels,
        dim=var_cfg.dim,
        depth=var_cfg.depth,
        num_heads=var_cfg.num_heads,
        mlp_ratio=var_cfg.mlp_ratio,
        dropout=var_cfg.dropout,
    )
    return model


def build_tokenizer(cfg: DictConfig) -> VQVAE:
    tok = cfg.tokenizer
    return VQVAE(
        vocab_size=tok.vocab_size,
        z_channels=tok.z_channels,
        ch=tok.ch,
        ch_mult=tuple(tok.ch_mult),
        num_res_blocks=tok.num_res_blocks,
        dropout=tok.dropout,
        beta=tok.beta,
        using_znorm=tok.using_znorm,
        patch_nums=tuple(tok.patch_nums),
        quantizer_type=tok.quantizer_type,
        quant_conv_ks=tok.quant_conv_ks,
        quant_resi=float(tok.get("quant_resi", 0.5)),
        share_quant_resi=int(tok.get("share_quant_resi", 4)),
        default_qresi_counts=int(tok.get("default_qresi_counts", 0)),
    )


def load_tokenizer_checkpoint(model: VQVAE, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model_keys = model.state_dict().keys()
    has_single_key = "quantizer.embedding.weight" in state
    has_multi_key = "quantizer.quantizer.embedding.weight" in state
    expects_single = "quantizer.embedding.weight" in model_keys
    expects_multi = "quantizer.quantizer.embedding.weight" in model_keys

    if has_single_key and expects_multi:
        state["quantizer.quantizer.embedding.weight"] = state.pop("quantizer.embedding.weight")
    elif has_multi_key and expects_single:
        state["quantizer.embedding.weight"] = state.pop("quantizer.quantizer.embedding.weight")

    model.load_state_dict(state, strict=True)


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
    tokenizer = None

    device = requested_device
    if device == "cuda":
        device = f"cuda:{local_rank}" if use_ddp else "cuda"

    if not cfg.tokenizer_checkpoint_path:
        raise ValueError(
            "tokenizer_checkpoint_path is required for VAR training with idx_to_var_input conditioning."
        )
    tokenizer = build_tokenizer(cfg).to(device)
    load_tokenizer_checkpoint(tokenizer, cfg.tokenizer_checkpoint_path)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

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
        tokenizer=tokenizer,
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
        early_stopping_patience=train_cfg.early_stopping_patience,
        early_stopping_min_delta=train_cfg.early_stopping_min_delta,
    )

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
