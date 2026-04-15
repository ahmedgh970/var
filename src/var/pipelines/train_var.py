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
from var.inference.decode import decode_indices_to_images, save_images
from var.inference.generator import generate_token_indices
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel
from var.training.ema import ModelEMA
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
        patch_nums=tuple(cfg.tokenizer.patch_nums),
        cvae_dim=cfg.tokenizer.z_channels,
        dim=var_cfg.dim,
        depth=var_cfg.depth,
        num_heads=var_cfg.num_heads,
        mlp_ratio=var_cfg.mlp_ratio,
        dropout=var_cfg.dropout,
        drop_path_rate=float(var_cfg.get("drop_path_rate", 0.0)),
        attn_l2_norm=bool(var_cfg.get("attn_l2_norm", True)),
        init_head=float(var_cfg.get("init_head", 0.02)),
        init_std=float(var_cfg.get("init_std", -1.0)),
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
    if bool(cfg.var.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if requested_device == "cuda" else None)

    train_cfg = cfg.train
    global_batch_size = int(train_cfg.batch_size) * (dist.get_world_size() if use_ddp else 1)
    lr = float(train_cfg.lr) * (global_batch_size / 256.0)
    optimizer = build_optimizer(
        model=model,
        lr=lr,
        weight_decay=train_cfg.weight_decay,
        betas=tuple(train_cfg.betas),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        name=cfg.scheduler.name,
        total_epochs=train_cfg.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        final_lr_ratio=float(cfg.scheduler.get("final_lr_ratio", cfg.scheduler.get("min_lr_ratio", 0.1))),
    )

    ema = None
    ema_cfg = cfg.get("ema", {})
    if bool(ema_cfg.get("enabled", False)) and is_main_process:
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.9999)))

    sample_fn = None
    if is_main_process and tokenizer is not None:
        _tok = tokenizer
        _temperature = float(cfg.get("val_temperature", 1.0))
        _top_k = int(cfg.get("val_top_k", 0))
        _top_p = float(cfg.get("val_top_p", 1.0))

        def sample_fn(model, out_dir, num_samples):
            ms_idx = generate_token_indices(
                model=model,
                tokenizer=_tok,
                batch_size=num_samples,
                temperature=_temperature,
                top_k=_top_k,
                top_p=_top_p,
            )
            images = decode_indices_to_images(_tok, ms_idx)
            save_images(images, out_dir)

    trainer = VARTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp=train_cfg.amp,
        grad_clip=train_cfg.grad_clip,
        weight_decay_start=float(train_cfg.weight_decay),
        weight_decay_end=float(cfg.scheduler.get("weight_decay_end", train_cfg.weight_decay)),
        save_dir=str(run_dir),
        log_file=str(log_file),
        is_main_process=is_main_process,
        ema=ema,
        sample_fn=sample_fn,
        num_val_samples=int(train_cfg.get("num_val_samples", 8)),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        eval_every=train_cfg.eval_every,
        save_every=train_cfg.save_every,
        early_stopping_patience=train_cfg.early_stopping_patience,
        early_stopping_min_delta=train_cfg.early_stopping_min_delta,
        sample_every=int(train_cfg.get("sample_every", 0)),
    )

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
