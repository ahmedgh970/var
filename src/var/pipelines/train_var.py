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

from var.datasets.data_sampler import DistInfiniteBatchSampler, InfiniteBatchSampler
from var.datasets.token_dataset import build_token_datasets
from var.inference.decode import decode_indices_to_images, save_images
from var.inference.generator import generate_token_indices
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel
from var.training.ema import ModelEMA
from var.training.optim import build_optimizer
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
    token_root = cfg.datasets.get("token_root", cfg.get("tokens_root", None))
    if token_root is None:
        raise ValueError("Missing token root. Set `datasets.token_root` in dataset config.")
    datasets = build_token_datasets(token_root=token_root)
    train_set = datasets["train"]
    val_set = datasets["val"]

    world_size = dist.get_world_size() if use_ddp else 1
    rank = dist.get_rank() if use_ddp else 0
    glb_batch_size = int(cfg.train.batch_size) * world_size
    train_shuffle_seed = int(cfg.get("seed", 0))
    if use_ddp:
        train_batch_sampler = DistInfiniteBatchSampler(
            world_size=world_size,
            rank=rank,
            dataset_len=len(train_set),
            glb_batch_size=glb_batch_size,
            same_seed_for_all_ranks=train_shuffle_seed,
            shuffle=True,
            fill_last=bool(cfg.train.drop_last),
        )
    else:
        train_batch_sampler = InfiniteBatchSampler(
            dataset_len=len(train_set),
            batch_size=int(cfg.train.batch_size),
            seed_for_all_rank=train_shuffle_seed,
            shuffle=True,
            fill_last=bool(cfg.train.drop_last),
        )
    val_sampler = DistributedSampler(val_set, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
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
    steps_per_epoch = len(train_batch_sampler)
    return train_loader, val_loader, steps_per_epoch


def build_model(cfg: DictConfig):
    var_cfg = cfg.var
    return VARModel(
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
        num_classes=int(var_cfg.get("num_classes", 1000)),
        cond_drop_rate=float(var_cfg.get("cond_drop_rate", 0.1)),
        label_smoothing=float(var_cfg.get("label_smoothing", 0.0)),
        init_adaln=float(var_cfg.get("init_adaln", 0.5)),
        init_adaln_gamma=float(var_cfg.get("init_adaln_gamma", 1.0e-5)),
        init_head=float(var_cfg.get("init_head", 0.02)),
        init_std=float(var_cfg.get("init_std", -1.0)),
    )


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

    train_loader, val_loader, train_steps_per_epoch = build_dataloaders(cfg, use_ddp=use_ddp)
    model = build_model(cfg)
    tokenizer = None

    device = requested_device
    if device == "cuda":
        device = f"cuda:{local_rank}" if use_ddp else "cuda"

    tokenizer_ckpt_path = cfg.tokenizer.get("checkpoint_path", cfg.get("tokenizer_checkpoint_path", None))
    if not tokenizer_ckpt_path:
        raise ValueError(
            "Missing tokenizer checkpoint path. Set `tokenizer.checkpoint_path` in tokenizer config."
        )
    tokenizer = build_tokenizer(cfg).to(device)
    load_tokenizer_checkpoint(tokenizer, tokenizer_ckpt_path)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    model = model.to(device)
    if bool(cfg.var.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if requested_device == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    train_cfg = cfg.train
    optim_cfg = cfg.get("optim", {})
    logging_cfg = cfg.get("logging", {})
    sampling_cfg = cfg.get("sampling", {})

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    global_batch_size = int(train_cfg.batch_size) * (dist.get_world_size() if use_ddp else 1) * grad_accum_steps
    base_lr = float(optim_cfg.get("base_lr", train_cfg.get("lr", 1.0e-4)))
    lr = base_lr * (global_batch_size / 256.0)
    weight_decay = float(optim_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.05)))
    betas = tuple(optim_cfg.get("betas", train_cfg.get("betas", (0.9, 0.95))))
    optimizer = build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
    ema = None
    ema_cfg = cfg.get("ema", {})
    if bool(ema_cfg.get("enabled", False)) and is_main_process:
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.9999)))

    sample_fn = None
    if is_main_process and tokenizer is not None:
        _tok = tokenizer
        _num_classes = int(cfg.var.get("num_classes", 1000))
        _cfg_scale = float(sampling_cfg.get("val_cfg_scale", cfg.get("val_cfg_scale", 1.5)))
        _temperature = float(sampling_cfg.get("val_temperature", cfg.get("val_temperature", 1.0)))
        _top_k = int(sampling_cfg.get("val_top_k", cfg.get("val_top_k", 0)))
        _top_p = float(sampling_cfg.get("val_top_p", cfg.get("val_top_p", 1.0)))
        _device = device

        def sample_fn(model, out_dir, num_samples):
            # cycle through class indices for diverse visual validation
            labels = torch.arange(num_samples, device=_device) % _num_classes
            ms_idx = generate_token_indices(
                model=model,
                tokenizer=_tok,
                batch_size=num_samples,
                class_labels=labels,
                cfg_scale=_cfg_scale,
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
        scheduler=None,
        device=device,
        amp=train_cfg.amp,
        grad_clip=train_cfg.grad_clip,
        grad_accum_steps=grad_accum_steps,
        weight_decay_start=weight_decay,
        weight_decay_end=float(cfg.scheduler.get("weight_decay_end", weight_decay)),
        schedule_name=str(cfg.scheduler.get("name", "none")),
        warmup_epochs=int(cfg.scheduler.get("warmup_epochs", 0)),
        final_lr_ratio=float(cfg.scheduler.get("final_lr_ratio", cfg.scheduler.get("min_lr_ratio", 0.1))),
        lr_warmup_start_ratio=float(cfg.scheduler.get("warmup_start_ratio", 0.005)),
        progressive_ratio=float(train_cfg.get("progressive_ratio", 0.0)),
        progressive_start_stage=int(train_cfg.get("progressive_start_stage", 0)),
        progressive_warmup_epochs=float(train_cfg.get("progressive_warmup_epochs", 0.0)),
        save_dir=str(run_dir),
        log_file=str(log_file),
        is_main_process=is_main_process,
        ema=ema,
        sample_fn=sample_fn,
        num_val_samples=int(logging_cfg.get("num_val_samples", train_cfg.get("num_val_samples", 8))),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        steps_per_epoch=int(train_steps_per_epoch),
        eval_every=int(logging_cfg.get("eval_every", train_cfg.get("eval_every", 1))),
        save_every=int(logging_cfg.get("save_every", train_cfg.get("save_every", 1))),
        early_stopping_patience=int(
            logging_cfg.get("early_stopping_patience", train_cfg.get("early_stopping_patience", 0))
        ),
        early_stopping_min_delta=float(
            logging_cfg.get("early_stopping_min_delta", train_cfg.get("early_stopping_min_delta", 0.0))
        ),
        sample_every=int(logging_cfg.get("sample_every", train_cfg.get("sample_every", 0))),
    )

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
