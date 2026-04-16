import math
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from var.inference.decode import decode_indices_to_images, save_images
from var.inference.generator import generate_token_indices
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_var_model(cfg: DictConfig) -> VARModel:
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
    tok_cfg = cfg.tokenizer
    return VQVAE(
        vocab_size=tok_cfg.vocab_size,
        z_channels=tok_cfg.z_channels,
        ch=tok_cfg.ch,
        ch_mult=tuple(tok_cfg.ch_mult),
        num_res_blocks=tok_cfg.num_res_blocks,
        dropout=tok_cfg.dropout,
        beta=tok_cfg.beta,
        using_znorm=tok_cfg.using_znorm,
        patch_nums=tuple(tok_cfg.patch_nums),
        quantizer_type=tok_cfg.quantizer_type,
        quant_conv_ks=tok_cfg.quant_conv_ks,
        quant_resi=float(tok_cfg.get("quant_resi", 0.5)),
        share_quant_resi=int(tok_cfg.get("share_quant_resi", 4)),
        default_qresi_counts=int(tok_cfg.get("default_qresi_counts", 0)),
    )


def _load_var_checkpoint(model, checkpoint_path: str, use_ema: bool = False):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_var_checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_var_checkpoint] unexpected keys: {len(unexpected)}")
    if use_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ckpt["ema"]:
                    param.data.copy_(ckpt["ema"][name])


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    var_model = build_var_model(cfg).to(device)
    tokenizer = build_tokenizer(cfg).to(device)
    tokenizer_ckpt_path = cfg.tokenizer.get("checkpoint_path", cfg.get("tokenizer_checkpoint_path", None))
    if not tokenizer_ckpt_path:
        raise ValueError("Missing tokenizer checkpoint path. Set `tokenizer.checkpoint_path` in tokenizer config.")
    _load_var_checkpoint(var_model, cfg.var_checkpoint_path, use_ema=bool(cfg.get("use_ema", True)))
    load_tokenizer_checkpoint(tokenizer, tokenizer_ckpt_path)

    if bool(cfg.var.get("torch_compile", False)) and hasattr(torch, "compile"):
        var_model = torch.compile(var_model)

    var_model.eval()
    tokenizer.eval()

    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.batch_size)
    num_classes = int(cfg.var.get("num_classes", 1000))

    # class_labels: null → cycle through 0..num_classes, or explicit list
    if cfg.get("class_labels") is None:
        all_labels = torch.arange(num_samples, device=device) % num_classes
    else:
        all_labels = torch.tensor(list(cfg.class_labels), device=device)

    saved = 0
    for _ in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - saved)
        labels = all_labels[saved: saved + bs]
        ms_idx = generate_token_indices(
            model=var_model,
            tokenizer=tokenizer,
            batch_size=bs,
            class_labels=labels,
            cfg_scale=float(cfg.get("cfg_scale", 1.5)),
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
        )
        images = decode_indices_to_images(tokenizer, ms_idx)
        save_images(images_pm1=images, out_dir=sample_dir, start_index=saved, prefix=str(cfg.sample_prefix))
        saved += bs

    lines = [
        f"var_checkpoint: {cfg.var_checkpoint_path}",
        f"tokenizer_checkpoint: {tokenizer_ckpt_path}",
        f"num_samples: {saved}",
        f"cfg_scale: {cfg.get('cfg_scale', 1.5)}",
        f"temperature: {cfg.temperature}",
        f"top_k: {cfg.top_k}",
        f"top_p: {cfg.top_p}",
        f"sample_dir: {sample_dir}",
    ]
    print("\n".join(lines))
    with (run_dir / "generate.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
