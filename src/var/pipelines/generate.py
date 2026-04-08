import math
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from var.inference.decode import decode_indices_to_images, save_images
from var.inference.generator import generate_token_indices
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_var_model(cfg: DictConfig) -> VARModel:
    var_cfg = cfg.var
    return VARModel(
        vocab_size=var_cfg.vocab_size,
        patch_nums=tuple(var_cfg.patch_nums),
        dim=var_cfg.dim,
        depth=var_cfg.depth,
        num_heads=var_cfg.num_heads,
        mlp_ratio=var_cfg.mlp_ratio,
        dropout=var_cfg.dropout,
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


def _load_state_dict_raw(model, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


def _load_tokenizer_checkpoint(model: VQVAE, checkpoint_path: str):
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

    model.load_state_dict(state)


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
    log_file = run_dir / "generate.log"

    var_model = build_var_model(cfg).to(device)
    tokenizer = build_tokenizer(cfg).to(device)

    _load_state_dict_raw(var_model, cfg.var_checkpoint_path)
    _load_tokenizer_checkpoint(tokenizer, cfg.tokenizer_checkpoint_path)

    var_model.eval()
    tokenizer.eval()

    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.batch_size)
    num_batches = math.ceil(num_samples / batch_size)

    saved = 0
    for _ in range(num_batches):
        bs = min(batch_size, num_samples - saved)
        ms_idx = generate_token_indices(
            model=var_model,
            tokenizer=tokenizer,
            batch_size=bs,
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
            start_token=cfg.start_token,
        )
        images = decode_indices_to_images(tokenizer, ms_idx)
        save_images(images_pm1=images, out_dir=sample_dir, start_index=saved, prefix=str(cfg.sample_prefix))
        saved += bs

    lines = [
        f"var_checkpoint: {cfg.var_checkpoint_path}",
        f"tokenizer_checkpoint: {cfg.tokenizer_checkpoint_path}",
        f"num_samples: {saved}",
        f"temperature: {cfg.temperature}",
        f"top_k: {cfg.top_k}",
        f"top_p: {cfg.top_p}",
        f"sample_dir: {sample_dir}",
    ]
    text = "\n".join(lines)
    print(text)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
