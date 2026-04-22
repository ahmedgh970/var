import math
from pathlib import Path

import hydra
import torch
from var.utils.seed import set_seed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from var.inference.decode import decode_indices_to_images, save_images
from var.inference.generator import generate_token_indices
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict


def _load_var_checkpoint(model, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _strip_prefix(state, "_orig_mod.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_var_checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_var_checkpoint] unexpected keys: {len(unexpected)}")


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

    var_model = VARModel.from_config(cfg).to(device)
    tokenizer = VQVAE.from_config(cfg).to(device)
    tokenizer_ckpt_path = cfg.tokenizer.checkpoint_path
    _load_var_checkpoint(var_model, cfg.var_checkpoint_path)
    load_tokenizer_checkpoint(tokenizer, tokenizer_ckpt_path)

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
