from pathlib import Path

import torch
from PIL import Image

from var.models.tokenizer.vqvae import VQVAE


def denormalize_pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


@torch.no_grad()
def decode_indices_to_images(tokenizer: VQVAE, ms_idx: list[torch.Tensor]) -> torch.Tensor:
    tokenizer.eval()
    return tokenizer.decode_from_indices(ms_idx)


def save_images(images_pm1: torch.Tensor, out_dir: str | Path, start_index: int = 0, prefix: str = "sample"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = denormalize_pm1_to_01(images_pm1.detach().cpu())
    for i in range(images.shape[0]):
        img = images[i]
        arr = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"{prefix}_{start_index + i:06d}.png")
