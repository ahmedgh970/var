from pathlib import Path

import hydra
import torch
from var.utils.seed import set_seed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from var.datasets.image_dataset import ImageDataset
from var.datasets.transforms import build_val_transform
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE
from var.inference.decode import denormalize_pm1_to_01
from var.training.losses import reconstruction_loss


def save_side_by_side(inp: torch.Tensor, rec: torch.Tensor, save_path: Path):
    inp = denormalize_pm1_to_01(inp.detach().cpu())
    rec = denormalize_pm1_to_01(rec.detach().cpu())
    side = torch.cat([inp, rec], dim=2)
    arr = (side.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(save_path)


@hydra.main(version_base=None, config_path="../../../configs", config_name="eval_tokenizer")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "eval.log"

    split_root = Path(cfg.datasets.data_root) / cfg.datasets.test_subdir
    val_tf = build_val_transform(
        image_size=cfg.datasets.image_size,
        mid_reso=cfg.datasets.mid_reso,
    )
    dataset = ImageDataset(root=split_root, transform=val_tf)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    model = VQVAE.from_config(cfg).to(device)
    load_tokenizer_checkpoint(model, cfg.tokenizer.checkpoint_path)
    model.eval()

    rec_meter = 0.0
    vq_meter = 0.0
    total_meter = 0.0
    n = 0
    saved = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", unit="batch"):
            images = batch.to(device, non_blocking=True)
            with autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                recon, _, vq_loss = model(images)
                rec_loss = reconstruction_loss(recon, images, loss_type=cfg.recon_loss_type)
                total = rec_loss + vq_loss

            bs = images.shape[0]
            rec_meter += float(rec_loss) * bs
            vq_meter += float(vq_loss) * bs
            total_meter += float(total) * bs
            n += bs

            if saved < int(cfg.num_save_images):
                num_left = int(cfg.num_save_images) - saved
                cur = min(num_left, bs)
                for i in range(cur):
                    save_side_by_side(images[i], recon[i], recon_dir / f"sample_{saved+i:06d}.png")
                saved += cur

    rec_mean = rec_meter / max(1, n)
    vq_mean = vq_meter / max(1, n)
    total_mean = total_meter / max(1, n)

    lines = [
        f"checkpoint: {cfg.tokenizer.checkpoint_path}",
        f"split: {cfg.datasets.test_subdir}",
        f"num_images: {n}",
        f"recon_{cfg.recon_loss_type}: {rec_mean:.6f}",
        f"vq_loss: {vq_mean:.6f}",
        f"total_loss: {total_mean:.6f}",
        f"saved_reconstructions: {saved}",
        f"recon_dir: {recon_dir}",
    ]
    text = "\n".join(lines)
    print(text)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
