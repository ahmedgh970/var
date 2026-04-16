import json
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from var.datasets.image_dataset import ImageDataset
from var.datasets.transforms import build_val_transform
from var.models.tokenizer.checkpoint import load_tokenizer_checkpoint
from var.models.tokenizer.vqvae import VQVAE


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: DictConfig) -> VQVAE:
    t = cfg.tokenizer
    return VQVAE(
        vocab_size=t.vocab_size,
        z_channels=t.z_channels,
        ch=t.ch,
        ch_mult=tuple(t.ch_mult),
        num_res_blocks=t.num_res_blocks,
        dropout=t.dropout,
        beta=t.beta,
        using_znorm=t.using_znorm,
        patch_nums=tuple(t.patch_nums),
        quantizer_type=t.quantizer_type,
        quant_conv_ks=t.quant_conv_ks,
        quant_resi=float(t.get("quant_resi", 0.5)),
        share_quant_resi=int(t.get("share_quant_resi", 4)),
        default_qresi_counts=int(t.get("default_qresi_counts", 0)),
    )


def tokenize_split(
    model: VQVAE,
    device: torch.device,
    split_name: str,
    split_root: Path,
    cfg: DictConfig,
    out_dir: Path,
):
    transform = build_val_transform(image_size=cfg.datasets.image_size, mid_reso=cfg.datasets.mid_reso)
    dataset = ImageDataset(root=split_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    with (split_dir / "manifest.jsonl").open("w", encoding="utf-8") as mf, torch.no_grad():
        offset = 0
        for images in tqdm(loader, desc=f"tokenize {split_name}", leave=False):
            images = images.to(device, non_blocking=True)
            ms_idx = model.encode_to_indices(images)

            bs = images.shape[0]
            for b, src_path in enumerate(dataset.samples[offset: offset + bs]):
                rel = src_path.relative_to(split_root)
                # rel = academic_gown,.../0400_imgid.jpg
                class_name = rel.parts[0]
                stem = rel.stem                          # e.g. "0400_imgid"
                label = int(stem.split("_")[0])          # e.g. 400

                token_dir = split_dir / class_name
                token_dir.mkdir(parents=True, exist_ok=True)
                token_filename = stem + ".pt"
                token_abs = token_dir / token_filename
                token_rel = class_name + "/" + token_filename   # relative to split_dir

                tokens = [ms_idx[si][b].detach().cpu().to(torch.int32) for si in range(len(ms_idx))]
                torch.save(tokens, token_abs)

                mf.write(json.dumps({
                    "token_path": token_rel,
                    "image_relpath": str(rel),
                    "class": class_name,
                    "label": label,
                    "num_scales": len(tokens),
                }) + "\n")
            offset += bs

    return len(dataset)


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenize_dataset")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    load_tokenizer_checkpoint(model, cfg.checkpoint_path)
    model.eval()

    split_to_subdir = {
        "train": cfg.datasets.train_subdir,
        "val":   cfg.datasets.val_subdir,
        "test":  cfg.datasets.test_subdir,
    }

    lines = [f"checkpoint: {cfg.checkpoint_path}"]
    for split in cfg.splits:
        split_root = Path(cfg.datasets.data_root) / split_to_subdir[split]
        n = tokenize_split(
            model=model, device=device, split_name=split,
            split_root=split_root, cfg=cfg, out_dir=run_dir,
        )
        lines.append(f"{split}: {n} images")

    text = "\n".join(lines)
    print(text)
    with (run_dir / "tokenize.log").open("w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
