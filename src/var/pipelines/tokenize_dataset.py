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
from var.models.tokenizer.vqvae import VQVAE


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: DictConfig) -> VQVAE:
    tokenizer_cfg = cfg.tokenizer
    model = VQVAE(
        vocab_size=tokenizer_cfg.vocab_size,
        z_channels=tokenizer_cfg.z_channels,
        ch=tokenizer_cfg.ch,
        ch_mult=tuple(tokenizer_cfg.ch_mult),
        num_res_blocks=tokenizer_cfg.num_res_blocks,
        dropout=tokenizer_cfg.dropout,
        beta=tokenizer_cfg.beta,
        using_znorm=tokenizer_cfg.using_znorm,
        patch_nums=tuple(tokenizer_cfg.patch_nums),
        quantizer_type=tokenizer_cfg.quantizer_type,
        quant_conv_ks=tokenizer_cfg.quant_conv_ks,
        quant_resi=float(tokenizer_cfg.get("quant_resi", 0.5)),
        share_quant_resi=int(tokenizer_cfg.get("share_quant_resi", 4)),
        default_qresi_counts=int(tokenizer_cfg.get("default_qresi_counts", 0)),
    )
    return model


def load_checkpoint(model: VQVAE, checkpoint_path: str):
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


def tokenize_split(
    model: VQVAE,
    device: torch.device,
    split_name: str,
    split_root: Path,
    cfg: DictConfig,
    out_dir: Path,
):
    transform = build_val_transform(
        image_size=cfg.datasets.image_size,
        mid_reso=cfg.datasets.mid_reso,
    )
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
    manifest_path = split_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as mf, torch.no_grad():
        offset = 0
        pbar = tqdm(loader, desc=f"tokenize {split_name}", leave=False)
        for images in pbar:
            images = images.to(device, non_blocking=True)
            ms_idx = model.encode_to_indices(images)

            bs = images.shape[0]
            sample_paths = dataset.samples[offset: offset + bs]
            for b in range(bs):
                rel = sample_paths[b].relative_to(split_root)
                stem = str(rel.with_suffix("")).replace("/", "__")
                token_path = split_dir / f"{stem}.pt"

                tokens = [idx[b].detach().cpu().to(torch.int32) for idx in ms_idx]
                torch.save({"tokens": tokens, "image_relpath": str(rel)}, token_path)

                mf.write(
                    json.dumps(
                        {
                            "token_path": token_path.name,
                            "image_relpath": str(rel),
                            "num_scales": len(tokens),
                        }
                    )
                    + "\n"
                )
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
    log_file = run_dir / "tokenize.log"

    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint_path)
    model.eval()

    split_to_subdir = {
        "train": cfg.datasets.train_subdir,
        "val": cfg.datasets.val_subdir,
        "test": cfg.datasets.test_subdir,
    }

    lines = [f"checkpoint: {cfg.checkpoint_path}"]
    for split in cfg.splits:
        split_root = Path(cfg.datasets.data_root) / split_to_subdir[split]
        n = tokenize_split(
            model=model,
            device=device,
            split_name=split,
            split_root=split_root,
            cfg=cfg,
            out_dir=run_dir,
        )
        lines.append(f"{split}: {n} images")

    text = "\n".join(lines)
    print(text)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
