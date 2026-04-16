import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)
        self.manifest = []
        with (self.split_dir / "manifest.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.manifest.append(json.loads(line))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        item = self.manifest[index]
        # token_path is relative to split_dir, e.g. "academic_gown,.../0400_imgid.pt"
        sample = torch.load(self.split_dir / item["token_path"], map_location="cpu", weights_only=False)
        tokens = [t.to(torch.long) for t in sample]
        label = torch.tensor(item["label"], dtype=torch.long)
        return tokens, label


def build_token_datasets(token_root):
    token_root = Path(token_root)
    return {
        "train": TokenDataset(token_root / "train"),
        "val":   TokenDataset(token_root / "val"),
    }
