import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        split_dir,
        return_image_relpath: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.return_image_relpath = return_image_relpath
        self.manifest = []

        manifest_path = self.split_dir / "manifest.jsonl"
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.manifest.append(json.loads(line))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        item = self.manifest[index]
        sample = torch.load(self.split_dir / item["token_path"], map_location="cpu")
        tokens = [t.to(torch.long) for t in sample["tokens"]]

        if self.return_image_relpath:
            return tokens, sample["image_relpath"]
        return tokens


def build_token_datasets(
    token_root,
):
    token_root = Path(token_root)
    datasets = {}
    datasets["train"] = TokenDataset(token_root / "train")
    datasets["val"] = TokenDataset(token_root / "val")
    return datasets
