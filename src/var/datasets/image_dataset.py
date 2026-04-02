from pathlib import Path

import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from .transforms import build_vqvae_transforms


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = {ext.lower() for ext in IMG_EXTENSIONS}
        self.samples = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = pil_loader(self.samples[index])
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_image_datasets(
    data_root,
    image_size,
    hflip=False,
    mid_reso=1.125,
    train_subdir="train",
    val_subdir="val",
):
    train_tf, val_tf = build_vqvae_transforms(
        image_size=image_size,
        hflip=hflip,
        mid_reso=mid_reso,
    )

    root = Path(data_root)
    train_set = ImageDataset(root=root / train_subdir, transform=train_tf)
    val_set = ImageDataset(root=root / val_subdir, transform=val_tf)

    return train_set, val_set
