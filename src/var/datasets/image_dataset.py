from pathlib import Path

import PIL.Image as Image
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

from .transforms import build_vqvae_transforms


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
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
    train_set = DatasetFolder(
        root=str(root / train_subdir),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_tf,
    )
    val_set = DatasetFolder(
        root=str(root / val_subdir),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=val_tf,
    )

    num_classes = len(train_set.classes)
    return num_classes, train_set, val_set
