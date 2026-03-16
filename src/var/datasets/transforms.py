import torch
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub_(1.0)


def build_train_transform(image_size: int, hflip: bool = False, mid_reso: float = 1.125):
    mid_size = round(image_size * mid_reso)
    ops = [
        transforms.Resize(mid_size, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((image_size, image_size)),
    ]
    if hflip:
        ops.insert(0, transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), normalize_01_to_pm1])
    return transforms.Compose(ops)


def build_val_transform(image_size: int, mid_reso: float = 1.125):
    mid_size = round(image_size * mid_reso)
    return transforms.Compose(
        [
            transforms.Resize(mid_size, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            normalize_01_to_pm1,
        ]
    )


def build_vqvae_transforms(image_size: int, hflip: bool = False, mid_reso: float = 1.125):
    train_tf = build_train_transform(image_size=image_size, hflip=hflip, mid_reso=mid_reso)
    val_tf = build_val_transform(image_size=image_size, mid_reso=mid_reso)
    return train_tf, val_tf
