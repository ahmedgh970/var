import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    name: str = "none",
    total_epochs: int = 200,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.1,
):
    if name == "none":
        return None

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_epochs,
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )

    if name == "warmup_cosine":
        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    return None
