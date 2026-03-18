import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
):
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
