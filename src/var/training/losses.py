import torch
import torch.nn.functional as F


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(recon, target)
    return F.mse_loss(recon, target)


def tokenizer_total_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    vq_loss: torch.Tensor,
    recon_weight: float = 1.0,
    vq_weight: float = 1.0,
    loss_type: str = "mse",
):
    rec_loss = reconstruction_loss(recon, target, loss_type=loss_type)
    total = recon_weight * rec_loss + vq_weight * vq_loss
    return total, rec_loss, vq_loss
