import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import VectorQuantizer


class MultiScaleQuantizer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 32,
        beta: float = 0.25,
        using_znorm: bool = False,
        patch_nums: tuple[int, ...] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.patch_nums = tuple(patch_nums)
        self.quantizer = VectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            beta=beta,
            using_znorm=using_znorm,
        )

    def forward(self, z: torch.Tensor):
        b, c, h, w = z.shape
        f_rest = z
        f_hat = torch.zeros_like(z)
        ms_idx = []
        vq_loss = 0.0

        for i, pn in enumerate(self.patch_nums):
            z_scale = F.interpolate(f_rest, size=(pn, pn), mode="area") if i < len(self.patch_nums) - 1 else f_rest
            z_q_scale, idx, loss_i = self.quantizer(z_scale)
            z_q = F.interpolate(z_q_scale, size=(h, w), mode="bicubic") if i < len(self.patch_nums) - 1 else z_q_scale

            f_hat = f_hat + z_q
            f_rest = f_rest - z_q
            ms_idx.append(idx)
            vq_loss = vq_loss + loss_i

        vq_loss = vq_loss / len(self.patch_nums)
        return f_hat, ms_idx, vq_loss

    def encode(self, z: torch.Tensor) -> list[torch.Tensor]:
        b, c, h, w = z.shape
        f_rest = z
        ms_idx = []

        for i, pn in enumerate(self.patch_nums):
            z_scale = F.interpolate(f_rest, size=(pn, pn), mode="area") if i < len(self.patch_nums) - 1 else f_rest
            idx = self.quantizer.encode(z_scale)
            z_q_scale = self.quantizer.decode(idx)
            z_q = F.interpolate(z_q_scale, size=(h, w), mode="bicubic") if i < len(self.patch_nums) - 1 else z_q_scale

            f_rest = f_rest - z_q
            ms_idx.append(idx)

        return ms_idx

    def decode(self, ms_idx: list[torch.Tensor]) -> torch.Tensor:
        h = ms_idx[-1].shape[-2]
        w = ms_idx[-1].shape[-1]
        b = ms_idx[-1].shape[0]
        c = self.quantizer.embed_dim
        f_hat = self.quantizer.embedding.weight.new_zeros((b, c, h, w))

        for i, idx in enumerate(ms_idx):
            z_q = self.quantizer.decode(idx)
            if i < len(ms_idx) - 1:
                z_q = F.interpolate(z_q, size=(h, w), mode="bicubic")
            f_hat = f_hat + z_q

        return f_hat
