import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import VectorQuantizer


class Phi(nn.Conv2d):
    def __init__(self, embed_dim: int, quant_resi: float):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(float(quant_resi))

    def forward(self, h_bchw: torch.Tensor):
        return h_bchw.mul(1.0 - self.resi_ratio) + super().forward(h_bchw).mul_(self.resi_ratio)


class PhiGroup(nn.Module):
    """Holds k Phi modules and selects by position ratio in [0, 1]."""
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        k = len(qresi_ls)
        if k == 4:
            self.ticks = torch.linspace(1.0 / (3 * k), 1.0 - 1.0 / (3 * k), steps=k)
        else:
            self.ticks = torch.linspace(1.0 / (2 * k), 1.0 - 1.0 / (2 * k), steps=k)

    def __getitem__(self, at_from_0_to_1: float):
        idx = int(torch.argmin(torch.abs(self.ticks - float(at_from_0_to_1))).item())
        return self.qresi_ls[idx]


class MultiScaleQuantizer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 32,
        beta: float = 0.25,
        using_znorm: bool = False,
        patch_nums: tuple[int, ...] = (1, 2, 4, 8, 16),
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        default_qresi_counts: int = 0,
    ):
        super().__init__()
        self.patch_nums = tuple(patch_nums)
        self.quantizer = VectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            beta=beta,
            using_znorm=using_znorm,
        )

        def _make_phi():
            return Phi(embed_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()

        n = share_quant_resi if share_quant_resi >= 1 else (default_qresi_counts or len(self.patch_nums))
        self.quant_resi = PhiGroup(nn.ModuleList([_make_phi() for _ in range(n)]))

    def _phi(self, si: int, sn: int):
        ratio = 0.0 if sn <= 1 else si / float(sn - 1)
        return self.quant_resi[ratio]

    def encode(self, z: torch.Tensor) -> list[torch.Tensor]:
        _, _, h, w = z.shape
        f_rest = z
        ms_idx = []
        sn = len(self.patch_nums)

        for i, pn in enumerate(self.patch_nums):
            z_scale = F.interpolate(f_rest, size=(pn, pn), mode="area") if i < sn - 1 else f_rest
            idx = self.quantizer.encode(z_scale)
            z_q_scale = self.quantizer.decode(idx)
            z_q = F.interpolate(z_q_scale, size=(h, w), mode="bicubic") if i < sn - 1 else z_q_scale
            z_q = self._phi(i, sn)(z_q)

            f_rest = f_rest - z_q
            ms_idx.append(idx)

        return ms_idx

    def decode(self, ms_idx: list[torch.Tensor]) -> torch.Tensor:
        h = ms_idx[-1].shape[-2]
        w = ms_idx[-1].shape[-1]
        b = ms_idx[-1].shape[0]
        c = self.quantizer.embed_dim
        f_hat = self.quantizer.embedding.weight.new_zeros((b, c, h, w))
        sn = len(ms_idx)

        for i, idx in enumerate(ms_idx):
            z_q = self.quantizer.decode(idx)
            if i < sn - 1:
                z_q = F.interpolate(z_q, size=(h, w), mode="bicubic")
            z_q = self._phi(i, sn)(z_q)
            f_hat = f_hat + z_q

        return f_hat

    def idx_to_var_input(self, ms_idx: list[torch.Tensor]) -> torch.Tensor | None:
        if len(ms_idx) <= 1:
            return None

        b = ms_idx[0].shape[0]
        c = self.quantizer.embed_dim
        h = self.patch_nums[-1]
        w = self.patch_nums[-1]
        sn = len(self.patch_nums)

        f_hat = self.quantizer.embedding.weight.new_zeros((b, c, h, w))
        next_scales = []
        for si in range(sn - 1):
            z_scale = self.quantizer.decode(ms_idx[si])
            z_up = F.interpolate(z_scale, size=(h, w), mode="bicubic")
            z_up = self._phi(si, sn)(z_up)
            f_hat = f_hat + z_up

            pn_next = self.patch_nums[si + 1]
            nxt = F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
            next_scales.append(nxt.flatten(2).transpose(1, 2).contiguous())

        return torch.cat(next_scales, dim=1)

    def get_next_autoregressive_input(
        self,
        si: int,
        sn: int,
        f_hat: torch.Tensor,
        h_bchw: torch.Tensor,
    ):
        hw = self.patch_nums[-1]
        if si != sn - 1:
            h = F.interpolate(h_bchw, size=(hw, hw), mode="bicubic")
            h = self._phi(si, sn)(h)
            f_hat = f_hat + h
            nxt = F.interpolate(f_hat, size=(self.patch_nums[si + 1], self.patch_nums[si + 1]), mode="area")
            return f_hat, nxt
        h = self._phi(si, sn)(h_bchw)
        f_hat = f_hat + h
        return f_hat, f_hat

    def forward(self, z: torch.Tensor):
        _, _, h, w = z.shape
        f_rest = z
        f_hat = torch.zeros_like(z)
        ms_idx = []
        vq_loss = 0.0
        sn = len(self.patch_nums)

        for i, pn in enumerate(self.patch_nums):
            z_scale = F.interpolate(f_rest, size=(pn, pn), mode="area") if i < sn - 1 else f_rest
            z_q_scale, idx, loss_i = self.quantizer(z_scale)
            z_q = F.interpolate(z_q_scale, size=(h, w), mode="bicubic") if i < sn - 1 else z_q_scale
            z_q = self._phi(i, sn)(z_q)

            f_hat = f_hat + z_q
            f_rest = f_rest - z_q
            ms_idx.append(idx)
            vq_loss = vq_loss + loss_i

        vq_loss = vq_loss / sn
        return f_hat, ms_idx, vq_loss
