import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .multiscale_quantizer import MultiScaleQuantizer
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        z_channels: int = 32,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        beta: float = 0.25,
        using_znorm: bool = False,
        patch_nums: tuple[int, ...] = (1, 2, 4, 8, 16),
        quantizer_type: str = "multi",
        quant_conv_ks: int = 3,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        default_qresi_counts: int = 0,
    ):
        super().__init__()
        self.quantizer_type = quantizer_type
        self.patch_nums = tuple(patch_nums)

        self.encoder = Encoder(
            in_channels=3,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
            using_sa=True,
            using_mid_sa=True,
        )
        self.decoder = Decoder(
            out_channels=3,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            dropout=dropout,
            using_sa=True,
            using_mid_sa=True,
        )

        # Pre-VQ latent adaptation.
        pad = quant_conv_ks // 2
        self.quant_conv = nn.Conv2d(
            z_channels, z_channels,
            kernel_size=quant_conv_ks,
            stride=1, padding=pad
        )
        # Post-VQ latent adaptation.
        self.post_quant_conv = nn.Conv2d(
            z_channels, z_channels, 
            kernel_size=quant_conv_ks, 
            stride=1, padding=pad
        )

        if self.quantizer_type == "single":
            self.quantizer = VectorQuantizer(
                vocab_size=vocab_size,
                embed_dim=z_channels,
                beta=beta,
                using_znorm=using_znorm,
            )
        else:
            self.quantizer = MultiScaleQuantizer(
                vocab_size=vocab_size,
                embed_dim=z_channels,
                beta=beta,
                using_znorm=using_znorm,
                patch_nums=self.patch_nums,
                quant_resi=quant_resi,
                share_quant_resi=share_quant_resi,
                default_qresi_counts=default_qresi_counts,
            )

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    def decode_latent(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z_q))

    def forward(self, x: torch.Tensor):
        z = self.encode_latent(x)

        if self.quantizer_type == "single":
            z_q, idx, vq_loss = self.quantizer(z)
            ms_idx = [idx]
        else:
            z_q, ms_idx, vq_loss = self.quantizer(z)

        rec = self.decode_latent(z_q)
        return rec, ms_idx, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.encode_latent(x)
        if self.quantizer_type == "single":
            return [self.quantizer.encode(z)]
        return self.quantizer.encode(z)

    def decode_from_indices(self, ms_idx: list[torch.Tensor]) -> torch.Tensor:
        if self.quantizer_type == "single":
            z_q = self.quantizer.decode(ms_idx[0])
        else:
            z_q = self.quantizer.decode(ms_idx)
        return self.decode_latent(z_q)

    def idx_to_var_input(self, ms_idx: list[torch.Tensor]) -> torch.Tensor | None:
        if self.quantizer_type == "single":
            return None
        return self.quantizer.idx_to_var_input(ms_idx)

    def get_next_autoregressive_input(
        self,
        si: int,
        sn: int,
        f_hat: torch.Tensor,
        h_bchw: torch.Tensor,
    ):
        if self.quantizer_type == "single":
            return f_hat + h_bchw, f_hat + h_bchw
        return self.quantizer.get_next_autoregressive_input(si=si, sn=sn, f_hat=f_hat, h_bchw=h_bchw)
