import torch
import torch.nn as nn
import torch.nn.functional as F

from var.models.common.blocks import AttnBlock, ResBlock, norm_layer


class Downsample2x(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 32,
        dropout: float = 0.0,
        using_sa: bool = True,
        using_mid_sa: bool = True,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + ch_mult
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()

        block_in = ch
        for i in range(self.num_resolutions):
            block_out = ch * ch_mult[i]
            level_blocks = nn.ModuleList()
            level_attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i]

            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(block_in, block_out, dropout))
                block_in = block_out
                level_attn.append(AttnBlock(block_in) if (i == self.num_resolutions - 1 and using_sa) else nn.Identity())

            self.down_blocks.append(level_blocks)
            self.down_attn.append(level_attn)
            self.downsample.append(Downsample2x(block_in) if i != self.num_resolutions - 1 else nn.Identity())

        self.mid_block1 = ResBlock(block_in, block_in, dropout)
        self.mid_attn = AttnBlock(block_in) if using_mid_sa else nn.Identity()
        self.mid_block2 = ResBlock(block_in, block_in, dropout)

        self.norm_out = norm_layer(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down_blocks[i][j](h)
                h = self.down_attn[i][j](h)
            h = self.downsample[i](h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
