import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.norm1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.drop(F.silu(self.norm2(h), inplace=True)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(self.norm(x))
        b, _, h, w = qkv.shape
        c = self.channels
        q, k, v = qkv.reshape(b, 3, c, h, w).unbind(1)

        q = q.view(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.view(b, c, h * w).contiguous()
        attn = torch.bmm(q, k).mul_(self.scale)
        attn = F.softmax(attn, dim=2)

        v = v.view(b, c, h * w).contiguous()
        attn = attn.permute(0, 2, 1).contiguous()
        out = torch.bmm(v, attn).view(b, c, h, w).contiguous()
        return x + self.proj(out)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
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

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid_block1 = ResBlock(block_in, block_in, dropout)
        self.mid_attn = AttnBlock(block_in) if using_mid_sa else nn.Identity()
        self.mid_block2 = ResBlock(block_in, block_in, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i]
            level_blocks = nn.ModuleList()
            level_attn = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                level_blocks.append(ResBlock(block_in, block_out, dropout))
                block_in = block_out
                level_attn.append(AttnBlock(block_in) if (i == self.num_resolutions - 1 and using_sa) else nn.Identity())
            self.up_blocks.insert(0, level_blocks)
            self.up_attn.insert(0, level_attn)
            self.upsample.insert(0, Upsample2x(block_in) if i != 0 else nn.Identity())

        self.norm_out = norm_layer(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid_block2(self.mid_attn(self.mid_block1(self.conv_in(z))))

        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up_blocks[i][j](h)
                h = self.up_attn[i][j](h)
            h = self.upsample[i](h)

        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
