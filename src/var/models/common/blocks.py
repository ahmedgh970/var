import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


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
        self.scale = channels ** -0.5  # 1/sqrt(channels) attention scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(self.norm(x))
        b, _, h, w = qkv.shape
        c = self.channels
        q, k, v = qkv.reshape(b, 3, c, h, w).unbind(1)

        q = q.view(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.view(b, c, h * w).contiguous()
        attn = F.softmax(torch.bmm(q, k).mul_(self.scale), dim=2)

        v = v.view(b, c, h * w).contiguous()
        out = torch.bmm(v, attn.permute(0, 2, 1).contiguous()).view(b, c, h, w)
        return x + self.proj(out)
