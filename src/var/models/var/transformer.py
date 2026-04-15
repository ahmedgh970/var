import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        attn_l2_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_l2_norm = bool(attn_l2_norm)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.attn_drop_p = float(dropout)

        self.kv_cache_enabled = False
        self.cached_k = None
        self.cached_v = None

    def kv_caching(self, enabled: bool):
        self.kv_cache_enabled = bool(enabled)
        if not enabled:
            self.cached_k = None
            self.cached_v = None

    def _reshape_qkv(self, x: torch.Tensor):
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attn_l2_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        return q, k, v

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self._reshape_qkv(x)

        if self.kv_cache_enabled:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                self.cached_k = torch.cat([self.cached_k, k], dim=2)
                self.cached_v = torch.cat([self.cached_v, v], dim=2)
            k = self.cached_k
            v = self.cached_v
            attn_mask = None

        if hasattr(F, "scaled_dot_product_attention"):
            mask = (attn_mask > 0) if (attn_mask is not None and attn_mask.dtype != torch.bool) else attn_mask
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                is_causal=False,
            )
            y = y.transpose(1, 2).contiguous().view(b, l, d)
            y = self.proj(y)
            y = self.drop(y)
            return y

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        y = self.proj(y)
        y = self.drop(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float = 0.0,
        attn_l2_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_l2_norm=attn_l2_norm,
        )
        self.ffn = FFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=dropout,
        )
        self.drop_path = DropPath(drop_prob=drop_path)

    def kv_caching(self, enabled: bool):
        self.attn.kv_caching(enabled)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
