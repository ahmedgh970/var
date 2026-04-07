import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        y = self.proj(y)
        y = self.drop(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VARModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        patch_nums: tuple[int, ...] = (1, 2, 4, 8, 16),
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_nums = tuple(patch_nums)
        self.num_scales = len(self.patch_nums)
        self.seq_len = sum(p * p for p in self.patch_nums)

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, dim))
        self.scale_embed = nn.Embedding(self.num_scales, dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def flatten_multiscale(self, ms_tokens: list[torch.Tensor]):
        flat = []
        scale_ids = []
        for si, t in enumerate(ms_tokens):
            b = t.shape[0]
            t = t.view(b, -1)
            flat.append(t)
            scale_ids.append(torch.full_like(t, fill_value=si))
        tokens = torch.cat(flat, dim=1)
        scale_ids = torch.cat(scale_ids, dim=1)
        return tokens, scale_ids

    def build_inputs_targets(self, ms_tokens: list[torch.Tensor]):
        tokens, scale_ids = self.flatten_multiscale(ms_tokens)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        scale_x = scale_ids[:, :-1]
        return x, y, scale_x

    def _causal_mask(self, length: int, device: torch.device):
        m = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
        return m.view(1, 1, length, length)

    def forward_tokens(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        b, l = x.shape
        tok = self.token_embed(x)
        pos = self.pos_embed[:, :l, :]
        scl = self.scale_embed(scale_x)
        h = tok + pos + scl

        mask = self._causal_mask(l, h.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        h = self.norm(h)
        logits = self.head(h)
        return logits

    def forward(self, ms_tokens: list[torch.Tensor]):
        x, y, scale_x = self.build_inputs_targets(ms_tokens)
        logits = self.forward_tokens(x, scale_x)

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        return logits, y, loss
