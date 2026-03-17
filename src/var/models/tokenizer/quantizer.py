import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 32,
        beta: float = 0.25,
        using_znorm: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.using_znorm = using_znorm
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def _nearest_indices(self, z_flat: torch.Tensor) -> torch.Tensor:
        emb = self.embedding.weight
        if self.using_znorm:
            z_flat = F.normalize(z_flat, dim=-1)
            emb = F.normalize(emb, dim=-1)
            return torch.argmax(z_flat @ emb.t(), dim=1)

        dist = z_flat.pow(2).sum(dim=1, keepdim=True) + emb.pow(2).sum(dim=1) - 2.0 * (z_flat @ emb.t())
        return torch.argmin(dist, dim=1)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        idx = self._nearest_indices(z_flat)
        return idx.view(b, h, w)

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        z_q = self.embedding(idx)
        return z_q.permute(0, 3, 1, 2).contiguous()

    def forward(self, z: torch.Tensor):
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        idx = self._nearest_indices(z_flat)

        z_q = self.embedding(idx).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q = z + (z_q - z).detach() # this trick lets encoder receive gradients
        return z_q, idx.view(b, h, w), vq_loss
