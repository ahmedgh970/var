import math

import torch
import torch.nn as nn
from omegaconf import DictConfig
from .transformer import AdaLNBlock, AdaLNBeforeHead


class VARModel(nn.Module):
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "VARModel":
        v = cfg.var
        return cls(
            vocab_size=v.vocab_size,
            patch_nums=tuple(cfg.tokenizer.patch_nums),
            cvae_dim=cfg.tokenizer.z_channels,
            dim=v.dim,
            depth=v.depth,
            num_heads=v.num_heads,
            mlp_ratio=v.mlp_ratio,
            dropout=v.dropout,
            drop_path_rate=v.drop_path_rate,
            attn_l2_norm=v.attn_l2_norm,
            num_classes=v.num_classes,
            cond_drop_rate=v.cond_drop_rate,
            label_smoothing=v.label_smoothing,
            init_adaln=v.init_adaln,
            init_adaln_gamma=v.init_adaln_gamma,
            init_head=v.init_head,
            init_std=v.init_std,
        )


    def __init__(
        self,
        vocab_size: int = 4096,
        patch_nums: tuple[int, ...] = (1, 2, 4, 8, 16),
        cvae_dim: int = 32,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_l2_norm: bool = True,
        num_classes: int = 1000,
        cond_drop_rate: float = 0.1,
        label_smoothing: float = 0.0,
        init_adaln: float = 0.5,
        init_adaln_gamma: float = 1e-5,
        init_head: float = 0.02,
        init_std: float = -1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_nums = tuple(patch_nums)
        self.num_scales = len(self.patch_nums)
        self.seq_len = sum(p * p for p in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.num_classes = num_classes
        self.cond_drop_rate = cond_drop_rate
        self.label_smoothing = label_smoothing

        self.begin_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        # class embedding: num_classes + 1 where index num_classes = unconditional token
        self.class_emb = nn.Embedding(num_classes + 1, dim)
        self.word_embed = nn.Linear(cvae_dim, dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.seq_len, dim))
        self.scale_embed = nn.Embedding(self.num_scales, dim)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, dim))

        dpr = torch.linspace(0.0, float(drop_path_rate), depth).tolist()
        self.blocks = nn.ModuleList([
            AdaLNBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, drop_path=dpr[i], attn_l2_norm=attn_l2_norm,
            )
            for i in range(depth)
        ])
        self.norm = AdaLNBeforeHead(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

        lvl_1l = torch.cat(
            [torch.full((pn * pn,), fill_value=si, dtype=torch.long) for si, pn in enumerate(self.patch_nums)],
            dim=0,
        )
        self.register_buffer("lvl_1l", lvl_1l)

        # Causal mask: token i can attend to j only if they belong to the same scale or j is at a coarser scale (level[i] >= level[j])
        d = lvl_1l.view(1, self.seq_len, 1)
        attn_mask = (d >= d.transpose(1, 2)).view(1, 1, self.seq_len, self.seq_len)
        self.register_buffer("attn_mask", attn_mask)
        self.init_weights(
            init_adaln=init_adaln,
            init_adaln_gamma=init_adaln_gamma,
            init_head=init_head,
            init_std=init_std,
        )

    def kv_caching(self, enabled: bool):
        for blk in self.blocks:
            blk.kv_caching(enabled)

    def init_weights(
        self,
        init_adaln: float = 0.5,
        init_adaln_gamma: float = 1e-5,
        init_head: float = 0.02,
        init_std: float = -1.0,
    ):
        if init_std < 0:
            init_std = (1.0 / self.head.in_features / 3.0) ** 0.5
        nn.init.trunc_normal_(self.pos_embed, std=init_std)
        nn.init.trunc_normal_(self.pos_start, std=init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=init_std)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(float(init_head))
        c = self.head.in_features
        depth = len(self.blocks)
        for blk in self.blocks:
            blk.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            blk.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            blk.ada_lin[-1].weight.data[2 * c:].mul_(init_adaln)
            blk.ada_lin[-1].weight.data[: 2 * c].mul_(init_adaln_gamma)
            if blk.ada_lin[-1].bias is not None:
                nn.init.zeros_(blk.ada_lin[-1].bias)

        self.norm.ada_lin[-1].weight.data.mul_(init_adaln)
        if self.norm.ada_lin[-1].bias is not None:
            nn.init.zeros_(self.norm.ada_lin[-1].bias)

    def _get_cond(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Apply conditional dropout and return class embedding (B, dim)."""
        if self.training and self.cond_drop_rate > 0:
            drop_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.cond_drop_rate
            class_labels = torch.where(drop_mask, torch.full_like(class_labels, self.num_classes), class_labels)
        return self.class_emb(class_labels)

    def _forward_stage(
        self,
        cond_blc_wo_first_l: torch.Tensor | None,
        cond_BD: torch.Tensor,
        batch_size: int,
        stage_idx: int,
    ) -> torch.Tensor:
        bg, ed = self.begin_ends[stage_idx]
        stage_len = ed - bg
        use_cached_infer = self.blocks[0].attn.kv_cache_enabled and not self.training

        if use_cached_infer:
            if stage_idx == 0:
                h = (
                    self.pos_start.expand(batch_size, self.first_l, -1)
                    + self.pos_embed[:, :self.first_l, :]
                    + self.scale_embed(self.lvl_1l[:self.first_l]).unsqueeze(0)
                )
            else:
                c_bg, c_ed = bg - self.first_l, ed - self.first_l
                cond_chunk = cond_blc_wo_first_l[:, c_bg:c_ed, :]
                h = (
                    self.word_embed(cond_chunk)
                    + self.pos_embed[:, bg:ed, :]
                    + self.scale_embed(self.lvl_1l[bg:ed]).unsqueeze(0)
                )
            mask = None
        else:
            sos = (
                self.pos_start.expand(batch_size, self.first_l, -1)
                + self.pos_embed[:, :self.first_l, :]
                + self.scale_embed(self.lvl_1l[:self.first_l]).unsqueeze(0)
            )
            if ed == self.first_l:
                h = sos
            else:
                cond = cond_blc_wo_first_l[:, :ed - self.first_l, :]
                cond_h = (
                    self.word_embed(cond)
                    + self.pos_embed[:, self.first_l:ed, :]
                    + self.scale_embed(self.lvl_1l[self.first_l:ed]).unsqueeze(0)
                )
                h = torch.cat([sos, cond_h], dim=1)
            mask = self.attn_mask[:, :, :ed, :ed]

        for blk in self.blocks:
            h = blk(h, cond_BD, attn_mask=mask)
        h = self.norm(h, cond_BD)
        logits = self.head(h[:, :stage_len] if use_cached_infer else h[:, bg:ed])
        return logits

    def forward(
        self,
        ms_tokens: list[torch.Tensor],
        cond_blc_wo_first_l: torch.Tensor | None,
        class_labels: torch.Tensor,
    ):
        cond_BD = self._get_cond(class_labels)
        flat_scales = [t.view(t.shape[0], -1) for t in ms_tokens]
        targets_cat = torch.cat(flat_scales, dim=1)
        b = targets_cat.shape[0]

        sos = (
            self.pos_start.expand(b, self.first_l, -1)
            + self.pos_embed[:, :self.first_l, :]
            + self.scale_embed(self.lvl_1l[:self.first_l]).unsqueeze(0)
        )
        if cond_blc_wo_first_l is None:
            h = sos
        else:
            cond_h = (
                self.word_embed(cond_blc_wo_first_l.float())
                + self.pos_embed[:, self.first_l:, :]
                + self.scale_embed(self.lvl_1l[self.first_l:]).unsqueeze(0)
            )
            h = torch.cat([sos, cond_h], dim=1)

        seq = h.shape[1]
        mask = self.attn_mask[:, :, :seq, :seq]
        for blk in self.blocks:
            h = blk(h, cond_BD, attn_mask=mask)
        logits_cat = self.head(self.norm(h, cond_BD))
        return logits_cat, targets_cat

    @torch.no_grad()
    def sample_next_scale(
        self,
        cond_blc_wo_first_l: torch.Tensor | None,
        cond_BD: torch.Tensor,
        batch_size: int,
        stage_idx: int,
    ) -> torch.Tensor:
        return self._forward_stage(
            cond_blc_wo_first_l=cond_blc_wo_first_l,
            cond_BD=cond_BD,
            batch_size=batch_size,
            stage_idx=stage_idx,
        )
