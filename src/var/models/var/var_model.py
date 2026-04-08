import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock

class VARModel(nn.Module):
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_nums = tuple(patch_nums)
        self.num_scales = len(self.patch_nums)
        self.seq_len = sum(p * p for p in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2

        self.begin_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.word_embed = nn.Linear(cvae_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, dim))
        self.scale_embed = nn.Embedding(self.num_scales, dim)
        self.stage_query = nn.Embedding(self.num_scales, dim)
        self.pos_start = nn.Parameter(torch.zeros(1, self.first_l, dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        lvl_1l = torch.cat(
            [torch.full((pn * pn,), fill_value=si, dtype=torch.long) for si, pn in enumerate(self.patch_nums)],
            dim=0,
        )
        self.register_buffer("lvl_1l", lvl_1l)

        d = lvl_1l.view(1, self.seq_len, 1)
        d_t = d.transpose(1, 2)
        # allow attending to previous scales and current scale, block future scales
        attn_mask = (d >= d_t).view(1, 1, self.seq_len, self.seq_len)
        self.register_buffer("attn_mask", attn_mask)

    def _flatten_scales(self, ms_tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        return [t.view(t.shape[0], -1) for t in ms_tokens]

    def _forward_stage(self, prefix_tokens: torch.Tensor, stage_idx: int) -> torch.Tensor:
        b = prefix_tokens.shape[0]
        bg, ed = self.begin_ends[stage_idx]
        n_stage = ed - bg

        if bg > 0:
            prefix_h = (
                self.token_embed(prefix_tokens)
                + self.pos_embed[:, :bg, :]
                + self.scale_embed(self.lvl_1l[:bg]).unsqueeze(0)
            )
        else:
            prefix_h = self.pos_embed[:, :0, :].expand(b, 0, -1)

        stage_h = (
            self.stage_query.weight[stage_idx].view(1, 1, -1).expand(b, n_stage, -1)
            + self.pos_embed[:, bg:ed, :]
            + self.scale_embed(self.lvl_1l[bg:ed]).unsqueeze(0)
        )

        h = torch.cat([prefix_h, stage_h], dim=1)
        mask = self.attn_mask[:, :, :ed, :ed]

        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        h = self.norm(h)
        logits = self.head(h[:, bg:ed, :])
        return logits

    def _forward_stage_with_var_input(
        self,
        cond_blc_wo_first_l: torch.Tensor | None,
        batch_size: int,
        stage_idx: int,
    ) -> torch.Tensor:
        bg, ed = self.begin_ends[stage_idx]

        sos = (
            self.pos_start.expand(batch_size, self.first_l, -1)
            + self.pos_embed[:, : self.first_l, :]
            + self.scale_embed(self.lvl_1l[: self.first_l]).unsqueeze(0)
        )
        if ed == self.first_l:
            h = sos
        else:
            if cond_blc_wo_first_l is None:
                raise ValueError("cond_blc_wo_first_l is required for stages after the first one")
            cond = cond_blc_wo_first_l[:, : ed - self.first_l, :]
            cond_h = (
                self.word_embed(cond)
                + self.pos_embed[:, self.first_l:ed, :]
                + self.scale_embed(self.lvl_1l[self.first_l:ed]).unsqueeze(0)
            )
            h = torch.cat([sos, cond_h], dim=1)

        mask = self.attn_mask[:, :, :ed, :ed]
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        h = self.norm(h)
        logits = self.head(h[:, bg:ed, :])
        return logits

    def forward(
        self,
        ms_tokens: list[torch.Tensor],
        cond_blc_wo_first_l: torch.Tensor | None = None,
    ):
        flat_scales = self._flatten_scales(ms_tokens)
        b = flat_scales[0].shape[0]

        prefix_tokens = flat_scales[0].new_zeros((b, 0), dtype=torch.long)
        logits_per_stage = []
        targets_per_stage = []
        loss_sum = 0.0
        num_tokens = 0

        for si, target in enumerate(flat_scales):
            if cond_blc_wo_first_l is None:
                logits = self._forward_stage(prefix_tokens=prefix_tokens, stage_idx=si)
            else:
                logits = self._forward_stage_with_var_input(
                    cond_blc_wo_first_l=cond_blc_wo_first_l,
                    batch_size=b,
                    stage_idx=si,
                )
            logits_per_stage.append(logits)
            targets_per_stage.append(target)

            loss_sum = loss_sum + F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target.reshape(-1),
                reduction="sum",
            )
            num_tokens += target.numel()

            prefix_tokens = torch.cat([prefix_tokens, target], dim=1)

        logits_cat = torch.cat(logits_per_stage, dim=1)
        targets_cat = torch.cat(targets_per_stage, dim=1)
        loss = loss_sum / max(1, num_tokens)
        return logits_cat, targets_cat, loss

    @torch.no_grad()
    def sample_next_scale(
        self,
        prefix_tokens: torch.Tensor,
        stage_idx: int,
    ) -> torch.Tensor:
        return self._forward_stage(prefix_tokens=prefix_tokens, stage_idx=stage_idx)
