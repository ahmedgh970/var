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
        drop_path_rate: float = 0.0,
        attn_l2_norm: bool = True,
        init_head: float = 0.02,
        init_std: float = -1.0,
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

        self.word_embed = nn.Linear(cvae_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, dim))
        self.scale_embed = nn.Embedding(self.num_scales, dim)
        self.pos_start = nn.Parameter(torch.zeros(1, self.first_l, dim))

        dpr = torch.linspace(0.0, float(drop_path_rate), depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                    attn_l2_norm=attn_l2_norm,
                )
                for i in range(depth)
            ]
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
        self.init_weights(
            init_head=init_head,
            init_std=init_std,
        )

    def kv_caching(self, enabled: bool):
        for blk in self.blocks:
            if hasattr(blk, "kv_caching"):
                blk.kv_caching(enabled)

    def init_weights(
        self,
        init_head: float = 0.02,
        init_std: float = -1.0,
    ):
        if init_std < 0:
            init_std = (1.0 / self.head.in_features / 3.0) ** 0.5

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

    def _forward_stage_with_var_input(
        self,
        cond_blc_wo_first_l: torch.Tensor | None,
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
                    + self.pos_embed[:, : self.first_l, :]
                    + self.scale_embed(self.lvl_1l[: self.first_l]).unsqueeze(0)
                )
            else:
                if cond_blc_wo_first_l is None:
                    raise ValueError("cond_blc_wo_first_l is required for stages after the first one")
                c_bg = bg - self.first_l
                c_ed = ed - self.first_l
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
        if use_cached_infer:
            logits = self.head(h[:, :stage_len, :])
        else:
            logits = self.head(h[:, bg:ed, :])
        #print(f"Stage {stage_idx}: logits shape {logits.shape}")
        return logits

    def forward(
        self,
        ms_tokens: list[torch.Tensor],
        cond_blc_wo_first_l: torch.Tensor | None,
    ):
        flat_scales = [t.view(t.shape[0], -1) for t in ms_tokens]
        b = flat_scales[0].shape[0]
        logits_per_stage = []
        targets_per_stage = []
        loss_sum = 0.0
        num_tokens = 0

        for si, target in enumerate(flat_scales):
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
                label_smoothing=0.1,
            )
            num_tokens += target.numel()

        logits_cat = torch.cat(logits_per_stage, dim=1)
        targets_cat = torch.cat(targets_per_stage, dim=1)
        loss = loss_sum / max(1, num_tokens)
        return logits_cat, targets_cat, loss

    @torch.no_grad()
    def sample_next_scale_with_var_input(
        self,
        cond_blc_wo_first_l: torch.Tensor | None,
        batch_size: int,
        stage_idx: int,
    ) -> torch.Tensor:
        if cond_blc_wo_first_l is None and stage_idx > 0:
            raise ValueError("cond_blc_wo_first_l is required for stages after the first one")
        return self._forward_stage_with_var_input(
            cond_blc_wo_first_l=cond_blc_wo_first_l,
            batch_size=batch_size,
            stage_idx=stage_idx,
        )
