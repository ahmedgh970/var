import torch

from var.inference.sampler import sample_from_logits
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel


def _unflatten_to_multiscale(flat_tokens: torch.Tensor, patch_nums: tuple[int, ...]) -> list[torch.Tensor]:
    ms_tokens = []
    offset = 0
    for pn in patch_nums:
        n = pn * pn
        ms_tokens.append(flat_tokens[:, offset: offset + n].reshape(flat_tokens.shape[0], pn, pn))
        offset += n
    return ms_tokens


@torch.no_grad()
def generate_token_indices(
    model: VARModel,
    tokenizer: VQVAE,
    batch_size: int,
    class_labels: torch.Tensor,       # (B,) int tensor of class indices
    cfg_scale: float = 1.5,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> list[torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    tokenizer.eval()

    # With CFG: double the batch — [conditioned | unconditioned]
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        uncond = torch.full_like(class_labels, model.num_classes)
        labels_in = torch.cat([class_labels, uncond], dim=0)  # (2B,)
        eff_bs = batch_size * 2
    else:
        labels_in = class_labels
        eff_bs = batch_size

    cond_BD = model.class_emb(labels_in)  # (eff_bs, dim)

    cvae_dim = tokenizer.codebook.embed_dim
    hw = model.patch_nums[-1]
    f_hat = torch.zeros((eff_bs, cvae_dim, hw, hw), device=device)

    cond_chunks = []
    cond_blc = None
    generated = []

    model.kv_caching(True)
    for si, pn in enumerate(model.patch_nums):
        logits = model.sample_next_scale(
            cond_blc_wo_first_l=cond_blc,
            cond_BD=cond_BD,
            batch_size=eff_bs,
            stage_idx=si,
        )  # (eff_bs, pn*pn, vocab_size)

        if use_cfg:
            ratio = float(si) / float(max(1, len(model.patch_nums) - 1))
            t = cfg_scale * ratio
            logits = (1 + t) * logits[:batch_size] - t * logits[batch_size:]

        stage_tokens = sample_from_logits(
            logits.reshape(-1, logits.shape[-1]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).view(batch_size, pn * pn)
        generated.append(stage_tokens)

        # feed the same sampled tokens to both CFG paths for next-scale conditioning
        idx = stage_tokens.view(batch_size, pn, pn)
        if use_cfg:
            idx = idx.repeat(2, 1, 1)  # (2B, pn, pn)

        h_bchw = tokenizer.codebook.decode(idx)

        f_hat, next_map = tokenizer.get_next_autoregressive_input(
            si=si, sn=len(model.patch_nums), f_hat=f_hat, h_bchw=h_bchw,
        )
        if si < len(model.patch_nums) - 1:
            cond_chunks.append(next_map.flatten(2).transpose(1, 2).contiguous())
            cond_blc = torch.cat(cond_chunks, dim=1)
    model.kv_caching(False)

    return _unflatten_to_multiscale(torch.cat(generated, dim=1), patch_nums=model.patch_nums)
