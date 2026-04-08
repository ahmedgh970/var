import torch

from var.inference.sampler import sample_from_logits
from var.models.tokenizer.vqvae import VQVAE
from var.models.var.var_model import VARModel

def _unflatten_to_multiscale(flat_tokens: torch.Tensor, patch_nums: tuple[int, ...]) -> list[torch.Tensor]:
    ms_tokens = []
    offset = 0
    for pn in patch_nums:
        n = pn * pn
        chunk = flat_tokens[:, offset : offset + n].reshape(flat_tokens.shape[0], pn, pn)
        ms_tokens.append(chunk)
        offset += n
    return ms_tokens


@torch.no_grad()
def generate_token_indices(
    model: VARModel,
    tokenizer: VQVAE,
    batch_size: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    start_token: int | None = None,
) -> list[torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    tokenizer.eval()

    # Build conditioning progressively, as in official VAR:
    # after each predicted scale, update f_hat and derive next-scale map.
    cond_chunks = []
    cond_blc = None
    cvae_dim = tokenizer.quantizer.quantizer.embed_dim if hasattr(tokenizer.quantizer, "quantizer") else tokenizer.quantizer.embed_dim
    hw = model.patch_nums[-1]
    f_hat = torch.zeros((batch_size, cvae_dim, hw, hw), device=device)

    generated = []
    for si, pn in enumerate(model.patch_nums):
        logits = model.sample_next_scale_with_var_input(
            cond_blc_wo_first_l=cond_blc,
            batch_size=batch_size,
            stage_idx=si,
        )
        stage_tokens = sample_from_logits(
            logits.reshape(-1, logits.shape[-1]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).view(batch_size, pn * pn)

        if si == 0 and start_token is not None:
            stage_tokens[:, 0] = int(start_token)

        generated.append(stage_tokens)

        idx = stage_tokens.view(batch_size, pn, pn)
        if tokenizer.quantizer_type == "single":
            h_bchw = tokenizer.quantizer.decode(idx)
        else:
            h_bchw = tokenizer.quantizer.quantizer.decode(idx)

        f_hat, next_map = tokenizer.get_next_autoregressive_input(
            si=si,
            sn=len(model.patch_nums),
            f_hat=f_hat,
            h_bchw=h_bchw,
        )
        if si < len(model.patch_nums) - 1:
            cond_chunks.append(next_map.flatten(2).transpose(1, 2).contiguous())
            cond_blc = torch.cat(cond_chunks, dim=1)

    flat = torch.cat(generated, dim=1)
    return _unflatten_to_multiscale(flat, patch_nums=model.patch_nums)
