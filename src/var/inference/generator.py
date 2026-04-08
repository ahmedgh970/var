import torch

from var.inference.sampler import sample_from_logits
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
    batch_size: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    start_token: int | None = None,
) -> list[torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    prefix = torch.zeros((batch_size, 0), device=device, dtype=torch.long)

    generated = []
    for si, pn in enumerate(model.patch_nums):
        logits = model.sample_next_scale(prefix_tokens=prefix, stage_idx=si)
        stage_tokens = sample_from_logits(
            logits.reshape(-1, logits.shape[-1]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).view(batch_size, pn * pn)

        if si == 0 and start_token is not None:
            stage_tokens[:, 0] = int(start_token)

        generated.append(stage_tokens)
        prefix = torch.cat([prefix, stage_tokens], dim=1)

    flat = torch.cat(generated, dim=1)
    return _unflatten_to_multiscale(flat, patch_nums=model.patch_nums)
