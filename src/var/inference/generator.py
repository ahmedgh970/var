import torch

from var.inference.sampler import sample_from_logits
from var.models.var.var_model import VARModel


def _build_scale_ids(batch_size: int, patch_nums: tuple[int, ...], device: torch.device) -> torch.Tensor:
    chunks = []
    for scale_id, pn in enumerate(patch_nums):
        chunks.append(torch.full((batch_size, pn * pn), fill_value=scale_id, device=device, dtype=torch.long))
    return torch.cat(chunks, dim=1)


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

    full_scale_ids = _build_scale_ids(batch_size=batch_size, patch_nums=model.patch_nums, device=device)

    if start_token is None:
        tokens = torch.randint(0, model.vocab_size, (batch_size, 1), device=device)
    else:
        tokens = torch.full((batch_size, 1), int(start_token), device=device, dtype=torch.long)

    for _ in range(model.seq_len - 1):
        cur_len = tokens.shape[1]
        scale_x = full_scale_ids[:, :cur_len]
        logits = model.forward_tokens(tokens, scale_x)
        next_token = sample_from_logits(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)

    return _unflatten_to_multiscale(tokens, patch_nums=model.patch_nums)
