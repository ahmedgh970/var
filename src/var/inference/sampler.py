import torch


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_sorted = cum_probs > top_p
    remove_sorted[..., 0] = False

    remove_mask = torch.zeros_like(remove_sorted, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    filtered = logits / temperature
    filtered = _apply_top_k(filtered, top_k=top_k)
    filtered = _apply_top_p(filtered, top_p=top_p)

    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
