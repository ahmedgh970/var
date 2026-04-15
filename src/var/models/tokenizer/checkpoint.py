from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any

import torch

from .vqvae import VQVAE


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _remap_official_var_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = OrderedDict()
    for k, v in state.items():
        nk = k
        if nk.startswith("quantize."):
            nk = "quantizer." + nk[len("quantize.") :]

        # Encoder naming diffs (official VAR -> 10-var).
        nk = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.\2.", nk)
        nk = re.sub(r"^encoder\.down\.(\d+)\.attn\.(\d+)\.", r"encoder.down_attn.\1.\2.", nk)
        nk = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.downsample.\1.", nk)
        nk = nk.replace("encoder.mid.block_1.", "encoder.mid_block1.")
        nk = nk.replace("encoder.mid.block_2.", "encoder.mid_block2.")
        nk = nk.replace("encoder.mid.attn_1.", "encoder.mid_attn.")

        # Decoder naming diffs (official VAR -> 10-var).
        nk = re.sub(r"^decoder\.up\.(\d+)\.block\.(\d+)\.", r"decoder.up_blocks.\1.\2.", nk)
        nk = re.sub(r"^decoder\.up\.(\d+)\.attn\.(\d+)\.", r"decoder.up_attn.\1.\2.", nk)
        nk = re.sub(r"^decoder\.up\.(\d+)\.upsample\.", r"decoder.upsample.\1.", nk)
        nk = nk.replace("decoder.mid.block_1.", "decoder.mid_block1.")
        nk = nk.replace("decoder.mid.block_2.", "decoder.mid_block2.")
        nk = nk.replace("decoder.mid.attn_1.", "decoder.mid_attn.")

        # Module field naming diffs.
        nk = nk.replace(".nin_shortcut.", ".skip.")
        nk = nk.replace(".proj_out.", ".proj.")

        remapped[nk] = v
    # Official VAR stores this training-only buffer; 10-var tokenizer does not.
    remapped.pop("quantizer.ema_vocab_hit_SV", None)
    return dict(remapped)


def _reconcile_quantizer_embedding_key(state: dict[str, torch.Tensor], model: VQVAE) -> dict[str, torch.Tensor]:
    state = dict(state)
    model_keys = model.state_dict().keys()

    has_single_key = "quantizer.embedding.weight" in state
    has_multi_key = "quantizer.quantizer.embedding.weight" in state
    expects_single = "quantizer.embedding.weight" in model_keys
    expects_multi = "quantizer.quantizer.embedding.weight" in model_keys

    if has_single_key and expects_multi:
        state["quantizer.quantizer.embedding.weight"] = state.pop("quantizer.embedding.weight")
    elif has_multi_key and expects_single:
        state["quantizer.embedding.weight"] = state.pop("quantizer.quantizer.embedding.weight")
    return state


def _validate_remap(
    state: dict[str, torch.Tensor],
    model: VQVAE,
    original_keys: set[str],
) -> None:
    """Raise a descriptive error when the remapped state dict doesn't match the model.

    Called before load_state_dict so the error message shows clearly what the
    key remapping produced instead of PyTorch's generic missing/unexpected-key dump.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if not missing and not unexpected:
        return

    added_by_remap = ckpt_keys - original_keys
    removed_by_remap = original_keys - ckpt_keys

    lines = [
        f"Checkpoint key mismatch after remapping "
        f"({len(missing)} missing, {len(unexpected)} unexpected).",
    ]
    if missing:
        lines.append(f"  Keys missing from checkpoint after remap ({len(missing)}):")
        for k in sorted(missing)[:20]:
            lines.append(f"    - {k}")
        if len(missing) > 20:
            lines.append(f"    ... and {len(missing) - 20} more")
    if unexpected:
        lines.append(f"  Unexpected keys in checkpoint after remap ({len(unexpected)}):")
        for k in sorted(unexpected)[:20]:
            lines.append(f"    - {k}")
        if len(unexpected) > 20:
            lines.append(f"    ... and {len(unexpected) - 20} more")
    if added_by_remap:
        lines.append(f"  Keys introduced by remap (new names, {len(added_by_remap)} total):")
        for k in sorted(added_by_remap)[:10]:
            lines.append(f"    + {k}")
    if removed_by_remap:
        lines.append(f"  Keys removed by remap (old names, {len(removed_by_remap)} total):")
        for k in sorted(removed_by_remap)[:10]:
            lines.append(f"    - {k}")
    raise KeyError("\n".join(lines))


def load_tokenizer_checkpoint(model: VQVAE, checkpoint_path: str, strict: bool = True):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original_state = _extract_state_dict(ckpt)
    original_keys = set(original_state.keys())

    state = _remap_official_var_keys(original_state)
    state = _reconcile_quantizer_embedding_key(state, model)

    if strict:
        _validate_remap(state, model, original_keys)

    return model.load_state_dict(state, strict=strict)
