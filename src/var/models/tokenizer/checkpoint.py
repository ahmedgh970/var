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


def load_tokenizer_checkpoint(model: VQVAE, checkpoint_path: str, strict: bool = True):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _remap_official_var_keys(state)
    state = _reconcile_quantizer_embedding_key(state, model)
    return model.load_state_dict(state, strict=strict)
