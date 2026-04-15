import torch
from torch import nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Only trainable parameters are tracked. Buffers (e.g. BN running stats)
    are left untouched — they belong to the live model and are always
    evaluated with the live model's buffers.

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        # after each optimizer step:
        ema.update(model)
        # for eval / sampling with EMA weights:
        ema.apply_shadow(model)
        try:
            ...
        finally:
            ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        for name, param in self._unwrap(model).named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in self._unwrap(model).named_parameters():
            if param.requires_grad and name in self._shadow:
                self._shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Swap model parameters with EMA shadow values; saves originals."""
        for name, param in self._unwrap(model).named_parameters():
            if param.requires_grad and name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def restore(self, model: nn.Module):
        """Restore parameters saved by apply_shadow."""
        for name, param in self._unwrap(model).named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        self._shadow = {k: v.clone() for k, v in state.items()}
