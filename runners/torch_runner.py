"""PyTorch backend -- the reference implementation of the Runner contract.

Portable by construction: the device comes from `edgebench.device.resolve()`,
so this same class runs on CUDA, MPS, or CPU with no branching. It is the
baseline every other backend in Phase 2 is measured against.
"""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRunner


@register
class TorchRunner(BaseRunner):
    name = "torch"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        self.model = None
        self._torch = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def _load(self, model_ref: str, *, module: Any = None, **kwargs: Any) -> None:
        """Load a model.

        `module` accepts an already-constructed `nn.Module`, which is how Phase 1
        hands over the variants it just trained. Otherwise `model_ref` is treated
        as a path to a serialized model.
        """
        import torch  # noqa: PLC0415

        self._torch = torch
        dev = torch.device(self.device)

        if module is not None:
            self.model = module
        else:
            # weights_only=False: these are our own artifacts, and Phase 1
            # variants carry the module structure, not just a state dict.
            self.model = torch.load(model_ref, map_location=dev, weights_only=False)

        self.model = self.model.to(dev).eval()

    def infer(self, inputs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("no model loaded; call load() first")
        with self._torch.no_grad():
            return self.model(inputs)

    def to_device(self, tensor: Any) -> Any:
        """Move a tensor to this runner's device."""
        return tensor.to(self._torch.device(self.device))

    def _info_detail(self) -> dict[str, Any]:
        detail = super()._info_detail()
        if self.model is not None:
            detail["parameters"] = sum(p.numel() for p in self.model.parameters())
        return detail

    def unload(self) -> None:
        self.model = None
        if self._torch is not None:
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(self._torch, "mps"):
                self._torch.mps.empty_cache()
        super().unload()
