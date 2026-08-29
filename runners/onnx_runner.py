"""ONNX Runtime backend -- the primary portable inference path.

The reason this backend matters more than the others: one exported graph runs
under CUDA, CoreML, TensorRT, OpenVINO, or plain CPU execution providers. It is
the closest thing in practice to the "deploy across diverse silicon without
rewrites" abstraction the curriculum is studying.

Fleshed out in **Phase 2**. `is_available()` and EP selection work today so
that backend discovery reports honestly before the phase begins.
"""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRunner

#: Execution providers in preference order. ORT silently falls back to the next
#: one when a provider is missing -- which is convenient and also exactly how
#: you end up unknowingly benchmarking the CPU. Phase 2 verifies the provider
#: that actually ran via `session.get_providers()`.
EP_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)


@register
class OnnxRunner(BaseRunner):
    name = "onnx"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        self.session = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    @classmethod
    def providers(cls) -> list[str]:
        """Execution providers this install offers, in preference order."""
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            return []
        offered = set(ort.get_available_providers())
        return [ep for ep in EP_PREFERENCE if ep in offered]

    def _load(self, model_ref: str, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OnnxRunner is a Phase 2 deliverable -- see curriculum/02-runtimes.md. "
            f"Providers available here: {self.providers()}"
        )

    def infer(self, inputs: Any) -> Any:
        raise NotImplementedError("OnnxRunner is a Phase 2 deliverable")

    def _info_detail(self) -> dict[str, Any]:
        detail = super()._info_detail()
        detail["providers_available"] = self.providers()
        if self.session is not None:
            detail["providers_active"] = self.session.get_providers()
        return detail
