"""Core ML backend -- optional, Darwin only.

Under ground rule 2 this is a *plug-in*, never the interface. Nothing in the
curriculum may depend on it, and every script that uses it must still run on a
Linux/CUDA box with this backend simply absent from `runners.available()`.

It earns inclusion for one reason: Core ML is the only way to reach the Apple
Neural Engine, and the ANE is a genuine NPU. Phase 2 uses it to study
accelerator residency -- which operators stay on the NPU and which silently fall
back to CPU. That is the same failure mode as an op with no CUDA kernel, learned
on the hardware that happens to be here.

Fleshed out in **Phase 2**.
"""

from __future__ import annotations

import sys
from typing import Any

from . import register
from .base import BaseRunner


@register
class CoreMLRunner(BaseRunner):
    name = "coreml"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        self.model = None

    @classmethod
    def is_available(cls) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import coremltools  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def _load(self, model_ref: str, *, compute_units: str = "ALL", **kwargs: Any) -> None:
        raise NotImplementedError(
            "CoreMLRunner is a Phase 2 deliverable -- see curriculum/02-runtimes.md. "
            "Phase 2 loads an .mlpackage with ct.ComputeUnit.<compute_units> and "
            "confirms ANE residency via powermetrics' ANE power reading."
        )

    def infer(self, inputs: Any) -> Any:
        raise NotImplementedError("CoreMLRunner is a Phase 2 deliverable")

    def _info_detail(self) -> dict[str, Any]:
        detail = super()._info_detail()
        detail["platform_specific"] = True
        detail["note"] = "optional plug-in; absent on non-Darwin platforms by design"
        return detail
