"""MLX backend -- optional, Darwin only.

Apple's array framework, built for unified memory. Like Core ML, it is a
plug-in and never the interface.

Its role in the curriculum is comparative. Phase 3 runs the same quantized model
through llama.cpp (portable: Metal here, CUDA elsewhere) and through MLX
(Apple-only), and the gap between them is the price of portability on this
hardware -- a number worth knowing rather than assuming.

Fleshed out in **Phase 3**.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from . import register
from .base import BaseRunner


@register
class MlxRunner(BaseRunner):
    name = "mlx"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        self.model = None
        self.tokenizer = None

    @classmethod
    def is_available(cls) -> bool:
        # MLX requires Apple silicon specifically, not merely macOS.
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False
        try:
            import mlx.core  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def _load(self, model_ref: str, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MlxRunner is a Phase 3 deliverable -- see curriculum/03-llms.md. "
            "Phase 3 compares it against the portable llama.cpp/GGUF path."
        )

    def infer(self, inputs: Any) -> Any:
        raise NotImplementedError("MlxRunner is a Phase 3 deliverable")

    def _info_detail(self) -> dict[str, Any]:
        detail = super()._info_detail()
        detail["platform_specific"] = True
        detail["note"] = "optional plug-in; comparison baseline only"
        return detail
