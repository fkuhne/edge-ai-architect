"""The backend abstraction -- ground rule 2, made concrete.

This is the interface that Phase 2 builds out and Phases 3, 4, and 6 consume. It
exists from day one so that no project accidentally grows a hard dependency on
one runtime.

The contract is deliberately small. A backend must be able to say whether it can
run here, load a model, run inference, and describe itself. Anything richer --
batching, streaming, KV-cache control -- belongs to a specific backend, not to
the interface everyone codes against.

The rule that makes this useful: **`is_available()` must never raise.** Backend
discovery runs on every platform, and a Core ML import error on a Linux box has
to be a `False`, not a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendInfo:
    """Self-description recorded alongside every benchmark row."""

    name: str
    version: str | None
    device: str
    device_name: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        version = f" {self.version}" if self.version else ""
        return f"{self.name}{version} on {self.device} ({self.device_name})"


class BackendUnavailable(RuntimeError):
    """Raised when a backend is instantiated on a platform that cannot run it."""


@runtime_checkable
class Runner(Protocol):
    """What every inference backend must provide."""

    @classmethod
    def backend_name(cls) -> str:
        """Short stable id, e.g. "torch", "onnx", "coreml"."""
        ...

    @classmethod
    def is_available(cls) -> bool:
        """True when this backend can run on this machine. Must not raise."""
        ...

    def load(self, model_ref: str, **kwargs: Any) -> None:
        """Prepare a model for inference."""
        ...

    def infer(self, inputs: Any) -> Any:
        """Run one forward pass."""
        ...

    def info(self) -> BackendInfo:
        """Describe the loaded backend."""
        ...

    def unload(self) -> None:
        """Release the model and any device memory it holds."""
        ...


class BaseRunner:
    """Shared scaffolding: availability guard, context manager, sane defaults.

    Subclasses implement `_load`, `infer`, and `_info_detail`.
    """

    name: str = "base"

    def __init__(self, device: str | None = None) -> None:
        if not self.is_available():
            raise BackendUnavailable(
                f"backend {self.backend_name()!r} is not available on this machine"
            )
        from edgebench import device as device_mod  # noqa: PLC0415

        self.device = device_mod.resolve(device)
        self.model_ref: str | None = None
        self._loaded = False

    @classmethod
    def backend_name(cls) -> str:
        return cls.name

    @classmethod
    def is_available(cls) -> bool:
        return False

    def load(self, model_ref: str, **kwargs: Any) -> None:
        self._load(model_ref, **kwargs)
        self.model_ref = model_ref
        self._loaded = True

    def _load(self, model_ref: str, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def infer(self, inputs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> BackendInfo:
        from edgebench import device as device_mod, provenance  # noqa: PLC0415

        return BackendInfo(
            name=self.backend_name(),
            version=provenance.backend_version(self.backend_name()),
            device=self.device,
            device_name=device_mod.info(self.device).name,
            detail=self._info_detail(),
        )

    def _info_detail(self) -> dict[str, Any]:
        return {"model_ref": self.model_ref, "loaded": self._loaded}

    def unload(self) -> None:
        self._loaded = False

    def __enter__(self) -> BaseRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
