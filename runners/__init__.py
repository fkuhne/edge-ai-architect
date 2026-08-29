"""Backend registry.

Backends register themselves here and callers ask for them by name -- or, better,
iterate `available()` and benchmark everything the machine can actually run. That
loop is the Phase 2 deliverable: one script, every backend, no code changes.

Discovery is defensive. A backend whose optional dependency is missing is simply
absent from the registry; it never breaks discovery for the others.

    from runners import available, get

    for name in available():        # e.g. ["torch", "onnx"] here,
        runner = get(name)()        #      ["torch", "onnx", "coreml"] on a Mac
        ...                         #      with coremltools installed
"""

from __future__ import annotations

import logging

from .base import BackendInfo, BackendUnavailable, BaseRunner, Runner

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseRunner]] = {}


def register(cls: type[BaseRunner]) -> type[BaseRunner]:
    """Class decorator adding a backend to the registry."""
    _REGISTRY[cls.backend_name()] = cls
    return cls


def _discover() -> None:
    """Import backend modules, tolerating missing optional dependencies.

    Ordered portable-first: the backends that run everywhere come before the
    platform-specific ones, so `available()[0]` is always a safe default.
    """
    modules = (
        "torch_runner",   # portable: cuda | mps | cpu
        "onnx_runner",    # portable: CUDA | CoreML | CPU execution providers
        "coreml_runner",  # optional, Darwin only
        "mlx_runner",     # optional, Darwin only
    )
    for module in modules:
        try:
            __import__(f"{__name__}.{module}", fromlist=["*"])
        except ImportError as exc:
            log.debug("backend module %s unavailable: %s", module, exc)


def registered() -> list[str]:
    """Every backend module that imported, whether or not it can run here."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY)


def available() -> list[str]:
    """Backends that can actually run on this machine, portable ones first."""
    if not _REGISTRY:
        _discover()
    return [name for name in _REGISTRY if _REGISTRY[name].is_available()]


def get(name: str) -> type[BaseRunner]:
    """Look up a backend class by name."""
    if not _REGISTRY:
        _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def describe() -> str:
    """Human-readable availability report -- useful when a backend is missing
    and you need to know whether it failed to import or failed to be usable."""
    if not _REGISTRY:
        _discover()
    if not _REGISTRY:
        return "no backend modules imported"

    lines = []
    for name, cls in sorted(_REGISTRY.items()):
        ok = cls.is_available()
        lines.append(f"  {'available    ' if ok else 'not available'}  {name}")
    return "\n".join(lines)


__all__ = [
    "BackendInfo",
    "BackendUnavailable",
    "BaseRunner",
    "Runner",
    "available",
    "describe",
    "get",
    "register",
    "registered",
]
