"""Power sampler discovery.

Callers use `get_sampler()` and never name a platform. Selection order puts the
accelerator-aware samplers first, since on a machine with a GPU the GPU is where
the interesting energy goes, and ends at `NullSampler`, which always succeeds.

Override with EDGEBENCH_POWER to pin a sampler (or disable measurement):

    EDGEBENCH_POWER=null uv run python -m edgebench.selftest
"""

from __future__ import annotations

import logging
import os

from .base import BaseSampler, PowerReading, PowerSampler
from .null import NullSampler

log = logging.getLogger(__name__)

ENV_VAR = "EDGEBENCH_POWER"


def _candidates() -> list[type[BaseSampler]]:
    """Sampler classes in selection order.

    Imported lazily and defensively: a sampler whose optional dependency is
    missing must not break discovery for the others.
    """
    found: list[type[BaseSampler]] = []

    try:
        from .nvml import NvmlSampler

        found.append(NvmlSampler)
    except ImportError:  # pragma: no cover
        log.debug("NVML sampler unavailable")

    try:
        from .powermetrics import PowerMetricsSampler

        found.append(PowerMetricsSampler)
    except ImportError:  # pragma: no cover
        log.debug("powermetrics sampler unavailable")

    try:
        from .rapl import RaplSampler

        found.append(RaplSampler)
    except ImportError:  # pragma: no cover
        log.debug("RAPL sampler unavailable")

    found.append(NullSampler)
    return found


def available() -> list[str]:
    """Names of samplers that could run here, best first."""
    return [c.name for c in _candidates() if c.is_available()]


def get_sampler(name: str | None = None) -> BaseSampler:
    """Return the best usable sampler, or the one named.

    Never raises for an unsupported platform: an explicit request that cannot be
    honoured logs a warning and falls back, because losing power data should not
    cost a whole benchmark run.
    """
    requested = name or os.environ.get(ENV_VAR)
    candidates = _candidates()

    if requested:
        requested = requested.strip().lower()
        for cls in candidates:
            if cls.name == requested:
                if cls.is_available():
                    return cls()
                log.warning(
                    "power sampler %r was requested but is not available here; "
                    "falling back to latency + memory only",
                    requested,
                )
                return NullSampler()
        log.warning("unknown power sampler %r; known: %s", requested,
                    [c.name for c in candidates])
        return NullSampler()

    for cls in candidates:
        if cls.is_available():
            return cls()
    return NullSampler()  # unreachable: NullSampler is always available


__all__ = [
    "BaseSampler",
    "NullSampler",
    "PowerReading",
    "PowerSampler",
    "available",
    "get_sampler",
]
