"""The power-sampling contract.

Power is the most platform-specific thing this library touches, so it is the
place ground rule 2 is most likely to be violated. The rule here: callers only
ever see `PowerSampler`. Whether that is NVML on a CUDA box, RAPL on a Linux
CPU, `powermetrics` on this Mac, or nothing at all is resolved at runtime.

A sampler that cannot run must report `is_available() -> False` rather than
raising, so an unsupported platform degrades to latency + memory instead of
failing the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PowerReading:
    """Result of one sampling window."""

    sampler: str
    avg_power_w: float | None
    energy_j: float | None
    duration_s: float
    #: Per-component breakdown where the platform exposes one, e.g.
    #: {"cpu": 3.1, "gpu": 8.4, "ane": 1.2}. Empty when unavailable.
    components_w: dict[str, float] | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.avg_power_w is not None


@runtime_checkable
class PowerSampler(Protocol):
    """Start/stop interface for a power measurement window."""

    #: Short stable id recorded in the results database.
    name: str

    @classmethod
    def is_available(cls) -> bool:
        """True when this sampler can run here, right now.

        Must be cheap and must never raise -- it is called during backend
        discovery on every platform.
        """
        ...

    def start(self) -> None:
        """Begin sampling. Must be idempotent-safe to call once per window."""
        ...

    def stop(self) -> PowerReading:
        """End sampling and return the reading."""
        ...


class BaseSampler:
    """Small shared base: timing bookkeeping every sampler needs."""

    name = "base"

    def __init__(self) -> None:
        self._t0: float | None = None

    @classmethod
    def is_available(cls) -> bool:
        return False

    def start(self) -> None:
        import time  # noqa: PLC0415

        self._t0 = time.perf_counter()

    def _elapsed(self) -> float:
        import time  # noqa: PLC0415

        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def stop(self) -> PowerReading:  # pragma: no cover - overridden
        raise NotImplementedError
