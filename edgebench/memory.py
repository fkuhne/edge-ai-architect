"""Memory measurement.

On an 8 GB machine this is often the number that decides whether a model is
usable at all, so it is tracked with the same care as latency.

Two values are reported and they are not interchangeable:

- **peak RSS** -- process resident set size, the honest "did this fit" number.
- **accelerator peak** -- what the framework's allocator reserved. On CUDA this
  is separate memory. On this Mac it is *unified*: the same physical DRAM as
  RSS, which is why a 3B model competes with the OS rather than living beside it.
"""

from __future__ import annotations

import resource
import sys
import threading
from dataclasses import dataclass

_POLL_INTERVAL_S = 0.05


@dataclass(frozen=True)
class MemoryStats:
    peak_rss_mb: float
    accel_peak_mb: float | None = None
    note: str = ""

    def __str__(self) -> str:
        accel = f"  accel={self.accel_peak_mb:.0f}MB" if self.accel_peak_mb else ""
        return f"peak_rss={self.peak_rss_mb:.0f}MB{accel}"


def peak_rss_mb() -> float:
    """Process peak RSS in megabytes.

    `ru_maxrss` units differ by platform -- bytes on macOS/BSD, kilobytes on
    Linux -- which is a classic source of results that look 1024x wrong.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return raw / divisor


def current_rss_mb() -> float:
    """Instantaneous RSS, for the polling tracker."""
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return peak_rss_mb()


def _accel_peak_mb(device: str) -> tuple[float | None, str]:
    """Accelerator high-water mark, where the framework exposes one."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None, "PyTorch not installed"

    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2, ""

    if device == "mps" and hasattr(torch, "mps"):
        # MPS has no max-watermark API, only current allocation. Reported as-is
        # rather than silently passed off as a peak.
        fn = getattr(torch.mps, "current_allocated_memory", None)
        if fn is not None:
            return fn() / 1024**2, "MPS reports current allocation, not peak"
        return None, "MPS allocator stats unavailable"

    return None, ""


def reset_accel_peak(device: str) -> None:
    """Reset the accelerator watermark so a measurement isn't polluted by setup."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


class MemoryTracker:
    """Context manager sampling RSS on a background thread.

    `ru_maxrss` is monotonic for the whole process lifetime, so it cannot
    attribute a peak to one phase of a long run. Polling gives a per-window
    figure, which is what makes stage-by-stage pipeline numbers possible in
    Phase 4.
    """

    def __init__(self, device: str | None = None, poll: bool = True) -> None:
        from . import device as device_mod  # noqa: PLC0415

        self.device = device or device_mod.resolve()
        self._poll = poll
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats: MemoryStats | None = None

    def __enter__(self) -> MemoryTracker:
        reset_accel_peak(self.device)
        self._peak_mb = current_rss_mb()
        if self._poll:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_mb = max(self._peak_mb, current_rss_mb())
            self._stop.wait(_POLL_INTERVAL_S)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._peak_mb = max(self._peak_mb, current_rss_mb())

        accel_mb, note = _accel_peak_mb(self.device)
        self.stats = MemoryStats(
            peak_rss_mb=self._peak_mb,
            accel_peak_mb=accel_mb,
            note=note,
        )
