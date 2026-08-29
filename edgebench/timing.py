"""Latency measurement.

The one subtlety worth internalising early: GPU and MPS work is dispatched
asynchronously. Timing a call that only *enqueues* kernels measures the enqueue,
not the compute, and produces impossibly fast numbers. Every measurement here
runs through a device-appropriate synchronise, which is why `make_sync()` exists
and why it is resolved from the device rather than hardcoded.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from . import device as device_mod


@dataclass(frozen=True)
class LatencyStats:
    """Distribution of per-iteration wall time, in milliseconds."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"p50={self.p50_ms:.2f}ms  p95={self.p95_ms:.2f}ms  "
            f"p99={self.p99_ms:.2f}ms  (n={self.n})"
        )


def make_sync(device: str | None = None) -> Callable[[], None]:
    """Return a no-arg callable that blocks until queued device work completes.

    Falls back to a no-op on CPU or when PyTorch is absent, so callers never
    need to branch on platform.
    """
    kind = device or device_mod.resolve()
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return lambda: None

    if kind == "cuda":
        return torch.cuda.synchronize
    if kind == "mps" and hasattr(torch, "mps"):
        return torch.mps.synchronize
    return lambda: None


def summarize(samples_s: list[float]) -> LatencyStats:
    """Build stats from a list of per-iteration durations in seconds."""
    if not samples_s:
        raise ValueError("no timing samples collected")

    ms = sorted(s * 1000.0 for s in samples_s)
    return LatencyStats(
        n=len(ms),
        mean_ms=statistics.fmean(ms),
        p50_ms=_percentile(ms, 0.50),
        p95_ms=_percentile(ms, 0.95),
        p99_ms=_percentile(ms, 0.99),
        min_ms=ms[0],
        max_ms=ms[-1],
        stdev_ms=statistics.stdev(ms) if len(ms) > 1 else 0.0,
    )


def _percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile. Deliberately simple: with the iteration counts
    used here, interpolation would imply a precision the samples do not have."""
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    idx = min(int(round(q * (len(sorted_ms) - 1))), len(sorted_ms) - 1)
    return sorted_ms[idx]


def measure_latency(
    fn: Callable[[], object],
    *,
    iters: int = 50,
    warmup: int = 10,
    sync: Callable[[], None] | None = None,
    device: str | None = None,
) -> LatencyStats:
    """Time `fn` over `iters` iterations after `warmup` untimed ones.

    Warmup is not optional in practice: the first calls pay for lazy kernel
    compilation, memory-pool growth, and (on this machine) clock ramp-up. Timing
    them buries the steady-state number the benchmark is actually after.
    """
    sync = sync if sync is not None else make_sync(device)

    for _ in range(warmup):
        fn()
    sync()

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)

    return summarize(samples)


@dataclass
class TokenStats:
    """Generation timings for LLM work.

    Time-to-first-token and steady-state throughput are reported separately
    because they are governed by different things -- TTFT by prefill over the
    whole prompt, tokens/sec by memory bandwidth during decode -- and averaging
    them into one number hides the tradeoff the curriculum is studying.
    """

    ttft_ms: float | None = None
    tokens_generated: int = 0
    decode_s: float = 0.0

    @property
    def tokens_per_s(self) -> float | None:
        if self.tokens_generated <= 0 or self.decode_s <= 0:
            return None
        return self.tokens_generated / self.decode_s


class GenerationTimer:
    """Hand-rolled TTFT / throughput timer for token streams.

    Usage::

        timer = GenerationTimer()
        timer.start()
        for token in stream:
            timer.token()
        stats = timer.finish()
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self._t_first: float | None = None
        self._count = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._t_first = None
        self._count = 0

    def token(self) -> None:
        if self._t_first is None:
            self._t_first = time.perf_counter()
        self._count += 1

    def finish(self) -> TokenStats:
        end = time.perf_counter()
        if self._t0 is None:
            return TokenStats()
        if self._t_first is None:
            return TokenStats(ttft_ms=None, tokens_generated=0, decode_s=0.0)

        # Decode excludes prefill: the first token's arrival is the boundary.
        return TokenStats(
            ttft_ms=(self._t_first - self._t0) * 1000.0,
            tokens_generated=max(self._count - 1, 0),
            decode_s=max(end - self._t_first, 0.0),
        )
