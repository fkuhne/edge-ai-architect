"""edgebench -- portable measurement for the edge-ai-architect curriculum.

Every phase of this project reports through here, so that a pruned CNN from
month one and a 4-bit LLM from month four land in the same table on the same
axes. The library holds four rules:

1. **Nothing platform-specific escapes.** Device and power selection happen at
   runtime behind `device.resolve()` and `power.get_sampler()`.
2. **Speed is never reported alone.** Every row has room for a quality metric.
3. **Degrade, don't fail.** A missing power sampler costs power data, not a run.
4. **Rows are self-describing.** Model hash, backend version, and git SHA travel
   with the numbers.

Typical use::

    import edgebench

    result = edgebench.benchmark(
        lambda: model(batch),
        label="resnet8-int8-ptq",
        project="p01-compress-cifar",
        backend="torch",
        quantization="int8",
        quality=("accuracy", 0.897),
    )
    print(result.summary())
"""

from __future__ import annotations

from collections.abc import Callable

from . import device, memory, power, provenance, timing
from .store import DEFAULT_DB, Result, Store
from .timing import GenerationTimer, LatencyStats, TokenStats

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DB",
    "GenerationTimer",
    "LatencyStats",
    "Result",
    "Store",
    "TokenStats",
    "benchmark",
    "device",
    "memory",
    "power",
    "provenance",
    "record",
    "timing",
]


def _base_result(
    *,
    label: str,
    project: str | None,
    model_ref: str | None,
    quantization: str | None,
    backend: str | None,
    dev: str,
    notes: str | None,
    extra: dict | None,
    quality: tuple[str, float] | None,
) -> Result:
    """Assemble the provenance-bearing shell every result shares."""
    prov = provenance.capture()
    dev_info = device.info(dev)

    model_hash = model_bytes = None
    if model_ref:
        model_hash, model_bytes = provenance.hash_path(model_ref)

    return Result(
        label=label,
        project=project,
        model_ref=model_ref,
        model_hash=model_hash,
        model_bytes=model_bytes,
        quantization=quantization,
        backend=backend,
        backend_version=provenance.backend_version(backend) if backend else None,
        device=dev,
        device_name=dev_info.name,
        quality_metric=quality[0] if quality else None,
        quality_value=quality[1] if quality else None,
        git_sha=prov.git_sha,
        git_dirty=prov.git_dirty,
        platform=prov.platform,
        python_version=prov.python_version,
        notes=notes,
        extra=extra or {},
    )


def benchmark(
    fn: Callable[[], object],
    *,
    label: str,
    project: str | None = None,
    model_ref: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    device_kind: str | None = None,
    batch_size: int | None = None,
    iters: int = 50,
    warmup: int = 10,
    quality: tuple[str, float] | None = None,
    notes: str | None = None,
    extra: dict | None = None,
    store: Store | None | bool = None,
) -> Result:
    """Measure `fn`, record latency/memory/power, and persist the row.

    `store=False` skips persistence (useful while iterating); `None` uses the
    default database; a `Store` writes there.
    """
    dev = device.resolve(device_kind)
    sampler = power.get_sampler()
    sync = timing.make_sync(dev)

    with memory.MemoryTracker(dev) as mem:
        sampler.start()
        try:
            latency = timing.measure_latency(
                fn, iters=iters, warmup=warmup, sync=sync, device=dev
            )
        finally:
            reading = sampler.stop()

    result = _base_result(
        label=label, project=project, model_ref=model_ref,
        quantization=quantization, backend=backend, dev=dev,
        notes=notes, extra=extra, quality=quality,
    )
    result.batch_size = batch_size
    result.iters = iters
    result.warmup = warmup
    result.lat_mean_ms = latency.mean_ms
    result.lat_p50_ms = latency.p50_ms
    result.lat_p95_ms = latency.p95_ms
    result.lat_p99_ms = latency.p99_ms
    result.lat_stdev_ms = latency.stdev_ms

    _attach_env(result, mem.stats, reading)
    _persist(result, store)
    return result


def record(
    *,
    label: str,
    project: str | None = None,
    latency: LatencyStats | None = None,
    tokens: TokenStats | None = None,
    mem_stats: memory.MemoryStats | None = None,
    power_reading: power.PowerReading | None = None,
    model_ref: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    device_kind: str | None = None,
    quality: tuple[str, float] | None = None,
    notes: str | None = None,
    extra: dict | None = None,
    store: Store | None | bool = None,
) -> Result:
    """Persist a result assembled by the caller.

    `benchmark()` assumes a cheap repeatable callable. Token generation and
    multi-stage pipelines are neither, so those measure themselves with
    `GenerationTimer` / `MemoryTracker` and hand the pieces here.
    """
    dev = device.resolve(device_kind)
    result = _base_result(
        label=label, project=project, model_ref=model_ref,
        quantization=quantization, backend=backend, dev=dev,
        notes=notes, extra=extra, quality=quality,
    )

    if latency is not None:
        result.iters = latency.n
        result.lat_mean_ms = latency.mean_ms
        result.lat_p50_ms = latency.p50_ms
        result.lat_p95_ms = latency.p95_ms
        result.lat_p99_ms = latency.p99_ms
        result.lat_stdev_ms = latency.stdev_ms

    if tokens is not None:
        result.ttft_ms = tokens.ttft_ms
        result.tokens_per_s = tokens.tokens_per_s

    _attach_env(result, mem_stats, power_reading)
    _persist(result, store)
    return result


def _attach_env(
    result: Result,
    mem_stats: memory.MemoryStats | None,
    reading: power.PowerReading | None,
) -> None:
    if mem_stats is not None:
        result.peak_rss_mb = mem_stats.peak_rss_mb
        result.accel_peak_mb = mem_stats.accel_peak_mb
        if mem_stats.note:
            result.extra["memory_note"] = mem_stats.note

    if reading is not None:
        result.power_sampler = reading.sampler
        result.avg_power_w = reading.avg_power_w
        result.energy_j = reading.energy_j
        if reading.components_w:
            result.extra["power_components_w"] = reading.components_w
        if reading.note:
            result.extra["power_note"] = reading.note


def _persist(result: Result, store: Store | None | bool) -> None:
    if store is False:
        return
    if isinstance(store, Store):
        store.add(result)
        return
    with Store() as db:
        db.add(result)
