"""End-to-end check that the harness works on this machine.

    uv run python -m edgebench.selftest
    EDGEBENCH_DEVICE=cpu uv run python -m edgebench.selftest

This is the Phase 0 acceptance test. It proves four things:

1. A benchmark runs and writes a complete, self-describing row.
2. Device resolution honours EDGEBENCH_DEVICE and refuses impossible requests.
3. The power sampler degrades to `null` rather than raising.
4. The whole thing works with **no ML framework installed at all** -- there is a
   stdlib fallback workload, so a fresh phase environment can be validated
   before spending 2.5 GB of a 15 GB disk on PyTorch.
"""

from __future__ import annotations

import argparse
import sys

import edgebench
from edgebench import device, power
from edgebench.store import Store


def _torch_workload():
    """A small conv stack -- shaped like real work, tiny enough to be instant."""
    try:
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
    except ImportError:
        return None

    dev = torch.device(device.resolve())
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(32, 10),
    ).to(dev).eval()
    batch = torch.randn(8, 3, 32, 32, device=dev)

    def run():
        with torch.no_grad():
            return model(batch)

    return run, "torch", "fp32", 8


def _stdlib_workload():
    """Pure-Python matmul. Slow by design -- it just has to be measurable."""
    import random  # noqa: PLC0415

    n = 40
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    b = [[random.random() for _ in range(n)] for _ in range(n)]

    def run():
        return [
            [sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]

    return run, "python", None, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="edgebench self-test")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-store", action="store_true", help="do not write to bench.db")
    args = parser.parse_args(argv)

    print("edgebench self-test")
    print("=" * 60)

    dev = device.resolve()
    info = device.info(dev)
    print(f"devices available : {device.available()}")
    print(f"device selected   : {info}")
    print(f"power samplers    : {power.available()}")

    sampler = power.get_sampler()
    print(f"power selected    : {sampler.name}")
    if sampler.name == "null":
        print("                    (latency + memory only -- expected without "
              "passwordless sudo or a CUDA GPU)")

    workload = _torch_workload() or _stdlib_workload()
    fn, backend, quantization, batch_size = workload
    print(f"workload          : {backend}"
          f"{' (PyTorch not installed -- stdlib fallback)' if backend == 'python' else ''}")
    print()

    result = edgebench.benchmark(
        fn,
        label=f"selftest-{backend}",
        project="p00-bootstrap",
        backend=backend,
        quantization=quantization,
        batch_size=batch_size,
        iters=args.iters,
        warmup=args.warmup,
        # A synthetic stand-in, so the quality column is exercised end to end.
        quality=("selftest_ok", 1.0),
        notes="edgebench self-test",
        store=False if args.no_store else None,
    )

    print(result.summary())
    print()

    checks = _verify(result, dev, store=not args.no_store)
    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed} check(s) failed")
        return 1
    print("all checks passed -- Phase 0 harness is working")
    return 0


def _verify(result, dev: str, store: bool) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "latency recorded",
        result.lat_p50_ms is not None and result.lat_p50_ms > 0,
        f"p50={result.lat_p50_ms:.3f}ms",
    ))
    checks.append((
        "memory recorded",
        result.peak_rss_mb is not None and result.peak_rss_mb > 0,
        f"peak_rss={result.peak_rss_mb:.0f}MB",
    ))
    checks.append((
        "power degraded cleanly",
        result.power_sampler is not None,
        f"sampler={result.power_sampler}"
        + (f", {result.avg_power_w:.2f}W" if result.avg_power_w else ", no data (expected)"),
    ))
    checks.append((
        "provenance captured",
        result.platform is not None and result.python_version is not None,
        f"git={result.git_sha or 'none'}{'*' if result.git_dirty else ''}, {result.platform}",
    ))
    checks.append((
        "device resolved",
        result.device == dev,
        f"{result.device} ({result.device_name})",
    ))

    # Rule 2 in action: an impossible device must be refused loudly, not
    # silently downgraded to CPU.
    bogus_rejected = False
    try:
        device.resolve("definitely-not-a-device")
    except RuntimeError:
        bogus_rejected = True
    checks.append((
        "bad device rejected",
        bogus_rejected,
        "unavailable device raises rather than silently downgrading",
    ))

    # CPU is always available, so this override must work on every platform.
    cpu_ok = device.resolve("cpu") == "cpu"
    checks.append(("cpu override works", cpu_ok, "EDGEBENCH_DEVICE=cpu is always honourable"))

    if store:
        with Store() as db:
            rows = db.query(project="p00-bootstrap", limit=1)
        checks.append((
            "row persisted",
            bool(rows),
            f"bench.db now holds {len(rows)} matching row(s)" if rows else "nothing written",
        ))

    return checks


if __name__ == "__main__":
    sys.exit(main())
