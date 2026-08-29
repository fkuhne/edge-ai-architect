# Phase 0 — Bootstrap & `edgebench`

**Week 1** · *How do I measure anything credibly, on any machine?*

Nothing else in this curriculum is trustworthy without this phase. Every later
claim — "quantization cost 1.2% accuracy for a 3.4x speedup" — is only as good as
the harness that produced it.

## Theory

Short phase; read while building.

- **MLPerf Inference** methodology — skim how the industry makes claims
  comparable: fixed scenarios, warmup rules, percentile reporting rather than
  means. This is why `edgebench` reports p95 and p99, not just an average.
- *Machine Learning Systems* (Reddi), **Ch. on benchmarking** — free at
  [mlsysbook.ai](https://mlsysbook.ai).

## Build

`edgebench/`, already scaffolded. What it does and why:

| Module | Responsibility | The subtlety |
|---|---|---|
| `device.py` | Resolve `cuda` → `mps` → `cpu` | An unavailable explicit request **raises** rather than downgrading. A benchmark that quietly ran on CPU is worse than one that failed. |
| `timing.py` | Latency percentiles, TTFT, tokens/sec | GPU work is async. Timing without a synchronise measures kernel *dispatch*. |
| `memory.py` | Peak RSS + accelerator watermark | `ru_maxrss` is **bytes on macOS, kilobytes on Linux** — the classic 1024x error. |
| `power/` | Pluggable: nvml, rapl, powermetrics, null | Falls back to `null` rather than failing. |
| `store.py` | One wide SQLite table | Flat on purpose: cross-phase comparison beats normalisation here. |
| `report.py` | Tables (no deps) and Pareto plots | Table path stays dependency-free so inspecting results never costs an install. |

## Done when

```bash
make selftest     # all checks pass
make check        # passes again under EDGEBENCH_DEVICE=cpu and EDGEBENCH_POWER=null
make report       # the row is in bench.db and renders
```

Specifically:

- [x] One command benchmarks a workload and writes a **complete** row: latency,
      peak memory, power (or an explicit "unavailable"), quality, provenance.
- [x] The same command runs unmodified with `EDGEBENCH_DEVICE=cpu`.
- [x] A missing power sampler degrades to `null` with a note, never an exception.
- [x] `runners.describe()` reports backend availability without importing
      anything that isn't installed.
- [ ] `git log` has at least one commit, so `git_sha` is populated in new rows.

## Optional: enable power measurement

`powermetrics` needs sudo, and `edgebench` declines to use it if sudo would
prompt — an unattended benchmark must not block on a password. To enable it:

```bash
sudo visudo -f /etc/sudoers.d/powermetrics
# add, replacing the username:
#   felipekuhne ALL=(root) NOPASSWD: /usr/bin/powermetrics
```

Then `python -c "from edgebench import power; print(power.available())"` should
list `powermetrics` first. This matters in Phase 2: the ANE power reading is the
cleanest available proof that a model really executed on the Neural Engine.

Skipping this is fine. The curriculum works on latency and memory alone; energy
just makes the Phase 1 Pareto plots more interesting.

## Before moving on

Run `make disk`. Phase 1 installs PyTorch (~2.5 GB) and downloads CIFAR-10
(~200 MB). Confirm the room exists.
