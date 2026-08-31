# Phase 0 — Bootstrap & `edgebench`

**Week 1** · *How do I measure anything credibly, on any machine?*

Nothing else in this curriculum is trustworthy without this phase. Every later
claim — "quantization cost 1.2% accuracy for a 3.4x speedup" — is only as good as
the harness that produced it.

## A transparency note

Phase 0 was fully implemented already — by Claude, in one pass, including
tests. That wasn't the right call for a project whose whole point is that you
build things yourself, and it won't happen again for Phases 1–7: those come as
scaffolding and guidance, not finished code, so the building is yours. See
[`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md) for the full reasoning.

`edgebench/` and `runners/` get to be the one exception, on the reasoning that
they're shared lab equipment — a measurement harness and a backend abstraction
every later phase leans on — rather than the subject being taught. Even so,
they're worth treating as a worked example, not a black box: read them with
the package READMEs open (`edgebench/README.md`, `edgebench/power/README.md`,
`runners/README.md`), and use the comprehension checkpoint below as a real
gate, not a formality. If you want the fuller version of this phase, pick one
module — `timing.py` is a good size — and reimplement it from scratch without
looking, then diff your version against the original.

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
- [x] `git log` has at least one commit, so `git_sha` is populated in new rows.
- [x] `make test` — 23 tests green, including AST checks that platform-specific
      imports have not escaped their modules.

## Comprehension checkpoint

Answer these in your own words — writing them into `notes/` is the point, not
just thinking them:

- Why does `resolve("cuda")` raise instead of returning `"cpu"` on this machine?
- What actually goes wrong if you delete the `sync()` call inside
  `measure_latency`'s loop?
- What's the unit bug that `peak_rss_mb()` specifically guards against, and
  what would your numbers look like if you got it backwards?
- Walk through, step by step, what happens when `power.get_sampler()` runs on
  a machine with no NVML, no readable RAPL counters, and no passwordless sudo
  for `powermetrics`. Why does it never raise?
- Why does `edgebench` report p50/p95/p99 instead of just a mean? Sketch a
  workload where the mean looks fine while p99 is terrible.
- `Result` carries an `extra` field alongside ~30 named columns. What's the
  tradeoff there, and why not just add a column for everything?

If any of these are shaky, reread the relevant module (or its README) before
treating Phase 0 as understood — the checklist above says the harness *works*;
this section is what says you know *why*.

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
