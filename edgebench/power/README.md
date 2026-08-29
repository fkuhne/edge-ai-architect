# edgebench/power — measuring watts on four different kinds of machine

Power is the least portable thing this whole curriculum measures, and this package is a small case study in handling that honestly instead of pretending it's not a problem.

## Why there's no single "get power" function

Every platform exposes energy/power information differently, if it exposes it at all:

- **NVIDIA GPUs** expose an *instantaneous* power draw through NVML (`nvmlDeviceGetPowerUsage`). There's no built-in "average over the last N seconds" — you get one number, right now, and averaging is your job.
- **Linux CPUs** (Intel, and some AMD) expose RAPL: cumulative *energy* counters in sysfs (`/sys/class/powercap/intel-rapl:*/energy_uj`), which increment monotonically and periodically wrap around. Energy over a window is just `end − start`, corrected for wraparound — no sampling error, because it isn't sampled.
- **macOS** has no public power API at all. `powermetrics` is a system tool that prints periodic text reports to stdout, requires root, and is the *only* thing on this machine that can see the Apple Neural Engine's power draw separately from the CPU and GPU — which is exactly why it's worth the sudo hassle in Phase 2.
- **Everything else** — a Linux box without readable RAPL counters, a Mac without passwordless sudo configured, a CI runner — gets nothing. `null.py` is that "nothing," made explicit instead of silently absent.

Four platforms, four completely different measurement mechanisms (poll-and-average, true energy counter, parse-a-CLI-tool's-text-output, admit-defeat). One `PowerSampler` protocol in `base.py` hides all four behind `start()` / `stop()`.

## The pattern: Strategy, via a `Protocol`

`base.py` doesn't define a base class every sampler must inherit — it defines a `typing.Protocol`, a structural contract: anything with the right methods satisfies it, inheritance optional. `BaseSampler` is a convenience implementation of the bookkeeping every sampler needs (mostly: tracking elapsed time), not a requirement.

The one rule every implementation obeys, and the one worth understanding deeply: **`is_available()` must never raise.** `power/__init__.py`'s `get_sampler()` calls `is_available()` on every candidate, on every platform, on every run — including platforms where the underlying package (`pynvml`, for instance) isn't even installed. If `NvmlSampler.is_available()` raised an `ImportError` instead of catching it and returning `False`, benchmarking on this Mac would crash before it ever got to the sampler that actually works here. Go read `nvml.py`'s `is_available()` — note the bare `except Exception` around the whole NVML probe. That's not sloppy error handling; on this specific method, it's the point.

## Reading order

1. `base.py` — the contract. Small; read it first.
2. `null.py` — the trivial case, and worth noticing that "no data" is represented as `avg_power_w = None`, not `0.0`. Zero would claim a measurement was taken and came out to zero watts, which is a different (false) claim.
3. `rapl.py` — the cleanest real implementation, because RAPL's cumulative counters mean there's no polling thread to reason about, just a before/after subtraction with a wraparound guard.
4. `nvml.py` — introduces polling: NVML only gives an instantaneous reading, so this one runs a background thread sampling every 100 ms and averages afterward. Compare its shape to `rapl.py`'s and notice why the shapes differ — it's a direct consequence of what each underlying API actually gives you.
5. `powermetrics.py` — the messiest one, because it's shelling out to a CLI tool, parsing text with regexes, and managing a subprocess and a temp file. Worth reading once for the mechanics, but the parsing approach is not something to imitate elsewhere in this codebase — it's what you're stuck with when a platform's only interface is a human-readable report.
6. `__init__.py` — ties the four together: `_candidates()` builds the preference-ordered list (defensively — an `ImportError` from one sampler module must not take down discovery for the others), `get_sampler()` walks it.

## Comprehension check

- Why is a *poll-and-average* approach (NVML) necessary for GPU power but not for CPU package energy on Linux (RAPL)?
- What would `NullSampler.stop()` returning `avg_power_w=0.0` instead of `None` cause downstream, in `edgebench/store.py` or a Pareto plot?
- `PowerMetricsSampler.is_available()` checks `sudo -n true` rather than just checking whether the `powermetrics` binary exists. Why isn't binary-exists enough?
- If you were adding a fifth sampler for, say, a Raspberry Pi's onboard power monitor, what two methods would you have to implement, and what's the one behavioral rule you couldn't violate no matter how the underlying hardware API behaves?
