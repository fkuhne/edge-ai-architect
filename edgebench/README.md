# edgebench

A portable measurement library — the thing every later phase in this curriculum reports through, so results stay comparable months and phases apart. Read this before treating the module as a black box; the design choices here are as much the lesson as any single number they produce. (New to how this repo works? Start with [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md) first.)

## The problem it solves

By month six of this curriculum there will be a pruned CIFAR classifier, a 4-bit quantized LLM, and an ONNX-exported vision transformer, all benchmarked at different times for different projects. `edgebench` exists so those three numbers can sit in one table and mean something next to each other — same units, same percentile definitions, same notion of what "the device" and "the backend" were at the time.

That requirement is what shapes every module below.

## Guided tour

### `device.py` — resolving "the device" without naming one

Ground rule 2 of this whole project is that nothing here is allowed to hardcode `mps` or `cuda`. `device.resolve()` is where that rule lives: it tries `cuda`, then `mps`, then `cpu`, and can be overridden with `EDGEBENCH_DEVICE`.

The detail worth sitting with: `resolve("cuda")` on this Mac **raises**, it does not fall back to CPU. Read `resolve()`'s docstring for why — the short version is that a benchmark which silently downgraded would produce a real number attached to the wrong device, and a wrong number that looks valid is worse than a crash.

### `timing.py` — why a stopwatch isn't enough on a GPU

`measure_latency()` looks like a simple loop with a stopwatch. The part that isn't simple is `make_sync()`, called before starting the clock and after every iteration. GPU (and MPS) work is dispatched *asynchronously* — a Python call that launches a kernel can return before the kernel has actually run. Time that call without a synchronize barrier and you're measuring how fast Python can hand off work, not how fast the model runs. This is a mistake every GPU benchmark gets wrong at least once; better to understand why the barrier is there than to memorize that it should be.

`GenerationTimer` exists for a related reason specific to LLMs: time-to-first-token (dominated by prefill, which is compute-bound) and tokens/sec (dominated by decode, which is memory-bandwidth-bound) are governed by different physical bottlenecks. Averaging them into one number would hide exactly the tradeoff Phase 3 is built to study.

### `memory.py` — the platform detail that silently corrupts results

`resource.getrusage().ru_maxrss` reports **bytes on macOS and BSD, kilobytes on Linux**. Get this backwards and every memory number you produce is off by 1024x in a way that often still looks plausible. `peak_rss_mb()` handles it once, here, so nothing downstream has to remember it.

The other subtlety: peak RSS (what the OS actually allocated to the process) and accelerator "peak memory" (what a framework's allocator reports reserving) are not the same number, and on this machine's *unified* memory they compete for the same physical DRAM in a way they wouldn't on a discrete GPU. `MemoryStats` keeps them as separate fields rather than collapsing them, on purpose.

### `power/` — the part of this library that's most platform-fragmented, made pluggable

There is no portable API for "how many watts is this process using." `power/base.py` defines the contract (`PowerSampler`: `start()`, `stop()`, and an `is_available()` that **must never raise**), and four implementations satisfy it: `nvml.py` (NVIDIA GPUs, the one that will actually run once this code moves to a CUDA box), `rapl.py` (Linux CPU energy via the kernel's RAPL counters), `powermetrics.py` (macOS, the only one that can see ANE power draw, needs sudo), and `null.py` (always available, measures nothing, and is the honest answer everywhere else). Full writeup in [`power/README.md`](power/README.md) — it's worth its own reading pass.

### `provenance.py` — making a number mean something later

A benchmark row six months old is worthless if you can't say what produced it. `capture()` grabs the git SHA (and whether the tree was dirty — a result from uncommitted code is a different kind of result), the platform, and the Python version. `hash_path()` content-addresses a model file or directory, capped at 64 MB of actual reading so hashing a multi-gigabyte GGUF doesn't dominate the benchmark it's describing.

### `store.py` — one wide table, on purpose

`Result` is a dataclass with roughly thirty fields, most of them nullable, all landing in one SQLite table. That's a real design choice against normalizing into several related tables: the entire point of this database is letting one query put a CNN's accuracy next to an LLM's tokens/sec, and joins across a normalized schema would make exactly that kind of ad-hoc comparison more tedious than it needs to be at this scale. `extra_json` is the escape hatch for anything that doesn't deserve a named column — the per-component power breakdown, for instance.

### `report.py` — tables with zero dependencies, plots with one

The table renderer (`render_table`) imports nothing beyond the standard library, deliberately: inspecting results should never require installing a plotting stack first. `plot_pareto()` is the one place `matplotlib` shows up, guarded by an `ImportError` that prints an actionable message instead of a traceback.

## Comprehension check

Before moving past this file, you should be able to answer, without looking back at the code:

- Why does `resolve("cuda")` raise instead of returning `"cpu"` on this machine?
- What actually goes wrong if you delete the `sync()` call inside `measure_latency`'s loop?
- What's the unit bug that `peak_rss_mb()` specifically guards against?
- Name the four power samplers and, for each, one platform fact that determines whether `is_available()` returns `True`.
- Why is `store.py` one wide table instead of several normalized ones?

If any of those are shaky, that's the signal to reread the relevant module rather than move on — this file was written to make the *why* legible, not just the *what*.

## If you want the harder version

Pick one module — `timing.py` is a good size, self-contained, and has real subtlety in it — and reimplement it from scratch in a scratch file, without looking at the original. Then diff your version against it. Where they disagree is usually where the actual learning is.
