# runners — the backend abstraction

This package is the working answer to the question Phase 2 is built around: how does one model run on whatever accelerator happens to be available, without the calling code knowing or caring which one it got?

## The shape of the problem

By Phase 2 there will be at least two ways to run the same model on this machine — PyTorch on MPS, and (once you build it) ONNX Runtime through its CoreML execution provider — and potentially a third and fourth (Core ML directly, MLX). On a CUDA machine, the exact same code should offer PyTorch-on-CUDA and ONNX-Runtime-on-CUDA instead, with no edits. The benchmark script that compares them shouldn't contain a single `if backend == "coreml"`.

That's what `runners/` is for.

## The two files that matter most

**`base.py`** defines the contract. `Runner` is a `Protocol` — structural typing, the same pattern as `edgebench/power/base.py`'s `PowerSampler` — describing what any backend must provide: `backend_name()`, `is_available()`, `load()`, `infer()`, `info()`, `unload()`. `BaseRunner` is a concrete convenience class implementing the parts every backend needs regardless of what it wraps (device resolution in `__init__`, the context-manager protocol, a default `info()`). Notice how little `BaseRunner.__init__` does beyond calling `is_available()` as a guard and resolving a device — that smallness is deliberate; a fat base class would start encoding assumptions that don't hold for every backend.

**`torch_runner.py`** is the one fully working implementation, and it exists to be *read*, not just used — it's the pattern every backend you add in Phase 2 should follow. Notice that it doesn't mention `mps` or `cuda` anywhere except by calling `edgebench.device.resolve()`; the device-agnosticism isn't a special feature of this runner, it's what falls out of using the shared resolver instead of hand-rolling device logic per backend.

## The registry: `__init__.py`

`register` is a class decorator — `@register` above a `BaseRunner` subclass adds it to a module-level dict keyed by `backend_name()`. `_discover()` imports every backend module inside a `try/except ImportError`, so a machine without `mlx` installed simply ends up with `mlx` absent from the registry rather than a crash on import. `available()` filters the registry down to backends whose `is_available()` (that same never-raise rule from the power package) returns `True` **right now, on this machine**.

The payoff is the loop this whole package exists to enable:

```python
for name in runners.available():
    with runners.get(name)() as r:
        r.load(model_ref)
        result = edgebench.benchmark(lambda: r.infer(batch), label=f"model-{name}", ...)
```

No backend-specific branch. On this Mac that loop currently runs zero iterations, because no backend has its ML dependency installed yet (`torch`, `onnxruntime`, `coremltools` are all absent from the core environment on purpose — see the root `README.md`). Install `torch` and it runs one iteration. That's the abstraction working correctly, not a bug: `runners.available()` is supposed to shrink and grow with what's actually installed.

## What's already done vs. what Phase 2 asks of you

- `torch_runner.py` — **done**, and it's the reference to imitate.
- `onnx_runner.py`, `coreml_runner.py` — registered, and `is_available()` (plus, for ONNX, execution-provider discovery) already works, but `_load()` and `infer()` raise `NotImplementedError` with a pointer back to `curriculum/02-runtimes.md`. That's not an oversight; it's the shape of the exercise. Open either file and you'll find the stub waiting.
- `mlx_runner.py` — same shape, held for Phase 3.

## Comprehension check

- What makes `Runner` a `Protocol` rather than an abstract base class, and why might that distinction matter for a class that wraps, say, a raw C library binding that can't inherit from Python classes cleanly?
- If `OnnxRunner.is_available()` returned `True` on a machine where `onnxruntime` wasn't actually importable, what's the first place that would break, and how far downstream would you have to debug before finding the real cause?
- Why does `_discover()` import backend modules inside individual `try/except` blocks instead of one `try` around the whole loop?
- You're adding a fifth backend for, say, TensorRT directly. Sketch (in words, not code) what `is_available()` would need to check, and what would go in `_load()` versus what belongs in `infer()`.
