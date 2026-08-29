# Phase 2 — Portable runtimes & backend abstraction

**Weeks 6–9** · *How does one model run on any accelerator without a rewrite?*

The phase where ground rule 2 stops being a convention and becomes the subject.
The artifact — a working `Runner` abstraction with several backends behind it — is
reused by Phases 3, 4, and 6, and is a miniature of what the Distinguished
Technologist posting calls *"abstraction layers that let AI workloads deploy
across diverse silicon without rewrites."*

## Theory

- **ONNX Runtime execution providers** — the docs on EP selection and fallback.
  The portability model itself.
- **HuggingFace `optimum`** — PyTorch → ONNX export, graph optimization, ORT
  quantization. The bridge between Phase 1's PyTorch work and portable runtimes.
- **MIT 6.5940**, the inference-engine and hardware-acceleration lectures.
- Sze, Chen, Yang & Emer, ***Efficient Processing of Deep Neural Networks*** —
  dataflow, memory hierarchy, why operator fusion matters. Dense; skim first.
- *Optional, Apple-specific:* Core ML Tools user guide, and Apple's
  **Deploying Transformers on the Apple Neural Engine**, which explains why the
  ANE wants particular tensor layouts.

## Build — `projects/p02-runtime-shootout/`

> **How this phase works.** Ask Claude for scaffolding — a project brief with
> the task broken into steps, stub files with TODOs where structure helps —
> not a finished `onnx_runner.py`/`coreml_runner.py`. Build the `_load()` and
> `infer()` methods yourself; that implementation *is* the lesson. Ask for
> concept explanations, hints, or a review of your attempt; ask for a
> reference implementation only after you've genuinely tried. Full version in
> [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

```bash
make clean-phase && make env-p02
make env-apple      # optional: adds the coreml + mlx plug-ins
```

Take the Phase-1 model plus one real pretrained vision model (MobileNetV3 or a
small ViT from `transformers`) and run both through every backend this machine
offers.

The scaffolding is already in place — `runners/base.py` defines the contract,
`runners/torch_runner.py` is the working reference. This phase implements:

| Backend | Status | Notes |
|---|---|---|
| `torch` | done | Reference baseline. Device-agnostic already. |
| `onnx` | **implement** | The important one. `is_available()` and EP selection work; `_load`/`infer` are yours. |
| `coreml` | **implement** | Optional plug-in. Reaches the ANE. Darwin only. |
| `mlx` | Phase 3 | Leave it. |

### The real work

**One benchmark script, every available backend, no branching.** Something like:

```python
for name in runners.available():
    with runners.get(name)() as r:
        r.load(model_for(name))
        edgebench.benchmark(lambda: r.infer(batch), label=f"mobilenetv3-{name}", ...)
```

If that loop needs an `if name == "coreml"` in it, the abstraction has failed and
is worth fixing before continuing.

### The lesson to hunt for

**Find the operators that silently fall back to CPU.** ONNX Runtime will happily
accept a model, place half its graph on the CPU provider, and report nothing.
Core ML will do the same with the ANE. Both look like "it worked" and benchmark
like a mystery.

- ORT: `session.get_providers()` for what is *active*, then enable profiling
  (`sess_options.enable_profiling = True`) and read the per-node placement.
- Core ML: `coremltools` compute-unit reports, plus **ANE power draw in
  `powermetrics`** — near-zero ANE power during inference means nothing ran there,
  whatever the config said.

This is the same failure mode as an op with no CUDA kernel on a GPU box. Learning
to detect it here transfers directly.

## Done when

- [ ] `onnx` and (on this machine) `coreml` backends implemented and registered.
- [ ] One script benchmarks **all** available backends with no per-backend
      branching, and skips unavailable ones cleanly.
- [ ] Results for both models across all backends in `results/bench.db`.
- [ ] ANE power draw confirms the Core ML path really used the Neural Engine —
      or you can explain why it didn't.
- [ ] `notes/` names **three operators** that broke accelerator residency, and
      why.

## Watch out

- **Export is where models break**, not inference. Dynamic shapes, unsupported
  ops, and control flow all fail at `torch.onnx.export`. Budget time for it.
- **fp16 on ANE vs. fp32 on CPU is not a fair comparison.** Record the precision
  in `quantization` and compare like with like.
- `onnxruntime` on Apple silicon ships the CoreML EP. `onnxruntime-gpu` is a
  *different package* and CUDA-only — a portability seam worth noting in
  `notes/`, since it means the install line differs per platform even though the
  code does not.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- What does it mean, concretely, for an ONNX Runtime session to "fall back to
  CPU" for one operator? How would you detect it happened without being told?
- Your benchmark loop iterates `runners.available()` with no backend-specific
  branching. What had to be true about `BaseRunner` for that to be possible?
- Why must `is_available()` never raise, even when the underlying import
  genuinely fails? What would break if it raised instead?
- ANE power reads near-zero during a Core ML call that "succeeded." What are
  the two different explanations for that, and how would you tell them apart?
- Why does `onnxruntime-gpu` being a separate package from `onnxruntime`
  matter for how you'd write install scripts, even though the Python code
  doesn't change?
