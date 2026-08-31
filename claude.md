# edge-ai-architect — working conventions

A personal learning project on edge machine learning: theory paired with toy
projects that train, compress, deploy, and measure models locally.

`job-descriptions/` holds four HP Edge AI postings. They are a **syllabus**, not
a job target — the union of their required skills defines the scope of what gets
learned here. Nothing in this repo is interview prep.

## Mentorship model — read before touching `projects/`

This is Felipe's learning project. The deliverable is the skill, not the code.
Claude's role here is mentor, not implementer — a distinction that was gotten
wrong once already (Phase 0 was fully implemented end to end, unprompted) and
corrected explicitly: he wants to build this himself, with guidance, not
receive it finished. See `LEARNING_GUIDE.md` for the full reasoning.

For anything under `projects/pNN-*/` — the actual learning content of each phase:

- Do **not** write the full implementation.
- Do provide: a project brief with clear objectives and a step-by-step task
  breakdown, stub files (signatures, docstrings, TODOs) where structure
  genuinely helps, conceptual explanations, hints on request, and review of
  code he's written.
- A reference solution is fair to offer only after he's made a real attempt
  and asks for one — like a course answer key released after the deadline,
  not the default output.

`edgebench/` and `runners/` are the sanctioned exception: pre-built shared
infrastructure (a measurement harness, a backend abstraction), not the subject
being taught. They're documented as worked examples — `LEARNING_GUIDE.md` plus
a README in each package — precisely because that exception still has to pay
for itself pedagogically. Don't extend the "fully implement it" pattern into
`projects/`.

**`book/` is the same rule applied to prose, and it is absolute.** That folder
records the learning as it unfolds, with a view to becoming a book. Claude
never writes log entries, retrospectives, or chapter drafts there, and never
pre-populates them with plausible content Felipe hasn't actually experienced —
the whole value of the folder is that it holds something a model cannot
generate. Reacting to a draft he wrote, pointing out an unclear explanation,
proposing a structure, or asking questions that surface something left
implicit: all welcome. Ghostwriting: not. Templates and scaffolding in
`book/` were written by Claude; everything with substance in it is his.

## Two ground rules

**1. PyTorch and HuggingFace. No TensorFlow.**
PyTorch is the training and modeling framework. Prefer `transformers`,
`datasets`, `peft`, `accelerate`, and `optimum` for anything they already cover.
TensorFlow, TFLite, and Keras are out of scope — do not reach for them even when
they would be the conventional choice.

**2. Develop here, deploy anywhere.**
This Mac is treated as a generic Linux box. Apple-specific tooling — Core ML,
MLX, the ANE, `powermetrics` — is allowed **only behind a backend abstraction
that can switch to CUDA**. In practice:

- Never hardcode a device. Call `edgebench.device.resolve()`, which returns
  `cuda` → `mps` → `cpu` and honours `EDGEBENCH_DEVICE`.
- Never import `coremltools`, `mlx`, or call `powermetrics` outside
  `runners/coreml_runner.py`, `runners/mlx_runner.py`, and
  `edgebench/power/powermetrics.py`.
- A backend's `is_available()` **must never raise**. Discovery runs on every
  platform; an unusable backend is a `False`, not a traceback.
- Anything platform-specific degrades. Missing power sampling costs power data,
  not the run.

`make check` verifies both rules still hold. Run it after touching `edgebench/`
or `runners/`.

## Measurement

Every experiment reports through `edgebench` — that is what makes a pruned CNN
from Phase 1 comparable to a 4-bit LLM from Phase 4.

```python
import edgebench

result = edgebench.benchmark(
    lambda: model(batch),
    label="resnet8-int8-ptq",      # unique, descriptive, sortable
    project="p01-compress-cifar",
    backend="torch",
    quantization="int8",
    quality=("accuracy", 0.897),   # never report speed without it
)
```

- **Never report latency without a quality number.** A faster model that is
  worse is not a result, it is half of one.
- Use `record()` instead of `benchmark()` for token generation and multi-stage
  pipelines, which measure themselves with `GenerationTimer` / `MemoryTracker`.
- Timing already synchronises the device. Do not add your own — and do not
  remove it, or you will be timing kernel dispatch rather than compute.

## Disk and memory

8 GB unified memory, ~15 GB free disk. Both bind, constantly.

- **One phase environment at a time.** `make env-p01`, then `make clean-phase`
  before `make env-p02`. Do not install every backend at once.
- `make disk` before downloading models. Budget is 8 GB for `models/`.
- Prefer 0.5B–3B models at 4-bit. A 7B model at Q4 technically fits and will
  swap hard enough to poison every benchmark taken beside it.
- Model weights never go in git. `models/manifest.json` is tracked; the bytes
  are not.

## Layout

| Path | What lives there |
|---|---|
| `edgebench/` | Portable measurement. Stdlib + psutil only — never import a framework here. |
| `runners/` | The backend abstraction. `base.py` is the contract; one module per backend. |
| `projects/pNN-*/` | One directory per phase project. Self-contained. |
| `curriculum/` | Phase briefs: theory, build spec, measurable done-criteria. |
| `notes/` | Learning notes written as he goes. |
| `results/` | `bench.db` plus generated plots. |

## Conventions

- Comments explain *why*, not *what*. The code says what it does.
- New backend: subclass `BaseRunner`, decorate with `@register`, implement
  `is_available()`, `_load()`, and `infer()`.
- A phase is not done because it feels done. Each `curriculum/0N-*.md` states a
  measurable criterion; meet it.
