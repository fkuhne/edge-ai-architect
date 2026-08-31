# edge-ai-architect

A hands-on curriculum for learning edge machine learning: train models, compress
them, deploy them across runtimes, and measure every step on real hardware.

Everything runs **locally**. Everything is **portable** — the code is written for
a generic Linux/CUDA box and merely happens to be developed on an Apple M3.

## Read this first: how this repo teaches

This is a mentor relationship, not a build contract — Claude scaffolds and
explains, you implement. Phase 0 (`edgebench/`, `runners/`) is the one
exception: fully pre-built as shared infrastructure, and documented as a
worked example rather than a black box. Phases 1–7, where the actual edge-ML
skills live, work as guided builds — theory, a task breakdown, stubs where
useful, code review — not finished code.

Full explanation, including how to ask well and what the comprehension
checkpoints are for: **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)**.

The learning also gets written down as it happens, in **[book/](book/README.md)** —
raw same-day log entries, a retrospective per phase, and cross-phase threads.
Possibly a book eventually, with this repo as its companion code. Claude writes
none of it; that folder is the one thing here a model couldn't generate.

## Start here

In this order:

1. **Check the harness runs on this machine.**
   ```bash
   make env-core     # edgebench + psutil, no ML frameworks (~15 MB)
   make selftest     # one benchmark, end to end
   make check        # same harness, device and power sampler swapped
   ```
   `make selftest` works without PyTorch installed — the harness must be
   verifiable before spending 2.5 GB on a framework.

2. **Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md).** How the mentoring works, how
   to phrase a request so you get scaffolding instead of finished code, and what
   happens when you're stuck. Ten minutes, and it changes what the rest of this
   repo is for.

3. **Work [Phase 0](curriculum/00-bootstrap.md).** The code is pre-built, but its
   comprehension checkpoint is not — answer it in `notes/` before treating
   `edgebench` as settled infrastructure.

4. **Start Phase 1.**
   ```bash
   make clean-phase && make env-p01
   ```
   Read [curriculum/01-foundations.md](curriculum/01-foundations.md), then ask
   Claude for the project brief.

**Every session after that:** say which phase you're on and what you've already
tried. Claude starts each session without memory of the last one, so that one
sentence is what makes the mentoring continuous. Tick the status column below as
you go — it's the progress record.

## Why this exists

`job-descriptions/` holds four HP Edge AI postings spanning silicon-level
deployment through platform architecture. They are a **syllabus** — the union of
their required skills sets the scope. The goal is capability, not applications.

## The spine: `edgebench`

Every phase reports through one measurement library into one SQLite database, so
a pruned CNN from month one and a 4-bit LLM from month four land on the same
axes. It records latency (p50/p95/p99, and TTFT vs. tokens/sec separately for
LLMs), peak memory, power where the platform exposes it, a quality metric, and
enough provenance — model hash, backend version, git SHA — to still mean
something six months later.

```bash
make report                    # the results table
make pareto                    # quality vs. latency
```

## Curriculum

Sized for ~6–10 hrs/week. Phases are sequential and the projects compound.

| | Phase | Weeks | Core question | Status |
|---|---|---|---|---|
| 0 | [Bootstrap & edgebench](curriculum/00-bootstrap.md) | 1 | How do I measure anything credibly, on any machine? | ✓ pre-built* |
| 1 | [Foundations & compression](curriculum/01-foundations.md) | 2–5 | What do I actually give up to make a model small? | ☐ your build |
| 2 | [Runtimes & backend abstraction](curriculum/02-runtimes.md) | 6–9 | How does one model run on any accelerator without a rewrite? | ☐ your build |
| 3 | [Language, tokens & transformers](curriculum/03-nlp.md) | 10–13 | Why is a transformer shaped the way it is, and which shape should I reach for? | ☐ your build |
| 4 | [On-device LLMs](curriculum/04-llms.md) | 14–18 | How does a transformer behave under a hard memory ceiling? | ☐ your build |
| 5 | [Streaming perception](curriculum/05-perception.md) | 19–22 | How do I hold a real-time latency budget end to end? | ☐ your build |
| 6 | [Agentic runtime & RAG](curriculum/06-agents.md) | 23–27 | How do I make a small model reliable enough to trust with tools? | ☐ your build |
| 7 | [Platform & lifecycle](curriculum/07-platform.md) | 28–32 | How does this become a system rather than a pile of scripts? | ☐ your build |

\* *Phase 0 was fully implemented by Claude, which — for a project about learning
edge ML by building it — was more than it should have done. It's kept as a
documented worked example rather than rebuilt as an exercise; every phase after
it is scaffolding and guidance, not finished code. See
[LEARNING_GUIDE.md](LEARNING_GUIDE.md).*

Full bibliography: [curriculum/resources.md](curriculum/resources.md).

## The machine

| | | |
|---|---|---|
| Memory | 8 GB unified (Apple M3) | Caps local LLMs to ~0.5B–3B at 4-bit |
| Disk | ~25GiB free — check with `make disk` | One phase environment at a time |
| Accelerators | Metal GPU, 16-core Neural Engine | Reached only through `runners/` |

The constraints are not obstacles to route around — they are the subject. Every
one of the four postings asks for optimization under latency, memory, and power
budgets, and a small machine teaches that honestly.

## Ground rules

**PyTorch and HuggingFace, no TensorFlow.** **Develop here, deploy anywhere** —
Core ML and MLX exist only as optional plug-ins behind `runners/`, never as the
interface anything is written against.

See [CLAUDE.md](CLAUDE.md) for the full conventions.

## Layout

```
LEARNING_GUIDE.md   the mentorship model -- read this before anything else
book/               the learning documented as it unfolds -- see its README
edgebench/          portable measurement (stdlib + psutil only) -- see its README
runners/            backend abstraction: torch | onnx | coreml | mlx -- see its README
projects/           one directory per phase project
curriculum/         phase briefs, each ending in a comprehension checkpoint
notes/              technical notes and checkpoint answers
results/            bench.db and generated plots
```

`edgebench/README.md`, `edgebench/power/README.md`, and `runners/README.md`
each walk through their package's design decisions as a worked example —
worth reading in full rather than treating as reference documentation.
