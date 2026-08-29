# edge-ai-architect

A hands-on curriculum for learning edge machine learning: train models, compress
them, deploy them across runtimes, and measure every step on real hardware.

Everything runs **locally**. Everything is **portable** — the code is written for
a generic Linux/CUDA box and merely happens to be developed on an Apple M3.

## Quickstart

```bash
make env-core     # edgebench + psutil, no ML frameworks (~15 MB)
make selftest     # prove the measurement harness works here
make check        # prove it still works with the device and power sampler swapped
```

`make selftest` runs without PyTorch installed. That is intentional: the harness
must be verifiable before spending 2.5 GB of a 15 GB disk on a framework.

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
| 0 | [Bootstrap & edgebench](curriculum/00-bootstrap.md) | 1 | How do I measure anything credibly, on any machine? | ▶ in progress |
| 1 | [Foundations & compression](curriculum/01-foundations.md) | 2–5 | What do I actually give up to make a model small? | ☐ |
| 2 | [Runtimes & backend abstraction](curriculum/02-runtimes.md) | 6–9 | How does one model run on any accelerator without a rewrite? | ☐ |
| 3 | [On-device LLMs](curriculum/03-llms.md) | 10–14 | How does a transformer behave under a hard memory ceiling? | ☐ |
| 4 | [Streaming perception](curriculum/04-perception.md) | 15–18 | How do I hold a real-time latency budget end to end? | ☐ |
| 5 | [Agentic runtime & RAG](curriculum/05-agents.md) | 19–23 | How do I make a small model reliable enough to trust with tools? | ☐ |
| 6 | [Platform & lifecycle](curriculum/06-platform.md) | 24–28 | How does this become a system rather than a pile of scripts? | ☐ |

Full bibliography: [curriculum/resources.md](curriculum/resources.md).

## The machine

| | | |
|---|---|---|
| Memory | 8 GB unified (Apple M3) | Caps local LLMs to ~0.5B–3B at 4-bit |
| Disk | ~15 GB free | One phase environment at a time |
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
edgebench/      portable measurement (stdlib + psutil only)
runners/        backend abstraction: torch | onnx | coreml | mlx
projects/       one directory per phase project
curriculum/     phase briefs and the bibliography
notes/          learning notes
results/        bench.db and generated plots
```
