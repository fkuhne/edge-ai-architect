# Learning Guide — how this project actually teaches

The operating manual for using Claude as a mentor on this repo, not a contractor.

## What happened, and what changes

Phase 0 — the `edgebench` measurement library and the `runners` backend abstraction — was built end to end by Claude in the first session: fully implemented, tested, documented. For a project whose entire point is Felipe building skills himself, that was the wrong call, and he corrected it directly: *"I told you that my objective was to LEARN. how would I learn if you do everything to me?"*

The code stays — rewriting it as an unsolved exercise now would just waste what's already there — but it's repositioned as a **worked example**, not a template for how the rest of this repo gets built. Phases 1 through 6, where the actual edge-ML skills live, work differently from here on.

## The loop, for every phase from here on

1. **Read the theory** in `curriculum/0N-*.md` — papers, courses, docs.
2. **Ask Claude for scaffolding, not code**: a project brief with the task broken into concrete steps, stub files (signatures + docstrings + TODOs) where structure helps, and pointers back to the relevant theory.
3. **Build it.** This part has to be yours. Struggling here isn't a sign anything's wrong — it's most of the point.
4. **Self-check** against the phase's "Done when" criteria and its comprehension checkpoint.
5. **Come back to Claude for** — roughly in order of how often you'll reach for each:
   - a concept explained before you start, or explained differently if the first pass didn't land
   - a hint when you're stuck — ask explicitly for a hint, not a fix, and say what you've already tried
   - a review of code once something runs, even if it's ugly
   - a reference implementation to compare against, once you've made a genuine attempt — treat it like a course's answer key released after the deadline, not the default output
6. **Write a short note** in `notes/`. Explaining a result back to yourself in writing is where a lot of the actual learning happens; `bench.db` captures what happened, the notes capture what you understood.

## Asking well

The phrasing changes what you get back:

| Instead of... | Try... |
|---|---|
| "write the quantization sweep script" | "explain what a quantization sweep needs to measure, then give me a stub with the loop structure and TODOs for the parts I should fill in" |
| "why isn't my code working" | "here's my attempt and the error — what's the concept I'm missing, not just the fix" |
| "give me the LoRA fine-tuning code" | "I've got the training loop running but the loss isn't dropping — review what I have and ask me questions rather than rewriting it" |
| *(nothing — just moving on)* | "here's my solution to 3(a) — is there a cleaner way, and did I miss an edge case?" |

None of this means Claude should refuse to write code here — sometimes the fastest way to understand a pattern is to see it once, well-explained. It means the default is scaffolding and explanation, and a full implementation is something asked for deliberately, having already tried.

## Comprehension checkpoints

Each phase brief now ends with a short list of questions that can't be answered by pattern-matching the code — they ask *why*, not *what*. Nobody's grading them, but treat one you can't answer as a real signal: reread the theory, or go back into the code, before calling the phase done. A phase that satisfies its checklist but can't survive its checkpoint questions isn't actually finished.

## Reading Phase 0 as a worked example

Rather than skim `edgebench/` and `runners/` as a library to import and forget, three READMEs walk through the design decisions:

- [`edgebench/README.md`](edgebench/README.md) — the measurement library: why percentiles over means, why timing needs a device sync, why memory units differ by platform, how results stay comparable across seven months of phases.
- [`edgebench/power/README.md`](edgebench/power/README.md) — why power measurement is the most fragmented part of the whole library, and the plug-in pattern used to handle that.
- [`runners/README.md`](runners/README.md) — the backend abstraction itself: the pattern Phase 2 asks you to extend, demonstrated once before you're asked to build on it.

Phase 0's curriculum brief also carries a comprehension checkpoint and a longer note on the reasoning above. Answer it before treating this code as settled infrastructure you don't need to think about again — and if you want the harder version, each README ends with a suggestion for reimplementing a piece of it from scratch to compare against.
