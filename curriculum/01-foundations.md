# Phase 1 — Foundations & the compression loop

**Weeks 2–5** · *What do I actually give up to make a model small?*

The central tradeoff of edge ML, learned on a model small enough to iterate on in
minutes. Every technique here reappears at LLM scale in Phase 4 — quantization,
pruning, and distillation are the same ideas whether the model has 300 K
parameters or 3 B.

## Theory

- **MIT 6.5940, *TinyML and Efficient Deep Learning Computing*** (Song Han) —
  lectures 1–8 cover exactly this ladder. Free on YouTube, slides public. The
  single most valuable resource in this curriculum.
- ***Machine Learning Systems*** (Reddi, Harvard) — free at
  [mlsysbook.ai](https://mlsysbook.ai). Read the model-optimization chapters for
  the systems framing Han's course assumes.
- **Papers**, in this order:
  1. Han et al., *Deep Compression* (2015) — pruning + quantization + Huffman
     coding, the paper that started the field.
  2. Jacob et al., *Quantization and Training of Neural Networks for Efficient
     Integer-Arithmetic-Only Inference* (2018) — the int8 QAT paper. Read the
     fake-quantization section carefully; it explains Phase 1's main result.
  3. Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) —
     short, and the temperature trick is worth understanding properly.

## Build — `projects/p01-compress-cifar/`

> **How this phase works.** The brief below and the theory above are what you
> get up front — not an implementation. Ask Claude for scaffolding: a
> `projects/p01-compress-cifar/README.md` with the task broken into concrete
> steps, and stub files with signatures/docstrings/TODOs where structure
> helps. Then build it yourself. Good reasons to come back to Claude: a
> concept explained before you start, a hint (not the answer) when stuck, a
> review once something runs, or a reference implementation to compare against
> — after a genuine attempt, not before. Full version in
> [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

```bash
make clean-phase && make env-p01
```

Train a small ResNet on CIFAR-10, then walk the ladder. **Benchmark every rung
in `edgebench`** — the point is the curve, not any single number.

| Rung | Technique | Expect |
|---|---|---|
| 0 | fp32 baseline | The reference. Everything else is measured against it. |
| 1 | Post-training int8 (PTQ) | ~4x smaller, near-free accuracy loss at int8 |
| 2 | Quantization-aware training (QAT) | Recovers most of what PTQ lost — and the gap **widens** as bits drop |
| 3 | Structured pruning | Real speedup, unlike unstructured sparsity without kernel support |
| 4 | Distillation into a smaller student | Often the best accuracy-per-millisecond of the five |

Use `torch.ao.quantization` — portable, not vendor-specific. Keep the training
loop plain PyTorch; this phase is about what happens *after* training.

Two things worth doing properly:

- **Measure at each bit width**, not just int8. The PTQ-vs-QAT gap is
  unremarkable at 8 bits and dramatic at 4. That divergence is the lesson.
- **Distinguish size from speed.** Unstructured pruning makes a model smaller on
  disk and no faster in wall time without sparse kernels. Discovering that from
  your own numbers is more durable than reading it here.

## Done when

- [ ] Five variants benchmarked, all in `results/bench.db` under project
      `p01-compress-cifar`.
- [ ] `make pareto` renders accuracy vs. latency with all five plotted.
- [ ] `notes/` contains a short write-up answering, **from your own numbers**:
      why does QAT beat PTQ at low bit-widths, and where is the knee?
- [ ] You can say which rung you would ship, and why.

## Watch out

- **Swap poisons benchmarks.** Close everything before a timing run. Check
  `make disk` — a nearly full disk on 8 GB RAM means aggressive swapping, and a
  swapped run can be several times slower for reasons having nothing to do with
  the model.
- **MPS quantization support is patchy.** Some `torch.ao` paths are CPU-only.
  That is fine — benchmark quantized variants on CPU and say so in the label
  (`-cpu` suffix). Comparing a GPU fp32 number against a CPU int8 number without
  noting it is the easiest way to produce a plot that means nothing.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- Why does the PTQ-vs-QAT accuracy gap widen as bit-width drops, rather than
  staying roughly constant?
- Structured vs. unstructured pruning: which one actually gets you a
  wall-clock speedup on this hardware, and why does the other not, even
  though it removes just as many parameters?
- Distillation uses a temperature on the softmax. What does raising it
  actually do to the teacher's output distribution, and why does that help
  the student learn?
- You benchmarked a quantized variant on CPU because an op wasn't supported
  on MPS. Why would comparing that number directly to an MPS fp32 number be
  misleading, even though both numbers are individually correct?
- If your five-rung Pareto curve had a point that was both smaller *and*
  slower than the baseline, what would that tell you about where the real
  bottleneck was?

## Before moving on

`make clean-phase`. Phase 2 needs a different environment.
