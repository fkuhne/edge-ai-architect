# Phase 3 — On-device LLMs

**Weeks 10–14** · *How does a transformer behave under a hard memory ceiling?*

The longest phase, and where the 8 GB constraint becomes genuinely interesting.
Everything from Phase 1 returns at a scale where the tradeoffs are unavoidable.

## Theory

- **Quantization**: Dettmers, *LLM.int8()* · Frantar, *GPTQ* · Lin, *AWQ* ·
  Xiao, *SmoothQuant*. Read `LLM.int8()` first — the outlier-feature explanation
  is what makes the rest make sense.
- **Adaptation**: Hu et al., *LoRA* · Dettmers et al., *QLoRA*.
- **Inference systems**: Leviathan et al., *Speculative Decoding* · Kwon et al.,
  *PagedAttention (vLLM)* · Dao, *FlashAttention*.
- **llama.cpp** k-quant documentation — what `Q4_K_M` actually means.
- **MIT 6.5940**, the efficient-LLM lectures.
- If attention internals are fuzzy: **Karpathy, *Let's build GPT*** — two hours,
  worth it before touching speculative decoding.

## Build — `projects/p03-llm-on-device/`

> **How this phase works.** Ask Claude for scaffolding for each of the three
> pieces below — a task breakdown and stubs, not finished scripts. (c) in
> particular is meant to be hand-written: the accept/reject loop is short and
> the understanding lives in writing it, not in reading it. Ask for concept
> explanations, hints, or review; ask for a reference implementation only
> after a genuine attempt. Full version in
> [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

```bash
make clean-phase && make env-p03
brew install cmake          # llama.cpp needs it
make disk                   # check before downloading weights
```

Three escalating pieces.

### (a) Quantization sweep

Qwen2.5-1.5B/3B and Llama-3.2-1B/3B across Q2/Q4/Q6/Q8 via **llama.cpp + GGUF**
— portable by construction: Metal here, CUDA elsewhere, identical model files.

Measure TTFT, tokens/sec, peak memory, and quality on a small **fixed** eval set
(30–50 prompts with checkable answers is enough; keep it constant across the
whole sweep or the comparison is worthless).

Deliverable: **the quality/speed knee on 8 GB**, written down. Below Q4 quality
usually falls off a cliff; find where it actually happens on these models rather
than repeating folklore.

Budget: a 3B at Q4 is ~2 GB. The full sweep will not fit at once — download,
measure, delete, next. `make disk` between models.

### (b) LoRA fine-tune

A 0.5B–1.5B model with **HF `peft` + `transformers`**, which runs identically on
MPS and CUDA. Then merge the adapter, convert to GGUF, quantize, and serve it
through the Phase-2 `Runner`. This closes the train→compress→deploy loop at LLM
scale.

> **Portability seam, worth writing up.** True QLoRA needs `bitsandbytes`, which
> is effectively CUDA-only. Here it is fp16 LoRA on a small model. Write the code
> so the quantization backend is **selected by device**, with the 4-bit path
> activating on CUDA, and note the divergence in `notes/`. This is a real example
> of an abstraction that cannot be perfectly leaky-free — the interesting kind.

### (c) Speculative decoding, by hand

0.5B draft + 3B target. Implement the accept/reject loop yourself rather than
calling a library — it is about 100 lines and the rejection-sampling step is the
part worth understanding.

Then measure honestly. **It may not help here.** Speculative decoding trades
memory for latency, and on 8 GB holding two models resident may cost more than
the speedup returns. Finding that out empirically, and being able to say at what
memory budget it flips, is a better outcome than a speedup.

## Done when

- [ ] Sweep complete; the quality/speed knee identified and defended with data.
- [ ] A LoRA-tuned model runs end to end: merged, quantized, served, benchmarked.
- [ ] Speculative decoding implemented and measured — including a clear answer on
      whether it helps on this machine, and why.
- [ ] `notes/` states **the largest model genuinely usable here**, with numbers.
- [ ] Optionally: MLX vs. llama.cpp on the same model, quantifying the price of
      portability on this hardware.

## Watch out

- **TTFT and tokens/sec are different problems.** Prefill is compute-bound,
  decode is memory-bandwidth-bound. `GenerationTimer` reports them separately;
  keep them separate in your conclusions too.
- **Quality needs a fixed eval set from day one.** Vibes will tell you Q2 is fine.
  It is not.
- **Watch for swap.** If tokens/sec collapses non-linearly as the model grows,
  you have found the memory cliff, not a quantization effect. Note the peak RSS
  alongside — that is what distinguishes the two.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- Why are TTFT and tokens/sec governed by different hardware bottlenecks
  (compute vs. memory bandwidth)? What would doubling the prompt length do to
  each, versus doubling the model's hidden size?
- In speculative decoding, what determines whether a drafted token gets
  accepted or rejected, and why is that step necessary instead of always
  trusting the draft model?
- Why might speculative decoding fail to speed anything up on an 8 GB machine
  even though it works on paper? What resource is actually being traded for
  what?
- Explain, in your own words, why `bitsandbytes` 4-bit quantization is
  effectively CUDA-only, and what that implies for writing "portable" LoRA
  fine-tuning code.
- If tokens/sec drops off a cliff at a certain context length, how do you
  tell whether that's an algorithmic effect (attention cost growing with
  sequence length) or a memory effect (swap)?
