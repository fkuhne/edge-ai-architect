# Phase 3 — Language, tokens & transformer architecture

**Weeks 10–13** · *Why is a transformer shaped the way it is, and which shape should I reach for?*

Phases 1 and 2 compressed and deployed a CNN. Phase 4 will ask you to quantize a
transformer, fine-tune it, and hand-implement speculative decoding. This phase
is the bridge, and without it Phase 4 is optimization of something you have not
studied — which is how people end up tuning knobs by folklore.

The goal is not NLP applications. It is enough architectural understanding to
make **informed decisions**: when a 60M encoder beats a 3B decoder, why Llama and
Qwen use grouped-query attention, what a "token" costs you, and which of those
choices actually bind on an 8 GB machine.

> **How this phase works.** Ask Claude for scaffolding — a task breakdown, stubs
> for the measurement plumbing, concept explanations before you start — not
> finished implementations. (b) in particular is meant to be hand-written; the
> understanding lives in deriving it, not reading it. Ask for hints when stuck,
> review once something runs, and a reference implementation only after a
> genuine attempt. Full version in [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

## Theory

**Core** — pick the format that suits you; these overlap heavily by design:

- **Jurafsky & Martin, *Speech and Language Processing* (3rd ed.)** — free draft
  at [stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/).
  The standard text. Chapters on subword tokenization, embeddings, transformers,
  and fine-tuning are the ones that matter here.
- **Stanford CS224N — NLP with Deep Learning** — lectures free on YouTube.
- **HuggingFace NLP Course** — free, hands-on, and uses exactly the stack this
  project already standardises on.
- **Karpathy, *Let's build the GPT Tokenizer*** — two hours, and the single best
  preparation for (a) below.
- **Jay Alammar, *The Illustrated Transformer*** — the visual explanation to read
  before the paper, not after.

**Papers — the architecture**
- Vaswani et al., **Attention Is All You Need** (2017)
- Devlin et al., **BERT** (2018) — encoder-only
- Raffel et al., **T5** (2019) — encoder-decoder, and the reason Whisper in
  Phase 5 is shaped the way it is
- Sennrich et al., **Neural Machine Translation of Rare Words with Subword
  Units** (2016) — BPE
- Kudo & Richardson, **SentencePiece** (2018)

**Papers — the edge angle.** These are the ones that pay off in Phase 4:
- Shazeer, **Fast Transformer Decoding** (2019) — multi-query attention
- Ainslie et al., **GQA** (2023) — the compromise Llama and Qwen actually ship
- Su et al., **RoFormer / RoPE** (2021) — positional encoding and why context
  extension is possible at all
- Sanh et al., **DistilBERT** (2019) — Phase 1's distillation, applied to language
- Hoffmann et al., **Chinchilla** (2022) — why small models are usually
  undertrained rather than fundamentally limited

## Build — `projects/p03-nlp-foundations/`

```bash
make clean-phase && make env-p03
```

Three pieces, escalating. Everything reports to `edgebench` as usual.

### (a) Tokenizer economics — ~3 days

Take several tokenizers (Llama 3, Qwen 2.5, GPT-2, BERT, and a SentencePiece
model) and measure **tokens per word** over the same corpus in **English and in
Portuguese**.

The finding is not academic. Most tokenizers are fit predominantly on English,
so the same sentence in Portuguese costs meaningfully more tokens. That means:
shorter effective context for the same text, more forward passes for the same
output, and higher latency for the same content. You are paying a tax the model
card does not mention.

Then convert it to consequences: at your measured tokens/word, how much of a
4096-token context does a 1,500-word Portuguese document actually consume? What
does that do to TTFT?

Also measure the **embedding table**: `vocab_size × hidden_dim × bytes`. For a
0.5B model with a 150k vocab this is a startling fraction of total parameters —
compute it rather than take my word for it, and note whether the model ties
input and output embeddings, because that halves the cost.

### (b) Attention and the KV cache, by hand — ~1 week

Implement single-head self-attention from scratch in PyTorch, verify it against
`torch.nn.functional.scaled_dot_product_attention`, then extend to multi-head.
Then implement the three variants and understand what separates them:

| | Query heads | KV heads | KV cache size |
|---|---|---|---|
| MHA | n | n | baseline |
| MQA | n | 1 | baseline / n |
| GQA | n | g (1 < g < n) | baseline × g/n |

Then derive the formula that decides whether a model fits on this machine:

```
kv_cache_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_elem
                 ^
                 K and V
```

**Do not take a worked example from me.** Open the `config.json` of a model you
will actually run in Phase 4, pull the real values, and compute the cache size at
2k, 8k, and 32k context. Then compute what the same model would cost with plain
MHA. The ratio is why GQA exists, and it is the difference between a model that
fits in your remaining memory and one that does not.

Add a KV cache to your implementation and measure the speedup on a short
generation loop. That number is the whole reason Phase 4's `GenerationTimer`
separates TTFT from tokens/sec.

### (c) Architecture bake-off — ~1.5 weeks

The phase's main deliverable, and the one that produces a *decision*.

Pick one concrete task — text classification is cleanest (sentiment, topic, or
intent; a small public dataset is fine). Solve it three ways:

1. **Fine-tuned encoder** — DistilBERT or MiniLM, roughly 20–70M parameters.
2. **Decoder LLM, zero-shot** — a Phase-4-class quantized model, prompted.
3. **Decoder LLM, few-shot** — the same model with examples in the prompt.

Measure all three in `edgebench`: **accuracy, p50 latency, peak memory, model
size on disk**. Same test set, same machine, same harness.

The result is usually decisive and surprises people: for a fixed, well-specified
task, the tiny encoder tends to win on every axis at once — often by an order of
magnitude on latency and memory. That is a genuine architecture decision backed
by your own numbers, and it is the answer to "should this feature be an LLM call?"
for a large fraction of real product work.

Then write down the honest caveat: the encoder needed labelled training data and
a fine-tuning run; the LLM did not. Under what conditions does that flip the
decision?

## Done when

- [ ] Tokens/word measured for English and Portuguese across ≥4 tokenizers, with
      the context and latency consequences worked out.
- [ ] Self-attention implemented from scratch and verified against PyTorch's.
- [ ] KV cache size computed from a real `config.json` at three context lengths,
      with the MHA-vs-GQA ratio stated.
- [ ] Three approaches to one task benchmarked on identical footing in
      `results/bench.db`.
- [ ] `notes/` contains a short written recommendation: for a task like this one,
      which architecture would you reach for, and what would change your mind?

## Watch out

- **Perplexity is not comparable across tokenizers.** Two models with different
  vocabularies produce perplexities that cannot be compared directly, because
  they are not predicting the same units. This trips up a lot of published
  comparisons, and it matters in Phase 4 when you use perplexity to measure
  quantization damage — it is valid *within* a model, not *across* models.
- **The bake-off must be fair.** Do not evaluate the fine-tuned encoder on data
  it saw. Use the same held-out test set for all three, and report the labelled-
  data requirement as part of the encoder's cost, not as free.
- **`max_length` truncates silently.** A tokenizer that quietly drops the second
  half of every input produces a model that looks bad for the wrong reason.
  Check token counts against your inputs before believing an accuracy number.
- **Fine-tuning even a 60M model on 8 GB needs attention to batch size.** Reduce
  batch size and use gradient accumulation before concluding it does not fit.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- What does the KV cache actually store, why does it grow linearly with sequence
  length, and why does GQA reduce it while MQA reduces it further — yet nearly
  everyone ships GQA rather than MQA?
- Why can a 60M-parameter encoder beat a 3B-parameter decoder on a classification
  task? What is the decoder spending its parameters and its compute on that the
  encoder is not?
- Your Portuguese text costs more tokens than the equivalent English. Explain the
  mechanism — what about how the tokenizer was built produces that, and what are
  the two distinct costs it imposes at inference time?
- Why can't you compare the perplexity of Llama 3.2 against the perplexity of
  Qwen 2.5 and conclude one is better?
- Encoder-only, decoder-only, encoder-decoder: what is each actually good at, and
  what property of its training objective makes it so?
