# Bibliography

Curated rather than exhaustive. Everything here is either free or worth buying;
nothing is listed that isn't actually read during a phase.

## Start here

**MIT 6.5940 — TinyML and Efficient Deep Learning Computing** (Song Han, MIT
HAN Lab). Lectures on YouTube, slides public. The single most valuable resource
in this curriculum: pruning, quantization, NAS, distillation, on-device training,
and efficient LLM inference, all with the systems detail that most courses skip.
Used in Phases 1, 2, and 4.

**Machine Learning Systems: Principles and Practices of Engineering Artificially
Intelligent Systems** (Vijay Janapa Reddi, Harvard). Free at
[mlsysbook.ai](https://mlsysbook.ai). The systems framing around the techniques.
Used in Phases 0 and 1.

## Books

| Book | Author | Phase | Why |
|---|---|---|---|
| *AI Engineering* | Chip Huyen (2025) | 6 | Best single text on evals, guardrails, LLM app architecture |
| *Designing Machine Learning Systems* | Chip Huyen | 7 | MLOps foundations: deployment, monitoring, continual learning |
| *Efficient Processing of Deep Neural Networks* | Sze, Chen, Yang, Emer | 2 | The definitive hardware/dataflow text. Dense; skim first |
| *TinyML* | Warden & Situnayake | optional | MCU/embedded framing. Dated on tooling, sound on concepts |
| *Speech and Language Processing* (3rd ed.) | Jurafsky & Martin | 3 | The standard NLP text. Free draft; tokenization, embeddings, transformers |

## Papers by phase

### Phase 1 — compression
- Han, Mao & Dally, **Deep Compression** (2015) — pruning + quantization + Huffman
- Jacob et al., **Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference** (2018) — the int8 QAT paper
- Hinton, Vinyals & Dean, **Distilling the Knowledge in a Neural Network** (2015)
- Howard et al., **MobileNets** (2017) / **MobileNetV3** (2019) — efficient
  architecture design
- Lin et al., **MCUNet** (2020) — TinyML at the extreme end of constraint

### Phase 2 — runtimes and hardware
- Chen et al., **TVM** (2018) — compiler approach to the same portability problem
- Apple ML Research, **Deploying Transformers on the Apple Neural Engine** —
  why the ANE wants particular layouts
- **MLPerf Inference** (Reddi et al., 2020) — benchmark methodology

### Phase 3 — language, tokens and architecture
- Vaswani et al., **Attention Is All You Need** (2017)
- Devlin et al., **BERT** (2018) — encoder-only
- Raffel et al., **T5** (2019) — encoder-decoder; the shape Whisper inherits
- Sennrich et al., **Neural Machine Translation of Rare Words with Subword
  Units** (2016) — BPE, the tokenizer everything descends from
- Kudo & Richardson, **SentencePiece** (2018)
- Shazeer, **Fast Transformer Decoding** (2019) — multi-query attention
- Ainslie et al., **GQA** (2023) — the compromise Llama and Qwen actually ship
- Su et al., **RoFormer / RoPE** (2021) — positional encoding, and why context
  extension is possible at all
- Sanh et al., **DistilBERT** (2019) — Phase 1's distillation applied to language
- Reimers & Gurevych, **Sentence-BERT** (2019) — sentence embeddings; the
  retrieval side of Phase 6
- Hoffmann et al., **Chinchilla** (2022) — why small models are usually
  undertrained rather than fundamentally limited

### Phase 4 — LLM inference and adaptation
- Dettmers et al., **LLM.int8()** (2022) — read first; the outlier-feature
  explanation makes the rest legible
- Frantar et al., **GPTQ** (2022)
- Lin et al., **AWQ** (2023)
- Xiao et al., **SmoothQuant** (2022)
- Hu et al., **LoRA** (2021)
- Dettmers et al., **QLoRA** (2023)
- Leviathan et al., **Fast Inference from Transformers via Speculative
  Decoding** (2023)
- Chen et al., **Accelerating LLM Decoding with Speculative Sampling** (2023)
- Kwon et al., **Efficient Memory Management for LLM Serving with
  PagedAttention** (2023) — the vLLM paper
- Dao et al., **FlashAttention** (2022)

### Phase 5 — perception and edge systems
- Radford et al., **Robust Speech Recognition via Large-Scale Weak Supervision**
  (2022) — Whisper
- Wu et al., **Machine Learning at Facebook: Understanding Inference at the
  Edge** (HPCA 2019) — still the best paper on real edge constraints

### Phase 6 — agents
- Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)
- Schick et al., **Toolformer** (2023)
- Lewis et al., **Retrieval-Augmented Generation** (2020) — the original framing

## Documentation worth reading properly

- **ONNX Runtime** — execution providers, EP fallback, profiling
- **HuggingFace `optimum`** — ONNX export, graph optimization, ORT quantization
- **HuggingFace `peft`** — LoRA and adapter mechanics
- **llama.cpp** — the k-quant scheme (`Q4_K_M` and friends), GBNF grammars
- **PyTorch `torch.ao.quantization`** — PTQ and QAT workflows
- **Core ML Tools** *(optional, Apple-specific)* — conversion, compute units
- **MLX / `mlx-examples`** *(optional, Apple-specific)*

## Video

- **MIT 6.5940 lectures** — the whole course, free
- **Stanford CS224N — NLP with Deep Learning** — the canonical NLP course, free
- **HuggingFace NLP Course** — free, hands-on, and uses this project's exact stack
- **Karpathy, *Let's build the GPT Tokenizer*** — two hours; the best preparation
  for Phase 3(a)
- **Karpathy, *Let's build GPT*** — two hours; worth it before Phase 3(b)
- **Jay Alammar, *The Illustrated Transformer*** — read before the paper, not after
- **Karpathy, *Neural Networks: Zero to Hero*** — if foundations need refreshing
- **Apple WWDC sessions on Core ML and the Neural Engine** *(optional)*

## Deliberately excluded

- **TensorFlow, TFLite, Keras** — out of scope by project convention. The
  PyTorch and HuggingFace equivalents cover the same ground.
- **TensorRT, OpenVINO deep-dives** — vendor-specific and not runnable on this
  machine. The ONNX Runtime EP abstraction covers the transferable concepts;
  revisit if hardware changes.
- **Kubernetes** — no Docker and no disk. Phase 7 learns control-plane concepts
  in Python instead.
