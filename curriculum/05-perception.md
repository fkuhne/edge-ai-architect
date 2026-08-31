# Phase 5 — Streaming multimodal perception

**Weeks 19–22** · *How do I hold a real-time latency budget end to end?*

Everything so far measured one model in isolation. Real edge systems are
pipelines, and a pipeline's latency is not the sum of its parts — it is whatever
the slowest stage plus the queueing between stages produces. This phase builds
one and instruments it properly.

## Theory

- Radford et al., ***Whisper*** — the architecture, and why the 30-second window
  matters for streaming.
- **VAD and streaming ASR**: chunking, buffering, and overlap strategies. The
  practical literature here is mostly in `whisper.cpp` and `faster-whisper`
  issue threads, which is worth reading directly.
- Wu et al., ***Machine Learning at Facebook: Understanding Inference at the
  Edge*** (HPCA 2019) — still the best paper on what actually constrains edge
  inference at scale. Short and worth full attention.
- Queueing basics: why a stage at 95% utilization has unbounded latency. Little's
  Law is enough theory.

## Build — `projects/p05-streaming-perception/`

> **How this phase works.** Ask Claude for scaffolding — a project brief
> breaking the pipeline into stages with a task list, and stubs for the
> trickier plumbing (queues, buffering) if useful — not a finished pipeline.
> The back-pressure policy in particular should be a decision you make and
> defend, not one handed to you. Ask for concept explanations, hints, or
> review; ask for a reference implementation only after a genuine attempt.
> Full version in [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

A **local real-time meeting assistant**:

```
mic capture → VAD → streaming ASR → local LLM summarization
```

- **Capture**: `sounddevice` or PyAudio. Fixed sample rate, explicit buffer size.
- **VAD**: Silero VAD (small, PyTorch, portable) or WebRTC VAD. Segments speech
  so ASR is not transcribing silence — the cheapest large win available.
- **ASR**: `faster-whisper` (CTranslate2) or `whisper.cpp`. **Both are portable**
  to CUDA, which is why they are preferred over an Apple-only path.
- **Summarization**: the Phase-4 quantized LLM through the Phase-2 `Runner`.

### The real work

**Declare a latency budget before writing the pipeline.** For example: speech end
to transcript within 2 s, transcript to running summary within 10 s. Then
instrument every stage with `edgebench.MemoryTracker` and `GenerationTimer` and
find out whether it holds.

Use `edgebench.record()` per stage with a shared label prefix so the breakdown
queries cleanly:

```python
edgebench.record(label="meeting-asr",  project="p04-streaming-perception", ...)
edgebench.record(label="meeting-vad",  project="p04-streaming-perception", ...)
edgebench.record(label="meeting-llm",  project="p04-streaming-perception", ...)
```

The interesting failures are structural, not per-model:

- **Back-pressure.** ASR slower than real time means the audio queue grows without
  bound. What is the policy — drop, degrade to a smaller model, or block capture?
  Choose deliberately and measure the choice.
- **Contention.** On 8 GB, ASR and the LLM resident simultaneously may not fit.
  Whether to keep both loaded or swap between them is a real architecture
  decision with a measurable answer.

## Done when

- [ ] Runs live on microphone input without unbounded queue growth.
- [ ] Holds the stated latency budget on a 10-minute recording.
- [ ] Per-stage breakdown in `results/bench.db` shows where the time goes.
- [ ] A deliberate, documented back-pressure policy — tested by deliberately
      overloading it.
- [ ] `notes/` records what happened when both models were resident at once.

## Watch out

- **Fix a test recording early.** Live microphone input is not reproducible, and
  a pipeline tuned against unrepeatable input cannot be compared against itself
  a week later.
- **Wall-clock, not model latency.** A model at 200 ms that runs every 5 s of
  audio is fine; the same model in a stage that receives audio every 100 ms is
  not. Measure arrival rate alongside service time.
- **Audio format bugs outnumber model bugs**, by a lot. Sample rate mismatches
  and int16/float32 confusion produce transcripts that look like a model problem
  and are not.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- Why can a pipeline's end-to-end latency be much worse than the sum of each
  stage's individually-measured latency?
- You chose a back-pressure policy (drop / degrade / block). What does each
  option actually cost the user, and how would someone listening notice the
  difference?
- Why does testing against a fixed recording matter more here than in earlier
  phases, where synthetic benchmark inputs were fine?
- If ASR and the LLM can't both stay resident in memory at once, what's the
  real latency cost of swapping one out and back in — and how did you
  measure that rather than guess it?
