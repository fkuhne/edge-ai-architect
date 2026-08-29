# Phase 6 — Platform: serving, lifecycle & fleet

**Weeks 24–28** · *How does this become a system rather than a pile of scripts?*

Five phases have produced models, runtimes, and pipelines. This one makes them
operable: something that can be deployed, observed, updated, and rolled back.
This is the architect-level layer, and it is only meaningful because the layers
underneath it were built first.

## Theory

- **Chip Huyen, *Designing Machine Learning Systems*** — MLOps foundations. The
  chapters on deployment, monitoring, and continual learning.
- **Model provenance and signing** — supply-chain concepts: content addressing,
  signature verification, promotion gates. Sigstore's model is a good reference
  even though the implementation here is simpler.
- **OpenTelemetry** — traces, spans, and context propagation.
- **MLPerf Inference** methodology, revisited: by now you have your own
  benchmark data, and it is worth comparing your methodology against theirs.

## Build — `projects/p06-edge-platform/`

> **How this phase works.** Ask Claude for scaffolding — component breakdowns
> and stubs for the fiddlier plumbing if useful — not a finished platform.
> The registry's promotion/rollback logic and the router's policy are the
> parts worth writing yourself. Ask for concept explanations, hints, or
> review; ask for a reference implementation only after a genuine attempt.
> Full version in [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md).

Four components, assembled from what already exists.

### Inference gateway

FastAPI in front of the Phase-2 `runners`. Routes by **policy**, not hardcoded
choice: latency target, model size, backend availability. The routing logic is
the interesting part — the HTTP layer is not.

Worth implementing: request batching, a queue with a bounded depth, and a
documented behaviour when the queue is full. Phase 4's back-pressure lesson
applies again.

### Model registry

Versioned local models with manifests. Each entry carries:

- content hash (reuse `edgebench.provenance.hash_path`)
- a signature, verified before load
- quantization, backend compatibility, benchmark results from `bench.db`
- a promotion state: `candidate` → `staging` → `production`

**Rollback must work.** A model can be promoted, found wanting, and rolled back
without touching the gateway. That single property is what makes the registry
worth building.

### Telemetry

OpenTelemetry traces through gateway → router → runtime, plus the `edgebench`
metrics already being collected. A small dashboard — even a static HTML page
reading `bench.db` — closes the loop.

### "Fleet of one" control plane

One machine, but the full shape: push a policy, roll a model forward, roll it
back, report health. The point is the interfaces, not the scale. A control plane
that manages one device correctly generalizes; one that assumes a single device
does not.

## Done when

- [ ] Gateway routes across backends by policy, with routing decisions traced.
- [ ] A model is promoted and rolled back **without touching the gateway**.
- [ ] Signature verification refuses a tampered model, with a test proving it.
- [ ] A trace shows one request crossing gateway → router → runtime with timings.
- [ ] The dashboard renders the accumulated results from all six phases.

## Watch out

- **Scope.** This phase could absorb a year. The deliverable is a working
  skeleton that demonstrates the interfaces, not a production platform. If the
  registry works and rollback is real, the phase succeeded.
- **Do not reach for Kubernetes.** There is no Docker on this machine and no
  room to install it. The concepts — control plane, declarative desired state,
  reconciliation loops — are learnable in a few hundred lines of Python and
  transfer better than YAML would.
- **Resist rewriting the earlier phases.** Wrap them. If a Phase-2 backend needs
  changing to be servable, that is a finding worth writing down — it means the
  abstraction was wrong, and that is the most valuable thing this phase can
  teach.

## Comprehension checkpoint

Answer these in your own words in `notes/` before calling the phase done:

- What specifically makes "rollback without touching the gateway" hard to get
  right, and what does the registry have to guarantee for it to work?
- Why is a bounded queue with a documented drop/reject policy better than an
  unbounded one, even though the unbounded one never rejects a request?
- What's actually being generalized when you build a "fleet of one" control
  plane correctly, versus one that quietly assumes a single machine?
- If a Phase-2 backend needed to change to become servable through the
  gateway, what would that tell you about the original abstraction?

---

## Capstone (week 28+)

Write the architecture document the Distinguished Technologist posting describes:

- **Cross-layer contracts** — the interfaces between runtime, registry, gateway,
  and control plane, and why they sit where they do.
- **ADRs** for the real decisions: why ONNX Runtime as the portable path, why
  GGUF over MLX, what the LoRA/QLoRA portability seam cost, where the abstraction
  leaked.
- **The data** — seven months of Pareto curves from `bench.db`, showing the
  quality/latency/memory/energy tradeoffs measured rather than asserted.

That document, backed by working code and your own measurements, is the artifact
worth discussing inside HP.
