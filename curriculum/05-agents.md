# Phase 5 — Local agentic runtime & RAG

**Weeks 19–23** · *How do I make a small model reliable enough to trust with tools?*

A 3B model that can call tools is a different engineering problem from a frontier
model that can. It will fail more often, in more ways, and the interesting work is
in the harness around it rather than the model itself. That is precisely the edge
agentic problem the job descriptions describe.

## Theory

- Yao et al., ***ReAct*** — the reasoning/acting loop most agent designs descend
  from.
- Schick et al., ***Toolformer*** — tool use as a learned capability.
- **Chip Huyen, *AI Engineering*** (2025) — the strongest single text on evals,
  guardrails, and LLM application architecture. Read the evaluation chapters
  properly; they are the point of this phase.
- Agent evaluation methodology generally: **task success rate over anecdote**.

## Build — `projects/p05-local-agent/`

An agent over your own documents. Components:

- **Retrieval**: `sentence-transformers` for embeddings (a small model — 
  `all-MiniLM-L6-v2` is ~90 MB and adequate), `sqlite-vec` for the index. Keeping
  the vector store in the same SQLite file as `bench.db`'s sibling is a
  reasonable simplification at this scale.
- **Generation**: the Phase-3 quantized LLM through the Phase-2 `Runner`.
- **Corpus**: personal notes, the papers from this curriculum, or HP-public
  material. Nothing leaves the machine.

### The parts that matter

**A capability/permission layer.** Tools declare what they can touch; the agent is
granted a subset; calls outside the grant are refused by the harness rather than
by prompting. This is the sandboxing theme from the Distinguished Technologist
posting, and building it small is how you learn why it is hard.

```python
@tool(capabilities={"fs:read"}, paths=["~/notes"])
def read_note(path: str) -> str: ...
```

**An eval harness, built before the agent works well.** A fixed suite of 20–30
tasks with checkable outcomes, scored as end-to-end success. Without it there is
no way to tell whether a prompt change helped. With it, the phase has a number
that goes up.

**Budgets and failure handling.** Latency and token budgets per task, retries with
backoff, and a defined behaviour when the model loops — which small models do,
frequently.

**A routing policy.** Decide local-vs-escalate explicitly: query complexity,
context length, confidence, latency budget. **Keep it local-only**; stub the
escalation path and document what would trigger it. The decision framework is the
transferable lesson, and leaving the door closed costs nothing.

## Done when

- [ ] Fixed task suite runs repeatedly with a tracked success rate in
      `results/bench.db`.
- [ ] Tightening a guardrail moves that number **measurably** — the harness is
      sensitive enough to detect a real change.
- [ ] A tool call outside its granted capabilities is refused by the harness, and
      there is a test proving it.
- [ ] Per-task latency and token budgets enforced, with a documented policy for
      exceeding them.
- [ ] `notes/` documents the routing policy and what would trigger escalation.

## Watch out

- **Small models fail at tool syntax constantly.** Constrained decoding or
  grammar-based generation (llama.cpp supports GBNF grammars) is usually a better
  fix than more prompt engineering. Try both; measure.
- **Retrieval quality dominates.** Most "the agent is dumb" failures are the
  retriever returning the wrong chunks. Evaluate retrieval separately — 
  recall@k on a fixed query set — before blaming generation.
- **Build the eval harness first.** It is tempting to iterate on the agent until
  it feels good and add evals later. That ordering wastes weeks; the whole point
  is having a number.
