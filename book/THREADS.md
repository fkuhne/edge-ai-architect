# Threads

Patterns that recur across phases.

Chronological logs make these invisible — you notice a thing in Phase 2, hit something similar in Phase 4, and never connect them because they are four months and thirty files apart. Cross-phase threads are usually the most interesting material a technical book has, because they are what separates a book from a sequence of tutorials.

**How to use this file.** When something feels familiar, add a dated sighting under the relevant thread. Two sightings make a coincidence; three make a chapter. Append, don't rewrite — the accumulation is the evidence.

```markdown
## Thread name

One line on what the pattern actually is.

- **2026-09-14, Phase 1** — what happened, one line. [log entry](log/YYYY-MM-DD-slug.md)
- **2026-11-02, Phase 2** — what happened again, and what was different.
```

---

## Watchlist

*Candidates, not findings — hypotheses drawn from the "watch out" sections of the curriculum briefs, offered as things to keep an eye on. **Delete the ones that don't actually recur for you**, and add your own, which will be better because they will be real.*

**Did the measurement lie to me?** The curriculum warns about this in several unrelated places — timing without a device sync, a model silently falling back to CPU, swap masquerading as a quantization effect, RSS units differing by platform. Watch whether "the number was real but measured the wrong thing" turns out to be one pattern or four unrelated ones.

**Where portability leaks.** Every phase seems to have one seam where "write once, run anywhere" doesn't quite hold: `bitsandbytes` being CUDA-only, `onnxruntime-gpu` shipping as a separate package, quantization ops missing on MPS. Watch whether the leaks share a shape — and whether the abstraction was still worth it.

**The bottleneck was somewhere else.** Retrieval blamed on generation, queueing blamed on model latency, memory blamed on the model. Watch how often the first place you looked was wrong, and whether there is a reliable way to look in the right place first.

**Build the measurement before the thing.** Phase 0 builds the harness first; Phase 3 insists on a fixed eval set before the sweep; Phase 5 says build the eval harness before the agent. Watch whether this holds up in practice or whether it is advice that sounds good and gets skipped.

**Learning alongside an AI.** Where it accelerated things, where it short-circuited the learning, and what you changed about how you asked. This project already has one sharp instance of the failure mode. If you go with angle B in `PREMISE.md`, this thread is the spine rather than a thread.

---

## Active threads

*Move a watchlist item here once it has two real sightings, with the entries filled in.*
