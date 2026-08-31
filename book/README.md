# book/ — writing this as it happens

A place to document the learning *while it is happening*, in case it becomes a book later. The repo is already the companion code; this is where the narrative lives.

## The one rule

**Claude does not write anything in this folder.** Not entries, not retrospectives, not chapter drafts. The structure, templates, and prompts here are scaffolding — everything with substance in it gets written by Felipe.

That is not a stylistic preference. The entire value of this folder is that it records something a language model cannot generate: what a specific person actually believed, got wrong, and figured out, in order, on real hardware. Plausible-sounding prose about learning edge ML is worthless — the internet has plenty. A real record of confusion is not.

What Claude *can* do here, on request: react to a draft, point out where an explanation is unclear, suggest a structure, ask questions that surface something the entry left implicit, or help you decide what a finding actually means. Editing and interrogation, yes. Ghostwriting, no.

## Three layers, different lifespans

The mistake most "I'll write a book from my notes" projects make is keeping only one layer — usually a chronological log — and discovering two years later that turning it into a book means rewriting all of it from scratch. Three layers avoids that.

### 1. `log/` — raw, same-day, unedited

Written the day it happened, in whatever state. Includes dead ends, wrong theories, and things that turned out to be embarrassing. Not for readers. **This is the irreplaceable layer**: the single thing that evaporates fastest is *what you believed before you knew the answer*, and once you understand something you genuinely cannot reconstruct what it felt like not to.

Cadence: whenever a session produced a surprise, a stuck point, or a decision. Not every day. Skipping a boring day is fine; skipping the day you spent three hours on a bug is not.

### 2. `retrospectives/` — one per phase, at phase end

Written with hindsight, while still fresh. **This is the bridge layer, and it is the one that makes a book actually happen.** A phase retrospective is already roughly chapter-shaped: it knows how the story ended, it can say what mattered and what was noise, and it can be honest about the parts that were harder than they should have been.

Eight phases, eight retrospectives, and you have a real draft instead of a pile of notes and a large unpleasant task.

### 3. The manuscript — much later

Deliberately does not exist yet. Starting it now would mean writing chapters about things you have not learned, which is exactly the failure mode this whole project has been correcting for. It gets created somewhere around Phase 4, once there is enough material to know what the book is actually about.

Plus `THREADS.md`, which is not a layer so much as an index: recurring patterns that span phases. Chronological logs make these invisible, and they are usually the most interesting thing a technical book has to offer.

## Workflow

```
during a phase   ->  log/YYYY-MM-DD-slug.md      (raw, same day)
                     THREADS.md                   (append when something recurs)
at phase end     ->  retrospectives/NN-name.md   (with hindsight, chapter-shaped)
around Phase 4   ->  start deciding what the book is
```

## Getting started

1. Fill in `PREMISE.md` — even roughly. Knowing who you are writing for changes what you notice.
2. Write your first log entry using `log/TEMPLATE.md`.

A suggestion for that first entry, since it already happened and is genuinely good material: **the moment in this project where you told the AI it was doing too much of the work.** Phase 0 arrived fully built, and you pushed back — "how would I learn if you do everything to me?" That is a real lesson about learning with an AI, it is unusually timely, and if you do not write it down now the specifics will blur into a generality within a month.
