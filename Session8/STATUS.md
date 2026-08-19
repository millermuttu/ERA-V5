# Session 8 — where the work stands

Last updated: 2026-08-19, after concept 23.

## What the assignment asks for

`reference/session8-lesson.md` §18 is the spec, quoted as a blockquote. In short: a web app that
explains every attention mechanism from the lesson **visually**, starting from plain scaled
dot-product attention, with everything after it presented as a response to a limitation of what came
before, **ordered by when each technique was actually published** — not by teaching order and not
grouped into a taxonomy. Each entry answers three questions honestly: what does it buy, what does it
give up, when would you actually choose it. Deliverables: a live link, the GitHub repo, and a README
listing the sources for the chronology, because dates are where an agent sounds confident and is
wrong.

## The deliverable

`Session8/webapp/` — a static app, no build step, no dependencies. Open `index.html` over http
(ES modules need a server):

```bash
cd Session8/webapp && python3 -m http.server 8765
# then http://localhost:8765/index.html
# ?selfcheck=1 runs the whole assertion suite and paints pass/fail at the foot
# #<concept-id> deep-links a card, e.g. #mla
```

A real (tiny) transformer runs in the page: 32 dimensions, 4 heads, 2 blocks, seeded **untrained**
weights. Every concept is that same model with one thing changed. The weights being untrained is
load-bearing for the deck's honesty — several cards measure that their own subject does not appear on
an untrained model and say so rather than faking it.

## Progress: 23 of 26 concepts built

Built, in timeline order: `transformer`, `sinusoidal`, `relative-positions`, `learned-absolute`,
`transformer-xl`, `sparse-transformer`, `mqa`, `sliding-window`, `linear-attention`, `performer`,
`delta-rule`, `rope`, `alibi`, `flashattention`, `gqa`, `position-interpolation`, `ntk-aware`,
`yarn`, `attention-sinks`, `mamba`, `mla`, `parallel-deltanet`, `gated-deltanet`.

**Remaining, in order:**

| # | id | source | research note? |
|---|---|---|---|
| 2.23 | `nsa` | arXiv 2502.11089 | **no — needs the research pass** |
| 2.24 | `dsa` | DeepSeek-V3.2-Exp release note, no paper | **no** |
| 2.25 | `drope` | course cookbook only, no public source | **no** |

Then sections 3 (self-check) and 4 (verify and ship) of
`openspec/changes/session8-attention-timeline-webapp/tasks.md`, of which 4.4 (the README with the
chronology source table) is the one the assignment explicitly grades.

## How a concept gets built

One at a time: **research → build → verify → stop for review.** Never batch two.

1. **Research.** Read the primary source — actually fetch it, do not recall it. Verify the arXiv v1
   date against the abstract page; two records have already been corrected this way
   (`mqa` → 2019-11-06, `parallel-deltanet` → 2024-06-10, `ntk-aware` → 2023-06-29). Diff the
   versions if they differ materially; concept 18's card exists in the shape it does because YaRN v1,
   v2 and v3 are not the same paper. Write `webapp/docs/research/<id>.md` **before any code**. The
   note records: what was read (and what could not be reached), the mechanism in precise terms with
   quoted equations, the numbers that matter, `[measured here]` figures obtained by driving the app's
   own model from node, what the live view must let the reader do, **what the source does not
   establish**, and what the concept leaves behind in both directions.
2. **Build.** `webapp/app/cards/<id>.js`, imported in `app/cards/index.js`, plus a `built(id, answers,
   {text, to})` entry in the `LINKS` array of `app/data/mechanisms.js`. A concept with no card entry
   renders as a placeholder, which is how the deck shows work in progress honestly.
3. **Verify.** Run `?selfcheck=1` in a browser, sweep every control, check the console, and confirm
   the measured numbers match the research note. Add assertions for anything the card claims.
4. **Commit** with the `[ERA-V5][muttu]:` prefix (use the `era-commit` skill), then stop.

## The seam

`app/model/mixers.js` is where every mechanism plugs in, and it has stayed small on purpose:

- `softmaxMixer({ readable, bias, rotate })` — `readable(i, j)` is windows, sparsity, top-k and the
  streaming cache policy; `bias` is relative position and ALiBi; `rotate` is RoPE and everything
  built on it.
- `stateMixer({ write, decay, beta, phi, features, sumNorm, attnNorm })` — the fixed-size state
  family. `decay` and `beta` may be **functions of the token**, which is concept 20's whole
  contribution in one line; `beta` receives the same token arguments `decay` does, and `decay`
  applies to the add rule too — concept 23's two one-line extensions, which between them make
  Mamba2's row of that paper's Table 1 expressible.
- `rope({ base, stretch, dims, modulus })` — `stretch` may be a number or a function of the pair
  index (concept 18's ramp); `modulus` is the attention temperature folded into the rotation.
- `forward(tokens, { mixer, position, kvGroups, latent })` — `latent` replaces a block's keys and
  values wholesale, which a joint compression needs because it is shared across heads.

Adding a mechanism should mean configuring this seam, not extending it. Where an extension was
genuinely required it was one line and the research note said in advance what it had to be.

## House style for a card

Read two or three existing cards before writing one — `mla.js`, `mamba.js` and `attention-sinks.js`
are the current reference. What they have in common:

- **Nothing is asserted that the page can compute.** Every number on screen is either measured live
  from the model or explicitly quoted with its source and its conditions.
- **Quoted numbers are labelled as quoted**, and the conditions come with them. Several papers'
  headline figures compare more than one thing at once; the cards print the controlled version
  beside the famous one.
- **What the app cannot show gets its own panel** rather than a suggestive picture. Concept 19
  measures that this untrained model has no attention sink and reports the negative result; concept
  20 refuses to draw a Mamba block or animate a parallel scan.
- Structure: `prose({problem, mechanism})`, a `.formula` line, the shared `flowPanel`, three to five
  live panels, then `tradeBlock({buys, givesUp, chooseWhen})` and `plainBlock({pros, cons, verdict})`.
  The self-check enforces that the last two exist and that the plain-language costs are non-empty.

## Verification

`app/lib/selfcheck.js` — **69 assertions, all passing.** No framework: model invariants, every
mechanism's degenerate setting reducing to the baseline, the cost formulas reproducing the lesson's
6.44 GB / 51.54 GB, and the integrity of the chronology (unique ids, parseable dates, nothing
answering a limitation that had not happened yet, every built card rendering without throwing and
carrying its trade-off record).

Also checked by hand each time: no console errors, every control moves something, the shared sentence
propagates, and no sideways scroll at 390px.

## Open items

- **`.claude/` is gitignored**, so the `scrap-session` skill that `CLAUDE.md` points at exists only on
  this machine. Fine if intended.
- **Task 4.6 needs a person**: the repo's Pages source has to be set to GitHub Actions before the
  deployment can be confirmed, and nothing is pushed without being asked.
- The three remaining concepts include two with **no paper at all** — `dsa` (a release note) and
  `drope` (the course cookbook only, no public source, and its record carries a placeholder sort-key
  date that the card must present as a sort key rather than as a date).
