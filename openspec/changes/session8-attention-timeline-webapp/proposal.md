## Why

ERA V5 Session 8 (*Modern Attention Variants*) is scraped into `Session8/reference/` but has no deliverable. The assignment (`Session8/reference/session8-lesson.md` §18) asks the cohort to build a hosted web app that explains every attention mechanism covered in the session visually, ordered **chronologically by when each technique was launched** rather than by teaching order, so the app shows the field thinking: what existed, what problem people hit, what somebody proposed, what it fixed, and what new trade-off it introduced. The instructor explicitly warns that launch dates are exactly where an AI agent sounds confident and is wrong, and asks for a README listing the sources behind the chronology.

## What Changes

- Add a static, no-build web app under `Session8/webapp/`: a chronological deck of 26 attention mechanisms, one concept in view at a time, moved through by next/previous controls, the arrow keys, or a timeline slider that is itself the chronology. Each concept answers what the mechanism buys, what it gives up, and when it would actually be chosen — and closes with the same verdict written in plain language, no formulas and no unexplained jargon, for a reader who has not taken the course.
- The app opens on standard scaled dot-product attention (Q×K → scale → mask → softmax → weighted sum of V) as the baseline; every later mechanism is framed as a response to a limitation of what came before, never as a standalone feature card.
- Each card shows its mechanism against plain attention on the same input, carries playback (play, pause, step through tokens, heads and layers), and a live cost readout. A single data module `Session8/webapp/app/data/mechanisms.js` stays the sole source of truth for the chronology and now also carries each entry’s mixer, defaults and controls.
- Every mechanism is implemented as a configuration of one real transformer that runs in the page — 32 dimensions, 4 heads, 2 blocks, seeded weights — over a sentence the reader edits. A mechanism changes which keys a query may read, how position enters the score, how many key/value heads are stored, or how the past is compressed into state, and the reader watches the attention pattern, the cost ledger and the next-token distribution move. Every displayed number comes from that forward pass or from a cost formula; nothing is transcribed.
- Every date in the data module is verified against the primary source (arXiv v1 submission date) during implementation. Entries with no paper — NTK-aware base scaling, the DeepSeek V3.2-Exp release, and DroPE — stay `verified: false` and render with a badge naming what the source actually is. DroPE additionally carries the lesson's own evidence boundary: the record establishes 8K → 256K and 32×, not the algorithm.
- Add a self-check reachable at `?selfcheck=1`: assertions over the math (softmax rows sum to 1, masked future weight is exactly 0, the regrouped path equals the direct path with softmax off, the delta rule lands on 55 where add-only lands on 95, the KV formula reproduces the lesson's 6.44 GB / 51.54 GB) and over the data (unique ids, parseable dates, no card missing its trade-off answers, every demo key resolves). No test framework.
- Add `Session8/webapp/README.md`: the assignment statement, the live link, the reproduce/serve command, and the full chronology source table that the assignment asks to submit. Everything the web app needs lives under `Session8/webapp/`; nothing app-related is written to `Session8/` itself.
- Add `.github/workflows/pages.yml` publishing `Session8/webapp/` to GitHub Pages, and index Session 8 in the root `README.md`.

## Capabilities

### New Capabilities
- `attention-timeline-webapp`: a hosted, chronologically ordered explanation of attention mechanisms — the sourced chronology data contract, the baseline-first narrative ordering, the honest per-mechanism trade-off record, the live-computed demos whose numbers match the lesson, and the self-check that keeps both the math and the data record from silently rotting.

### Modified Capabilities
(none — the only existing capability, `data-execution-system`, is Session 6's pipeline and is untouched)

## Impact

- New directory tree, entirely self-contained: `Session8/webapp/` (`index.html`, `README.md`, `app/styles.css`, `app/main.js`, `app/data/`, `app/lib/`, `app/demos/`).
- New CI: `.github/workflows/pages.yml` — the repo has no `.github/` today. This diverges from the existing convention of one manually deployed Netlify site per session (Sessions 1–4, recorded in the root README's `Netlify` column); keeping the app self-contained under `Session8/webapp/` means a Netlify drop of that folder still works unchanged.
- Root `README.md` gains the Session 8 row and its live link.
- Dependencies: none. Vanilla ES modules, no npm, no build step, no framework. The only external request is the Google Fonts link already used by Sessions 3 and 4.
- Reuses the Session 3/4 visual system (the `:root` palette and the Archivo / Hanken Grotesk / IBM Plex Mono stack from `Session4/index.html`) rather than introducing a new one.
- No changes to Sessions 1–7 or to `Session8/reference/`, which stays verbatim scraped course material.
