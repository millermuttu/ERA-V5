## Why

ERA V5 Session 8 (*Modern Attention Variants*) is scraped into `Session8/reference/` but has no deliverable. The assignment (`Session8/reference/session8-lesson.md` §18) asks the cohort to build a hosted web app that explains every attention mechanism covered in the session visually, ordered **chronologically by when each technique was launched** rather than by teaching order, so the app shows the field thinking: what existed, what problem people hit, what somebody proposed, what it fixed, and what new trade-off it introduced. The instructor explicitly warns that launch dates are exactly where an AI agent sounds confident and is wrong, and asks for a README listing the sources behind the chronology.

## What Changes

- Add a static, no-build web app under `Session8/webapp/`: a chronological timeline of ~25 attention mechanisms, each rendered as a card answering what the mechanism buys, what it gives up, and when it would actually be chosen.
- The app opens on standard scaled dot-product attention (Q×K → scale → mask → softmax → weighted sum of V) as the baseline; every later mechanism is framed as a response to a limitation of what came before, never as a standalone feature card.
- A single data module `Session8/webapp/app/data/mechanisms.js` is the sole source of truth for the chronology: name, ISO launch date, primary source (arXiv id + URL), a `verified` flag, thread, problem, mechanism, `buys`, `givesUp`, `chooseWhen`, and an optional demo key. Every view is a projection of that array.
- Six mechanisms get live-computed demos with real arithmetic in the page (baseline attention with a causal-mask toggle; the softmax-off regrouping and the delta rule; RoPE relative rotation; the KV-cache formula across MHA/GQA/MQA and cache precision; top-k selection; sequence compression plus top-k blocks). Every other mechanism gets a static inline-SVG diagram plus its trade-off card, which the assignment explicitly permits.
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
