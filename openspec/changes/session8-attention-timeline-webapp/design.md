## Context

See `proposal.md` — Why. Constraints that actually shape the approach:

- The repo has no JS toolchain: no `package.json`, no `node_modules`, no bundler config anywhere. Sessions 1–4 are hand-written single-file static HTML with inline `<style>` and vanilla JS; Sessions 3 and 4 share one visual system (`Session4/index.html` `:root` palette, Archivo / Hanken Grotesk / IBM Plex Mono).
- The course's own widgets under `Session8/reference/widgets/` are the same shape — vanilla, single file, real arithmetic computed in the page (e.g. `widget_0b_attention_flow.html` runs a 32-dim, 4-head, 2-block transformer live). They are read-only reference material, not code to import.
- The content is already transcribed: `Session8/reference/session8-lesson.md` holds every formula and figure this app must agree with (6.44 GB / 51.54 GB cache example, the 140 = 140 regrouping, 40 → 55 versus 95, the 32× extension, DDDGDDDG).
- The single largest correctness risk is not code, it is dates. The assignment calls this out by name.
- The user asked that everything web-app-related stay inside `Session8/webapp/`.

## Goals / Non-Goals

**Goals:**

- One data record per mechanism, one place, driving every view — timeline, card, source table.
- Demos whose numbers are computed, so they cannot drift from the lesson silently.
- A chronology whose every date can be traced to a primary source, with unverifiable entries visibly marked rather than quietly rounded into fact.
- Something that stays runnable in three years: open the folder, serve it, it works.

**Non-Goals:**

- No build step, framework, router library, or package manager. No dark/light toggle — the Session 3/4 system is dark by design.
- Not a reimplementation of the course widgets. They are reference; this app is the deliverable and stands alone.
- Not a benchmark. Demos illustrate mechanism and cost arithmetic, never claim model-quality results.

## Decisions

**Vanilla ES modules over a bundler.** Native `<script type="module">` gives file separation across ~25 mechanism records and 6 demos without adding npm to a repo that has never had it. Cost: it needs an HTTP server (`file://` blocks module loads) — acceptable, since the deliverable is hosted anyway, and the README states the one-line serve command. Alternative considered: one large single-file page like Sessions 3/4. Rejected because the chronology data is the thing that gets edited and re-verified most, and burying it in a 4,000-line HTML file makes that edit dangerous.

**`app/data/mechanisms.js` is the single source of truth.** The timeline, the card, the README source table, and the self-check all read the same array. A mechanism cannot appear in the timeline with a trade-off missing from its card, because they are the same record. Alternative considered: JSON + `fetch`. Rejected — a module import is one fewer failure mode under a sub-path deploy, and JSON cannot carry comments explaining an unverified date.

**Verification is a data field, not a footnote.** Each record carries `verified: true|false` plus its source. The renderer keys the unverified badge off that field, so a date nobody checked cannot be displayed as if it were checked. DroPE is the deliberate test case: the lesson establishes 8K → 256K and 32×, and explicitly not the algorithm, so the card states both what the record supports and what it does not.

**Six live demos, the rest illustrated.** The mechanisms where arithmetic *is* the lesson get real computation: baseline attention (mask on/off), softmax-off regrouping plus the delta rule, RoPE relative rotation, the KV-cache formula, top-k selection cost, sequence compression. YaRN, NTK scaling, MLA and DroPE do not become clearer as live arithmetic — they get diagrams and honest cards, which the assignment explicitly allows. Alternative considered: a demo for all 25. Rejected as work that buys illustration, not understanding.

**One shared `lib/mathx.js`, and only what the demos call.** `dot` and `softmax`. The demos are 6×4 and 2D, so a general `matmul` has no caller yet, the causal mask is `j > i ? -Infinity : 0` written inline where the scores are computed, and demo weights are literal arrays — which makes every reload reproducible without a seeded PRNG. Demos import the shared softmax; none carries its own. That is what makes the self-check meaningful: it tests the function the demos actually run.

**No router, no SVG builder library.** Every card renders into the document flow with `id="<mechanism id>"`; the year rail is anchor links and the browser does the navigating, which also makes the baseline card the landing view for free. Diagrams are literal SVG in the module that needs them — a builder API for `line`/`arrow`/`grid` before any diagram exists would be an abstraction with no second caller. Extract one only if the repetition turns up.

**Diagram where the text cannot carry it, not one per card.** The assignment says a clear static explanation beats a clever interaction; text plus the trade-off record already *is* the explanation. Four or so mechanisms are genuinely spatial (sliding window, attention sinks, MLA, NSA block selection) and get a drawing. A blanket "no card is text-only" rule buys decoration, not understanding.

**The timeline starts at the Transformer.** The assignment demands both "start with standard scaled dot-product attention" and "arrange them chronologically" — which collide if RNN-era attention (Bahdanau 2014, Luong 2015) and pre-Transformer learned positions are timeline entries, because then the baseline is fourth in a date sort. Beginning at 1706.03762 makes the baseline both the first card read and the earliest date, so no prologue or ordering exception is needed. The 2014–2015 lineage is not erased: it is one line inside the baseline card saying attention predates the Transformer and the Transformer is what removed the recurrence. Learned absolute position tables stay in the timeline at their BERT-era date, which is where they actually became the default and where the length wall came from.

**`thread` is a card label, not a lane.** The field's shift over time should be readable from the chronology itself. The assignment explicitly says not to group the techniques into a neat taxonomy first, so colour lanes and a legend drawn over the timeline work against the requirement they were meant to serve.

**Self-check as a query parameter, not a test framework.** `?selfcheck=1` runs asserts against `lib/mathx.js` and the data array and paints a pass/fail banner. It covers the math the lesson pins down *and* the data integrity rules from the spec. Adding a runner, fixtures, or a CI test job would be more machinery than the check itself.

**Reuse the Session 3/4 visual system.** Same palette tokens and font stack, different layout (sticky year rail + cards, not a scroll report). The repo reads as one body of work, and no time goes into inventing a second look.

**GitHub Pages via workflow, folder kept self-contained.** Pages cannot serve an arbitrary sub-directory from a branch, so publishing `Session8/webapp/` needs `upload-pages-artifact` + `deploy-pages` — about 25 lines, the repo's first `.github/`. Because the folder is self-contained and all asset paths are relative, the same folder still works as a Netlify drop, which is what Sessions 1–4 use.

## Risks / Trade-offs

- **A confidently wrong launch date** — the exact failure the instructor warned about, and the one that most damages the deliverable → each date is fetched from its primary source and compared before shipping; anything unfetchable ships as `verified: false` with the real provenance named.
- **Course-record-only claims (DroPE, LightningLM V4 numbers, DeepSeek V4) presented as public fact** → these render in the unverified style and are attributed to the course cookbook in both the card and the README table.
- **Demo arithmetic drifting from the lesson figures** → the self-check pins the numbers the lesson states (6.44 GB / 51.54 GB, 140 = 140, 55 versus 95); a drift fails the check rather than shipping quietly.
- **ES modules need a server; opening `index.html` from disk shows a blank page** → README leads with the serve command, and the Pages deploy is the primary way anyone views it.
- **Google Fonts is an external request; if it fails the page falls back** → font stack ends in `system-ui`, so layout survives; nothing else is external.
- **Pages publishes the whole app publicly at a repo-derived URL** → the content is coursework built from lesson material and public papers, which is the assignment's intent.
- **25 hand-written mechanism records is the bulk of the effort and the part most likely to be rushed** → the spec's completeness rules are enforced by the self-check, so an under-written record fails loudly instead of looking finished.

## Migration Plan

Additive only — no existing file changes behaviour. Order: data module and `lib/` first (they gate everything), then shell and renderer, then demos, then date verification, then the Pages workflow and the root README row. Rollback is deleting `Session8/webapp/` and the workflow; nothing else in the repo depends on them.
