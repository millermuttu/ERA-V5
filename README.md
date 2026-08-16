# ERA V5

Coursework and deliverables for ERA V5 — building the data and training
pipeline behind an LLM, session by session.

| Session | Topic | Deliverable | Netlify |
|---|---|---|---|
| [Session 1](Session1/) | Interactive proofs — linear vs. ReLU, depth without nonlinearity, embeddings from next-token prediction, memorization vs. generalization | [`index.html`](Session1/index.html) | [Link](https://era-v5-session1-mallikarjun.netlify.app/) |
| [Session 2](Session2/) | BPE tokenizer from scratch | [`index.html`](Session2/index.html) (playground) · [`src/`](Session2/src/) | [Link](https://era-v5-session2-mallikarjun.netlify.app/) |
| [Session 3](Session3/) | Data strategy for a 40B-token India-first corpus | [`index.html`](Session3/index.html) | [Link](https://era-v5-session3-mallikarjun.netlify.app/) |
| [Session 4](Session4/) | Data cleaning & deduplication, applied to CC-News | [`README.md`](Session4/README.md) | [Link](https://era-v5-session4-mallikarjun.netlify.app/) |
| [Session 5](Session5/) | Data mixture & curriculum plan for the V5 pretraining run | [`submission.md`](Session5/submission.md) | - |
| [Session 6](Session6/) | Training data execution system — shards/manifests through checkpoint, crash, resume, replay, fork, audit | [`README.md`](Session6/README.md) | - |
| [Session 7](Session7/) | Embeddings and model internals — the Kronecker byte codec, its collisions, and five open "V2" problems | [`README.md`](Session7/README.md) | - |
| [Session 8](Session8/) | Modern attention variants — 25 mechanisms in launch order, each a reply to what the last one could not do | [`webapp/`](Session8/webapp/) | GitHub Pages, link below |

## Session 8 — how attention got here

A static web app: every attention mechanism from the session on one timeline, ordered by the date it
was launched rather than the order it was taught, so the page reads as the field changing its mind.
It opens on scaled dot-product attention and works forward; six mechanisms compute their arithmetic
live in the browser, and every date is the arXiv v1 submission date, checked against the source
rather than recalled — the three entries with no paper behind them say so on the card.

`?selfcheck=1` runs 25 assertions over both the arithmetic and the chronology data.

## Session 7 — the Kronecker V2 problems

The Session 7 assignment offered five open problems and asked for **one**. All five are
solved, each with code that proves it and a README that states what it does *not* claim.

| | problem | solution | the result |
|---|---|---|---|
| [P1](Session7/kronmath/) | arithmetic inside the embedding | additive **and multiplicative** characters of `Z/p` — 22 fixed dims on top of 8192 | exact `+ − × ÷` and powers on `Z/15015`; a constructed readout scores 1.0000 on operands it never saw |
| [P2](Session7/kronmm/) | one codec for text, images, audio | the filler alphabet is already universal (all three are 8-bit); only the role space differs | one `Linear` for all three, no per-modality table; the role must match the modality's invariance axis, measured |
| [P3](Session7/kronbudget/) | the 32-byte position budget | the waste premise is wrong — `R` buys projection *width*, not per-token space; so fix the crop at fixed `R` | the per-script collision count the lesson asks for, and wrapping at R=16 that collides less than the release at R=32 |
| [P4](Session7/kronfourier/) | a real Fourier alternative | Fourier HRR — phasor per byte, position as phase, word as sum | `D` independent of length; exact recovery to 256 bytes vs 32; the suffix limitation fixed by shift equivariance, not by being Fourier |
| [P5](Session7/kroninv/) | drop the output head | `κ` is sparse, so recovering it is compressed sensing — no inverse of `W_proj` required | `d ≈ 64·L`, and `V/D` = 122× fewer output params at a 1M vocabulary — at a cost of more than half the top-1 accuracy |

Three of the five record a prediction that the measurement killed: fp16 was expected to break
the Fourier superposition (it does not), the byte codec's locality advantage turned out to be
mostly an artifact of its z-normalisation, and removing collisions turned out to be necessary
but not sufficient downstream. Two silent bugs are documented alongside the numbers that
caught them.

## Planning artifacts

Spec-driven change proposals (used for larger sessions) live under
[`openspec/`](openspec/) — `openspec/specs/` holds the current capability
specs, `openspec/changes/archive/` holds completed change proposals with
their design docs and task breakdowns.
