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

## Planning artifacts

Spec-driven change proposals (used for larger sessions) live under
[`openspec/`](openspec/) — `openspec/specs/` holds the current capability
specs, `openspec/changes/archive/` holds completed change proposals with
their design docs and task breakdowns.
