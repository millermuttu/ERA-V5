# How attention got here

**Session 8 assignment.** *"Work with your AI agent and build a web app that explains every
attention mechanism we covered today visually… Do not arrange the techniques in the order I taught
them today. Arrange them chronologically, by when the technique actually appeared… For each
technique, answer three questions: What does it buy? What does it give up? When would I actually
choose it?"*

A static page: 25 attention mechanisms in launch order, starting from *Attention Is All You Need*,
each framed as a reply to something the previous ones could not do. Six of them compute their
arithmetic live in the page; the rest are prose, a diagram where the shape matters, and an honest
trade-off card.

- **Live:** *(deployed on push — link added once the first GitHub Pages run is green)*
- **Repository:** https://github.com/millermuttu/ERA-V5/tree/master/Session8/webapp
- **Lesson it implements:** [`../reference/session8-lesson.md`](../reference/session8-lesson.md)

## Run it

ES modules need HTTP; opening `index.html` from disk will not work.

```bash
python3 -m http.server 8000     # from this directory
# http://localhost:8000/
# http://localhost:8000/?selfcheck=1   → runs the assertions and paints the result
```

## What is in it

```
index.html               shell: header, sticky year rail, card container
app/styles.css           Session 3/4 palette and type stack
app/main.js              sorts by date, renders rail + cards, mounts demos. No router:
                         cards carry ids and the browser handles #anchors
app/data/mechanisms.js   THE CHRONOLOGY — one record per mechanism, single source of truth
app/lib/mathx.js         dot, softmax. Everything else is local to the demo that needs it
app/lib/selfcheck.js     25 assertions over the arithmetic and the data
app/demos/*.js           six live demos + the four static diagrams
```

`mechanisms.js` is the only file that needs editing to change what the page says. The rail, the
cards, the badges and the table below are all projections of it.

## The demos, and what each one is for

| demo | on which card | the point |
|---|---|---|
| `attention` | the Transformer | 6 tokens, 4 dims, real Wq/Wk/Wv. Turn the causal mask off and watch weight land on tokens that have not happened yet |
| `linear` | linear attention, delta rule, Gated DeltaNet | the lesson's `q=2, k=[.5,1,1.5], v=[10,20,30]`: both routes give 140 with softmax off and stop agreeing with it on. Below it, 40 → 95 against 40 → 55 |
| `rope` | RoPE | move both tokens by the same amount — the arrows sweep round together, the gap holds, the score does not move |
| `cache` | MQA, GQA, MLA | `2 × layers × kv_heads × head_dim × T × batch × bytes`, reproducing the lesson's 6.44 GB and 51.54 GB |
| `topk` | DeepSeek sparse attention | values summed falls with k; keys scored to choose k never moves |
| `compress` | NSA | storing fewer positions and reading fewer of them are two different savings |

## The self-check

`?selfcheck=1` runs 25 assertions and prints pass or fail at the foot of the page. It covers the
arithmetic the lesson pins down — softmax rows summing to one, masked future weight being exactly
zero, the two routes agreeing only without softmax, the delta rule landing on 55, the cache formula
returning 6.44 GB and 51.54 GB — and the integrity of the chronology: unique ids, parseable dates,
no entry missing its buys / gives-up / when, every demo and diagram reference resolving, the
baseline being the earliest entry, and nothing marked verified unless a paper backs it.

That last group is the one that matters most. A card silently missing its trade-off is the failure
this assignment is graded on, so it fails a check rather than looking finished.

## Sources for the chronology

Every date below is the **arXiv v1 submission date**, read from the arXiv API rather than from
memory. Two of my first-pass dates were wrong and were corrected by that check: MQA (Nov 6, not
Nov 5, 2019) and parallelizable DeltaNet (Jun 10, not Jun 11, 2024).

Three entries have no paper behind them. They are marked unverified in this table and carry a badge
on their card saying what the source actually is. DroPE's row is the clearest case: the Session 8
lesson records an 8K → 256K result, a 32× extension and that it was applied before annealing, and
states explicitly that the record does **not** establish the algorithm — so the card does not invent
one, and its date field is a sort key, not a claim.

| # | date | mechanism | source | provenance |
|---|---|---|---|---|
| 1 | 2017-06-12 | Scaled dot-product attention, multi-head, sinusoidal positions | [Vaswani et al., Attention Is All You Need — arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | verified against arXiv v1 |
| 2 | 2018-03-06 | Relative position representations | [Shaw et al. — arXiv:1803.02155](https://arxiv.org/abs/1803.02155) | verified against arXiv v1 |
| 3 | 2018-10-11 | Learned absolute position tables | [Devlin et al., BERT — arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | verified against arXiv v1 |
| 4 | 2019-01-09 | Segment recurrence across contexts | [Dai et al., Transformer-XL — arXiv:1901.02860](https://arxiv.org/abs/1901.02860) | verified against arXiv v1 |
| 5 | 2019-04-23 | Sparse Transformer, strided and fixed patterns | [Child et al. — arXiv:1904.10509](https://arxiv.org/abs/1904.10509) | verified against arXiv v1 |
| 6 | 2019-11-06 | Multi-query attention | [Shazeer — arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | verified against arXiv v1 |
| 7 | 2020-04-10 | Sliding window attention with global tokens | [Beltagy et al., Longformer — arXiv:2004.05150](https://arxiv.org/abs/2004.05150) | verified against arXiv v1 |
| 8 | 2020-06-29 | Linear attention, the kernel regrouping | [Katharopoulos et al. — arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | verified against arXiv v1 |
| 9 | 2020-09-30 | Performer, FAVOR+ | [Choromanski et al. — arXiv:2009.14794](https://arxiv.org/abs/2009.14794) | verified against arXiv v1 |
| 10 | 2021-02-22 | The delta rule, fast-weight programmers | [Schlag et al. — arXiv:2102.11174](https://arxiv.org/abs/2102.11174) | verified against arXiv v1 |
| 11 | 2021-04-20 | RoPE, rotary position embedding | [Su et al., RoFormer — arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | verified against arXiv v1 |
| 12 | 2021-08-27 | ALiBi, attention with linear biases | [Press et al. — arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | verified against arXiv v1 |
| 13 | 2022-05-27 | FlashAttention | [Dao et al. — arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | verified against arXiv v1 |
| 14 | 2023-05-22 | Grouped-query attention | [Ainslie et al., GQA — arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | verified against arXiv v1 |
| 15 | 2023-06-27 | Position interpolation | [Chen et al. — arXiv:2306.15595](https://arxiv.org/abs/2306.15595) | verified against arXiv v1 |
| 16 | 2023-06-30 | NTK-aware base scaling | [bloc97, r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | **unverified** — community post, no paper |
| 17 | 2023-08-31 | YaRN | [Peng et al. — arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | verified against arXiv v1 |
| 18 | 2023-09-29 | Attention sinks, StreamingLLM | [Xiao et al. — arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | verified against arXiv v1 |
| 19 | 2023-12-01 | Selective state space | [Gu and Dao, Mamba — arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | verified against arXiv v1 |
| 20 | 2024-05-07 | Multi-head latent attention | [DeepSeek-AI, DeepSeek-V2 — arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | verified against arXiv v1 |
| 21 | 2024-06-10 | Parallelizable DeltaNet | [Yang et al. — arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | verified against arXiv v1 |
| 22 | 2024-12-09 | Gated DeltaNet | [Yang et al. — arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | verified against arXiv v1 |
| 23 | 2025-02-16 | Natively trainable sparse attention | [Yuan et al., NSA — arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | verified against arXiv v1 |
| 24 | 2025-09-29 | DeepSeek sparse attention | [DeepSeek-V3.2-Exp release](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | **unverified** — release note, no paper |
| 25 | — | DroPE | LightningLM V4 cookbook, via the Session 8 lesson | **unverified** — course record only |

### Coverage against the assignment's list

Named in the assignment and present above: standard attention (1), learned absolute positions (3),
sinusoidal (1 — same paper, same date as scaled dot-product, so it is one card rather than two),
RoPE (11), ALiBi (12), MQA (6), GQA (14), sliding window (7), attention sinks (18), NTK-aware
scaling (16), YaRN (17), linear attention (8), the delta rule (10) and Gated DeltaNet (22), MLA
(20), sparse and top-k attention (5, 23, 24), DeepSeek's compressed sparse attention (23, 24),
DroPE (25).

Added beyond the list: relative position representations (2), Transformer-XL segment recurrence (4)
— the ancestor of the Memory Stream, Performer (9), FlashAttention (13) — the reminder that some
attention bills are paid to the memory hierarchy rather than the asymptotics, position
interpolation (15) — without which the NTK card has no problem to answer, Mamba (19), and
parallelizable DeltaNet (21) — the work that made the delta rule trainable at scale.

## Notes on what this app does not claim

- Demo arithmetic is illustration at toy scale. It shows mechanism and cost, never model quality.
- Numbers attributed to LightningLM V4 or DeepSeek-V4 come from the course cookbook via the Session
  8 lesson, not from a public paper, and are labelled as such wherever they appear.
- A reported extension factor is evidence for one model and one procedure. It is not a promise that
  the same factor transfers.
