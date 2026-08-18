## How this change is worked

One concept at a time. For each mechanism, in order: **research → build → verify → stop for review**,
then the next. Research means reading the primary source, not recalling it, and writing what was
learned to `Session8/webapp/docs/research/<id>.md` before any code for that concept is written. Each
note records: what was read, the mechanism in precise terms, the parameters and numbers that matter,
what the live view must let the reader do, and what the source does **not** establish. The note is
the spec for that card and the citation for the README.

## 1. Reset and foundation

- [x] 1.1 First implementation built and rejected: static tables where a live model was needed. Kept in git at `0668e67`
- [x] 1.2 Delete `Session8/webapp/`, carrying forward only the prose and the already-verified dates from that commit
- [x] 1.3 New shell: `index.html`, `app/styles.css` on the Session 3/4 palette and type stack, sized for a card with a live visual panel
- [x] 1.4 `app/model/ops.js`: seeded generator, layer norm, GELU, matvec, softmax — the primitives the forward pass needs, nothing more
- [x] 1.5 `app/model/vocab.js`: a fixed small vocabulary and tokenizer, so an edited sentence maps to ids and the prediction bars name real words
- [x] 1.6 `app/model/transformer.js`: 32 dims, 4 heads, 2 blocks, seeded weights, tied output; a forward pass returns per-block per-head scores and weights, hidden states, and the output distribution
- [x] 1.7 `app/model/mixers.js`: the seam every mechanism plugs into — softmax attention parameterised by a readability rule, a position scheme and key/value sharing; and fixed-size state parameterised by a write rule and decay
- [x] 1.8 `app/model/cost.js`: cache bytes, keys read per query, mixing work per token, plus the serving-scale formula that must reproduce 6.44 GB / 51.54 GB
- [x] 1.9 `app/runner.js`: one shared sentence, playback position and head/layer selection; the mounted concept subscribes. Only the current concept computes, which the deck gives for free
- [x] 1.10 `app/deck.js`: one concept in view at a time; next / previous controls, left and right arrow keys, and a position readout ("7 of 26"); mount and unmount so only the current concept computes
- [x] 1.11 `app/timeline.js`: the slider that is also the chronology — year ticks, entries spaced by real date, drag to move, current entry marked, and the drift in the field's priorities readable from it
- [x] 1.12 URL fragment names the current concept, kept in step with `history.pushState`, honoured on load, and the browser's back control returns to the previous concept
- [x] 1.13 Card chrome: date, source link, unverified badge, problem → mechanism → buys / gives up / when, the live panel, and the plain-language verdict at the foot

## 2. Concepts, in timeline order

Each line is: research note written, card built on the live model, plain-language verdict written, verified in the browser, reviewed.

- [x] 2.1 Scaled dot-product attention, multi-head — the baseline (arXiv 1706.03762)
- [x] 2.1b Sinusoidal position encoding — its own card, same paper and date (arXiv 1706.03762)
- [x] 2.2 Relative position representations (arXiv 1803.02155)
- [x] 2.3 Learned absolute position tables (arXiv 1810.04805)
- [x] 2.4 Segment recurrence across contexts, Transformer-XL (arXiv 1901.02860)
- [x] 2.5 Sparse Transformer, strided and fixed patterns (arXiv 1904.10509)
- [x] 2.6 Multi-query attention (arXiv 1911.02150)
- [x] 2.7 Sliding window with global tokens, Longformer (arXiv 2004.05150)
- [x] 2.8 Linear attention, the kernel regrouping (arXiv 2006.16236)
- [x] 2.9 Performer, FAVOR+ (arXiv 2009.14794)
- [x] 2.10 The delta rule, fast-weight programmers (arXiv 2102.11174)
- [x] 2.11 RoPE (arXiv 2104.09864)
- [x] 2.12 ALiBi (arXiv 2108.12409)
- [x] 2.13 FlashAttention (arXiv 2205.14135)
- [x] 2.14 Grouped-query attention (arXiv 2305.13245)
- [x] 2.15 Position interpolation (arXiv 2306.15595)
- [x] 2.16 NTK-aware base scaling (community post, no paper)
- [x] 2.17 YaRN (arXiv 2309.00071)
- [ ] 2.18 Attention sinks, StreamingLLM (arXiv 2309.17453)
- [ ] 2.19 Selective state space, Mamba (arXiv 2312.00752)
- [ ] 2.20 Multi-head latent attention (arXiv 2405.04434)
- [ ] 2.21 Parallelizable DeltaNet (arXiv 2406.06484)
- [ ] 2.22 Gated DeltaNet (arXiv 2412.06464)
- [ ] 2.23 Natively trainable sparse attention (arXiv 2502.11089)
- [ ] 2.24 DeepSeek sparse attention (release note, no paper)
- [ ] 2.25 DroPE — the reported 8K → 256K, the 32×, the boundary, and no invented algorithm (course cookbook only)

## 3. Self-check

- [ ] 3.1 Model assertions: softmax rows sum to 1, masked future weight exactly 0, forward pass deterministic across two runs
- [ ] 3.2 Mechanism assertions: every degenerate setting matches the baseline — GQA at one query head per group, top-k at k = T, window = T, block size 1; the softmax-free regrouping reproduces the direct sum; the delta rule corrects where add-only accumulates
- [ ] 3.3 Cost assertions: the serving formula returns 6.44 GB at the lesson's configuration and 51.54 GB at eight conversations
- [ ] 3.4 Data assertions: every entry has a plain-language verdict whose pros and cons match the technical record in count and direction; unique ids, parseable dates, complete trade-off records, every entry resolving to a mechanism the model can run, every entry having a research note, and every entry naming both the earlier limitation it answers and what it left for later work
- [ ] 3.5 `?selfcheck=1` paints pass or fail and names every failure; run it and confirm

## 4. Verify and ship

- [ ] 4.1 Walk every card in a browser: no console errors, every control moves something, the sentence propagates
- [ ] 4.2 Type into the input and confirm the visuals keep up
- [ ] 4.3 Phone width: no sideways scroll on the body, wide visuals scroll inside their own container
- [ ] 4.4 `Session8/webapp/README.md`: what it is, how to run it, the architecture, the chronology source table, links to the research notes, and the untrained-model caveat
- [ ] 4.5 Commit with the `[ERA-V5][muttu]:` prefix
- [ ] 4.6 Push and confirm the Pages deployment, once the repo's Pages source is set to GitHub Actions
