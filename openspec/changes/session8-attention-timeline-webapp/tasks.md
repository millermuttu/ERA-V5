## 1. Skeleton and shared library

- [x] 1.1 Create `Session8/webapp/` with `index.html` (module shell: header, sticky year rail of anchor links, one container the cards render into) and `app/styles.css` carrying the `Session4/index.html` `:root` palette and the Archivo / Hanken Grotesk / IBM Plex Mono stack
- [x] 1.2 Write `app/lib/mathx.js`: `dot` and `softmax` only — every demo imports from here, none defines its own softmax. Masking is `j > i ? -Infinity : 0` inline in the score loop; demo weights are literal arrays, so reloads are reproducible without a PRNG. Add `matmul` if and when a demo needs one

## 2. The chronology data

- [x] 2.1 Define the record shape in `app/data/mechanisms.js` (`id`, `name`, `date`, `source`, `verified`, `thread`, `problem`, `mechanism`, `buys[]`, `givesUp[]`, `chooseWhen`, `demo`) with the baseline Transformer entry written out in full as the shape example
- [x] 2.2 Write the pre-2020 entries, the timeline opening on the Transformer: scaled dot-product + MHA + sinusoidal (Transformer, the baseline and the first entry), relative positions (Shaw), learned absolute position tables (BERT-era default, the hard length wall Session 7 named), Transformer-XL segment recurrence, Sparse Transformer, MQA. RNN-era attention (Bahdanau 2014, Luong 2015) is one line of context inside the baseline card — the Transformer is what removed the recurrence — not a timeline entry
- [x] 2.3 Write the 2020–2022 entries: Longformer sliding window, linear attention, Performer/FAVOR+, delta rule / fast-weight programmers, RoPE, ALiBi, FlashAttention
- [x] 2.4 Write the 2023 entries: GQA, NTK-aware base scaling, YaRN, attention sinks / StreamingLLM, Mamba selective state
- [x] 2.5 Write the 2024–2025 entries: MLA, parallelizable DeltaNet, Gated DeltaNet, NSA, DeepSeek sparse attention
- [x] 2.6 Write the course-record-only entry: DroPE — state what the record establishes (8K → 256K, 32×, applied before annealing) and what it does not (the algorithm), `verified: false` — then read the finished set against the assignment's named minimum list and add whatever is missing
- [x] 2.7 Confirm every entry's `problem` names a limitation of an entry that appears earlier in the sorted timeline

## 3. Timeline and card rendering

- [x] 3.1 Write `app/main.js`: sort by date, render every card into the document flow with `id="<mechanism id>"` — no router. The rail is `<a href="#id">` links, the browser handles navigation and `:target`, and the baseline is first in the DOM so it is what loads
- [x] 3.2 Render the mechanism card: date + source link, problem → mechanism → buys → gives up → choose when, and the demo mount when the record has one
- [x] 3.3 Render the unverified badge from the `verified` field, naming the actual provenance (community post, release note, course cookbook)
- [x] 3.4 Show `thread` as a one-word label on the card — no colour lanes, no legend: the assignment asks for a chronology, not a taxonomy drawn over it

## 4. Live demos

- [x] 4.1 `app/demos/attention.js` — 6 tokens × 4 dims, real Wq/Wk/Wv, all 36 scores, scale, softmax, weighted sum, with a causal-mask toggle that visibly leaks weight onto future tokens when off
- [x] 4.2 `app/demos/linear.js` — the lesson's `q=2`, `k=[0.5,1,1.5]`, `v=[10,20,30]` example computed both ways (direct and pre-built state) with a softmax toggle, plus the delta rule panel showing 40 → 55 against add-only 40 → 95
- [x] 4.3 `app/demos/rope.js` — two 2D arrows rotated by `iθ` and `jθ` with a live dot product; moving both tokens together holds the gap, the relative angle and the score fixed
- [x] 4.4 `app/demos/cache.js` — `2 × layers × kv_heads × head_dim × T × batch × bytes`, with MHA/GQA/MQA head sharing, cache precision and concurrency
- [x] 4.5 `app/demos/topk.js` — 12 keys with a `k` slider: value work falls with `k` while naive selection cost stays at all 12
- [x] 4.6 `app/demos/compress.js` — block size `m` giving `T/m` stored positions, and top-k blocks read, shown as two separate savings
- [x] 4.7 Draw a static inline-SVG diagram only where the card text cannot carry the mechanism (sliding window, attention sinks, MLA, NSA blocks are the likely four) — written as literal SVG in the module, not through a builder library. Prose plus the trade-off card is a complete explanation everywhere else

## 5. Self-check

- [x] 5.1 Write `app/lib/selfcheck.js` with the math assertions: softmax rows sum to 1, masked future weight exactly 0, regrouped equals direct with softmax off and differs with it on, delta rule reaches 55 where add-only reaches 95
- [x] 5.2 Add the lesson-figure assertion: the cache formula returns ≈6.44 GB at 48 layers / 8 KV heads / head dim 128 / bf16 / 32,768 tokens, and ≈51.54 GB at eight users
- [x] 5.3 Add the data-integrity assertions: unique ids, parseable ISO dates, every entry has a source and non-empty `buys`, `givesUp`, `chooseWhen`, every `demo` key resolves to a registered demo
- [x] 5.4 Wire `?selfcheck=1` to run the assertions and paint a visible pass/fail banner naming any failure
- [x] 5.5 Run it and confirm every assertion passes

## 6. Date verification

- [x] 6.1 For every paper-backed entry, fetch its arXiv abstract page and read the v1 submission date
- [x] 6.2 Correct `date` in `app/data/mechanisms.js` wherever it disagrees with the source, then set `verified: true`
- [x] 6.3 Leave NTK-aware scaling, the DeepSeek sparse-attention release and DroPE at `verified: false`, each with its real provenance recorded in the `source` field
- [x] 6.4 Re-sort and re-read the timeline after corrections; fix any `problem` text whose "what came before" claim the corrected dates broke

## 7. Docs and hosting

- [x] 7.1 Write `Session8/webapp/README.md`: the assignment statement, the live link, the serve command, the app structure, and the full chronology source table generated to match the data exactly
- [x] 7.2 Add `.github/workflows/pages.yml` publishing `Session8/webapp/` on push to `master`
- [x] 7.3 Add the Session 8 row to the root `README.md` index table, with the live link in the deploy column
- [x] 7.4 Serve locally and walk the whole app: timeline order reads chronologically, every card opens, all six demos respond, no console errors, no 404s

## 8. Ship

- [ ] 8.1 Commit with the `[ERA-V5][muttu]:` prefix
- [ ] 8.2 Push, then confirm the Pages deployment serves and no asset 404s under the `/ERA-V5/` sub-path
- [ ] 8.3 Record the live URL in both READMEs once the deployment is confirmed
