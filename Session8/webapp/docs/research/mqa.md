# Concept 7 — Multi-query attention

**Card id:** `mqa` · **Date:** 2019-11-06 (arXiv v1) · **Pressure:** what generation must remember

## What was read

- [arXiv:1911.02150](https://arxiv.org/abs/1911.02150), Noam Shazeer (sole author) — *Fast Transformer
  Decoding: One Write-Head is All You Need*. Abstract page for the version history and abstract text.
- [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1911.02150) for §2 (multi-head attention and its
  performance analyses), §2.4.1 (incremental multi-head), §3 (multi-query attention), §3.1
  (multi-query performance analysis), §4 (experiments), Tables 1–3, and the conclusion.
- Version history: **v1, 6 Nov 2019**. The timeline uses v1.
- One forward-link source, read only for the forward-link claim:
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), Ainslie, Lee-Thorp, de Jong, Zemlyanskiy,
  Lebrón, Sanghai — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints*, v1 **22 May 2023**. Abstract only.

Everything quoted below is from the 1911.02150 text unless labelled otherwise.

## The mechanism, precisely

**The change is one sentence long, and the paper writes it as one sentence.** §3 opens:

> *"We introduce multi-query Attention as a variation of multi-head attention as described in
> [Vaswani et al. 2017]."*

and immediately states the difference:

> *"Multi-query attention is identical except that the different heads share a single set of keys and
> values."*

So: **Q keeps its per-head projection. K and V become shared across all heads.** The paper describes
the edit at the level of the code it is modifying:

> *"we remove the letter 'h' from the tf.einsum equations where it represents the 'heads' dimension of
> K, V, P_k, or P_v."*

That is the whole mechanism. `P_q` and `P_o` keep their `h` dimension; `P_k` goes from `[h, d, k]` to
`[d, k]` and `P_v` from `[h, d, v]` to `[d, v]`. Every query head still computes its own scores and its
own output — it just reads the same keys and the same values as every other head. The title's "one
write-head" is the metaphor: many readers, one writer. (Note: the phrase *write-head* appears in the
title and nowhere in the body — the paper never explains it.)

**Why this is not a compute optimisation.** The paper's argument is arithmetic, and it is worth getting
exactly right because the popular summary — "the cache is smaller" — is a consequence, not the claim.
The claim is about a **ratio**.

Symbols, as the paper uses them: `n` query positions, `m` key/value positions, `d` model dimension,
`h` heads, `k` (= `d_k`) key dimension, `b` batch size. §2.3 states the simplifying assumptions used
throughout the analyses: *"m=n, k=v=d/h, as suggested by [Vaswani et al. 2017], n≤d"*.

**Training / all-positions-at-once (§2.3.1).** With the whole sequence available, the ratio of memory
access to arithmetic is

> *"the ratio of memory access to arithmetic operations is O(1/k + 1/(b·n))"*

Both terms are small when `k` and `b·n` are large, so this regime is compute-bound. This is why the
paper says training is fine and only inference hurts.

**Incremental decoding, multi-head (§2.4.1).** One token at a time. First, why it cannot be
parallelised:

> *"the output of the self-attention layer at a particular position affects the token that is generated
> at the next position, which in turn affects the input to that layer at the next position. This
> prevents parallel computation."*

Then the counting:

> *"Across n calls, the total number of arithmetic operations is again Θ(bnd²). Across n calls, the
> total amount of memory access is Θ(bn²d + nd²), the first term due to K and V and the second term
> due to Pq, Pk, Pv and Po. Dividing the memory by the computations, we find that the ratio of memory
> access to arithmetic operations is **Θ(n/d + 1/b)**. When n≈d or b≈1, the ratio is close to 1,
> causing memory bandwidth to be a major performance bottleneck on modern computing hardware."*

That last sentence is the paper's actual argument, and it is a hardware claim, not a memory-footprint
claim. Hardware does far more arithmetic per second than it does memory loads per second. When the
ratio of loads to arithmetic approaches 1, the arithmetic units idle waiting on memory. `n≈d` is
long-context; `b≈1` is a single user. Both make the ratio bad.

The paper then splits the two terms by how hard they are to fix:

> *"The 1/b term is the easier one — we can just use a larger batch size, memory size permitting."*

> *"Reducing the n/d term is harder. This term is related to the expense of reloading at each step the
> K and V tensors representing the memory which have size bhmk. One solution is to limit the sequence
> length n. Another is to reduce the number of positions being attended-to, either by attending to a
> local neighborhood, or by otherwise compressing the number of memory positions…"*

Under the paper's own assumptions `bhmk = b·h·n·(d/h) = bnd` per step, and over `n` steps that is the
`Θ(bn²d)` first term above. (The rendered ar5iv text prints this product as `bhmk = bn²`, which does
not follow from `m=n, k=d/h` — it should be `bnd`. Treat it as a typo in the source; the `Θ(bn²d)`
expression it feeds is internally consistent and is the one to quote.)

**Incremental decoding, multi-query (§3.1).** Same arithmetic, less memory:

> *"the total amount of memory access is Θ(bnd + bn²k + nd²), the first term due to x, q, o and y, the
> second term due to K and V and the third term due to P_q, P_k, P_v, P_o."*

> *"Dividing the memory by the computations, we find that the ratio of memory access to arithmetic
> operations is **Θ(1/d + n/(dh) + 1/b)**. We have reduced the offensive n/d by a factor of h."*

**That factor of `h` is the whole paper.** The arithmetic term `Θ(bnd²)` is unchanged — multi-query
does not save a single multiply-accumulate in the attention mixing. It changes only what has to be
loaded to feed those multiplies, and that is enough because the bottleneck was never the multiplies.

## Numbers that matter

**Model configuration (§4, WMT14 EN-DE).** 6 layers, `d_model = 1024`, `d_ff = 4096`, `h = 8`,
`d_k = d_v = 128`. Trained 100,000 steps (20 epochs), batch of 128 examples, each a 256-token input
and 256-token target sequence, on a 32-core TPUv3 cluster (~2 hours).

The multi-query variant is **parameter-matched, not parameter-reduced** — a control the card should
mention, because it rules out "it's just a smaller model":

> *"We widen the feed-forward hidden layers from 4096 to 5440 to make the total parameter-count equal
> to that of the baseline."*

**Table 1 — WMT14 EN-DE quality.** `ln(PPL)` on dev, BLEU on dev, BLEU on test at beam 1 / beam 4.

| attention | h | d_k, d_v | d_ff | ln(PPL) dev | BLEU dev | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|---|
| **multi-head** | 8 | 128 | 4096 | **1.424** | 26.7 | 27.7 / **28.4** |
| **multi-query** | 8 | 128 | 5440 | **1.439** | 26.5 | 27.5 / **28.5** |
| multi-head local | 8 | 128 | 4096 | 1.427 | 26.6 | 27.5 / 28.3 |
| multi-query local | 8 | 128 | 5440 | 1.437 | 26.5 | 27.6 / 28.2 |
| multi-head | 1 | 128 | 6784 | 1.518 | 25.8 | — |
| multi-head | 2 | 64 | 6784 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 4 | 32 | 6784 | 1.488 | 26.1 | — |
| multi-head | 8 | 16 | 6784 | 1.513 | 25.8 | — |

The ablation block at the bottom is the paper's real defence. It asks: if you want a cheaper cache,
why not just use fewer heads, or smaller heads? Answer, in the paper's words:

> *"the multi-query attention model seems to be slightly worse than the baseline, but much closer than
> any of the alternatives involving decreasing h, d_k and d_v."*

Read the numbers: multi-query costs **+0.015 in ln(PPL)** (1.424 → 1.439). Cutting to `h = 1` costs
**+0.094**; `h = 2, d_k = 64` costs +0.056; `h = 8, d_k = 16` costs +0.089. Multi-query is roughly
four to six times cheaper in quality than any other way of shrinking the same tensors.

**Table 2 — cost. Caption:** *"Amortized training and inference costs for WMT14 EN-DE Translation
Task with sequence length 128. Values listed are in TPUv2-microseconds per output token."*

| attention | training (enc. + dec.) | inference (enc. + dec.) | beam-4 search |
|---|---|---|---|
| multi-head | 13.2 | 1.7 + **46** | 2.0 + **203** |
| multi-query | 13.0 | 1.5 + **3.8** | 1.6 + **32** |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | 13.0 | 1.5 + 3.3 | 1.6 + 16 |

Three things the card should read off this table:

1. **Decoder, greedy: 46 → 3.8 µs/token, a 12.1× speedup.** Beam-4: 203 → 32 µs/token, 6.3×.
2. **Encoder barely moves: 1.7 → 1.5 µs (greedy), 2.0 → 1.6 (beam-4).** The encoder runs all
   positions at once, so it lives in the `O(1/k + 1/(b·n))` compute-bound regime where there was
   nothing to win. The near-zero encoder gain is direct evidence that the decoder gain is a
   memory-bandwidth effect, not a general "fewer parameters" effect.
3. **Training is a wash: 13.2 → 13.0 µs/token, ~1.5%.** Multi-query is not a training optimisation and
   the paper does not claim it is. The paper offers no explanation for the 0.2 µs; do not build one.

Also worth noting: local attention (the other fix for `n/d` the paper floated in §2.4.1) gets the
multi-head decoder from 46 to 23 µs — a 2× — while multi-query gets it to 3.8. And the two compose:
multi-query local is 3.3 greedy, 16 at beam-4, the fastest row in the table.

**Table 3 — Billion-Word Language Modeling Benchmark (§4.3).** 6 layers, `d_model = 1024`,
`d_ff = 8192`, `h = 8`, `d_k = d_v = 128`; 136K steps (10 epochs) at a batch size of 64K tokens.

| model | dev-PPL |
|---|---|
| multi-head (h=8, d_k/v=128) | **29.9** |
| multi-query (h=8, d_k/v=128) | **30.2** |
| multi-head h=1, d_k/v=128 | 31.2 |
| multi-head h=2, d_k/v=64 | 31.1 |
| multi-head h=4, d_k/v=32 | 31.0 |
| multi-head h=8, d_k/v=16 | 30.9 |

Same shape as Table 1: multi-query costs **+0.3 perplexity**, every head-shrinking alternative costs
**+1.0 to +1.3**.

**The paper's own summary of the trade**, from the abstract and conclusion:

> *"We verify experimentally that the resulting models can indeed be much faster to decode, and incur
> only minor quality degradation from the baseline."*

> *"We have proposed multi-query attention — an alternative to multi-head attention with much lower
> memory-bandwidth requirements in the incremental setting. We believe that this enables wider
> adoption of attention-based sequence models in inference-performance-critical applications."*

**Training stability: the paper says nothing.** Searched the full text for *stability*, *unstable*,
*instability*, *diverge* — none appear. There is no loss-curve discussion, no learning-rate
adjustment for the multi-query runs, no mention of a failed or restarted run. The card must not
imply the paper addressed this either way. (The instability folklore around MQA is later and from
other sources; see *What the source does not establish*.)

### Measured in this app's model, before building the card

Preset 0, *"The lighthouse keeper wrote the code in a notebook and hid it under the third stair"*,
16 tokens, block 1, `d_k = 8`, 4 heads, seeded untrained weights. Run via `forward(tokens, {kvGroups})`
and `toyCost(result, {kvGroups})`.

| quantity | kvGroups = 4 (MHA) | kvGroups = 2 (GQA-ish) | kvGroups = 1 (MQA) |
|---|---|---|---|
| mean cross-head **key** cosine | 0.0070 | 0.3429 | **1.0000** |
| mean cross-head **attention-row** cosine | 0.3336 | 0.3532 | 0.3654 |
| heads agreeing on top attended token | 24.0% | 24.0% | 24.0% |
| cache numbers stored (2 blocks) | 2048 | 1024 | **512** |
| cache bytes @ bf16 | 4096 | 2048 | **1024** |
| **mixing arithmetic (`mixOps`)** | **4352** | **4352** | **4352** |
| keys read per query | 8.50 | 8.50 | 8.50 |

Four findings, and one of them is a warning:

1. **The key cosine is the proof.** It goes 0.0070 → 1.0000, *exactly* 1, because at `kvGroups = 1`
   every head literally slices the same `K`. This is the one number that shows the mechanism did what
   the paper says, and it is exact rather than approximate, so it cannot be argued with.
2. **`mixOps` does not move at all — 4352 in every column — while cache bytes fall 4×.** This *is* the
   paper's ratio argument, reproduced. Same arithmetic, a quarter of the memory. The card can put
   these two rows side by side and let the reader see that multi-query is not a compute optimisation.
3. **The attention rows barely converge: 0.3336 → 0.3654, and the top-attended-token agreement does
   not move at all (24.0% in all three).** Do not oversell "heads become alike" — with shared keys but
   still-independent query projections, the heads still disagree. This is honest and it is also
   *correct about the mechanism*: MQA shares what is read, not who reads it.
4. **The output does change**, confirming the seam is wired and not decorative: total-variation
   distance between the two next-token distributions is **0.714**, and the top prediction changes
   (`her` 0.205 → `context` 0.292). This proves the switch is real. It proves nothing about quality —
   the weights are noise.

Serving-scale cache, from `cacheBytes` at the panel's verified configuration (48 layers, head dim 128,
32768 tokens, batch 1, bf16):

| kv heads | cache |
|---|---|
| 8 (as configured) | **6.442 GB** |
| 4 | 3.221 GB |
| 2 | 1.611 GB |
| 1 (multi-query) | **0.805 GB** |

The paper's ratio `Θ(n/d + 1/b)` vs `Θ(1/d + n/(dh) + 1/b)`, evaluated directly:

| regime | n | d | h | b | MHA ratio | MQA ratio | improvement |
|---|---|---|---|---|---|---|---|
| this app | 16 | 32 | 4 | 1 | 1.500 | 1.156 | 1.30× |
| this app, batched | 16 | 32 | 4 | 64 | 0.516 | 0.172 | 3.00× |
| paper's WMT model | 128 | 1024 | 8 | 1 | 1.125 | 1.017 | 1.11× |
| paper's WMT model | 128 | 1024 | 8 | 128 | 0.133 | 0.024 | 5.44× |
| modern serving | 32768 | 4096 | 32 | 1 | 9.000 | 1.250 | 7.20× |
| modern serving | 32768 | 4096 | 32 | 64 | 8.016 | 0.266 | **30.15×** |

Read the last two rows against the first: **the win grows with context length and with batch size**,
and only approaches the full factor of `h` when the `1/b` term has already been beaten down by
batching. At `b = 1` the batch term floors the improvement no matter how much you fix the keys — which
is exactly what the paper says when it calls `1/b` the easy term. The toy row (1.30×) is honestly
unimpressive, and the card should show it anyway, next to the serving row, because the gap between
them *is* the lesson: this mechanism was designed for a regime the toy is not in.

## What the live view must let the reader do

The seam is `kvGroups`, already implemented and verified to change the output. Everything below moves
a real computed quantity; nothing below reports a quality result.

1. **The kvGroups slider: 4 → 2 → 1, with the key-identity readout.** One control, three positions
   (multi-head, an intermediate group, multi-query). Display: the mean cross-head key cosine, live.
   It reads **0.007 → 0.343 → 1.000**. The proof is that it lands on exactly 1.000 — heads are now
   reading a single set of keys. Show the four heads' key vectors as small strips alongside, so the
   reader sees four different strips collapse into four copies of one strip.

2. **The two rows that must sit next to each other: arithmetic vs memory.** Same slider. Display
   `mixOps` and `cacheBytes` from the existing cost model as a two-row panel. `mixOps` stays at
   **4352** at every position; `cacheBytes` falls **4096 → 2048 → 1024**. The number that proves the
   point is the *unchanged* one: a mechanism that saves memory and no arithmetic is a
   memory-bandwidth mechanism, which is the paper's entire argument. Label the panel with the paper's
   sentence about the ratio, not with "the cache is smaller".

3. **The paper's ratio, recomputed live, with `n` and `b` as sliders.** Two formulas printed as
   written in the paper — `Θ(n/d + 1/b)` and `Θ(1/d + n/(d·h) + 1/b)` — with `d` and `h` taken from
   whatever scale the reader picks (toy, or the serving panel), and `n` (context length) and `b`
   (batch) on sliders. Display both ratios and their quotient. At the toy's `n=16, d=32, h=4, b=1` it
   shows **1.500 vs 1.156, a 1.30×**; drag `n` to 32768, `d` to 4096, `h` to 32, `b` to 64 and it
   shows **8.016 vs 0.266, a 30.15×**. The number that proves the point is the quotient rising
   towards `h` as `b` rises — it makes visible that `1/b` is the floor and that multi-query only pays
   off once you are batching, which the paper states and almost every retelling drops.

4. **Attention rows, four heads, before and after — as a negative result.** Show the four per-head
   attention rows for the last token at kvGroups=4 and kvGroups=1 side by side, with the mean
   cross-head row cosine printed: **0.334 → 0.365**, and top-attended-token agreement **24.0% →
   24.0%**. The honest caption: the heads read identical keys now, but they still ask different
   questions, so they still land in different places. This is the interaction most likely to be built
   wrong (as "watch the heads collapse"); building it as a measured non-collapse is both truer to the
   mechanism and more interesting.

5. **The serving-scale cache slider, tied to `kvGroups`.** Reuse the existing panel: 48 layers, head
   dim 128, 32768 tokens, bf16, batch 1. Sliding kv heads 8 → 1 takes the cache from **6.442 GB** to
   **0.805 GB** — 5.64 GB freed for one conversation. Pair it with a batch-size control so the reader
   can see 6.442 GB × batch become infeasible and 0.805 GB × batch stay affordable; that is the actual
   reason the mechanism was adopted, and it is arithmetic the app already does correctly.

**A banner the card must carry, not bury:** the model on this page is untrained. `kvGroups` visibly
changes its output (TV distance 0.714 between the two next-token distributions) — that shows the
switch is real, and nothing more. The card may say *what changes structurally*; it may never say
multi-query is better, worse, or "almost as good" here. The only quality evidence on this page is the
paper's, quoted from Tables 1 and 3.

## What the source does *not* establish

- **It does not establish that multi-query is quality-neutral.** The paper measured a real, consistent
  cost and named it: +0.015 ln(PPL) on WMT14 dev, +0.3 dev perplexity on Billion-Word, and *"slightly
  worse than the baseline"*. The higher test BLEU at beam-4 (28.5 vs 28.4) is a single decoding
  configuration on a single test set and moves in the opposite direction to both perplexity numbers
  and to dev BLEU (26.5 vs 26.7); it is not evidence multi-query is better and the paper does not
  claim it is. **The degradation was measured, not merely noted** — that distinction matters for the
  timeline.
- **It says nothing about training stability.** Not a word, in either direction. If the card wants to
  raise stability it must attribute it elsewhere and date it later.
- **It is not a long-context result.** The experiments run at sequence length 128–256. Every
  extrapolation to 32K contexts on this page is *our* arithmetic applied to the paper's formula, not
  the paper's measurement, and the card should say so where it shows the serving numbers.
- **It is not a decoder-only-LLM result in the modern sense.** The headline speedups come from a
  6-layer encoder-decoder machine-translation model on TPUv2/v3 in 2019. The mechanism generalises;
  the specific 12× does not transfer to other hardware or architectures unexamined.
- **`Θ` is asymptotic.** `Θ(n/d + 1/b)` drops all constants. Our recomputed table treats it as if the
  constants were 1, which is fine for showing *shape* and dishonest if read as a predicted speedup.
  Label those numbers "ratio, up to constants".
- **It never says "KV cache".** The term is not in the paper. The concept is (the `prev_K` / `prev_V`
  tensors reloaded each step); the vocabulary is later. Use the paper's framing — reloading K and V —
  when quoting it.
- **The ar5iv text contains at least one typo we caught**: §2.4.1 renders the K/V tensor size as
  `bhmk = bn²`, which contradicts the paper's own assumptions `m = n`, `k = d/h` (giving `bnd`). The
  `Θ(bn²d)` total that this sentence supports is consistent; the inline product is not. Quote the `Θ`
  expressions, not the product.
- **The app's own outputs establish nothing.** Seeded noise weights, no training, 32 dims.

## Leaves behind

**Backward.** Concept 1 ended by naming two costs the Transformer bought its O(1) path length with:
the T² score matrix (compute) and the key/value cache that grows with every generated token (memory).
The Sparse Transformer took the first. **This is the paper that takes the second** — and it is worth
being precise that it does not shrink the cache by attending to fewer positions (that is local
attention, which this paper tries as a baseline and gets only 2×). It shrinks the cache by noticing
that the `h` copies of the keys were never the expensive part of what heads do. Concept 1's cache
grew as `b·n·d` per layer per step; multi-query makes it `b·n·k`, `h` times smaller, and leaves the
arithmetic exactly where it was.

**Forward.** The paper hands the future two things. The first is the ratio itself — once you can say
"memory access divided by arithmetic", every subsequent inference paper is arguing about that
quotient, and the `1/b` term explains why batched serving and long context are the regimes where this
all matters.

The second is the residue this paper measured but declined to remove: **that +0.015 ln(PPL) and +0.3
perplexity**. Going from `h` key/value heads to exactly 1 is the most aggressive possible version of
the idea, and the paper never asks what happens in between. Four years later, GQA
([arXiv:2305.13245](https://arxiv.org/abs/2305.13245), 22 May 2023) asks exactly that, opening with
*"MQA can lead to quality degradation"* and proposing *"an intermediate (more than one, less than
number of query heads) number of key-value heads"*, reporting *"quality close to multi-head attention
with comparable speed to MQA"*. Note the GQA abstract says nothing about training stability either —
if the card mentions instability at all, it belongs to neither of these two sources.

This app's `kvGroups` seam already spans that whole span: `kvGroups = heads` is 2017, `kvGroups = 1`
is this paper, and every value in between is 2023. That intermediate position on the slider is not a
convenience — it is a later concept, and the card should say so rather than let the reader assume
Shazeer offered it.
