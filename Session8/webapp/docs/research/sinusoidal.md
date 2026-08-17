# Concept 2 — Sinusoidal position encoding

**Card id:** `sinusoidal` · **Date:** 2017-06-12 (arXiv v1) · **Pressure:** where a token sits

## What was read

- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), Vaswani, Shazeer, Parmar, Uszkoreit, Jones,
  Gomez, Kaiser, Polosukhin — *Attention Is All You Need*. Abstract page for the version history;
  [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1706.03762) for §3.5 (Positional Encoding),
  §3.1 (stacks, `d_model`), §3.4 (Embeddings and Softmax), §5.4 (Regularization), Table 3.
- Version history: v1 **12 Jun 2017**, v2 19 Jun, v3 20 Jun, v4 30 Jun, v5 6 Dec 2017, v6 24 Jul
  2023, v7 2 Aug 2023. The timeline uses **v1**.

**Same paper, same date as concept 1 — why it is still its own card.** Concept 1 is the mechanism;
this is the *repair* the mechanism forced. Scaled dot-product attention is permutation-equivariant:
shuffle the tokens and every score follows the shuffle unchanged. The paper says so in its own first
sentence of §3.5 — *"Since our model contains no recurrence and no convolution, in order for the
model to make use of the order of the sequence, we must inject some information about the relative or
absolute position of the tokens in the sequence."* So order is not a property attention has; it is a
separate thing bolted on at the bottom of the stack, with its own design space (*"There are many
choices of positional encodings, learned and fixed"*), its own hypothesis, and its own long line of
successors — RoPE, ALiBi, relative-position biases. Every one of those replaces **this** card, not
concept 1. Collapsing the two would hide the seam that the rest of the timeline pushes on. In the
app the split is literal: concept 1 plugs into the score path, this one adds a vector to the input
embedding — different seams.

## The mechanism, precisely

Two interleaved sinusoids, one pair per two dimensions:

    PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

*"where pos is the position and i is the dimension. That is, each dimension of the positional
encoding corresponds to a sinusoid."* Note the paper's own wording is loose: it calls `i` "the
dimension", but the formulas index dimensions `2i` and `2i+1`, so `i` runs over **pairs**,
`0 … d_model/2 − 1`. Even dimensions get the sine, odd dimensions the cosine of the same frequency.
The constant is **10000** and the paper gives no derivation for it.

**Frequencies.** The `2i/d_model` exponent makes the angular frequency `ω_i = 10000^(−2i/d_model)`
fall geometrically as `i` climbs: *"The wavelengths form a geometric progression from 2π to
10000·2π."* Dimension pair 0 turns over every ~6.28 positions; the last pair is effectively a
constant across any sentence a person would type.

**Where it goes.** *"To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks."* Once, at the bottom — not per layer, not per head.

**Why it can be added at all.** *"The positional encodings have the same dimension d_model as the
embeddings, so that the two can be summed."* This is a real constraint, not bookkeeping: §3.1 says
*"all sub-layers in the model, as well as the embedding layers, produce outputs of dimension
d_model=512"*, and §3.4 adds *"In the embedding layers, we multiply those weights by √d_model."* So
what actually enters block 1 is a **sum** of a √d_model-scaled token embedding and a fixed
unit-amplitude sinusoid vector — content and position share one set of channels and one budget.
(The paper states both facts but never spells out the ordering of the √d_model scaling against the
addition; implementations scale first, then add.)

**Dropout on the sum.** §5.4: *"We apply dropout to the sums of the embeddings and the positional
encodings in both the encoder and decoder stacks."* Rate for the base model, *"we use a rate of
P_drop=0.1."* Position is therefore trained under the same noise as content — it is not a protected
signal.

**The stated reason for sinusoids.** This is the load-bearing sentence of the section:

> *"We chose this function because we hypothesized it would allow the model to easily learn to attend
> by relative positions, since for any fixed offset k, PE_(pos+k) can be represented as a linear
> function of PE_pos."*

Read it exactly: **hypothesized**, and **can be represented as** — an availability claim about the
representation, not a demonstration that the trained model uses it. The paper offers no proof of the
linearity and no experiment testing whether relative attention was learned. (The linearity itself is
just the angle-addition identity — each `(sin, cos)` pair rotates by a fixed angle `k·ω_i`, so the
map is a block-diagonal rotation depending only on `k`. That derivation is *not* in the paper; it is
what makes the property checkable live, which is the point of this card.)

**The learned-embedding experiment, and the choice.**

> *"We also experimented with using learned positional embeddings instead, and found that the two
> versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
> because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
> during training."*

Again exactly: **may allow**. The extrapolation rationale is the sole stated tiebreaker, it is
speculative in the paper's own grammar, and the paper never tests it.

## Numbers that matter

Table 3 is *"Variations on the Transformer architecture. Unlisted values are identical to those of
the base model."*

| variant | PPL (dev) | BLEU (dev) |
|---|---|---|
| base | 4.92 | 25.8 |
| (E) positional embedding instead of sinusoids | 4.92 | 25.7 |

Identical perplexity, 0.1 BLEU apart. That is the whole empirical content of the sinusoid-vs-learned
comparison — and the direction, such as it is, favours the *sinusoids* by a margin no one should
read as a result.

| quantity | paper | this app | note |
|---|---|---|---|
| d_model | 512 | 32 | 16 sinusoid pairs instead of 256 |
| base constant | 10000 | 10000 | keep it, then make it a slider |
| wavelength range | 2π … 10000·2π | 2π … 10000·2π | ~6.28 to ~35,300 positions |
| pairs that complete a cycle in the sentence | — | **2 of 16** | i=0 (6.28 tok), i=1 (11.2 tok); i=2 is 19.9 tok > 16 |
| ‖PE(pos)‖ | √(d_model/2) = 16 | √16 = **4**, every pos | sin²+cos² = 1 per pair |
| PE(pos)·PE(pos) | d_model/2 = 256 | **16** | the k=0 point of the offset curve |
| dropout on the sum | 0.1 | none (inference only) | say so on the card |

The `2 of 16` row is the honest headline for a 32-dim toy: at this width and a ~16-token sentence,
**most of the encoding is nearly constant across the sentence** and only the lowest pairs carry
usable positional contrast. That is a property of the geometric schedule, visible live, and it is
why the base constant deserves a slider rather than a footnote.

## What the live view must let the reader do

Each of these changes a real computed quantity in the running toy model (32 dims, 4 heads, `d_k = 8`,
2 blocks, causal mask, seeded untrained weights).

1. **Kill position and shuffle the sentence.** Toggle the PE addition off, then permute the input
   tokens. Display `max |A_shuffled[π(i),π(j)] − A[i,j]|` over the pre-mask score matrix. With PE off
   it is **~1e-7 (float noise)** — attention is exactly permutation-equivariant, the defect §3.5
   exists to fix. Turn PE on and the same number jumps to something large. One toggle, one number,
   and the reason for the whole card is proved rather than asserted.

2. **The offset curve — the paper's hypothesis, checkable.** Plot `PE(pos) · PE(pos+k)` against `k`
   for the reader's chosen `pos`, and overlay the curve for a *second* `pos`. The two curves must
   land on top of each other: the readout shows e.g. `PE(3)·PE(7) = PE(9)·PE(13)` to 6 decimals.
   That equality is exactly the "depends only on the fixed offset k" property the paper hypothesizes,
   and `k = 0` reads **16.000** = `d_model/2`, which also proves every PE vector has the same norm.
   Let the reader drag `pos` and watch the curve *not move*.

3. **Drag the 10000.** A log slider on the base constant (say 10 → 100000) that re-derives the PE
   table and **re-runs the forward pass**. Show three things updating together: the per-pair
   wavelength ladder with a marker at the sentence length, the count of pairs whose wavelength fits
   inside the sentence (2 at base 10000; ~7 at base 100), and the attention heatmap for a chosen
   head. Small bases make the encoding position-discriminative but aliased — neighbouring positions
   with near-identical vectors far apart — and the offset curve from (2) starts oscillating instead
   of decaying. The paper picks 10000 and never justifies it; here the reader gets to see what the
   knob buys.

4. **Swap in a learned-style table, then run off the end of it.** Offer a second positional scheme:
   a seeded random embedding table of shape `(max_len, 32)` — Table 3 row (E) in miniature. Two
   consequences must be visible as numbers. (a) The offset curve from (2) collapses: `PE(3)·PE(7) ≠
   PE(9)·PE(13)`, so the relative-offset structure the paper hypothesized about is simply absent.
   (b) Set `max_len` to 12, type an 18-token sentence: the sinusoid produces vectors for positions
   12–17 with the *same* norm 4 and the *same* offset curve, and the table has nothing at all. That
   is the extrapolation rationale, made concrete — and the card must immediately say the paper only
   claims this *may* help, and never measured it.

5. **Watch position compete with content for the same channels.** A per-token bar of
   `‖PE(pos)‖ / ‖√d_model · E(token)‖` and the cosine between the pre- and post-addition embedding,
   with a gain slider on the PE term (0 → 2×). At gain 0 you are back to case (1); at high gain the
   attention rows become dominated by position and the token identity stops mattering — visible in
   the heatmap and in the top-1 attended token changing. This is what "the two can be summed" costs:
   one shared budget, no separate position channel.

## What the source does *not* establish

- **It does not prove the linearity claim.** The word is *"hypothesized"*, and the property is
  stated as *"can be represented as"*. No derivation, no rotation matrix, no experiment showing the
  trained model exploits it. The card may show the rotation identity — but as the card's own math,
  not as the paper's.
- **It does not show the model learns relative attention.** Commonly attributed; entirely absent.
  §3.5 makes an availability argument about the encoding, never a claim about what the heads learned.
- **It does not demonstrate extrapolation.** *"may allow the model to extrapolate"* is the whole of
  it — no long-sequence evaluation anywhere in the paper. This is the belief most worth correcting:
  later work (ALiBi and others) found sinusoidal PE extrapolates *poorly*, and nothing in this paper
  ever contradicted that, because this paper never looked.
- **It does not show sinusoids beat learned embeddings.** *"nearly identical results"* — 4.92/4.92
  PPL, 25.8/25.7 BLEU. Anyone citing row (E) as evidence of sinusoidal superiority is citing a
  0.1-BLEU non-difference. The choice was made on the speculative extrapolation argument, not on the
  measurement.
- **It does not justify 10000**, nor the geometric schedule, nor the interleaved `2i / 2i+1` layout.
  (Most real implementations use a half-split layout — first half sine, second half cosine — which is
  a permutation of the paper's and is never mentioned by it.)
- **It says nothing about the dot product decaying with distance.** The `PE(pos)·PE(pos+k)` curve is
  widely described as a smooth decay; the paper neither plots it, names it, nor claims it. It is a
  consequence the app can display, labelled as such.
- **It does not claim absolute position is recoverable** from the encoding, nor that the encoding is
  injective over positions.
- The app's own predictions establish nothing: seeded noise weights, no training. Whatever the PE
  does to the attention map here is geometry, not learned behaviour.

## Leaves behind

Position enters **once, at the bottom, as an addition to content**, sharing the residual stream and
the `d_model` budget with the token identity — and it enters as *absolute* position, with the
relative structure only implicit and only hypothesized. Both of those become targets. Adding to the
input means position has to survive every layer's projections intact; injecting it into the scores
instead (relative position biases, T5) or rotating Q and K directly (RoPE, 2021) puts it exactly
where the comparison happens. And "may allow the model to extrapolate" is a promissory note that
comes due: ALiBi (2021) is written against precisely this sentence. The app's mixer seam anticipates
the answers — `rotate(vector, pos)` on q and k, and additive `bias(i, j)` — so this card should end
by naming the seam it *does not* use: it only adds a vector to the input embedding, and everything
after it moves further down the pipe.
