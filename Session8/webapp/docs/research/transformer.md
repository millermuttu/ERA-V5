# Concept 1 — Scaled dot-product attention, multi-head

**Card id:** `transformer` · **Date:** 2017-06-12 (arXiv v1) · **Pressure:** the mechanism itself

## What was read

- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), Vaswani, Shazeer, Parmar, Uszkoreit, Jones,
  Gomez, Kaiser, Polosukhin — *Attention Is All You Need*. Abstract page for the version history,
  and the full text for §3.2 (attention), §3.2.2 (multi-head), §3.5 (positional encoding), Table 1.
- Version history: v1 **12 Jun 2017**, then v2 19 Jun, v3 20 Jun, v4 30 Jun, v5 6 Dec 2017, v6 24 Jul
  2023, v7 2 Aug 2023. The timeline uses **v1**, which is the date the idea became public.

## The mechanism, precisely

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Six stages, and the assignment asks for exactly these six to be obvious:

1. **Q × K** — every query against every key, one dot product per pair.
2. **scores** — the T×T matrix those dot products form.
3. **scale** — divide by `√d_k`.
4. **mask** — set illegal positions to −∞.
5. **softmax** — each row becomes weights that are positive and sum to one.
6. **weighted sum of V** — those weights combine the value vectors.

**Why the scale, in the paper's own terms.** Not hand-waving about "keeping numbers small" — a
variance argument: *"for large values of d_k, the dot products grow large in magnitude, pushing the
softmax function into regions where it has extremely small gradients."* The footnote gives the
reason: if the components of `q` and `k` are independent, mean 0, variance 1, then `q · k` has mean 0
and **variance d_k**. Dividing by `√d_k` returns the variance to 1.

The paper also records what this fixed: *"while for small values of d_k the two mechanisms perform
similarly, additive attention outperforms dot product attention without scaling for larger values of
d_k."* Dot-product attention is *"much faster and more space-efficient in practice"* because it is a
matrix multiply — so the scale is what let the faster mechanism win.

**Multi-head.**

    MultiHead(Q, K, V) = Concat(head₁, …, head_h) Wᴼ
    head_i = Attention(Q Wᵢ^Q, K Wᵢ^K, V Wᵢ^V)

Paper's configuration: `h = 8`, `d_model = 512`, `d_k = d_v = d_model / h = 64`. The stated reason is
worth quoting on the card because it is a claim about *averaging*: *"Multi-head attention allows the
model to jointly attend to information from different representation subspaces at different
positions. With a single attention head, averaging inhibits this."*

**Masking.** *"We implement this inside of scaled dot-product attention by masking out (setting to
−∞) all values in the input of the softmax which correspond to illegal connections."* Combined with
the output embeddings being offset by one, predictions at position i depend only on positions < i.
This is exactly what the app's `softmaxMixer` does, which is why the mask toggle is honest.

**The cost, from Table 1.**

| layer type | complexity per layer | sequential operations | maximum path length |
|---|---|---|---|
| self-attention | O(n²·d) | O(1) | O(1) |
| recurrent | O(n·d²) | O(n) | O(n) |

That row is the whole timeline in miniature: the Transformer bought O(1) sequential operations and an
O(1) path between any two tokens, and paid O(n²) per layer. Everything after it is somebody
attacking one of those terms.

## Numbers that matter

| quantity | paper | this app | note |
|---|---|---|---|
| d_model | 512 | 32 | small enough to re-run per keystroke |
| heads | 8 | 4 | |
| d_k = d_v | 64 | 8 | so the scale is √8 ≈ 2.83, not √64 = 8 |
| blocks | 6 | 2 | |
| scale | √d_k | √8 | same formula, smaller model |

The card must say the app's model is smaller than the paper's and untrained, so nobody reads the
predictions as competence.

## What the live view must let the reader do

1. **Walk the six stages** on their own sentence, each with its own numbers: raw Q·K, the scaled
   version, the mask applied, the softmax result, and the output vector. Not one finished heatmap.
2. **Turn the scale off** and watch the variance of the scores blow up and the softmax collapse
   towards one-hot — the paper's argument, reproduced rather than asserted. Report the measured score
   variance against d_k so the footnote's claim is visible as a number.
3. **Turn the mask off** and see weight land on tokens that have not happened yet, in their own
   sentence.
4. **Step through the heads** and see that they are not the same picture — and collapse them to a
   single averaged head to see what the paper means by *averaging inhibits this*.
5. **Watch the cost**: keys read, the T² growth, and the O(1) path length that bought it.

## Measured in this app's model, before building the card

Checked on the default sentence, block 1 head 1, `d_k = 8`:

| | unscaled | scaled by √8 |
|---|---|---|
| score variance | 107.96 | 13.49 |
| ratio | | **8.00 — exactly d_k** |
| largest weight in row 9 | 0.998 | 0.823 |
| entropy of row 9 | 0.016 | 0.682 |

The ratio is exactly `d_k` because dividing every score by `√d_k` divides the variance by `d_k` —
that part is mechanical and the card can state it. The *absolute* variance is 108, not the paper's
idealised `d_k`, because these queries and keys are projections of layer-normed embeddings rather
than the independent unit-variance components the footnote assumes. The card says the ratio, not the
absolute, so it does not overclaim the footnote.

The consequence the paper describes is plainly visible: unscaled, the row collapses to almost
one-hot (0.998, entropy 0.016) — softmax in the region where its gradients nearly vanish.

## What the source does *not* establish

- It does **not** claim heads are individually interpretable or that any head is a "syntax head".
  The claim is about representation subspaces; later work has argued both ways. The card says
  subspaces, not roles.
- √d_k is **not** shown to be optimal — it is a variance normalisation with a stated motivation.
- O(n²·d) is **per layer**, not the model's total cost, and it says nothing about wall-clock time on
  real hardware. FlashAttention later makes exactly that point.
- Nothing here is a claim about long context. The paper trains at 512-ish lengths; every long-context
  claim on this timeline comes later and separately.
- The app's own predictions establish nothing at all: seeded noise weights, no training.

## Leaves behind

Two costs, which the rest of the timeline answers: the T² score matrix (**compute**, picked up first
by the Sparse Transformer, 2019) and, at generation time, a key/value cache that grows with every
token (**memory**, picked up first by multi-query attention, 2019). Position is not solved either —
attention is permutation-equivariant, so the very same paper has to add positional encoding, which is
concept 2.
