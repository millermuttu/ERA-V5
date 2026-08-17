# Concept 13 — ALiBi, attention with linear biases

**Card id:** `alibi` · **Date:** 2021-08-27 (arXiv v1) · **Pressure:** where a token sits

## What was read

- [arXiv:2108.12409](https://arxiv.org/abs/2108.12409), Ofir Press, Noah A. Smith, Mike Lewis —
  *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*.
  Abstract page for the version history and abstract;
  [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2108.12409) for §1 (Introduction), §2
  (Current Approaches Do Not Extrapolate Efficiently), §3 (Attention with Linear Biases), §4
  (Results), §5 (Related Work), Appendix A (Tables 1–12), and **Appendix B (Analysis)** — which is
  where the honest limit lives and where almost nobody who cites this paper has read.
- Version history: v1 **27 Aug 2021**, v2 22 Apr 2022 (the ICLR 2022 camera-ready). The timeline
  uses **v1**. Code and models: `https://github.com/ofirpress/attention_with_linear_biases`.
- Quoted numbers come from the appendix tables, reconstructed cell-by-cell from the ar5iv HTML
  rather than from any secondary summary. Where a table's columns stop (T5 bias ran out of memory
  on a 32 GB GPU past `L_valid ≈ 13k`) that is noted rather than papered over.

**Why this card comes after concept 2 and settles a debt it left open.** The sinusoidal card ends
on *"may allow the model to extrapolate to sequence lengths longer than the ones encountered during
training"* and the observation that the Transformer paper never tested it. This paper opens by
quoting that exact sentence — *"Vaswani et al. (2017), introducing the transformer, speculated that
it 'may […] extrapolate to sequence lengths longer than the ones encountered during training.'"* —
and then goes and measures it. The answer is no. That is the card's first job; the mechanism is its
second.

## The mechanism, precisely

**The definition being measured.** Load-bearing, because half the confusion downstream comes from
people using "extrapolation" to mean something else:

> *"We define extrapolation as a model's ability to continue performing well as the number of input
> tokens during validation increases beyond the number of tokens on which the the model was
> trained."*

(The doubled *"the the"* is in the source.) Note what this does **not** say: it says nothing about
the model *using* the extra tokens. It is a claim about the perplexity curve staying flat or falling
as `L_valid` grows. §B revisits this, and the card must too.

**The baseline it replaces.** §3 restates unmodified attention for the `i`th query, given the first
`i` keys, *"where d is the head dimension"*:

    softmax(q_i K^T)

**The modification, in the paper's own words:**

> *"When using ALiBi, we do not add position embeddings at any point in the network. The only
> modification we apply is after the query-key dot product, where we add a static, non-learned
> bias:"*

    softmax( q_i K^T + m · [ −(i−1), …, −2, −1, 0 ] )

> *"where scalar m is a head-specific slope fixed before training."*

Every clause is doing work. **After the dot product** — not at the input, not on q and k. **Static,
non-learned** — no gradient reaches it, ever. **Head-specific** — one scalar per head, not per layer,
not per position. And the vector is written right-to-left: the last entry, `0`, is the penalty on the
key at the query's own position; `−(i−1)` is the penalty on the very first token of the sequence. The
bias depends on `i − j` and on nothing else — **not on the query vector**, which is precisely the
line that separates it from concept 3.

**The scaling footnote, which implementations get wrong.** Footnote 10, in full:

> *"The ALiBi bias is not multiplied by the √d_k scaling factor from Equation 1 of Vaswani et al.
> (2017)."*

So the composition is `dot(q,k)/√d_k + m·(j−i)`, not `(dot(q,k) + m·(j−i))/√d_k`. The app's seam
already has this right — `mixers.js` computes `dot(q, k) / scale + bias(i, j, q)` — but it is worth a
line on the card, because the slope's *units* are nats-of-logit-per-token only under this convention,
and every "effective span" number below depends on it.

**Figure 3's caption** is the cleanest one-sentence statement of the whole method and is worth
quoting on the card verbatim:

> *"When computing attention scores for each head, our linearly biased attention method, ALiBi, adds
> a constant bias (right) to each attention score (q_i · k_j, left). As in the unmodified attention
> sublayer, the softmax function is then applied to these scores, and the rest of the computation is
> unmodified. m is a head-specific scalar that is set and not learned throughout training. […] When
> using ALiBi, we do not add positional embeddings at the bottom of the network."*

**The slopes — the actual content of the method.**

> *"For our models with 8 heads, the slopes that we used are the geometric sequence:
> 1/2¹, 1/2², …, 1/2⁸. For models that require 16 heads, we interpolate those 8 slopes by
> geometrically averaging every consecutive pair, resulting in the geometric sequence that starts at
> 1/√2 and has the ratio of 1/√2: 1/2^0.5, 1/2¹, 1/2^1.5, …, 1/2⁸. In general, for n heads, our set
> of slopes is the geometric sequence that starts at 2^(−8/n) and uses that same value as its
> ratio."*

Read the general rule carefully: the sequence always **ends at 1/2⁸ = 1/256**, whatever `n` is. The
head count changes how finely the range `[2^−8, 2^−8/n]` is subdivided, not where it stops. For the
app's `n = 4` the rule gives first term and ratio `2^−2 = 1/4`:

    m = { 1/4, 1/16, 1/64, 1/256 } = { 0.25, 0.0625, 0.015625, 0.00390625 }

That set is not in the paper — the paper never runs a 4-head model — but it is what the paper's rule
returns for `n = 4`, and the card should label it that way.

**How the slopes were chosen.** No derivation. Footnote 11, in full:

> *"In our experiments, trainable slopes also slowed down the training speed by 3%. A brief manual
> exploration of around ten slope sets led us to discover the set of slopes that we finally picked.
> Our main insight from this exploration is that the slope sets that work best are those with slopes
> in the (0,1) range, with the slopes' density increasing as we get closer to 0. We also found our
> method to be robust to slope choice. Even randomly sampling from the exponential distribution
> worked well in some cases (although that method had high variance)."*

**Whether learning them helps — the exact sentence, because it is short and often mis-cited:**

> *"We initially experimented with making the slopes trainable, but this did not yield strong
> extrapolation results."*

That is the whole of it in the body. No table, no perplexity, no ablation — an assertion plus a
footnote saying trainable slopes cost 3% training speed. It is *not* a demonstration that the chosen
slopes are optimal; it is a report that the authors tried the obvious thing, did not like the
extrapolation, and moved on. The app must not upgrade this into a result.

**The claimed inductive bias:**

> *"ALiBi has an inductive bias towards recency; it penalizes attention scores between distant
> query-key pairs, with the penalty increasing as the distance between a key and a query grows. The
> different heads increase their penalties at different rates, depending on the slope magnitude."*

And from the abstract: *"ALiBi's inductive bias towards recency also leads it to outperform multiple
strong position methods on the WikiText-103 benchmark."*

**Where the position information lives.**

> *"Since ALiBi is a relative position method, we add position information at every layer to the keys
> and queries but not to the values, as is done in the T5 bias and rotary methods. We hypothesize
> that these properties might be beneficial for extrapolation."*

*Hypothesize*, again — the segregation-of-position argument (§2.2, credited to observing rotary) is
never isolated by an ablation.

**Implementation, and why the cost is zero.**

> *"ALiBi is easy to implement, with all changes accomplished in a few lines of code. We implement it
> by modifying the mask matrix by adding the linear biases to it (in practice, when training a
> transformer LM, query q_i attends only to keys 1 to i; this is implemented by adding a mask matrix
> to the query-key dot product before the softmax operation is applied). This means that there is no
> runtime penalty when using our method since we add no operations to the network."*

The causal mask is already an additive `−∞` matrix on the scores. ALiBi rides along inside it. That
is the entire trick behind "no additional runtime": the addition was already being performed.

> *"Compared to the sinusoidal model trained on the same input lengths, AliBi incurs a memory
> increase (up to 100MB in some of our experiments): in the unmodified transformer, the mask is of
> size L×L; when using ALiBi, the mask is a slightly larger n×L×L (where n is the number of heads)
> since the linear biases added for each head uses a different slope."*

So the honest cost accounting is: **zero parameters, zero extra FLOPs, and one mask tensor that grows
by a factor of `n`** — which is exactly the price of the per-head slopes.

## Numbers that matter

**How badly the incumbents extrapolate.** Appendix Table 3, WikiText-103, all models trained at
`L = 1024`, evaluated with nonoverlapping inference at increasing `L_valid` (perplexity, lower
better):

| `L_valid` | sinusoidal | rotary | T5 bias | ALiBi |
|---|---|---|---|---|
| 1024 (`= L`) | 19.34 | 19.33 | 18.80 | **18.66** |
| 1124 | 19.26 | 19.18 | 18.62 | 18.46 |
| 1524 | 29.82 | 22.59 | 18.42 | 18.22 |
| 2024 | 51.09 | 31.17 | 18.34 | 18.05 |
| 3024 | 96.46 | 35.67 | 18.62 | **17.92** |
| 6024 | 214.02 | 54.78 | 21.76 | 18.01 |
| 10024 | 337.48 | 77.70 | 29.54 | 17.97 |
| 16024 | **453.32** | **106.99** | (OOM) | **17.98** |

The sinusoidal column is the headline of §2 and the fact concept 2's card is missing: **19.34 →
453.32**, a 23× degradation, from a model that was supposed to extrapolate by construction. Rotary
is better and still catastrophic — **19.33 → 106.99**, 5.5×. Same story at `L = 512` (Table 2):
sinusoidal 20.05 → **406.01** at `L_valid = 15512`; rotary 20.07 → 79.25; ALiBi 19.73 → **18.31**.

The prose gives the widths precisely:

> *"while the model improves perplexity up to k=20, performance stops improving and stays steady from
> k=20 to k=50 and then begins degrading"* (sinusoidal, `L=512`); the `L=1024` model *"improves for
> up to L_valid = L+50 tokens, after which performance declines."*
> Rotary: *"the model with L=512 (L=1024) improves perplexity with up to k=200 (k=100) more tokens
> than it saw during training, but this comes at the cost of slower training and inference."*
> T5 bias: *"improves perplexity with longer sequences than the ones it was trained on, i.e., k=600
> (k=800) extra tokens"* — but *"training is at least twice as slow as with the sinusoidal model.
> Therefore, this model's extrapolation ability provides no efficiency advantage."*

So the honest ranking of extrapolation *headroom* before this paper: sinusoidal ~20–50 tokens, rotary
~100–200 tokens, T5 bias ~600–800 tokens. Not one of them is a length regime anyone would call long.
And the learned-embedding method, per footnote 4, *"does not have a way to encode positions greater
than L; it therefore has no ability to extrapolate"* — a hard wall, not a decay. Concept 4's card
already owns that wall; this card owns the decay.

**Cost, exactly.** Abstract and §1: *"Compared to a sinusoidal model trained on the same input
length, our method requires no additional runtime or parameters and incurs a negligible (0–0.7%)
memory increase."* Figure 2's caption: *"The speed differences between our method and the sinusoidal
are within 1% during training and 3% for inference, which is insignificant on our hardware. ALiBi
uses 100MB of extra memory when training on input lengths 1024 and 3072 in this setting."*
Appendix Table 1, one V100, WikiText-103:

| method | `L` | train WPS | eval WPS | memory |
|---|---|---|---|---|
| sinusoidal | 512 / 1024 / 3072 | 28.5k / 26.0k / 15.3k | 82.1k / 77.8k / 42.4k | 15.3 / 19.2 / 15.1 GB |
| rotary | 512 / 1024 / 3072 | 20.0k / 17.7k / 11.5k | 43.4k / 39.4k / 29.5k | 17.8 / 22.8 / 17.8 GB |
| T5 bias | 512 / 1024 / 3072 | 14.4k / 13.0k / 4.3k | 21.8k / 20.2k / 4.9k | 16.9 / 20.9 / 15.9 GB |
| **ALiBi** | 512 / 1024 / 3072 | **28.3k / 25.8k / 15.5k** | **85.8k / 76.4k / 42.2k** | **15.3 / 19.3 / 15.2 GB** |

The `0–0.7%` is literally `15.3→15.3` (0%), `19.2→19.3` (+0.5%), `15.1→15.2` (+0.7%). T5 bias at
`L=3072` runs at 4.3k WPS against sinusoidal's 15.3k — 3.6× slower. That gap is the reason ALiBi
exists rather than "just use the T5 bias".

**The headline result, both halves.** §1: *"a 1.3 billion parameter LM trained on L=1024 tokens with
ALiBi achieves the same perplexity as a sinusoidal model trained on L=2048 when both are tested on
sequences of 2048 tokens, even though our model is 11% faster and uses 11% less memory."* The cells
behind it, Appendix Table 11 (CC100+RoBERTa, 461 GB, 1.3B params, 25 layers, 16 heads, d=2048, one
epoch on 128 V100s):

| model | memory | updates | hours | PPL @ `L_valid`=2048 |
|---|---|---|---|---|
| sinusoidal, `L_train` = 2048 | 29.3 GB | 44.2k | 5.9k | 9.01 |
| **ALiBi, `L_train` = 1024** | **26.2 GB** | 50.0k | 5.9k | **8.92** |

Same wall-clock, 3.1 GB less memory, and it *wins* by 0.09. §4.2 spells out the speed claim: *"By
sampling five evenly distributed points across the training process, we compute that our L=1024 model
reaches a given perplexity value, on average, 11% faster than the sinusoidal model does."*

**And the smaller half of the same result, which is less flattering.** Appendix Table 12, all models
at 50k updates, same corpus:

| model | PPL @512 | PPL @1024 | PPL @2048 |
|---|---|---|---|
| sinusoidal, `L_train`=512 | 9.71 | **37.05** | **105.42** |
| ALiBi, `L_train`=512 | 9.79 | 9.30 | 9.54 |
| sinusoidal, `L_train`=1024 | — | 9.15 | 48.85 |
| ALiBi, `L_train`=1024 | — | 9.16 | 8.92 |
| sinusoidal, `L_train`=2048 | — | — | 8.83 |
| ALiBi, `L_train`=2048 | — | — | 8.84 |

At `L_valid = L` and 1.3B scale, ALiBi is a *wash*: 9.79 vs 9.71, 9.16 vs 9.15, 8.84 vs 8.83 — losing
each time by a hair. The paper says so: *"ALiBi performs similarly to the sinusoidal baseline when not
extrapolating. This contrasts with the results presented on the smaller datasets, where ALiBi
consistently outperforms other position methods even when not extrapolating, suggesting that ALiBi's
inductive bias provides additional benefits for lower-resource language modeling."* The entire 1.3B
win is the *extrapolation* column, where the baseline is at 105.42 and 48.85.

**Where the peak sits.** §1: *"Though performance peaks at around two times the number of tokens that
the model was trained on, ALiBi maintains strong performance even on sequences of length 10,000."*
§4.2 with the cells: the `L=512` model *"(that obtains 9.79 perplexity when L_valid = 512) achieves
its best score (9.3) when extrapolating to 1012 tokens, and the L=1024 model (that obtains 9.16
perplexity when L_valid = 1024) achieves its best score (8.9) when extrapolating to 2024 tokens."*
The paper's own guess at why, worth quoting because it is a counting argument rather than a claim
about attention: *"When performing inference on subsequences of length 2L, half of the subsequences
the model consumes are as long as the examples seen during training. When inference is performed on
subsequences of length 2L+1 or longer, less than half of the predictions the model makes are on
subsequences of lengths seen during training, and that might degrade performance."*

**The honest limit, with the table that shows it.** Appendix B.2 re-runs the evaluation with a sliding
window at stride `S = 1`, *"giving each prediction the maximum number of context tokens that the model
can use"* — which removes the early-token curse from every arm. Appendix Table 15, ALiBi, WikiText-103
validation:

| `L_train` \ eval len | 512 | 1024 | 1536 | 2048 | 3072 |
|---|---|---|---|---|---|
| 512 | 17.98 | 17.92 | 18.2 | 18.28 | 18.3 |
| 1024 | — | 17.46 | 17.47 | 17.62 | 17.92 |
| 3072 | — | — | — | — | 16.96 |

Compare against the nonoverlapping numbers for the same `L=512` ALiBi model: 19.73 at 512 falling to
18.40 at 3072 — an apparent 1.33-point *gain* from length. Under stride-1 evaluation the same model
goes 17.98 → **18.3**, i.e. it gets very slightly **worse** with more context. The gain was the
evaluation protocol, not the model. (Table 13, sinusoidal, for contrast: 18.35 at 512 → 204.42 at
1024 → 360.12 at 3072 — it explodes even *with* full context, so the sinusoidal failure is real and
not an artifact.)

**Effective span, computable from the slopes alone** (this is the app's material, derived here, not
in the paper). With bias `−m·d` at distance `d`, the softmax weight ratio between the nearest key and
one `d` away is `exp(m·d)` before any content term. Define a head's "4-nat horizon" as the distance at
which the penalty reaches 4 nats — where a key is down 55× on position alone — i.e. `d = 4/m`:

| heads | slope `m` | 1-nat horizon `1/m` | 4-nat horizon `4/m` |
|---|---|---|---|
| n=8, head 1 | 1/2 | 2 | 8 |
| n=8, head 8 | 1/256 | 256 | 1024 |
| **n=4 (this app), head 0** | **0.25** | 4 | **16** |
| n=4, head 1 | 0.0625 | 16 | 64 |
| n=4, head 2 | 0.015625 | 64 | 256 |
| n=4, head 3 | 0.00390625 | 256 | 1024 |

The coincidence is a gift: at `n = 4` the sharpest head's 4-nat horizon is **exactly 16 tokens**, the
app's sentence length, and the flattest head's is 1024 — 64× longer than the sentence. So in the toy,
head 0 is a hard recency head and head 3 is very nearly position-blind, and both facts are visible in
the same 16-token sentence.

| quantity | paper | this app | note |
|---|---|---|---|
| heads | 8 (WikiText) / 16 (1.3B) | 4 | slopes from the general rule `2^(−8/n)`, `n=4` |
| slopes | 1/2 … 1/256 | 1/4, 1/16, 1/64, 1/256 | the paper never runs `n=4`; say so |
| `√d_k` on the bias | **not applied** (fn. 10) | `dot/√8 + bias` | already correct in `mixers.js` |
| parameters added | **0** | 0 | the honest headline |
| mask tensor | `L×L` → `n×L×L` | 16×16 → 4×16×16 | the only real cost |
| `L` vs `L_valid` | 512…3072 vs up to 16024 | 16 vs 16 | **the app cannot show extrapolation** |

## What the live view must let the reader do

Each of these changes a real computed quantity in the running toy model (32 dims, 4 heads, `d_k = 8`,
2 blocks, causal mask, seeded untrained weights, editable ~16-token sentence).

**A seam note first, because interaction 1 needs it.** `mixers.js` calls `bias(i, j, q)` — the `at`
object carrying `{block, head, kvHead}` reaches `readable(i, j, at)` but *not* `bias`. Per-head slopes
need `bias(i, j, q, at)`, a one-argument change at `mixers.js:33`, and the `alibi()` stub in
`position.js` (currently `({slope = 0.5}) => ({ bias: (i,j) => -(i-j)*slope })`) generalised to take
the slope *set*. The sign and the `i−j` form in the stub are already right. This is the one code
change the card depends on, and it is small.

1. **The four heads, four slopes, side by side — this is the mechanism.** Show four attention
   heatmaps for block 0 with `m = {0.25, 0.0625, 0.015625, 0.00390625}`, and under each one number:
   the **fraction of the last query's attention mass falling on the nearest 4 keys**. With the bias
   dominating the near-random untrained scores, the closed form is
   `Σ_{d<4} e^{−m d} / Σ_{d<16} e^{−m d}`, giving roughly **64% for head 0** and **26% for head 3**
   (uniform over 16 keys would be 25%). Print alongside it the **nearest-to-farthest weight ratio**
   `e^{15m}`: **42.5× for head 0, 1.06× for head 3**. Those two numbers, read across four heads, are
   the paper's *"the different heads increase their penalties at different rates"* made arithmetic.
   Give the reader a toggle between "paper slopes for n=4" and a single shared slope applied to all
   heads — with one slope the four heatmaps become near-identical and the span table collapses to one
   row, which is the clearest possible demonstration that the *geometric spread*, not the linear
   penalty, is what the method actually contributes.

2. **The bias does not depend on the query — the direct contrast with concept 3.** The app already
   measures Shaw's query-dependence. Reuse that instrument: display
   `spread(d) = max_i b(i, i−d) − min_i b(i, i−d)` over all queries `i` at each distance `d`. For
   ALiBi this reads **exactly 0.000 at every `d`** (float-exact, since the same scalar is subtracted);
   for `relativeBuckets` the same readout is large and varies with `d`, because the term is
   `q_i · w_clip(j−i)/√d_z`. One column of zeros against one column of nonzeros. Then state the
   consequence the zeros buy: ALiBi needs no lookup table, so `d` is unbounded — set `d` to 500 and
   the bias is still defined (`−m·500`), whereas Shaw's table clips at `k` and the learned table of
   concept 4 has nothing at all. That is extrapolation's *mechanical* precondition, and it is the most
   of extrapolation the app can honestly show at 16 tokens.

3. **What the monotone penalty costs — the far-back token that deserves to win.** Let the reader mark
   one token as the "answer" (e.g. a name at position 1 that position 15 should retrieve). Show, for
   the marked pair, the **content score `dot(q,k)/√8`**, the **bias `−m·d`**, and the **deficit**: how
   many nats the content term must beat the best near neighbour by, merely to tie. At head 0, distance
   15, that deficit is **3.75 nats** — the content logit has to win by 3.75 before position stops
   deciding — while at head 3 it is **0.059 nats**. Then display the marked token's actual attention
   weight in each of the four heads and let the reader watch it be readable *only* by the flat heads.
   This is the card's honest cost: a recency prior is a prior, and it fights genuine long-range
   evidence with a force that grows without bound in `d`. Nothing in the paper measures this; label it
   as the app's own demonstration.

4. **Curve view: the bias against distance, one line per head, with the softmax consequence.** Plot
   `−m·d` for `d = 0…40` (past the sentence, to make the unboundedness visible) for all four slopes on
   a shared axis, and beneath it the same four as post-softmax weight profiles for the last query.
   Two readouts: the **4-nat horizon** `4/m` marked on each line (4, 16, 64, 256 for `n=4`), and a
   marker at the sentence length showing which heads have already saturated inside the sentence
   (head 0 has; head 3 has not, by a factor of 64). Add a slope-set selector — paper `n=4`, paper
   `n=8` (using the first four, `1/2…1/16`), and all-equal — so the reader can see the paper's rule
   place the horizons *geometrically* rather than uniformly, which is the footnote-11 remark
   (*"slopes' density increasing as we get closer to 0"*) drawn rather than asserted.

5. **Turn the slopes into learned parameters, and count what that costs.** Expose the four slopes as
   four draggable numbers and show a running **parameter count delta**: `+0` for ALiBi as published,
   `+4` if the slopes were learned, `+(2k+1)·8 = 72` for concept 3's relative buckets at `k = 4`, and
   `+rows·32` for concept 4's learned table. The point is the zero: the seam's cost view can display
   ALiBi's *entire* footprint as one number that does not move. Then let dragging the slopes wreck it
   — set all four to 2.0 and watch every head collapse onto the diagonal (nearest-to-farthest ratio
   `e^30 ≈ 1e13`, attention numerically one-hot), set all four to 0 and recover plain causal attention
   bit-for-bit. Both endpoints are checkable equalities, and between them sits the paper's remark that
   *"the slope sets that work best are those with slopes in the (0,1) range"* — an observation from
   *"a brief manual exploration of around ten slope sets"*, which the card should quote next to the
   slider so nobody mistakes the defaults for something derived.

## What the source does *not* establish

- **It does not show that ALiBi uses the longer context.** This is the single most-misread thing in
  the paper, and the paper itself says it plainly, twice. Appendix B: *"We find that ALiBi's decrease
  in perplexity when given longer sequences is largely explained by its improved avoidance of the
  early token curse."* And in the analysis: *"our analysis reveals that when L_valid > L, ALiBi might
  not be using contexts longer than the ones it was trained on. This highlights a research direction
  that could be pursued in future work."* Figure 11's caption: *"This might mean that ALiBi increases
  performance when L_valid > L not because it uses longer contexts, but because fewer tokens suffer
  from the early token curse."* The evidence is Table 15 above: with stride-1 evaluation, the `L=512`
  model's perplexity goes **17.98 → 17.92 → 18.2 → 18.28 → 18.3** as evaluation length grows. Flat,
  drifting slightly the wrong way. So the demonstrated property is **"does not blow up"**, not
  **"reads further back"**. The paper's own summary of what remains: *"These findings do not lessen
  the value of ALiBi. When L_valid = L, ALiBi achieves either superior or similar results to the
  sinusoidal method and other alternatives even though it is simpler and requires no learned
  parameters. When evaluating L_valid > L tokens, even if ALiBi does not attend to more than L tokens,
  it yields better results than the other alternatives that can be used in this case."* That is a real
  and useful claim — it is just a much narrower one than "trains short, tests long" sounds.
- **It never measures an effective attention span.** The sliding-window-like behaviour that everyone
  attributes to ALiBi — that a slope `m` gives each head a soft window of roughly `1/m` tokens — is
  *not* in this paper. There is no plot of attention mass against distance, no entropy measurement, no
  span statistic anywhere. The paper argues from perplexity only. The app's horizon table in §Numbers
  is derived from the slope definition, and the card must label it as the app's arithmetic. The
  paper's closest statement is the qualitative *"the different heads increase their penalties at
  different rates, depending on the slope magnitude."*
- **It says nothing about early-token behaviour of the attention itself.** "Early token curse" in this
  paper is strictly an *evaluation-protocol* fact — predictions near the start of a nonoverlapping
  chunk have little left context, so their perplexity is bad — defined by citation to Press et al.
  (2021). It is not a claim about attention sinks, about the first token absorbing mass, or about what
  ALiBi does to position 0. None of that is here.
- **It does not show the slopes are good, only that they worked.** *"A brief manual exploration of
  around ten slope sets"*, and *"we found our method to be robust to slope choice. Even randomly
  sampling from the exponential distribution worked well in some cases (although that method had high
  variance)."* Robustness cuts both ways: if random exponential draws work, the specific geometric
  sequence is not carrying much, and the paper offers no experiment separating "geometric spread of
  slopes" from "some spread of slopes".
- **It does not show learned slopes fail.** *"this did not yield strong extrapolation results"* is one
  sentence with no numbers attached. Treat it as a negative anecdote, not a finding.
- **It does not test ALiBi on anything but causal language modelling.** No bidirectional encoder, no
  machine translation, no classification, no downstream task. Every number in the paper is a
  perplexity. §5 notes that Wennberg & Henter's concurrent RBF-bias method *"present experiments on
  text classification, not on language modeling"* and *"do not explore extrapolation"* — the two
  literatures never meet inside this paper.
- **It does not establish superiority at `L_valid = L` at scale.** Table 12: 9.79/9.71, 9.16/9.15,
  8.84/8.83 — ALiBi loses all three by a rounding error. The paper is honest about it. Anyone citing
  ALiBi as a straightforward quality improvement at 1B+ scale is citing the extrapolation column.
- **It gives no theory.** There is no argument for why a *linear* penalty rather than logarithmic,
  bucketed, or learned-decay; the one comparison offered is empirical and parenthetical — of
  multiplying rather than adding the bias, §5 says only *"In our experiments (not presented),
  multiplying attention scores by the bias (instead of adding, as in ALiBi) degraded performance."*
  Not presented.
- The app's own predictions establish nothing: seeded noise weights, no training, 16 tokens. With
  untrained weights the content logits are small and near-random, so the bias dominates almost
  totally — the heatmaps will look *cleaner* than a trained model's, and the card should say so. And
  at 16 tokens the app cannot show extrapolation at all; it can only show that the bias is *defined*
  everywhere, which is the precondition, not the result.

## Leaves behind

Position stops being a vector and becomes **a scalar per (head, distance) pair, added to the score and
never learned** — the smallest thing that can possibly encode order, and the end of the line that
started with concept 2's `d_model`-wide sinusoid competing with content for the residual stream.
Concept 2 put position at the bottom, once, added to content; concept 3 moved it into the score but
kept it learned and query-dependent; ALiBi keeps the score-side injection and throws away both the
learning and the query. What is left is one number: `m`.

**Backward.** This card retires two beliefs the timeline has been carrying. Sinusoidal encoding was
*chosen* over learned embeddings on the strength of *"may allow the model to extrapolate"* — a
sentence the app's concept 2 card correctly flags as never tested — and this paper tests it and finds
19.34 → 453.32. RoPE is worse-off than its reputation too: it is *defined* at any position (the
rotation angle `pos·ω_i` needs no table), so it looks like it should extrapolate, and Figure 1 shows it
buys ~100–200 tokens before decaying to 106.99. "Defined beyond `L`" and "works beyond `L`" turn out to
be different properties, and this is the paper that separated them. The T5 bias — the closest ancestor,
a learned scalar per bucketed distance added to the score — already had the right *shape*; ALiBi's
contribution is deleting the learning and the bucketing from it, at 3.6× the training speed.

**Forward.** The monotone penalty is the fork in the road. Because `−m·d` grows without bound, a
distant token can never be recovered by content alone (interaction 3 puts a number on it: 3.75 nats at
15 tokens in the sharpest of four heads), and because §B could not show the longer context being *used*,
ALiBi's extrapolation is closer to a graceful soft window than to a longer memory. Frontier models
mostly went the other way: keep RoPE and stretch it — position interpolation, NTK-aware scaling, YaRN —
buying real long context at the cost of a little fine-tuning, rather than accepting a permanent recency
prior for free. ALiBi's branch is not dead (BLOOM and MPT shipped it, and the soft-window reading is
exactly what concept 11's sliding window makes explicit and cheap), but the timeline should show two
branches from here, not one road. The seam records the split: ALiBi is the case where `bias(i, j, q)`
ignores `q` entirely and needs `{head}`, RoPE is the case where nothing enters the score at all and
`rotate(v, pos)` does the work — and the extension methods are all a `stretch` factor on that rotate,
which `position.js` already has a parameter for.
