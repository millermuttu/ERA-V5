# Concept 3 — Relative position representations
**Card id:** `relative-positions` · **Date:** 6 March 2018 (arXiv v1; v2 12 April 2018) · **Pressure:** where a token sits

## What was read

- `https://arxiv.org/abs/1803.02155` — abstract page, for the version history and the abstract as published.
  Peter Shaw, Jakob Uszkoreit, Ashish Vaswani. Submitted **v1: 6 Mar 2018**, **v2: 12 Apr 2018**.
- `https://ar5iv.labs.arxiv.org/html/1803.02155` — full text, fetched raw and read end to end
  (sections 1–5, Tables 1–3, all equations). ar5iv renders the **latest** version, so everything
  below is v2 text. The card date is v1 because that is when the idea landed.

The paper is short: five sections, three tables, five equations. Everything load-bearing is
quoted verbatim below, marked with `>`. Nothing here is from recall.

## The mechanism, precisely

### The baseline it modifies

Standard self-attention, as the paper restates it (eq. 1 and eq. 2):

```
z_i = Σ_{j=1..n} α_ij (x_j W^V)                        (1)

α_ij = exp(e_ij) / Σ_{k=1..n} exp(e_ik)

e_ij = ( (x_i W^Q)(x_j W^K)^T ) / sqrt(d_z)            (2)
```

> "W^Q, W^K, W^V ∈ ℝ^{d_x × d_z} are parameter matrices. These parameter matrices are unique per
> layer and attention head."

And the framing of why position is a problem at all:

> "For the Transformer, which employs neither convolution nor recurrence, incorporating explicit
> representations of position information is an especially important consideration since the model
> is otherwise entirely invariant to sequence ordering."

### The two edge terms

The input is recast as a graph:

> "We propose an extension to self-attention to consider the pairwise relationships between input
> elements. In this sense, we model the input as a labeled, directed, fully-connected graph."

> "The edge between input elements x_i and x_j is represented by vectors a^V_ij, a^K_ij ∈ ℝ^{d_a}.
> The motivation for learning two distinct edge representations is that a^V_ij and a^K_ij are
> suitable for use in eq. (3) and eq. (4), respectively, without requiring additional linear
> transformations. These representations can be shared across attention heads. We use d_a = d_z."

**Into the value sum** (eq. 3), the paper's modification of eq. 1:

```
z_i = Σ_{j=1..n} α_ij ( x_j W^V + a^V_ij )             (3)
```

> "This extension is presumably important for tasks where information about the edge types selected
> by a given attention head is useful to downstream encoder or decoder layers. However, as explored
> in 4.3, this may not be necessary for machine translation."

**Into the score** (eq. 4), the paper's modification of eq. 2:

```
e_ij = ( x_i W^Q ( x_j W^K + a^K_ij )^T ) / sqrt(d_z)  (4)
```

> "The primary motivation for using simple addition to incorporate edge representations in eq. (3)
> and eq. (4) is to enable an efficient implementation described in 3.3."

Note the shape of the score term carefully, because the whole timeline turns on it. The edge vector
is added **to the key**, then dotted with the query. Distributing that product (the paper does this
itself in eq. 5) gives:

```
e_ij = [ x_i W^Q (x_j W^K)^T  +  x_i W^Q (a^K_ij)^T ] / sqrt(d_z)   (5)
```

So the position term is `q_i · w_{clip(j-i,k)} / sqrt(d_z)` — an additive bias on the score, but a
**query-dependent** one. It is not a scalar attached to an offset; it is a dot product between the
querying token's own query vector and a learned per-offset vector. Two different tokens at the same
relative offset get different amounts of bias. (This is exactly the property ALiBi later throws
away; see *Leaves behind*.)

### Clipping and k

> "For linear sequences, edges can capture information about the relative position differences
> between input elements. The maximum relative position we consider is clipped to a maximum absolute
> value of k. **We hypothesized that precise relative position information is not useful beyond a
> certain distance.** Clipping the maximum distance also enables the model to generalize to sequence
> lengths not seen during training. Therefore, we consider 2k+1 unique edge labels."

(Bold mine; the sentence is otherwise verbatim, including the past tense "hypothesized".)

```
a^K_ij   = w^K_{clip(j-i, k)}
a^V_ij   = w^V_{clip(j-i, k)}
clip(x, k) = max(−k, min(k, x))
```

> "We then learn relative position representations w^K = (w^K_{−k}, …, w^K_k) and
> w^V = (w^V_{−k}, …, w^V_k) where w^K_i, w^V_i ∈ ℝ^{d_a}."

Two consequences worth stating flatly. `clip` is a **saturating** function, not a modulus: every
pair further apart than k collapses onto the same single label, so beyond k the model literally
cannot tell 20 tokens back from 200 tokens back. And the label is `j - i`, **signed** — the graph is
directed, so "3 to my left" and "3 to my right" are different learned vectors.

### Sharing: what is possible vs. what they did

This is the most commonly mis-stated part of the paper, so both statements go here side by side.

What §3.1 / §3.3 say is *possible*:

> "These representations can be shared across attention heads."

> "For a sequence of length n and h attention heads, we reduce the space complexity of storing
> relative position representations from O(hn²d_a) to O(n²d_a) by sharing them across each heads.
> Additionally, relative position representations can be shared across sequences."

What §4.1 says they *actually trained*:

> "When using relative position encodings, we used clipping distance k = 16, and used **unique edge
> representations per layer and head**." (base model)

> "When using relative position encodings, we used k = 8, and used **unique edge representations per
> layer**." (big model)

So: **never shared across layers, in either configuration.** Shared across heads only in the big
model. The base model — the one all of §4.3's ablations run on — shares nothing. Head-sharing is
presented as a complexity lever the method *permits*, not as the configuration that produced the
headline BLEU. There is no ablation of shared vs. unshared.

### Efficient implementation, and the exact cost

The memory statement:

> "Therefore, the overall self-attention space complexity increases from O(bhnd_z) to
> O(bhnd_z + n²d_a). Given d_a = d_z, the size of the relative increase depends on n/(bh)."

The compute problem, verbatim, and this is the sentence the two successor papers exist to fix:

> "The Transformer computes self-attention efficiently for all sequences, heads, and positions in a
> batch using parallel matrix multiplication operations. Without relative position representations,
> each e_ij can be computed using bh parallel multiplications of n × d_z and d_z × n matrices. Each
> matrix multiplication computes e_ij for all sequence positions, for a particular head and
> sequence. For any sequence and head, this requires sharing the same representation for each
> position across all compatibility function applications (dot products) with other positions."

> "When we consider relative positions the representations differ with different pairs of positions.
> **This prevents us from computing all e_ij for all pairs of positions in a single matrix
> multiplication.** We also want to avoid broadcasting relative position representations."

Unpacked: the plain `QK^T` works because *one* key vector for token j is reused against every query
i. The relative term destroys that reuse — the operand paired with query i depends on the pair
(i, j), not on j alone. Materialising the pair-wise key tensor would be an n × n × d_a broadcast,
which they explicitly refuse.

Their fix is the distributive split of eq. 5 plus a transpose trick:

> "The first term is identical to eq. (2), and can be computed as described above. For the second
> term involving relative position representations, tensor reshaping can be used to compute n
> parallel multiplications of bh × d_z and d_z × n matrices. Each matrix multiplication computes
> contributions to e_ij for all heads and batches, corresponding to a particular sequence position.
> Further reshaping allows adding the two terms. The same approach can be used to efficiently
> compute eq. (3)."

So the second term is parallelised over **sequence position** instead of over batch×head — the loop
axis moves. It stays one fused op rather than n kernel launches, but it is a *second* matmul with a
different data layout, plus reshapes, plus an add. Measured price:

> "For our machine translation experiments, the result was a modest 7% decrease in steps per second,
> but we were able to maintain the same model and batch sizes on P100 GPUs."

### Replaces, does not augment

> "In this work we present an efficient way of incorporating relative position representations in
> the self-attention mechanism of the Transformer. **Even when entirely replacing its absolute
> position encodings**, we demonstrate significant improvements in translation quality on two
> machine translation tasks."

> "In our experiments we did not observe any benefit from including sinusoidal position encodings in
> addition to relative position representations."

And from the abstract:

> "Notably, we observe that combining relative and absolute position representations yields no
> further improvement in translation quality."

All of §4.3 runs "without any absolute position representations". Note this is an *observation
reported in prose* — there is no table row for relative+absolute. The combined-condition number is
not printed anywhere in the paper.

### Backward link — what absolute/sinusoidal could not do

The paper states the gap itself, in §2.1:

> "In contrast to learned, absolute position representations, the authors hypothesized that
> sinusoidal position encodings would help the model to generalize to sequence lengths unseen during
> training by allowing it to learn to attend also by relative position. **This property is shared by
> our relative position representations which, in contrast to absolute position representations, are
> invariant to the total sequence length.**"

That is the whole backward edge. Sinusoids make relative attention *learnable in principle* — the
model must discover, from data, that a phase difference between two encodings means an offset, and
it must rediscover it separately for every offset and every head. Nothing in the architecture
enforces it. Absolute encodings are also injected **once, at the input, before layer 1**
("Position encodings based on sinusoids of varying frequency are added to encoder and decoder input
elements prior to the first layer… Residual connections help propagate position information to
higher layers") — so deeper layers get position only as whatever survived the residual stream. Shaw
instead makes offset a **first-class term inside every attention score in every layer**, structurally
identical for a given offset no matter where in the sentence that pair sits.

## Numbers that matter

**Table 1 — WMT 2014, newstest2014.** Baselines were regenerated by the authors, not copied:
> "We generated baseline results to isolate the impact of relative position representations from any
> other changes to the underlying library and experimental configuration."

| Model | Position information | EN-DE BLEU | EN-FR BLEU |
|---|---|---|---|
| Transformer (base) | Absolute | 26.5 | 38.2 |
| Transformer (base) | Relative | **26.8** (+0.3) | **38.7** (+0.5) |
| Transformer (big) | Absolute | 27.9 | 41.2 |
| Transformer (big) | Relative | **29.2** (+1.3) | **41.5** (+0.3) |

> "For English-to-German our approach improved performance over our baseline by 0.3 and 1.3 BLEU for
> the base and big configurations, respectively. For English-to-French it improved by 0.5 and 0.3
> BLEU for the base and big configurations, respectively."

The abstract's "1.3 BLEU and 0.3 BLEU" pairs the **big** EN-DE gain with the **big** EN-FR gain.
Both headline numbers are the big model. The base model's gains are 0.3 / 0.5 — the EN-DE base gain
is the smallest of the four.

**Table 2 — clipping distance k.** Base config, no absolute positions, EN-DE dev set newstest2013.

| k | 0 | 1 | 2 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| BLEU | 12.5 | 25.5 | 25.8 | 25.9 | 25.8 | 25.9 | 25.8 |

> "Notably, for k ≥ 2, there does not appear to be much variation in BLEU scores. However, as we use
> multiple encoder layers, precise relative position information may be able to propagate beyond the
> clipping distance."

k = 0 (12.5) is the no-position floor — one edge label for all pairs is worth nothing. k = 1 recovers
almost everything (25.5). From k = 2 to k = 256 the spread is 0.4 BLEU and non-monotonic. The trained
base model used k = 16 and the big model k = 8, but Table 2 says k = 2 would have done the same job.

**Table 3 — ablating the two terms.** Same setup (base, no absolute, newstest2013).

| a^V_ij | a^K_ij | EN-DE BLEU |
|---|---|---|
| Yes | Yes | 25.8 |
| No | Yes | **25.8** |
| Yes | No | 25.3 |
| No | No | 12.5 |

Dropping the value term `a^V_ij` costs **exactly zero** BLEU. Dropping the score term `a^K_ij` costs
0.5. Dropping both lands on 12.5 — identical to k = 0, i.e. the position-free floor.

> "Including relative position representations solely when determining compatibility between elements
> may be sufficient, but further work is needed to determine whether this is true for other tasks."

**Complexity and cost.**
- Storage without head sharing: `O(h n² d_a)`; with head sharing: `O(n² d_a)`.
- Total self-attention space: `O(bhnd_z)` → `O(bhnd_z + n² d_a)`; relative increase scales as `n/(bh)`.
- Learned parameters: `2k+1` vectors of width `d_a` for `w^K`, same again for `w^V`, with `d_a = d_z`.
- Throughput: **7% decrease in steps per second**. Same model and batch sizes on P100s.

**Setup numbers.** tensor2tensor; WMT14 EN-DE ≈ 4.5M sentence pairs, EN-FR ≈ 36M; 32,768 word-piece
vocab; 4096 tokens/GPU/batch (≈25,000 source + 25,000 target per batch); Adam β1 = 0.9, β2 = 0.98,
ε = 1e−9, 4,000 warmup steps; label smoothing 0.1; beam 4, length penalty α = 0.6.
Base: 6+6 layers, d_x = 512, d_z = 64, 8 heads, FFN 1024, dropout 0.1, **k = 16**, unique per layer
and head, 100,000 steps on 8 K40s, no checkpoint averaging.
Big: 6+6 layers, d_x = 1024, d_z = 64, 16 heads, FFN 4096, dropout 0.3 (EN-DE) / 0.1 (EN-FR),
**k = 8**, unique per layer, 300,000 steps on 8 P100s, last 20 checkpoints averaged.

## What the live view must let the reader do

App context: 32 dims, 4 heads (d_k = 8), 2 blocks, seeded untrained weights, causal mask, ~16
user-editable tokens. Mechanisms attach at the mixer seam: `readable(i,j)`, additive score
`bias(i,j)`, `rotate(vector,pos)`, key/value head sharing.

**Seam note, before anything else.** Shaw's score term is `q_i · w_{clip(j-i,k)} / sqrt(d_k)`. It
needs the querying token's **query vector**, not just the pair of indices. If `bias(i,j)` is handed
only `(i, j)`, it cannot express Shaw — it can only express ALiBi. So the one seam change this card
requires is that `bias` receives `q_i` (and `d_k`) alongside the indices. That is the honest minimum;
everything below rides on the existing bias hook once that argument exists. The value term `a^V_ij`
needs a *second* seam inside the weighted sum, which the mixer does not have — see interaction 5 for
why not building it is the right call.

**1. Drag k and watch pairs become indistinguishable.**
Slider `k` from 0 to 15. Display the n × n grid of edge labels `clip(j-i, k)` beside the live bias
matrix, one colour per label. As k drops, the lower-left corner of the causal triangle floods to a
single colour.
*The number that proves it:* pick two pairs at different true offsets that share a clipped label —
say (i=15, j=2) and (i=15, j=5) at k=3, both labelled −3 — and print `bias(15,2) − bias(15,5)`. It
reads **exactly 0.000000**, and it goes nonzero the instant k crosses 13. Also print the live count
of distinct labels actually in use. Under a causal mask only `j ≤ i`, so offsets are in `[-k, 0]` and
just **k+1** of the 2k+1 learned vectors ever fire — half the parameter budget is dead weight in a
decoder-only model. Show that count next to the nominal 2k+1.
*Anchor:* k = 0 must be reachable, and the resulting bias matrix must be visibly constant — that is
the paper's 12.5-BLEU floor, reproduced as a picture.

**2. Shift the whole sentence and show the pattern rides along.**
A "prepend token" control that inserts a word at position 0 and re-runs. Show the bias matrix before
and after, with the shared sub-block outlined.
*The number that proves it:* max absolute difference over the shared sub-block, displayed live. Under
relative positions it is **0.000000** (the matrix is Toeplitz — constant along every diagonal, up to
clipping). Run the same shift with sinusoidal absolute encodings selected instead and display the same
statistic: it is nonzero, and the per-word attention row for an unmoved word visibly rewrites. That
single pair of numbers is the entire backward link to Concept 2, and it is a real computed quantity,
not an assertion. Untrained random weights are fine here — arbitrary-but-identical is exactly the
point.

**3. Toggle query-dependence: Shaw vs. a scalar per offset.**
Two modes on the same slider: `q_i · w_{offset} / sqrt(d_k)` (Shaw) and `s_{offset}` (one scalar per
offset). Fix an offset — say −3 — and plot the bias value down the whole diagonal, one point per row i.
*The number that proves it:* the variance (or max−min) of that diagonal. Shaw: **nonzero** — every
token gets a different amount of positional pull at the same distance, because it depends on what the
token is asking for. Scalar mode: **exactly 0** — the diagonal is flat. Show a token whose ranking of
its neighbours changes between the two modes. This is the precise thing ALiBi later gives up, and the
reader should have seen the number before meeting that card.

**4. Price the mechanism.**
A cost panel that updates from the live config: `n`, `d_a`, heads, blocks, and two checkboxes,
"share across heads" and "share across layers".
*The numbers that prove it:* learned parameter count `2·(2k+1)·d_a` per unshared unit — with the
current defaults, unchecking head-sharing multiplies it by 4, unchecking layer-sharing by 2. Beside
it, the extra score-term memory `n² · d_a` floats, and a matmul counter: baseline `QK^T` is 1 batched
matmul reusing one K per column; with the relative term it is 2 matmuls plus reshapes, because — quote
it on the panel — *"the representations differ with different pairs of positions. This prevents us
from computing all e_ij for all pairs of positions in a single matrix multiplication."* Pin the
paper's measured 7% throughput loss next to it as the real-world calibration. Default both boxes to
**unchecked**, which is what the paper's base model actually trained.

**5. The value term, shown but not built (optional).**
`a^V_ij` needs a hook inside the weighted sum that the mixer does not have. The paper's own ablation
says it is worth 0.0 BLEU (25.8 → 25.8). So do not add the seam. Instead show a static two-row table
of Table 3's first two rows with the delta printed as `0.0`, and one sentence: the attention weights
`α_ij` are **identical** either way — the value term changes what gets summed, never who gets
attended to. If the seam ever exists for another reason, the live proof is a diff of `z_i` (nonzero)
against a diff of the `α` row (exactly zero).

## What the source does *not* establish

- **No length-extrapolation experiment exists in this paper.** Clipping "enables the model to
  generalize to sequence lengths not seen during training" and invariance to total sequence length are
  both stated as properties, and the sinusoid rationale in §2.1 is explicitly flagged as something the
  *original* authors "hypothesized". Not one number in Tables 1–3 tests generalisation to unseen
  lengths. Every result is in-distribution WMT translation. The length claim is a motivation, not a
  finding.
- **"Precise relative position information is not useful beyond a certain distance" is a hypothesis,
  and Table 2 does not cleanly confirm it.** It shows BLEU is *insensitive* to k above 2 — which is
  equally consistent with the model routing long-range position information through stacked layers.
  The authors say so themselves: "as we use multiple encoder layers, precise relative position
  information may be able to propagate beyond the clipping distance."
- **The relative+absolute combination has no table row.** "We did not observe any benefit" is prose;
  the combined score is not printed. No ablation, no dev-set number.
- **Sharing across heads and layers is never ablated.** §3.3 gives the complexity saving, §4.1 says
  the trained models used unique representations per layer (and per head, in base). Nobody measured
  what sharing costs in quality.
- **Tables 2 and 3 have no absolute-position baseline.** They are all on newstest2013 dev, and the
  absolute-position model was never run there — so "25.8" cannot be compared to Table 1's 26.5.
  The only in-table reference point is the 12.5 floor.
- **Single runs, no variance.** No seeds, no confidence intervals, no significance test. The base
  EN-DE gain is 0.3 BLEU, which is inside the range where seed noise usually lives.
- **Two language pairs, one task, one metric.** Encoder-decoder MT, BLEU only. No language modelling,
  no classification, no decoder-only setting. The `a^V` result is explicitly scoped: "further work is
  needed to determine whether this is true for other tasks."
- **Nothing about inference.** No KV-cache interaction, no per-token decoding cost, no long-context
  memory analysis. The 7% figure is *training* steps per second.
- **No analysis of what the `w` vectors learn.** They are never visualised or probed.
- **The read is v2.** ar5iv renders the latest version; whether any equation or number differs from
  v1 (6 Mar 2018) was not verified.
- **RoPE and ALiBi are not in this paper.** Everything in *Leaves behind* below is framing for the
  timeline, drawn from the cost this paper measures — not claims the source makes.

## Leaves behind

What this card hands to the next one, stated as the problem the successors inherit:

**The win.** Position stops being a vector you add once at the input and hope survives, and becomes a
term inside every score in every layer. Offset is structural, not learned-by-luck. The bias matrix is
Toeplitz, so a pattern learned at one offset is the *same parameter* everywhere in the sentence — and
it holds for sentence lengths the model never saw, because nothing in `clip(j-i, k)` mentions `n`.
Replacing absolute encodings entirely was better than keeping both.

**The bill, exactly.** `a^K_ij` sits inside the dot product against a per-pair operand. Plain
attention is fast because token j has *one* key vector, reused against all n queries — one batched
`(n×d_z)(d_z×n)` matmul per head per sequence. The relative term breaks that reuse: what query i is
dotted against depends on the pair, so there is no single K matrix to multiply by. Shaw's rescue is
algebraic — distribute the product (eq. 5) and compute the position half as n parallel
`(bh×d_z)(d_z×n)` matmuls, re-laid-out so the parallel axis is *sequence position* rather than
batch×head. It works, at **7% throughput** plus `O(n²d_a)` memory plus `2(2k+1)d_a` learned parameters
per unshared unit. Two structural debts remain: attention is no longer one matmul, and the cost of the
extra one grows with `n²`.

**What the successors do with that bill.**
- **RoPE** removes the extra term entirely. Instead of adding a per-pair vector to the key, it
  *rotates* q and k by an angle proportional to their own absolute positions; the dot product of two
  rotated vectors depends only on the difference of angles, so relative dependence falls out of the
  standard `QK^T` — one matmul, zero extra parameters, zero extra memory, no clipping and therefore no
  distance ceiling. In this app that is the `rotate(vector,pos)` seam, not the `bias` seam. Concept 3
  earns the reader the question RoPE answers: *can we get the Toeplitz property without paying for a
  per-pair term?*
- **ALiBi** goes the other way and makes the bias cheaper than cheap: a fixed, unlearned scalar
  `−m_h · |i−j|`, one slope per head, no parameters and no clip. That is precisely interaction 3's
  "scalar mode" — it discards the query-dependence Shaw paid for. The reader should already have seen,
  as a number, what that costs on the diagonal.

**The open thread this card should not close.** Table 2 says k ≥ 2 is enough and Table 3 says the
value term is worth nothing — both on one task, one metric, single runs. Two of Shaw's four moving
parts turn out not to be load-bearing for translation. The next card inherits a mechanism whose
*shape* mattered far more than its *capacity*.
