# Concept 6 — Sparse Transformer, strided and fixed patterns

**Card id:** `sparse-transformer` · **Date:** 2019-04-23 (arXiv v1) · **Pressure:** how many comparisons

## What was read

- [arXiv:1904.10509](https://arxiv.org/abs/1904.10509), Child, Gray, Radford, Sutskever (OpenAI) —
  *Generating Long Sequences with Sparse Transformers*. Abstract page for the version history, and
  the full text (ar5iv, cross-checked against the LaTeX source from `arxiv.org/e-print/1904.10509`,
  `sparse_transformers.tex`) for §4 (Factorized Self-Attention), §5 (Sparse Transformer), §7
  (Experiments), and Tables 1–4.
- Version history: **v1, Tue 23 Apr 2019** — and that is the *only* version. Unlike the Transformer
  paper there is no revision trail to disambiguate; the timeline uses v1.
- Every quotation below is from the LaTeX source, so it is the authors' wording, not a summariser's.

## The mechanism, precisely

### The empirical observation that motivated it

This is the load-bearing part, and it is more ambivalent than the folklore version. The authors did
not start from "attention is probably sparse". They **trained a dense model and looked at it**:

> "To motivate our approach, we first perform a qualitative assessment of attention patterns learned
> by a standard Transformer on an image dataset."

The model was a **128-layer self-attention network on CIFAR-10 trained with full attention**. The
figure caption catalogues four things they saw:

> "a) Many early layers in the network learn locally connected patterns, which resemble convolution.
> b) In layers 19 and 20, the network learned to split the attention across a row attention and
> column attention, effectively factorizing the global attention calculation. c) Several attention
> layers showed global, data-dependent access patterns. d) Typical layers in layers 64-128 exhibited
> high sparsity, with positions activating rarely and only for specific input patterns."

Item (b) is the direct origin of the two-head factorization: **the dense model discovered a row/column
factorization on its own**, and the paper's move is to hard-code it. Item (d) is the licence to be
sparse at all:

> "Visual inspection showed that most layers had sparse attention patterns across most data points,
> suggesting that some form of sparsity could be introduced without significantly affecting
> performance."

And then, in the same paragraph, the honest caveat — which the card must carry, because it is the
seed of everything that comes after this concept on the timeline:

> "Several layers (Figure 2c) clearly exhibited global patterns, however, and others exhibited
> data-dependent sparsity (Figure 2d), both of which would be impacted by introducing a predetermined
> sparsity pattern into all of the attention matrices."

They know a fixed pattern cannot reproduce what they measured:

> "We aimed to empirically validate the performance of these factorized patterns on a range of tasks,
> given that they are unable to learn the exact same mappings as those in Figure 2."

So the honest reading is: *the sparsity was observed, the fixity was assumed.* The observation
supports "attention could be sparse"; it does **not** support "attention could be sparse in a pattern
chosen ahead of time by the architect". That gap is the whole forward edge of this card.

### Factorized self-attention, formally

A self-attention layer is parameterized by a connectivity pattern `S = {S₁, …, Sₙ}`, "where `Sᵢ`
denotes the set of indices of the input vectors to which the `i`th output vector attends":

    Attend(X, S) = ( a(xᵢ, Sᵢ) )_{i ∈ 1..n}
    a(xᵢ, Sᵢ)   = softmax( (W_q xᵢ) K_{Sᵢ}ᵀ / √d ) V_{Sᵢ}
    K_{Sᵢ} = (W_k x_j)_{j ∈ Sᵢ}      V_{Sᵢ} = (W_v x_j)_{j ∈ Sᵢ}

Note what this framing does: the softmax, scale and weighted sum are *untouched*. The only thing that
changes is the index set. **This is exactly a `readable(i, j)` predicate**, which is why the mechanism
drops into the app's mixer seam without touching the attention math.

- Full causal self-attention is the special case `Sᵢ = {j : j ≤ i}` — "allowing every element to
  attend to all previous positions and its own position."
- "Factorized self-attention instead has `p` separate attention heads, where the `m`th head defines a
  subset of the indices `A_i^(m) ⊂ {j : j ≤ i}` and lets `Sᵢ = A_i^(m)`."
- Efficiency criterion: "We are chiefly interested in *efficient* choices for the subset `A`, where
  `|A_i^(m)| ∝ ᵖ√n`" (the `p`-th root of `n`) — each head reads on the order of `ᵖ√n` keys.
- Validity criterion: "for the time being we consider *valid* choices of `A`, where all input
  positions are connected to all future output positions across the `p` steps of attention."

### The reachability claim, quoted exactly

> "For every `j ≤ i` pair, we set every `A` such that `i` can attend to `j` through a path of
> locations with maximum length `p+1`. Specifically, if `(j, a, b, c, ..., i)` is the path of indices,
> then `j ∈ A_a^(1)`, `a ∈ A_b^(2)`, `b ∈ A_c^(3)`, and so forth."

For the `p = 2` case used throughout the paper, the path has **maximum length 3** (`p+1 = 3`), which
means **at most two hops of attention**: `i → a → j`. Read carefully, "maximum length `p+1`" counts
*locations on the path*, not edges. The consequence they claim:

> "These two criteria allow us keep the ability of Transformers to propagate signals from arbitrary
> input positions to arbitrary output positions in a constant number of steps, while reducing the
> total effective computation to `O(n ᵖ√n)`."

The abstract states the `p = 2` instance: **`O(n √n)`**.

### STRIDED

> "A natural approach to defining a factorized attention pattern in two dimensions is to have one head
> attend to the previous `l` locations, and the other head attend to every `l`th location, where `l`
> is the *stride* and chosen to be close to `√n`, a method we call *strided* attention."

    A_i^(1) = { t, t+1, ..., i }   for t = max(0, i − l)     ← local window, the previous l positions
    A_i^(2) = { j : (i − j) mod l = 0 }                      ← every l-th position back from i

**That is where the √ comes from, and it is worth being blunt about it on the card**: head 1 reads
`≈ l` keys, head 2 reads `≈ n/l` keys. The sum `l + n/l` is minimised at `l = √n`, giving `≈ 2√n` keys
per query and `O(n√n)` for the layer. The square root is *not* a property of language or images — it
is the minimum of `l + n/l`. The paper's own generalisation makes this explicit: with `p` heads the
right stride is the `p`-th root and the cost is `O(n·ᵖ√n)`.

Which data it suits:

> "This formulation is convenient if the data naturally has a structure that aligns with the stride,
> like images or some types of music."

Why the alignment matters: for a 32×32 image flattened row-major, a stride of 32 *is* the column
neighbour. Head 1 becomes row attention, head 2 becomes column attention — precisely pattern (b) that
the dense CIFAR-10 model had learned by itself.

### FIXED

> "For data without a periodic structure, like text, however, we find that the network can fail to
> properly route information with the strided pattern, as spatial coordinates for an element do not
> necessarily correlate with the positions where the element may be most relevant in the future."
>
> "In those cases, we instead use a *fixed* attention pattern, where specific cells summarize previous
> locations and propagate that information to all future cells."

    A_i^(1) = { j : ⌊j/l⌋ = ⌊i/l⌋ }                          ← the current block of l positions
    A_i^(2) = { j : j mod l ∈ { t, t+1, ..., l } },  t = l − c   ← the last c "summary" columns of every block

`c` is a hyperparameter. Their worked example, verbatim:

> "Concretely, if the stride is 128 and `c = 8`, then all future positions greater than 128 can attend
> to positions 120-128, all positions greater than 256 can attend to 248-256, and so forth."

And the cost of `c`:

> "A fixed-attention pattern with `c = 1` limits the expressivity of the network significantly, as
> many representations in the network are only used for one block whereas a small number of locations
> are used by all blocks. We instead found choosing `c ∈ {8, 16, 32}` for typical values of
> `l ∈ {128, 256}` to perform well, although it should be noted that this increases the computational
> cost of this method by `c` in comparison to the strided attention."

Plus a multi-head detail worth implementing if the app ever splits heads across the pattern: "having
them attend to distinct subblocks of length `c` within the block of size `l` was preferable to having
them attend to the same subblock."

> **Implementation snag, flagged honestly.** As written, `A_i^(2)` uses the residue set
> `{t, …, l}` — but `j mod l` can only take values `0 … l−1`, so the endpoint `l` is unreachable. The
> set is effectively `{l−c, …, l−1}`, which is exactly `c` residues and matches their "120-128 for
> `l = 128, c = 8`" example only if that range is read as the 8 positions `120…127`. The app should
> implement residues `l−c … l−1` and say so; do not silently paper over it.

### How the two heads are combined — three approaches, all in §5.1

Dense attention is `attention(X) = W_p · attend(X, S)`. The three integrations:

1. **Interleave across layers** — "The simplest technique for integrating factorized self-attention is
   to use one attention type per residual block, and interleave them sequentially or at a ratio
   determined as a hyperparameter": `attention(X) = W_p · attend(X, A^(r mod p))`, where "`r` is the
   index of the current residual block". Layer 0 gets pattern 1, layer 1 gets pattern 2, and so on.
   This is the version where the two-hop argument is literally two layers.
2. **Merged head** — "a single head attend to the locations of the pixels that both factorized heads
   would attend to": `attention(X) = W_p · attend(X, ⋃_{m=1..p} A^(m))`. "This is slightly more
   computationally intensive, but only by a constant factor." Both patterns in one layer, union of
   index sets. **This is the one the Enwik8 result used.**
3. **Multi-head** — `attention(X) = W_p ( attend(X, A)^(i) )_{i ∈ 1..n_h}`, the standard concat, where
   "the `A` can be the separate attention patterns, the merged patterns, or interleaved". Weight
   matrices shrink by `1/n_h` "such that the number of parameters are invariant across values of `n_h`".

Their practical note: "We typically find multiple heads to work well, though for extremely long
sequences where the attention dominates the computation time, it is more worthwhile to perform them
one at a time and sequentially."

### The other three contributions — separate from the sparsity

The abstract names four things, and only the first is sparsity. **The card must not let "Sparse
Transformer" mean "sparsity".** From the abstract:

> "In this paper we introduce sparse factorizations of the attention matrix which reduce this to
> `O(n √n)`. We also introduce a) a variation on architecture and initialization to train deeper
> networks, b) the recomputation of attention matrices to save memory, and c) fast attention kernels
> for training. **We call networks with these changes Sparse Transformers**."

1. **Restructured residual block + initialization (§5.2).** "We found that Transformers were difficult
   to train with many layers". They use the pre-activation residual block of He et al., with
   `resblock(H) = a(H) + b(H)`, `a(H) = dropout(attention(norm(H)))`,
   `b(H) = dropout(ff(norm(H + a(H))))`, GELU as `f`. The initialization rule: "We scale the
   initialization of `W₂` and `W_p` … by `1/√(2N)` to keep the ratio of input embedding scale to
   residual block scale invariant across values of `N`." This is what buys **128 layers**, and it has
   nothing to do with sparsity.
2. **Recomputation / gradient checkpointing (§5.4).** "In our experiments, we recompute the attention
   and feed-forward blocks during the backwards pass." And the killer line for the timeline:
   > "Using recomputation alone, we are able to train dense attention networks with hundreds of layers
   > on sequence lengths of 16,384, which would be infeasible on modern hardware otherwise."

   **Recomputation alone gets dense attention to 16,384 tokens.** That is a memory win, not a compute
   win, and it is orthogonal to the sparsity. It is also the direct ancestor of what FlashAttention
   later does properly.
3. **Block-sparse kernels + mixed precision (§5.5, §5.6).** "Attention over a local window can be
   computed as-is, whereas attention with a stride of `k` can be computed by transposing the matrix
   and computing a local window." "The softmax operation is fused into a single kernel and also uses
   registers to eliminate loading the input data more than once, allowing it to run at the same speed
   as a simple nonlinearity. The upper triangle of the attention matrix is never computed, moreover,
   removing the need for the negative bias term … and halving the number of operations." Weights in
   fp32, activations and gradients in fp16 with dynamic loss scaling, on V100 Tensor Cores; queries
   and keys cast back to fp32 when sampling "as the query-key product can sometimes overflow the max
   value of half-precision."

   The sparsity is only cheap **because someone wrote the kernel**. A `readable(i,j)` mask applied to a
   dense score matrix saves exactly zero FLOPs — which is what the app does, and the card must say so.

## Numbers that matter

**Datasets, sequence lengths, results (Table 1 / §7).** All modelled as sequences of raw bytes, same
architecture across all three modalities.

| dataset | sequence length | pattern | params | result | previous SOTA |
|---|---|---|---|---|---|
| CIFAR-10 | 3,072 bytes | strided, 128 layers, `d`=256, 2 heads | 59M | **2.80** bits/dim | 2.85 (PixelSNAIL) |
| Enwik8 | 12,288 tokens | fixed, 30 layers, `d`=512, 8 heads, `l`=128, `c`=32, merged | 95M | **0.99** bits/byte | 1.03 (Transformer-XL 88M) |
| ImageNet 64×64 | 12,288 bytes | strided, 48 layers, `d`=512, 16 heads, `l`=128 | 152M | **3.44** bits/dim | 3.52 (SPN 150M) |
| Classical music, 12 kHz | 65,536 samples (≈5 s) | strided | 152M | **1.97** bits/byte | — (no comparison offered) |

**Ablation — Table 2, the table that complicates the story.**

| | bits/byte | time/iter |
|---|---|---|
| **Enwik8, 12,288 context** | | |
| Dense attention | 1.00 | 1.31 |
| Sparse (fixed) | **0.99** | 0.55 |
| Sparse (strided) | 1.13 | 0.35 |
| **CIFAR-10, 3,072 context** | | |
| Dense attention | 2.82 | 0.54 |
| Sparse (fixed) | 2.85 | 0.47 |
| Sparse (strided) | **2.80** | 0.38 |

Read it in both directions. Strided is the fastest row in both blocks — and on text it is **13% worse
than dense** (1.13 vs 1.00), a catastrophic loss for a bits/byte metric. Fixed is worse than dense on
images. **The pattern must match the data or the sparsity costs you accuracy**, and the paper says so
plainly: "Strided attention failed to do well on this dataset, whereas fixed patterns were able to
recover and surpass the performance of dense attention."

Their own reading of the wins is a hedge, not a claim: "sparse patterns also converged to lower error…
This **may** point to a useful inductive bias from the sparsity patterns we introduced, **or an
underlying optimization issue with full attention**."

**Scaling to a million.** "we found that increasing the sequence length by a factor of 4 requires a
reduction in model capacity of approximately `4√4 = 8`. Thus we found we could use factorized
self-attention on sequences over 1 million timesteps long, albeit with extremely few parameters
(3 million)." Table 4: 65,536 → 152M params → 1.97 bpb; 262,144 → 25M → 2.17; 1,048,576 → 3M → 2.99.
"Sample quality quickly degrades for greater sequence lengths due to reduced model capacity."

**Positional embeddings the patterns require (§5.3).** Not optional garnish: "For images, we used data
embeddings, where `d_data = 3` for the row, column, and channel location of each input byte. For text
and audio, we used two-dimensional attention embeddings, where `d_attn = 2` and the index corresponds
to each position's row and column index **in a matrix of width equal to the stride**." The model is
told, positionally, where the stride boundaries are.

**Translated to this app: `n = 16`, causal.** Computed directly from the paper's index sets, not
estimated.

| pattern | keys read (union of both heads) | vs full causal (136) |
|---|---|---|
| full causal | 136 of 256 cells | — |
| strided, `l = 4` (= √16) | 82 | 60% |
| strided, `l = 3` | 80 | 59% |
| strided, `l = 8` | 108 | 79% |
| fixed, `l = 4`, `c = 1` | 64 | 47% |
| fixed, `l = 4`, `c = 2` | 88 | 65% |
| fixed, `l = 4`, `c = 3` | 112 | 82% |

**And the honest caveat the card must print next to that table:** at `n = 16` the saving is ~40%,
which is unimpressive, because `√n` scaling only pays at length. At the paper's Enwik8 setting
(`n = 12,288`, `l = 128`) the same strided rule reads **2,148,416 of 75,503,616 causal cells — 2.85%**.
The app should show both numbers side by side: the reader's 40% and the paper's 97% saving. That
contrast *is* the concept.

Sample index sets for query `i = 15`, `l = 4`:

| | head 1 | head 2 |
|---|---|---|
| strided | `{11,12,13,14,15}` (local) | `{3,7,11,15}` (every 4th) |
| fixed, `c = 1` | `{12,13,14,15}` (block) | `{3,7,11,15}` (summaries) |

## What the live view must let the reader do

The seam is `readable(i, j)`. Both patterns are one expression each; nothing about softmax, scale,
mask or the value sum changes. The card should make that visible — the sparse view is the *same*
heatmap with cells greyed out.

1. **Switch between `full` / `strided` / `fixed`, and see the heatmap change and nothing else.**
   The Q·K, scale and softmax panels stay the mechanism from concept 1. Display the predicate itself
   as text under the switch, in the paper's own notation, so the reader can check the picture against
   the formula:
   - strided: `j ≥ max(0, i−l)  OR  (i−j) mod l = 0`
   - fixed: `⌊j/l⌋ = ⌊i/l⌋  OR  j mod l ≥ l−c`
   Toggle head 1 / head 2 / merged (§5.1's third and second options) so the reader sees that the two
   heads are *different pictures whose union is connected*, not two copies.

2. **A stride slider, `l = 2 … 16`, with `√n` marked on the track.** Displayed number: **keys read,
   summed over all queries** — e.g. `82 / 136 keys (60%)` at `l = 4`, alongside `l + n/l = 8.0 keys
   per query, minimised at l = √16 = 4`. The slider must make the U-shape visible: at `l = 2` the
   count is 87, at `l = 4` it is 82, at `l = 8` it is 108. **The minimum sitting at `l = 4 = √16` is
   the number that proves where the square root comes from** — it is not a fact about text, it is the
   minimum of `l + n/l`, and the reader should watch it bottom out.

3. **Reachability of a word the reader actually chose.** Two dropdowns over their own sentence:
   *"query = [word i], needs to read = [word j]"*. Then a verdict in three states:
   - **read directly** — the cell is lit, show which head supplies it;
   - **reachable in 2 hops via [word a]** — show the path highlighted on the heatmap as two arrows,
     `i → a → j`, and name `a` in their sentence;
   - **unreachable** — which, for a valid pattern, must never appear; if it does, the pattern or the
     causal mask is wrong, and the app should say so rather than hide it.
   Worked example on a 16-token sentence, `l = 4`, strided: query 15 cannot read key 1 directly
   (`A₁₅^(1) = {11..15}`, `A₁₅^(2) = {3,7,11,15}`, and 1 is in neither), but `3 ∈ A₁₅^(2)` and
   `1 ∈ A₃^(1) = {0,1,2,3}`, so **15 → 3 → 1**. The app should verify this exhaustively and print the
   result: at `l = 4`, **54 of 136 causal pairs are not directly readable, and 0 of those 54 survive
   two hops** — every one is reachable. Under fixed, `l=4, c=1`: 72 pairs missed directly, 0 fail at
   two hops, and `15 → 3 → 1` again. That "0" is the paper's `p+1` claim, checked on the reader's own
   sentence rather than asserted.

4. **The cost counter, at their length and at the paper's.** Two rows, always both visible:
   `your sentence, n = 16: 136 → 82 keys (60%)` and `Enwik8, n = 12,288, l = 128: 75,503,616 →
   2,148,416 (2.85%)`. Under it, one line: *the saving is small here because √n scaling only pays at
   length*. Without the second row the reader concludes sparse attention is barely worth it, which is
   the wrong lesson from a correct number.

5. **The `c` slider for fixed, and the trap it exposes.** `c = 1 … 4`, showing keys read (64 / 88 /
   112 …) rising toward dense. Pair it with the paper's own warning — `c = 1` "limits the expressivity
   of the network significantly", they used `c ∈ {8,16,32}`, and the cost rises "by `c` in comparison
   to the strided attention". The displayed number that proves it: at `c = 3`, fixed reads 112 of 136
   keys — 82% — so the reader can see that you can tune a "sparse" pattern until it is not sparse.

**Optional but cheap, and it lands the forward argument:** a "which pattern would the data have
picked?" overlay — draw the top-`k` cells of the *dense* attention row for query `i`, then overlay the
strided/fixed mask, and count how many of the dense model's largest weights the fixed pattern actually
covers. The pattern is chosen without ever looking at the scores; this shows the reader, in their own
sentence, that it sometimes misses the cell the model wanted. That is the entire motivation for the
adaptive methods that come later.

## What the source does *not* establish

- **It does not claim sparsity alone produced the results.** The abstract lists four contributions and
  defines "Sparse Transformer" as the network with *all* of them. In particular, "using recomputation
  alone, we are able to train dense attention networks with hundreds of layers on sequence lengths of
  16,384" — a memory result with no sparsity in it. The card must not attribute the depth (128 layers)
  or the long training contexts solely to the attention pattern; the `1/√(2N)` initialization and the
  checkpointing are doing separate, named work.
- **It does not establish that fixed patterns beat dense in general.** They win one row each and lose
  one row each (Table 2), and the authors' own explanation is a disjunction — "a useful inductive bias
  … **or** an underlying optimization issue with full attention." Do not report the 0.99-vs-1.00 as a
  demonstration that sparsity improves modelling.
- **It does not show the patterns match what attention wants to do.** The opposite: they state the
  factorized patterns "are unable to learn the exact same mappings" as the global and data-dependent
  layers they measured, and that predetermined sparsity "would be impacted" by exactly those layers.
- **The empirical motivation was measured on images, not text.** The CIFAR-10 visualisation is the only
  evidence offered for the sparsity assumption, and the row/column factorization it revealed is an
  image-shaped fact. The fixed pattern for text is introduced because strided *failed* on text, not
  because a comparable analysis of a text model suggested it.
- **`O(n√n)` is asymptotic and per-layer**, and it is `O(n·ᵖ√n)` in general — `√n` is the `p = 2`
  case. Nothing about wall-clock follows without the block-sparse kernels; and fixed attention costs a
  further factor of `c`.
- **The million-timestep claim is heavily hedged.** 3M parameters, 2.99 bits/byte, and "sample quality
  quickly degrades". "Possible in principle" is the abstract's own wording.
- **The app proves none of this.** A `readable(i,j)` mask over a dense score matrix saves zero FLOPs —
  it demonstrates *connectivity*, which is the concept, not *speed*, which needs a kernel. And the
  model is 32-dim, 2-block, untrained; the attention patterns it shows are seeded noise shaped by a
  rule, so the reader should read the geometry, never the predictions.

## Leaves behind

**Backward — what this answers.** Concept 1 leaves a T² score matrix and measures it. This is the
first card on the timeline that attacks that number directly, and it attacks it in the cheapest
possible way: *don't compute most of the cells*. The path-length property the Transformer bought
(`O(1)` between any two positions, Table 1 of Vaswani et al.) is the thing at risk, and the validity
criterion is precisely the promise to keep it — degraded from 1 step to `p+1 = 3`, i.e. **two hops**,
but still constant in `n`. The trade is stated exactly: constant-factor path length in exchange for
`n√n` instead of `n²`.

**Forward — the weakness, stated precisely.** The distinction the card must draw, and it is the whole
hinge of the timeline:

> Sparse Transformer is cheaper because **we skip work**. It is not cheaper because **we found the
> right work to skip**.

`readable(i, j)` is a function of the *indices only*. It never looks at `q_i`, at `k_j`, or at the
content of the sentence. The stride is a hyperparameter the architect sets — 128 for Enwik8, 128 for
ImageNet — chosen before a single token is seen. Two consequences follow, and the paper hands us the
evidence for both:

1. **A fixed pattern can be wrong for the data**, and when it is, it is *badly* wrong: strided on
   Enwik8, 1.13 vs 1.00. The fix in 2019 was to have a human notice and pick the other pattern.
2. **A fixed pattern cannot express the layers the dense model actually learned** — the global and
   data-dependent ones the authors themselves photographed and then set aside.

Everything downstream is an attack on that fixity. Content-based routing (learn or hash which keys are
worth reading, per input, rather than legislating it); learned/adaptive sparsity; and, in the other
direction entirely, the argument that you should stop skipping work and instead *do all the work
faster* by fixing the memory traffic — which is the FlashAttention answer, and which quietly finishes
the job §5.4 and §5.5 of this very paper started with recomputation and fused kernels. The pressure
this card measures — *how many comparisons* — is the one all of them are pushing on; they just
disagree about whether the answer is "fewer, chosen in advance", "fewer, chosen by the data", or
"all of them, but cheaper".
