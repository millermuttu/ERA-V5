# Concept 22 — Parallelizing DeltaNet over sequence length
**Card id:** `parallel-deltanet` · **Date:** 2024-06-10 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:2406.06484](https://arxiv.org/abs/2406.06484), Songlin Yang, Bailin Wang, Yu Zhang,
  Yikang Shen, Yoon Kim (MIT, Soochow University, MIT-IBM Watson AI Lab) — *Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length*. **NeurIPS 2024** (v6 comment field reads
  "Final camera ready").
- **Version history read off the abstract page and verified**: v1 **10 Jun 2024**, v2 26 Aug 2024,
  v3 31 Oct 2024, v4 5 Nov 2024, v5 25 Nov 2024, v6 15 Jan 2025. The timeline uses **v1, 2024-06-10**,
  which is the date already in `mechanisms.js` — confirmed rather than assumed.
- **v1 was fetched separately and compared with v6.** The title is identical, and the v1 abstract is
  the same text as v6's down to the clause order; the only visible v1→v6 change in the abstract is
  the removal of "(including on tasks that focus on recall)" from the sentence about outperforming
  Mamba and GLA. That deletion is not cosmetic — §4.2 of the final version says in as many words
  that at 1.3B DeltaNet **underperforms** GLA on recall-intensive tasks. The paper walked back its
  own abstract, and the card should not restore the claim the authors removed. The comments field
  also moved from "Preprint" to "Final camera ready".
- Full text read from the **arXiv HTML render of v6** (`arxiv.org/html/2406.06484v6`), downloaded
  and de-marked-up locally so the equations could be read in source form rather than as glyphs.
  Sections used: §1, §2.1 (linear attention, the chunkwise parallel form and its complexity), §2.2
  (DeltaNet as the paper restates it), **§3.1** (the memory-efficient reparameterization — the
  proof), **§3.2** (the chunkwise parallel form, the UT transform, the fully parallel form), §3.3
  (the DeltaNet transformer, feature map and normalization), §3.4 (hybrids), §4.1 (MQAR, MAD), §4.2
  (language modelling, Table 1, ablations, throughput), §5.1 (Table 2, the unifying recurrence),
  **§5.3 (Limitations)**, and the appendix section list for B.1/B.2.
- Released as part of **FlashLinearAttention**: <https://github.com/fla-org/flash-linear-attention>.
- **In addition**, every quantitative claim this card intends to put on screen was re-derived
  numerically against this app's own model — 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817,
  the default 16-token sentence — by driving `app/model/transformer.js` and `app/model/mixers.js`
  from node. Those numbers are labelled **[measured here]**. Two of them decided the card's shape,
  and one of them is the card.

## Scope, before anything else

**This concept does not change what the model computes.** It is the second card on this timeline
whose subject is an algorithm rather than a mechanism — concept 14 was the first — and the honest
framing is the same one that card used: the output is bit-for-bit the thing the previous concept
already produced, and the entire contribution is *where the cost is paid*.

That has a consequence the card must not dodge. Every panel that shows "the mechanism working" would
show concept 11's picture unchanged, because it *is* concept 11's picture unchanged. So the card's
job is the opposite of the usual one: rather than showing what changed in the output, it has to
**prove that nothing changed in the output** — to a stated number of decimal places — and then show
the thing that did change, which is the number of steps that have to happen one after another.

There is one exception, and the card should give it its own panel rather than bury it: §3.3 changes
the feature map and the normalization (SiLU + L₂ instead of Schlag et al.'s elu+1 + sum), and that
**does** change the model's output. It is a real second contribution, it is ablated in Table 1, and
it is the part of this paper that a reader would otherwise never learn about.

## The mechanism, precisely

### Step 0 — the bill this pays, in the previous card's own words

Concept 11's `tradeBlock` already names the debt, because the research for it found the same gap:

> "Parallel training. Each write must read the state the previous write left, so the sequence can no
> longer be computed as one batched product — the paper never mentions this, and it is why the rule
> waited three years to be scaled."

This paper is the collection of that debt, and it agrees on the diagnosis:

> "However, the original work used a sequential algorithm that did not parallelize across sequence
> length, thus resulting in hardware-inefficient training, and it has not been clear how to scale
> DeltaNet to larger models and datasets."

> "Schlag et al. 2021 demonstrate that DeltaNet outperforms ordinary linear transformers on
> small-scale language modeling and synthetic in-context retrieval tasks. However, their training
> algorithm … is strictly sequential and thus hardware inefficient."

Note the precise nature of the complaint, because the card's cost panel depends on getting it right.
The sequential algorithm is **not** expensive in arithmetic. §2.2: *"The complexity of this recurrent
form is the same as that of vanilla linear attention, i.e. `O(Ld²)`"* — which is **cheaper** than the
parallel form's `O(L²d + Ld²)`. The problem is that `O(Ld²)` arithmetic arranged as `L` dependent
steps of `O(d²)` each is the worst possible shape for a GPU. §2.1 states both halves:

> "The parallel form takes `O(L²d + Ld²)` and thus requires more FLOPs than the recurrent form,
> which takes `O(Ld²)`. However, the parallel form is often much faster in practice for moderate-
> length sequences as it can be done in `O(1)` steps. This sequence-level parallelism also enables
> high GPU occupancy. The recurrent form requires fewer FLOPs but cannot be parallelized across
> sequence length … and the elementwise operations involved in recurrence moreover cannot make use
> of specialized matmul accelerators (e.g., tensor cores)."

**So the trade this paper makes is: buy fewer sequential steps with more arithmetic.** That is a
sentence the app can put a measured number against in both directions, and it should.

Footnote 1 closes the obvious escape route, and it is the same objection concept 20's card had to
handle:

> "It is possible in theory to use parallel scan to parallelize the recurrent form, which would
> enable the computations to be performed in `O(log L)` steps and `O(Ld²)` FLOPs. However, this
> approach requires materializing the 2D hidden state for each time step, which would incur
> significant memory I/O cost unless the state size is small enough such that materialization can
> happen in faster memory (i.e., as in Mamba)."

A parallel scan over `L` matrix-valued states means writing `L` matrices of size `d × d` to memory.
That is concept 14's argument arriving in a different family: the arithmetic was never the problem,
the traffic was.

### Step 1 — the reparameterization, which is one line of algebra and the whole paper

§3.1 begins by rewriting the delta update — concept 11's Eq. 24 — so the previous state appears
exactly once, as a factor:

    S_t = S_{t-1} − v_old kᵀ + v_new kᵀ
        = S_{t-1} − β_t (S_{t-1} k_t) k_tᵀ + β_t v_t k_tᵀ
        = S_{t-1} ( I − β_t k_t k_tᵀ ) + β_t v_t k_tᵀ

> "which can be seen as applying a generalized Householder transformation (i.e., matmul with an
> identity plus rank-one matrix) to the previous state."

Nothing has been approximated or even chosen; the three expressions are the same expression. But the
last one has a property the first two hide: **the state's dependence on its predecessor is a matrix
multiply by a structured matrix**, and products of such matrices have a compact representation that
numerical linear algebra worked out in 1987 (Bischof and Van Loan's WY representation, the paper's
reference [11]).

Unrolling gives Eq. 4:

    S_t = Σ_{i=1..t} β_i (v_i k_iᵀ) ( Π_{j=i+1..t} ( I − β_j k_j k_jᵀ ) )        (4)

Every earlier write is still there, but each one has been multiplied by every Householder factor
that came after it. That product is what "the past has been edited since you wrote it" looks like
when it is written down.

### Step 2 — the pseudo-value `u`, and the proof that it needs no state

The observation that makes everything else possible, §3.1:

> "We first observe that `S_t` admits a purely additive representation of the form
> `S_t = Σ_{i=1..t} u_i k_iᵀ` … since we can simply set `u_i = v_i^new − v_i^old = β_i(v_i − v_i^old)`."

> "Recall … that simple linear attention has the form `S_t = Σ_{i=1..t} v_i k_iᵀ`. Thus, **DeltaNet
> simply replaces the value vector `v_i` in linear attention with the "pseudo" value vector `u_i`.**
> Once the `u_i`'s have been constructed, the rest of computation can proceed as in ordinary linear
> attention, i.e. `O = (QKᵀ ⊙ M) U`."

This is the sentence to build the card's explanation around. **DeltaNet is linear attention with a
different value matrix.** Not a different rule, not a different read — a different `V`. Everything
that made concept 9's form parallel still applies, provided you can get `U`.

The obstruction, stated and then removed:

> "However, computing `u_t` naïvely requires explicitly materializing `S_{t-1}` to compute
> `v_t^old`, which would require `O(d²)` memory. We now show that we can obtain the `u_t`'s without
> explicitly materializing `S_{t-1}` in `O(d)` memory."

The induction (Eq. 3), with the base case `S_1 = β_1 v_1 k_1ᵀ`, so `u_1 = β_1 v_1`:

    S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
        = Σ_{i=1..t-1} u_i k_iᵀ  +  β_t ( v_t − Σ_{i=1..t-1} u_i (k_iᵀ k_t) ) k_tᵀ
                                    └──────────── defined as u_t ────────────┘
        = Σ_{i=1..t} u_i k_iᵀ                                                     (3)

Read what `u_t` is made of: **the incoming value, minus a weighted sum of the earlier pseudo-values,
weighted by how much each earlier key overlaps this one.** No state matrix appears. The "read before
you write" that concept 11 performs against a `d × d` grid has become a read against the list of
previous `u`'s and `k`'s — vectors, not matrices.

And the honest accounting of what that costs, which the paper supplies immediately:

> "While we have avoided materializing the `S_t`'s, computing `u_t`'s for all `L` (that is, `U`)
> takes `O(L²d)` and moreover **cannot be fully parallelized**, unlike in linear attention where we
> can calculate all the value vectors `V` in parallel in `O(1)` steps."

So Eq. 3 alone does not solve the problem. It converts a sequential dependence between *states* into
a sequential dependence between *vectors* — which is a much better object to chunk.

### Step 3 — chunking, and the two matrices that live in `O(d)` per token

§3.2 defines, for `L/C` chunks of size `C`:

    P_i^j = Π_{t=i..j} ( I − β_t k_t k_tᵀ )   ∈ ℝ^{d×d}      (the decay factor from S_i to S_j)
    H_i^j = Σ_{t=i..j} β_t (v_t k_tᵀ) P_{t+1}^j ∈ ℝ^{d×d}    (contributions to S_j starting at i)

with `P_i^j = I` whenever `i > j`, so that `S_t = H_1^t`, and the chunk recurrence is exactly

    S_[t]^r = S_[t]^0 P_[t]^r + H_[t]^r                                          (5)

Then the same WY trick is applied *within* the chunk, so that neither `d × d` matrix is ever built:

    P_[t]^r = I − Σ_{i=1..r} w_[t]^i k_[t]^iᵀ ,   H_[t]^r = Σ_{i=1..r} u_[t]^i k_[t]^iᵀ    (6)

    w_[t]^r = β_[t]^r ( k_[t]^r − Σ_{i<r} w_[t]^i (k_[t]^iᵀ k_[t]^r) )
    u_[t]^r = β_[t]^r ( v_[t]^r − Σ_{i<r} u_[t]^i (k_[t]^iᵀ k_[t]^r) )                     (7)

Note the symmetry, which is worth pointing out on the card because it makes `W` intelligible: **`w`
is `u` with the key in place of the value.** `U` carries what the chunk writes; `W` carries what the
chunk erases from whatever state it inherits. Both are `C × d` — vectors per token, never matrices.

Substituting Eq. 6 into Eq. 5 and stacking gives the two lines an implementation actually runs:

    S_[t+1] = S_[t] + ( U_[t] − W_[t] S_[t]ᵀ )ᵀ K_[t]                            (8)
    O_[t]   = Q_[t] S_[t]ᵀ + ( Q_[t] K_[t]ᵀ ⊙ M ) ( U_[t] − W_[t] S_[t]ᵀ )       (9)

`U_[t] − W_[t] S_[t]ᵀ` appears in both and is computed once. It is the chunk's pseudo-values
*corrected for what the inherited state already holds* — the delta rule applied to a whole chunk at
a time instead of a token at a time. Everything in Eq. 8 and Eq. 9 is a matrix product.

Compare Eq. 9 with plain linear attention's chunk form, Eq. 2:

    O_[t] = Q_[t] S_[t]ᵀ + ( Q_[t] K_[t]ᵀ ⊙ M ) V_[t]                            (2)

They are the same line with `V` replaced by `U − W Sᵀ`. §3.1's sentence — *DeltaNet is linear
attention with a different value matrix* — survives all the way to the kernel.

**Two transcription notes, both confirmed by implementing it.** First, the paper's inline derivation
of `o_[t]^r` prints the inner product as `(k_[t]^{iᵀ} q_[t]^i)`; the index must be `q_[t]^r`, since
the sum builds the output for row `r`. The matrix form (Eq. 9) is unambiguous and correct. Second,
the mask `M` in Eq. 9 must **include** the diagonal, because `o_[t]^r` reads `S_[t]^r`, the state
*after* the `r`-th write. Both readings were settled numerically rather than by argument: the
implementation reproduces the sequential recurrence to 1e-15 with `q_r` and an inclusive mask, and
does not with either alternative.

### Step 4 — the UT transform, which is what makes it a matmul

Eq. 7 is still a recurrence — over `C` instead of `L`, but a recurrence.

> "In the above, Eq. 7 is fully recurrent and thus cannot use tensor cores written as is. To solve
> this, we further leverage the UT transform:"

    T_[t] = ( I + tril( diag(β_[t]) K_[t] K_[t]ᵀ , −1 ) )⁻¹ diag(β_[t])          (10)
    W_[t] = T_[t] K_[t] ,      U_[t] = T_[t] V_[t]                               (11)

> "to rewrite most operations in matmuls. The inverse of lower triangular matrices could be solved
> efficiently using forward substitution."

This is the step most summaries skip, and it is the one that decides whether any of it is fast. Read
what `T` is: a `C × C` matrix built from the chunk's **key–key** inner products — not query–key. A
strictly-lower-triangular matrix of "how much does key `i` overlap key `j`, scaled by how hard `i`
writes", plus the identity, inverted. The inverse of a unit lower-triangular matrix is the closed
form of "undo the chain of corrections all at once".

`tril(·, −1)` is strictly below the diagonal, so `I + tril(...)` is unit lower triangular, hence
always invertible — the algorithm cannot hit a singular matrix, whatever the data. That is a real
robustness property and worth one sentence on the card.

The remaining cost picture, from §2.1 and carried over: chunkwise is `O(LCd + Ld²)` in arithmetic and
`O(L/C)` in sequential steps, so

> "`C = L` recovers the fully parallel form and `C = 1` recovers the recurrent form. The chunkwise
> parallel form allows us to interpolate between the two forms, in essence trading off the number of
> sequential computations against sequence-level parallelism. In practice `C` is set to a small
> constant (usually 64 or 128), allowing for subquadratic training."

**A single dial with the two endpoints being the two things the reader already knows.** That is the
card's central interaction, and it needs no invention — the paper hands it over.

Implementation detail worth quoting because it is the same trick as concept 14's:

> "We adapt FlashLinearAttention to implement Eq. 8 and 9 with hidden states recomputed during the
> backward pass for saving GPU memory."

### Step 5 — the fully parallel form, which gives this family an attention matrix

For completeness the paper derives what happens at `C = L`, and this is the most useful paragraph in
the paper for a *visual* app:

> "From Eq. 4, it is straightforward to compute the attention matrix `A`: `A_ij = k_jᵀ P_{j+1}^i q_i`
> if `j ≤ i` and 0 otherwise. Notably, `A` has the matrix form `A = (QKᵀ ⊙ M) T`, obtained by
> combining Eq. 3 and 11. However, computing `T` requires a matrix inverse (Eq. 10), which scales
> cubically with sequence length without further algorithmic changes. Due to the above we avoid using
> the fully parallel form for training DeltaNet; **however the "attention" matrix derived from this
> form could be of interest to the interpretability study for RNNs.**"

Take that invitation. Every state-family card on this timeline so far — concepts 9, 10, 11, 20 — has
had to say "there is no score matrix here, that is the point", and has drawn a state grid instead.
This paper proves that DeltaNet *does* have one, exactly, computable in closed form, and that
`O = A V`. So the deck can, for the first and only time in this family, put a recurrent model's
attention pattern next to softmax's on the same sentence and the same head.

And the comparison is not flattering to the analogy, which is why it is worth drawing rather than
asserting: `A` is not a probability distribution. It has no softmax anywhere in its derivation.
Its rows do not sum to one and its entries can be negative — and a negative entry has a meaning,
which is that the association that token wrote was later erased by a key that overlapped it. **The
delta rule's ability to remove, which concepts 11 onward described in words, shows up as a minus
sign in a picture.** Measured below.

### Step 6 — §3.3, the part that does change the model

Two substitutions, both away from Schlag et al.'s choices:

> "Our key/query vectors are given by `k_t = SiLU(W_K x_t) / ‖SiLU(W_K x_t)‖₂`, `q_t = …`. Schlag et
> al. 2021 originally follow Katharopoulos et al. 2020 and apply a "ELU+1" to nonlinearly transform
> the key/query vectors. We instead use the SiLU activation, which was found to perform better."

> "For stability, it is crucial to ensure that the norm of each eigenvalue of the transition matrices
> does not exceed one. The eigenvalues of `I − β_t k_t k_tᵀ` are 1 with multiplicity `d−1` and
> `1 − β_t‖k_t‖₂` with multiplicity 1. Schlag et al. 2021 used the `L₁` norm to normalize query/key
> vectors, ensuring that `0 ≤ 1 − β_t‖k_t‖₂ ≤ 1`. We instead apply `L₂` normalization, which we found
> to perform better and offers a more intuitive interpretation: **when `β_t = 1`, `I − k_t k_tᵀ`
> becomes a projection matrix, erasing information in one subspace while preserving the other `d−1`
> subspaces.** This is beneficial for retaining information while enabling more targeted forgetting."

Three things about this passage.

1. **It reframes concept 11's normalisation crisis as an eigenvalue condition.** Concept 11's note
   found empirically that the delta step is a gradient step with learning rate `β‖φ(k)‖²`, that this
   averages 33.9 on the app's keys, that the stable band is `(0, 2)`, and that Schlag's models
   "diverged otherwise". This paper says the same thing in the language of linear algebra: the
   transition matrix must not have an eigenvalue outside the unit disk. Same requirement, better
   vocabulary, and it generalises — which is why the next concept can put a decay in front of it and
   still reason about stability.
2. **The printed eigenvalue is off by a square.** For eigenvector `k`,
   `(I − βkkᵀ)k = (1 − β‖k‖₂²)k`, so the non-trivial eigenvalue is `1 − β‖k‖₂²`, not `1 − β‖k‖₂`.
   The distinction is invisible in their own setting — under `L₂` normalisation `‖k‖₂ = ‖k‖₂² = 1` —
   which is presumably how it survived six revisions. It is very visible on unnormalised keys, and
   the measurement below shows by how much. The card should state the correct form and can note the
   coincidence in one clause; it should not make a meal of it.
3. **The projection reading is exactly right and is the better mental image.** At `β = 1` with
   `‖k‖₂ = 1`, the transition matrix annihilates the one direction `k` and leaves the orthogonal
   complement untouched. "Erase this key's direction, keep everything else" is what concept 11 spent
   an appendix arguing the delta rule does; here it is a one-line property of the matrix.

## Numbers that matter — from the paper

### Table 1, and what the abstract does not say

Trained on the same SlimPajama subset with the Mistral tokenizer, evaluated with
lm-evaluation-harness. "State exp." is the recurrent state size relative to layers × model dim.

**340M params / 15B tokens**

| model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ | SWDE ↑ | SQuAD ↑ | FDA ↑ | state exp. |
|---|---|---|---|---|---|---|---|
| Transformer++ | 28.39 | 42.69 | 41.2 | 42.2 | 22.1 | 21.4 | N/A |
| RetNet (w/o conv) | 32.33 | 49.19 | 41.0 | 13.3 | 27.6 | 2.9 | 512× |
| Mamba (w. conv) | 28.39 | 39.66 | 41.8 | 12.4 | 23.0 | 2.1 | 64× |
| GLA (w. conv) | 29.47 | 45.53 | 41.8 | 24.0 | 24.7 | 7.3 | 128× |
| **DeltaNet (w. conv)** | **28.24** | **37.37** | **42.1** | **26.4** | **28.9** | **12.8** | 128× |
| + sliding attn | 27.06 | 38.17 | 42.1 | 39.3 | 32.5 | 18.8 | N/A |
| + global attn (2 layers) | 27.51 | 35.04 | 42.1 | 42.9 | 32.1 | 23.1 | N/A |

**1.3B params / 100B tokens**

| model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ | SWDE ↑ | SQuAD ↑ | FDA ↑ | state exp. |
|---|---|---|---|---|---|---|---|
| Transformer++ | **16.85** | 13.44 | 50.9 | **66.6** | 31.5 | **27.4** | N/A |
| Mamba (w. conv) | 17.06 | 13.89 | 50.0 | 41.4 | 35.2 | 6.2 | 64× |
| GLA (w. conv) | 17.25 | 14.92 | 50.4 | 52.4 | 37.4 | 22.3 | 256× |
| **DeltaNet (w. conv)** | 16.87 | **12.21** | 51.6 | 49.5 | 37.4 | 17.2 | 128× |
| + sliding attn | 16.56 | 11.74 | 52.1 | 53.3 | 43.3 | 22.3 | N/A |
| + global attn (2 layers) | 16.55 | 12.40 | 51.8 | **71.0** | 43.0 | 29.8 | N/A |

The honest reading, which the paper itself supplies and the abstract does not:

> "For recall-intensive tasks (i.e., SWDE, SQuAD, FDA), we find that under the same state size at the
> 340M scale, DeltaNet outperforms GLA, confirming the effectiveness of the delta rule. **However, at
> the 1.3B scale, DeltaNet underperforms GLA due to its poorer state size scalability (see §5.3),**
> since state size plays an important role in recall-intensive tasks."

Three things a card that quotes only the headline would get wrong:

- At 1.3B the pure DeltaNet does **not** beat the transformer on Wikitext perplexity: 16.87 against
  16.85. It beats it on LAMBADA and on average accuracy; it loses badly on SWDE (49.5 vs 66.6) and
  FDA (17.2 vs 27.4). **Only the hybrids beat Transformer++ across the board**, and the paper is
  clear that this is what it claims: *"we also experiment with two hybrid models … and find that
  these hybrids outperform strong transformer baselines."*
- GLA's advantage at 1.3B comes with **twice the state** (256× vs 128×), and the reason DeltaNet
  cannot simply take a bigger state is a kernel limitation, not a modelling one — §5.3, below.
- The 3B/1T run "slightly underperforms a Transformer architecture trained with the same setting
  (PowerLM-3B)": DeltaNet-3B scores 60.4 ARC and 72.8 HellaSwag against PowerLM-3B's 60.5 and 74.6.

### The synthetic results, which are the cleaner argument

MAD benchmark (Figure 3 in v6; all rows but DeltaNet's borrowed from Poli et al. 2024):

| model | Compress | Fuzzy Recall | In-Context Recall | Memorize | Noisy Recall | Selective Copy | Average |
|---|---|---|---|---|---|---|---|
| Transformer | 51.6 | 29.8 | 94.1 | **85.2** | 86.8 | 99.6 | **74.5** |
| Mamba | **52.7** | 6.7 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| GLA | 38.8 | 6.9 | 80.8 | 63.3 | 81.6 | 88.6 | 60.0 |
| **DeltaNet** | 42.2 | **35.7** | **100** | 52.8 | **100** | **100** | 71.8 |

Perfect on three recall tasks including one the transformer does not solve perfectly, better than the
transformer on fuzzy recall — and **worse than every baseline on Memorize (52.8)**, which the paper
flags without explaining: *"although it somehow struggles on the 'Memorize' task."* On MQAR (Figure
4, sequence length 512, 64 key-value pairs) DeltaNet *"performs perfectly (even without convolution)
in the hardest setting and outperforms Mamba (which uses convolutions) in the low-dimension
setting."*

### Ablations — the §3.3 changes, isolated (340M)

| variant | Wiki ppl ↓ | LMB ppl ↓ | Avg ↑ | SWDE ↑ | FDA ↑ |
|---|---|---|---|---|---|
| L₁-norm & 1+ELU (**Schlag et al.'s choices**) | 31.12 | 55.96 | 40.1 | 14.5 | 6.2 |
| L₂-norm & 1+ELU | 28.03 | 37.62 | 42.1 | 23.8 | 13.1 |
| L₂-norm & ReLU | 28.75 | 43.53 | 40.9 | 27.2 | 9.0 |
| **L₂-norm & SiLU** (the paper's model) | 28.24 | 37.37 | 42.1 | 26.4 | 12.8 |

> "We find that simply replacing the `L₁`-norm with the `L₂`-norm greatly increases performance."

Worth being precise about, because it is easy to over-credit the feature map: **the normalisation is
doing nearly all of the work.** L₁→L₂ with the feature map held at 1+ELU moves Wikitext from 31.12 to
28.03 — 3.09 perplexity. Swapping 1+ELU for SiLU with L₂ held fixed moves it from 28.03 to 28.24,
which is *worse* on Wikitext and better on LAMBADA (37.62 → 37.37). The paper's claim that SiLU
"performs the best" is not visible in the Wikitext column of its own ablation table. The card should
say the normalisation is the change that matters and not oversell the activation.

### Speed, and how thin the reported evidence is

> "Our chunkwise algorithm achieves greater speed-ups as sequence length `L` and head dimension
> `d_head` increase, where the use of sequence-level parallelism (for high GPU occupancy) and tensor
> core (for fast matmuls) become more important."

Figure 1 plots speed-up of chunkwise over recurrent against sequence length 0.5K–16K for head
dimensions 64, 128 and 256, with the y-axis gridded to **30×**. The paper gives **no table of
values** for this figure, so the card must not quote a specific multiplier as if it were tabulated;
"the axis runs to 30×" is what the source supports. Both forms are implemented in Triton, and
footnote 4 notes that *"our recurrent kernel is already 2× faster than the original CUDA kernel from
Schlag et al. 2021"* — so the speed-up is measured against a baseline already twice as fast as the
2021 code, which makes it a conservative comparison rather than a flattering one.

End-to-end training throughput (Figure 6, 1.3B models, single H100) is reported only qualitatively:

> "The training speed of DeltaNet is close to GLA and significantly faster than Mamba. All
> linear-time models outperform Transformers for longer-sequence training."

### §5.3 — the limitations, which are the next card

Quoted nearly in full, because both halves are load-bearing:

> "First, in terms of computation, although we propose a new hardware-efficient algorithm, **the
> training speed still lags behind that of GLA.** This is due to the overhead caused by modeling
> state-to-state dependencies as described above, which requires "marginalizing" over the head
> dimension inside the kernel, similar to the case of softmax attention. However, for GLA since there
> are no intra-state dependencies (everything is elementwise), and thus it is easy to use tiling to
> support arbitrary size of head dimension … **This limitation would potentially limit DeltaNet's
> memory size, consequently lowering the recall-intensive task performance as we observed in §4.2.**"

> "We also found that **the length generalization of DeltaNet was limited**, while GLA and RetNet
> (and Mamba to an extent) have been found to be able to extrapolate beyond the training length. We
> speculate that this is because **DeltaNet lacks explicit decay factors. This could be improved
> through incorporating a gating term in the recurrence, as demonstrated in a recent work by Yang et
> al. 2024.**"

That last sentence names concept 23 (`gated-deltanet`, arXiv 2412.06464, same first author) as the
fix, from inside this paper. The link text on the timeline should come from here rather than be
invented.

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

All figures below were produced by driving the app's own model from node on the default sentence
("The lighthouse keeper wrote the code in a notebook and hid it under the third stair", 16 tokens).
Queries and keys are SiLU'd and L₂-normalised per §3.3; values are the model's own projections.

### 1. The equivalence, which is the card's central claim — and it is exact

Sequential recurrence (§2.2) against the chunkwise WY/UT form (Eq. 8–11), **whole model, both
blocks, all four heads, every token**, compared at the head outputs and at the final logits:

| chunk size `C` | sequential steps per head | multiplies, all 8 head-passes | max head-output Δ | max logit Δ | top word |
|---|---|---|---|---|---|
| 1 | 16 | 28,672 | 6.22e-15 | 3.55e-15 | in 0.358989 |
| 2 | 8 | 31,296 | 5.77e-15 | 4.44e-15 | in 0.358989 |
| 3 | 6 | 33,632 | 6.44e-15 | 4.88e-15 | in 0.358989 |
| 4 | 4 | 36,672 | 6.66e-15 | 4.44e-15 | in 0.358989 |
| 5 | 4 | 38,752 | 4.88e-15 | 4.00e-15 | in 0.358989 |
| 8 | 2 | 47,936 | 7.99e-15 | 5.33e-15 | in 0.358989 |
| 16 | 1 | 72,512 | 1.27e-14 | 4.88e-15 | in 0.358989 |

Sequential form for comparison: **16 steps per head, 25,600 multiplies** across all 8 head-passes
(`3d² + d` per token, `d = 8`).

The difference is at the last bit of a double. **This is the strongest exactness result in the deck**
— concept 14's tiled attention matched to a comparable tolerance, and this matches it while changing
the *order of operations* far more radically. The card can therefore make the claim in its hardest
form: same next-word bars, same probability to six decimal places, computed with sixteen dependent
steps or with one.

Also confirmed: the equivalence holds with `β` constant at 1, constant at 0.9, and **varying per
token** (a seeded stand-in for `σ(W_β x_t)`), and under both L₂ and Schlag's L₁ normalisation — max
output difference between 2.2e-16 and 2.2e-15 in all twelve combinations. The algorithm is not
relying on `β` being constant, which matters because a reader could reasonably suspect it was.

### 2. The trade, counted rather than timed

At the app's own scale the arithmetic overhead is visible but small. At a realistic head dimension it
is the whole story. Counted multiplies (not timed — see below), one head, `d_head = 64`:

| `L` | form | sequential steps | multiplies | vs sequential |
|---|---|---|---|---|
| 4096 | sequential | 4096 | 50.59M | 1.00× |
| 4096 | chunkwise `C=16` | 256 | 61.38M | 1.21× |
| 4096 | chunkwise `C=32` | 128 | 72.39M | 1.43× |
| 4096 | **chunkwise `C=64`** | **64** | **95.46M** | **1.89×** |
| 4096 | **chunkwise `C=128`** | **32** | **145.80M** | **2.88×** |
| 4096 | chunkwise `C=256` | 16 | 263.24M | 5.20× |
| 4096 | chunkwise `C=4096` (fully parallel) | 1 | 14,188M | 280.44× |

The same shape at `L = 512`: sequential 6.32M / 512 steps; `C=64` 11.93M / 8 steps; `C=512` 70.65M /
1 step. **The ratios are identical at both lengths** — 1.21×, 1.43×, 1.89×, 2.88×, 5.20× for the same
chunk sizes — because the overhead is `O(LCd)` against `O(Ld²)` and depends on `C/d`, not on `L`.
That is a clean, checkable statement and it explains the paper's "usually 64 or 128" without the
paper having to justify it: **at `C = d_head` the arithmetic roughly doubles and the sequential depth
falls by a factor of `d_head`.** The knee of the curve is at the head dimension.

This measurement is also the honest form of the claim. The paper's argument is about wall-clock on an
H100; what a JS interpreter can count is multiplications. Trading 1.89× the multiplies for 1/64th the
steps is only a good deal on a machine that can do many multiplies at once — which is exactly the
premise the app cannot test and must state.

### 3. The attention matrix, which is the card's picture

`A = (QKᵀ ⊙ M) T` computed from Eq. 10 over the whole 16-token sentence, block 0 head 0, verified by
checking `O = A V` against the sequential recurrence (max difference **1.78e-15**, so the matrix is
the real thing and not an illustration):

| | DeltaNet `A`, β = 1 | DeltaNet `A`, β varying | softmax, same head |
|---|---|---|---|
| negative entries | **54 / 136 (39.7%)** | 41 / 136 (30.1%) | **0** |
| minimum entry | −0.5911 | −0.4960 | 0 |
| maximum entry | 0.8719 | 0.6181 | — |
| row sums | −0.0542 to 1.3206, mean 0.7525 | −0.0018 to 1.2926, mean 0.5643 | **exactly 1, every row** |

Last row (the query that predicts the next word), β = 1:

    [0.008, −0.017, 0.207, −0.077, −0.217, 0.209, 0.015, −0.406,
     −0.033, 0.135, −0.045, 0.344, 0.423, −0.063, 0.337, 0.495]

against softmax on the same head and the same tokens:

    [0.002, 0.000, 0.013, 0.054, 0.002, 0.633, 0.000, 0.000,
     0.000, 0.014, 0.000, 0.272, 0.001, 0.002, 0.002, 0.004]

Two facts to build the panel on. **Two in five entries are negative**, and softmax's are structurally
incapable of being so. And the row sums wander from −0.05 to 1.32 where softmax's are exactly 1 by
construction. This is the delta rule's "remove" operation made visible: a negative weight on token
`j` says that whatever `j` wrote was subsequently *un*-written by a later key that overlapped it, and
the read at this query has to subtract it back out. Concepts 11 and 20 could only assert that; this
one can draw it, and the drawing is the reason to build the panel rather than describe it.

Softmax attention cannot express this at all — every weight is non-negative, so every token can only
ever be added in. That is not a small difference in degree; it is a different set of functions.

### 4. §3.3's normalisation, on real keys

Eigenvalue of `I − β k kᵀ` along `k`, i.e. `1 − β‖k‖₂²` at `β = 1`, over all 64 key vectors in block
0 (4 heads × 16 tokens):

| features | ‖k‖₂ mean | ‖k‖₂ min–max | `1 − ‖k‖₂²` mean | min | max | any \|eigenvalue\| > 1 |
|---|---|---|---|---|---|---|
| SiLU, unnormalised | 3.9010 | 0.7190 – 7.3410 | **−17.0313** | **−52.8898** | 0.4831 | **yes** |
| SiLU + **L₂** (the paper) | 1.0000 | 1.0000 – 1.0000 | **−0.0000** | −0.0000 | 0.0000 | no |
| SiLU + L₁ (Schlag's convention) | 0.5421 | 0.3996 – 0.7805 | 0.7011 | 0.3908 | 0.8403 | no |

And what that does to the recurrence, `β = 1`, 16 tokens, head 0 block 0:

| features | largest number in the state | largest output |
|---|---|---|
| SiLU, unnormalised | **2.886e+8** | 6.907e+8 |
| SiLU + L₂ | 5.252 | 5.195 |
| SiLU + L₁ | 2.697 | 1.546 |

Every claim in §3.3 reproduces exactly. L₂ pins the non-trivial eigenvalue to **exactly zero** at
`β = 1` — the projection matrix the paper describes, erasing one direction and preserving the other
seven, measured rather than asserted. L₁ keeps it inside `[0, 1]` as the paper says of Schlag's
scheme, but never reaches zero, so a full-strength write under L₁ **cannot completely erase** the
direction it is overwriting: at 0.70 mean, 70% of the old association survives its own replacement.
That is a sharper statement of why L₂ wins than the paper offers, and it is measurable on screen.

Unnormalised, the transition matrix has eigenvalues down to −52.9 and the state reaches 2.9e8 in
sixteen tokens — the same divergence concept 11's note found by a different route.

The measurement also settles the printed-eigenvalue question in Step 6: on unnormalised keys
`1 − ‖k‖₂` averages −2.90 while the true `1 − ‖k‖₂²` averages −17.03. Under the paper's own L₂
normalisation both equal 0, which is why the typo is invisible in context.

### 5. The seam needs no extension at all

The paper's DeltaNet is reachable from the existing `stateMixer` with no new options:

```js
stateMixer({ write: "delta", beta, features: (v) => l2(silu(v)), sumNorm: false, attnNorm: false })
```

Checked against an independent sequential implementation written from the paper's §2.2 equation:
**max difference exactly 0.00e+0**. `features` was added for Performer's random projection and
happens to be exactly the right shape for "SiLU then L₂"; `sumNorm: false` turns off Schlag's Eq. 29,
which this paper replaces.

So **concept 22 adds nothing to `mixers.js`**. The chunkwise algorithm lives in the card, because the
seam's job is to describe *models* and this concept is not a model — it is a second way to evaluate
one that is already there. Building it into the seam would be the wrong shape and would also destroy
the card's main demonstration, which requires two independent implementations to compare.

For the app's own scale the L₂ variant predicts **"in" at p = 0.358989**, where the Schlag-normalised
delta rule of concept 11 predicts "keeps" at p = 0.4485. Both are untrained noise and neither means
anything; the card must not present either as better. It is only evidence that §3.3 changes the
model, unlike §3.1–3.2, which do not.

## What the live view must let the reader do

### The one thing this card must prove

Two implementations, one comparison, stated as a number: **the chunkwise form and the sequential form
produce the same output.** Not "approximately", not "in our experiments" — to 1.3e-14 on the logits,
which is the arithmetic noise floor of a double. Everything else on the card is secondary to that,
because if it is not exact then the trade is not the trade the paper claims.

### Interaction 1 — the chunk-size dial

One slider, `C` from 1 to the sentence length, with both endpoints labelled by the concepts the
reader already has: `C = 1` **is** concept 11's recurrence, `C = L` is the fully parallel form. For
each setting show, live: sequential steps per head, multiplies counted by the running code, and the
maximum difference from the sequential result. The difference stays at 1e-15 across the entire range
while the other two numbers move by 3× and 16× respectively.

The counter must be a real counter incremented inside the loops that produce the output, not a
formula evaluated alongside them. Anything else is an assertion wearing a number's clothes.

### Interaction 2 — the chunk laid out

For a chosen chunk, show what is actually computed: the `C × C` matrix `diag(β)KKᵀ` strictly below
the diagonal, then `T` after the inverse, then `W` and `U` as `C × d` blocks. The reader should be
able to see that no `d × d` matrix appears anywhere inside a chunk, and that the only thing crossing
a chunk boundary is the single state `S`. That is the memory argument, and it is a picture rather
than a sentence.

Keep the chunk small (4 or 8) so the matrices are legible. This is the panel where the reader learns
what "WY representation" means operationally: two thin matrices standing in for a product of `C`
`d × d` Householder factors.

### Interaction 3 — the attention matrix. This is the card.

Draw `A = (QKᵀ ⊙ M)T` with the deck's existing attention grid, alongside softmax attention on the
same head and sentence, and report the four measured facts: fraction of negative entries, the range,
the row sums, and the verification that `O = A V` reproduces the recurrence.

The grid's `heat()` is written for non-negative weights, so negative entries need their own colour
rather than being clamped to the background — otherwise the panel would hide the exact thing it
exists to show. That is a real change to a shared view and should be made as an opt-in second colour,
not a redefinition of the default.

This is the only place in the entire deck where a fixed-state model's attention pattern can be put
beside softmax's without either being an analogy.

### Interaction 4 — the normalisation, §3.3

A three-way choice — unnormalised / L₁ / L₂ — showing the eigenvalue `1 − β‖k‖₂²` across the real
keys and the largest number in the resulting state. Unnormalised diverges to 2.9e8; L₁ sits at 0.70
and cannot fully erase; L₂ sits at exactly 0 and is a projection. One control, three regimes, all
measured, and it carries the only part of this paper that changes the model.

### What this card must not build

- **No timing.** Not a stopwatch, not a "speedup" number, not an animated GPU. The paper's claim is
  wall-clock on an H100 with Triton kernels and tensor cores; a JS interpreter in a browser tab
  cannot measure any part of that, and a fake number would be worse than none. Count operations and
  count steps — both are exact, both are honest, and the gap between "fewer steps" and "less time" is
  itself the thing to explain.
- **No animation of a parallel scan or a tensor core.** Same reason concept 20 refused to draw a
  Mamba block: a picture of hardware the app does not have is decoration.
- **No claim that this concept improves quality.** §3.1 and §3.2 change nothing. §3.3 does, and it is
  the panel's job to keep them apart, not to blend them into "DeltaNet got better in 2024".
- **No re-drawing of the state grid.** Concept 11 owns that picture and it is unchanged here. The
  shared `flowPanel` covers the continuity; the card's own panels should be about the algorithm.

## What the source does *not* establish

- **That chunkwise training is faster than sequential in general.** It is faster on hardware with
  wide parallel arithmetic units and expensive memory traffic. Figure 1's speed-ups grow with both
  sequence length and head dimension, which is the tell: the gain is a property of the machine, not
  of the algorithm. On a single-threaded interpreter the chunkwise form does 1.2× to 5× *more* work
  and would be strictly slower. The paper never claims otherwise, but a summary of it easily would.
- **A specific speed-up multiplier.** Figure 1 is a plot with no accompanying table. The y-axis
  reaches 30×; individual values are not stated in text, and the card must not invent them.
- **That DeltaNet beats a transformer.** At 1.3B the pure model loses on Wikitext (16.87 vs 16.85),
  SWDE (49.5 vs 66.6) and FDA (17.2 vs 27.4). Only the hybrids — which contain softmax attention —
  beat Transformer++ across the board, and the 3B model "slightly underperforms" PowerLM-3B. The
  claim the paper actually defends is that DeltaNet beats *linear-time* baselines at matched scale,
  with the 1.3B recall exception it states itself.
- **That the delta rule beats gated linear attention.** It does at 340M under matched state size, and
  loses at 1.3B where GLA has twice the state. The paper attributes this to a kernel constraint on
  head dimension rather than to the rule — a plausible explanation that it does not test by, for
  instance, running GLA at 128× to match.
- **Why DeltaNet is bad at MAD's "Memorize" task** (52.8, worse than every baseline including GLA's
  63.3). The paper notes it — "it somehow struggles" — and offers nothing further. A rule whose
  entire purpose is overwriting doing badly at pure memorisation is at least suggestive, but the
  paper does not go there and neither should the card.
- **That SiLU is the right feature map.** Its own ablation has L₂+1+ELU at 28.03 Wikitext against
  L₂+SiLU's 28.24. The normalisation is what carries the result; the activation is a wash on that
  metric and is justified by citation ("was found to perform better") rather than by the table.
- **Any numerical-precision analysis of the triangular inverse.** The paper says forward substitution
  solves it efficiently and says nothing about conditioning, error growth with `C`, or the precision
  the kernel runs at. `I + tril(·, −1)` is unit triangular so it is never singular, but "invertible"
  and "well-conditioned" are different claims and only the first is established. This app measured
  1e-15 agreement at `C ≤ 16` in float64; nothing here supports a statement about `C = 128` in bf16.
- **That the WY representation is novel here.** It is Bischof and Van Loan, 1987, cited as such. The
  contribution is recognising that the delta rule's transition matrix is a generalized Householder
  matrix and that forty-year-old numerical linear algebra therefore applies.

## Leaves behind

### Backward — three debts collected at once

- **Concept 11 (`delta-rule`)** is the direct parent, and this paper is the answer to the cost that
  card's trade-off record lists first. Its "why the rule waited three years to be scaled" is resolved
  here, and the resolution is that nothing about the rule needed changing — only the way it is
  evaluated. The `LINKS` entry should say so.
- **Concept 9 (`linear-attention`)** supplied the chunkwise parallel form (Eq. 1–2) that this paper
  extends. Eq. 9 is Eq. 2 with `V` replaced by `U − W Sᵀ`, which is as close to a one-line
  description of this paper as exists.
- **Concept 14 (`flashattention`)** is the same *kind* of contribution in a different family — an
  exact restructuring that changes no output and moves the cost — and the same argument appears in
  footnote 1 here: a parallel scan would be fine on arithmetic and unaffordable on memory traffic.
  The deck now has two cards whose subject is "the arithmetic was never the problem".

### Forward — what this paper leaves on the table, and who collects

- **Concept 23 (`gated-deltanet`)** is named from inside §5.3: length generalisation is limited,
  the speculated cause is the absence of an explicit decay, and the stated fix is "incorporating a
  gating term in the recurrence, as demonstrated in a recent work by Yang et al. 2024" — which is
  arXiv 2412.06464, the next card, same first author, six months later. The forward link text should
  come from this sentence rather than be invented.
- **The state-size ceiling.** §5.3 says the head dimension cannot grow because state-to-state
  dependencies force marginalisation inside the kernel, and that this is what costs DeltaNet the
  1.3B recall comparison against GLA. The suggested remedy — "block diagonal generalized Householder
  transition matrices with block sizes fitting GPU SRAM (e.g., 128)" — is left unexplored.
- **The attention matrix as an interpretability object.** §3.2 raises it, notes it "could be of
  interest to the interpretability study for RNNs", and drops it because the `O(L³)` inverse makes it
  useless for training. This card is, as far as this deck goes, the thing the paper suggested and did
  not do.
