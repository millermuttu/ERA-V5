# Concept 11 — The delta rule, fast-weight programmers
**Card id:** `delta-rule` · **Date:** 2021-02-22 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:2102.11174](https://arxiv.org/abs/2102.11174), Imanol Schlag\*, Kazuki Irie\*, Jürgen
  Schmidhuber (The Swiss AI Lab IDSIA, USI & SUPSI; \* equal contribution) — *Linear Transformers
  Are Secretly Fast Weight Programmers*. **ICML 2021.**
- Version history from the abstract page: **v1 Mon 22 Feb 2021 16:51:38 UTC**, v2 23 Feb 2021,
  v3 9 Jun 2021. The timeline uses v1.
- Full text read from the **ar5iv render of v3** (`ar5iv.labs.arxiv.org/html/2102.11174`), fetched
  and de-marked-up locally so the equations could be read in their LaTeX source form rather than as
  rendered glyphs. Sections used: §1 (Introduction), §2 (Background on FWPs), §3.1 (Self-Attention
  Without Softmax Is a Fast Weight Programmer), §3.2 (Linearising Self-Attention), §4.1 (Capacity
  Limitation), §4.2 (Improving the FWP's Programming Instruction), §5.1–5.4 (Linear Attention
  Functions, incl. DPFP), §6.1.1/6.1.2 (both synthetic settings), §6.2 (MT), §6.3 (LM), §7, and
  appendices **A.1** (update-rule derivation), **A.2** (key sum normalisation — the derivation that
  explains *why* the normaliser is a sum and not a norm), **B** (formal comparison to Peng et al.'s
  gated rule — this is concept 22 in advance), **C** (the four-line DPFP implementation), and
  **D.3** (the non-overcapacity LM ablation, Table 5).
- Every quotation below is the authors' own wording from that source.
- Code released by the authors: <https://github.com/ischlag/fast-weight-transformers>.
- **In addition**, and clearly separated below, every quantitative claim this card intends to put on
  screen was re-derived numerically against this app's own model — 32 dims, 4 heads, `d_k = 8`,
  2 blocks, seed 20260817, the default 16-token sentence — by driving `app/model/mixers.js`
  directly. Those numbers are labelled **[measured here]**. They are not the paper's, and one of
  them contradicts the current implementation.

## The mechanism, precisely

### Step 0 — the 1991 machine the paper is reaching back to

§2 sets up Fast Weight Programmers before a single Transformer equation appears, and the order
matters: the paper's claim is not "linear attention is *like* a 1991 model", it is that they are the
same equations.

> "A traditional slow net with slow weights continually changes or reprograms the fast weights of a
> fast net, making the fast weights effectively dependent on the spatio-temporal context of a given
> input stream. Simply put, the slow net learns to program its fast net."

The elementary instruction is an outer product (Eqs. 1–3):

    a^(i), b^(i) = W_a x^(i), W_b x^(i)                        (1)
    W^(i)        = σ( W^(i−1) + a^(i) ⊗ b^(i) )                (2)
    y^(i)        = W^(i) x^(i)                                 (3)

> "where `⊗` denotes the outer product, `σ` is an activation function, `W_a` and `W_b` are trainable
> slow weights, while the fast weights `W^(i)` are generated at each time step `i` and serve as a
> short-term memory. This is a key-value associative memory model in which the write operation is
> based on a summation (Eq. 2) and the retrieval is a matrix-vector multiplication (Eq. 3)."

That last sentence is the whole card in miniature. **Write is a sum. Read is a matrix-vector
product.** Everything that follows — the capacity ceiling, the delta rule, the normalisation
trouble — is a consequence of those two choices.

### Step 1 — the identification: the state *is* a fast weight matrix

§3.1 removes the softmax from ordinary attention (Eq. 7 → Eq. 8) and watches the associativity
regroup the product:

    y^(i) = V^(i)( (K^(i))ᵀ q^(i) ) = ( V^(i)(K^(i))ᵀ ) q^(i)
          = ( Σ_{j=1..i} v^(j) ⊗ k^(j) ) q^(i)                 (8)

> "Denoting by `W^(i)` the corresponding weight matrix generated from key and value vectors:"

    W^(i) = ( Σ_{j=1..i} v^(j) ⊗ k^(j) )                       (9)

and then rewrites attention itself as Eqs. 4, 10, 11 — which are Eqs. 1–3 with `σ = identity`:

    k^(i), v^(i), q^(i) = W_k x^(i), W_v x^(i), W_q x^(i)      (4)
    W^(i)               = W^(i−1) + v^(i) ⊗ k^(i)              (10)
    y^(i)               = W^(i) q^(i)                          (11)

§3.2 repeats the derivation with the softmax *linearised* rather than deleted — Katharopoulos'
kernel `κ'(k, q) = φ(k)ᵀφ(q)` — and lands on the same shape with `φ` wrapped around the key, plus
the running denominator:

    W^(i) = W^(i−1) + v^(i) ⊗ φ(k^(i))                         (17)
    z^(i) = z^(i−1) + φ(k^(i))                                 (18)
    y^(i) = ( 1 / (z^(i) · φ(q^(i))) ) W^(i) φ(q^(i))          (19)

> "which is a Fast Weight Programmer (Sec. 2) with normalisation. Hence, the core of linear
> Transformer variants are outer product-based Fast Weight Programmers."

Two things to hold on to, because the app depends on both:

1. **Eq. 17 is line 89 of `mixers.js`.** `S[a][b] = S[a][b]*g + target*k[b]` with `write === "add"`
   *is* Eq. 17, with `g = 1`. Concept 9's card already draws this state. This card's job is to say
   what is wrong with it, not to re-derive it.
2. The identification runs **both ways**, and the paper's own framing is that the 2020 direction is
   the derived one. Linear attention did not invent a state; it rediscovered a fast weight matrix.
   The rhetorical force of the title — *secretly* — is that the linear-attention papers wrote down
   Eq. 17 without noticing they had written down Eq. 2.

### Step 2 — the capacity argument (§4.1). This is the paper.

§4 opens by naming the two defects the FWP view exposes:

> "Viewing linear Transformer variants as Fast Weight Programmers provides us with two insights
> which we investigate in this work: their capacity limits as associative memories (Sec. 4.1), and
> their ineptness to edit previously stored associations (Sec. 4.2)."

Then §4.1, "Intuition", in full — this is the load-bearing paragraph of the entire paper and it is
five sentences long:

> "Endlessly adding new associations to a memory of finite size, as in Eq. 17, inevitably will reach
> a limit. In linear attention, information is stored in a matrix and is retrieved using matrix
> multiplication (see Eq. 19). As a consequence, **to prevent associations from interfering with
> each other upon retrieval, the respective keys need to be orthogonal.** Otherwise, the dot product
> will attend to more than one key and return a linear combination of values. With keys embedded in
> a `d_dot` space, there cannot be more than `d_dot` orthogonal vectors. **That is, storing more
> than `d_dot` associations will result in a retrieval error.** In linear Transformers, when the
> length of the sequence is longer than `d_dot`, the model might be in such an overcapacity regime."

Unpack the mechanism, because the card must show it rather than assert it. Store two pairs with the
sum rule, `W = v₁ ⊗ k₁ + v₂ ⊗ k₂`, and read with `k₁`:

    W k₁ = v₁ (k₁·k₁) + v₂ (k₂·k₁)

The second term is the leak. It vanishes only if `k₂ ⊥ k₁`. With `n` pairs the read at `k_i` returns
`v_i (k_i·k_i)` plus `n − 1` contaminating terms, so the "retrieved value" is a linear combination of
every value ever written, weighted by key overlap. In a `d_dot`-dimensional space you can arrange at
most `d_dot` keys with zero mutual overlap; the `(d_dot + 1)`-th key is necessarily a linear
combination of the others and its retrieval is necessarily contaminated. **The bound is linear
algebra, not an empirical observation** — which is why the paper can state it before running
anything.

The paper then grounds it in older theory rather than claiming novelty:

> "A tensor product representation (TPR) of a structured symbolic system consisting of a set of
> variables and values constructed from outer products of the so called role and filler vectors.
> These terms directly translate into keys and values in our context. The fast weight memories of
> Eq. 17 are the most basic form of such representations (second order tensors). … In particular,
> Theorem 3.3 and 3.1 of Smolensky 1990 discuss more formally the crosstalk and retrieval error
> intuitively described in the previous paragraph."

and marks the one real difference from Smolensky:

> "the classic TPRs of Smolensky 1990 are constructed with a priori knowledge of the symbolic
> structure. In contrast, our FWPs since 1991 … learn all the vectors involved in constructing such
> a representation."

**The consequence that determines what `φ` is for.** §5.1 draws the line the app must not blur:

> "Another property of `φ` derives from the discussion of memory capacity in Sec. 4.1. **The
> dimensionality of its codomain `d_dot` defines the model's capacity.** Therefore, by including a
> transformation which projects the input dimension `d_key` to a larger dimension `d_dot`, the `φ`
> function can potentially increase the upper bound of the capacity."

So: **capacity is set by `φ`, not by the update rule.** The delta rule does not raise the ceiling.
Anyone who says "the delta rule fixes the capacity problem" has merged §4.1 and §4.2 into one claim
the paper is careful to keep apart, and the app's own measurement (below) reproduces the separation
exactly. Getting this right is the difference between this card being honest and being folklore.

### Step 3 — why addition specifically fails (§4.2 and §1)

§1, stated as the motivation for the whole paper:

> "When the sequence length exceeds storage capacity, the model may end up in an overcapacity regime
> … To properly operate under such a regime, the model should learn to dynamically interact with the
> memory contents and selectively decide which key-value associations to keep and which ones to
> delete. **The purely additive instruction may be inappropriate for this purpose.**"

§4.2, more precisely:

> "Once in overcapacity, an ideal memory model should dynamically interact with the memory contents
> and selectively determine which associations to remember or to forget. This is in stark contrast
> to the standard Transformer which stores immutable pairs of key and value vectors by
> concatenation, thus increasing the storage size. While such models work well in practice, we
> consider a model's capability to update previously acquired knowledge to be critical for many
> problems. Hence, from the perspective of dynamic interaction with the memory, **the purely
> additive update rule of Eqs. 17 may be sub-optimal.**"

Note the shape of the argument. The failure is not that addition is inaccurate; it is that addition
has **no inverse operation available to it**. `W^(i) = W^(i−1) + v ⊗ φ(k)` contains no term that can
be negative-going by design. To un-write an association the model would have to arrange for a *later*
value vector to be the negation of an earlier one — possible in principle, impossible to learn
reliably, and it still leaves both outer products physically present in the sum, both consuming rank.
Addition can only ever say "and also". It cannot say "instead".

The standard Transformer is exempt because it does not compress: it keeps `K^(i)` and `V^(i)` as
literal growing lists (Eqs. 5–6) and the softmax *selects* from them. The price of throwing the list
away is that "select" becomes "superpose", and superposition is not undoable by more superposition.

### Step 4 — the delta rule (§4.2, Eqs. 20–25)

The prose statement first, because it names the three moves in order:

> "Given a new input key-value pair `(k^(i), v^(i))`, the FWP **first accesses the current state of
> the memory `W^(i−1)` and retrieves the value `v̄^(i)` currently paired with the key `k^(i)`**.
> Then the model stores a **convex combination `v_new^(i)` of the retrieved value `v̄^(i)` and the
> input `v^(i)`** using an **interpolation weight `0 ≤ β^(i) ≤ 1` also generated by the model.**"

and the equations:

    k^(i), v^(i), q^(i) = W_k x^(i), W_v x^(i), W_q x^(i)                       (4)
    v̄^(i)              = W^(i−1) φ(k^(i))                                      (20)
    β^(i)               = σ( W_β x^(i) )                                        (21)
    v_new^(i)           = β^(i) v^(i) + (1 − β^(i)) v̄^(i)                       (22)

> "where `W_β ∈ ℝ^{1×d}`, and `σ` is the sigmoid function. **The interpolation weight `β^(i)` is the
> "write-strength" as it defines to which extent the new value will replace the previous value.** We
> note that while `β^(i)` only depends on `x^(i)`, in a multi-layer model, `x^(i)` has the full
> context information except in the first layer."

> "We set `W^(0) = 0` and `z^(0) = 0`."

    W^(i) = W^(i−1) + v_new^(i) ⊗ φ(k^(i))   − v̄^(i) ⊗ φ(k^(i))                (23)
                     └──── write ────┘         └──── remove ────┘
          = W^(i−1) + β^(i) ( v^(i) − v̄^(i) ) ⊗ φ(k^(i))                       (24)
    y^(i) = W^(i) φ(q^(i))                                                      (25)

> "As shown in Eq. 24, our programming instruction or update rule is effectively a delta rule with a
> dynamic learning rate `β^(i)`. The model thus learns to correct the current key to value
> association."

Every part of the brief's "read-then-write-the-difference" is answerable from these six lines:

| question | answer, from the paper |
|---|---|
| where does `v_old` come from? | Eq. 20 — a **read of the state at the incoming key**, `v̄ = W^(i−1) φ(k^(i))`. It is not stored anywhere; it is recomputed at write time. |
| where does `β` come from? | Eq. 21 — a **learned linear map of the token's own hidden state**, `W_β ∈ ℝ^{1×d}`, one scalar per token per head, through a sigmoid. |
| what is `β`'s range? | `0 ≤ β^(i) ≤ 1`, guaranteed by the sigmoid. `β = 0` is "leave the memory alone", `β = 1` is "overwrite this key completely". |
| is `β` a forget gate? | **No.** It multiplies only the *correction* `(v − v̄)`, never `W^(i−1)`. Nothing in Eq. 24 scales the existing state. That distinction is the whole of Appendix B and the whole of concept 22. |
| why is Eq. 23 → Eq. 24 legal? | Appendix A.1: group the two new terms, `v_new − v̄ = βv + (1−β)v̄ − v̄ = β(v − v̄)`, substitute. Two lines, no approximation. |

**Naming clash the card copy must not inherit.** The paper's `v_new` (Eq. 22) is the *blend* that
ends up stored; the *incoming* value is plain `v^(i)`. The brief for this card writes the rule as
`W_new = W_old + β(v_new − v_old)k^ᵀ`, where "v_new" means the incoming value and "v_old" means `v̄`.
Same equation, opposite convention for the word "new". On screen, label the three vectors
**`v_in`** (this token's value), **`v_held`** (what the state currently returns for this key), and
**`Δ = v_in − v_held`**, and never use the bare word "new".

**Why Eq. 23 is the more instructive form for a visual app.** Eq. 24 is compact but Eq. 23 is
*two operations with names the paper itself supplies* — `write` and `remove`. The state grid should
be able to show them separately: one frame that subtracts `v̄ ⊗ φ(k)` (the old association being
lifted out) and one that adds `v_new ⊗ φ(k)`. That is the animation. Eq. 24 is the frame after both.

### Step 5 — Appendix B: delta vs. gating, which is concept 22 arriving early

Peng et al. (2021)'s concurrent gated rule:

    W^(i) = (1 − β^(i)) W^(i−1) + β^(i) v^(i) ⊗ φ(k^(i))                        (52)

The paper's comparison is a four-line worked example and it is the sharpest thing in the appendix.
Take `W = v₁ ⊗ k₁ + v₂ ⊗ k₂` with `k₁, k₂` orthonormal, and write a third pair `(k₃, v₃)` with
`k₃ = k₂`. Under the gated rule:

    W' k₃ = (1−β) v₂ + β v₃      ✓ the intended update
    W' k₁ = (1−β) W k₁ = (1−β) v₁   ✗ "it also modifies or in the worst case erases the value
                                        associated with the key k₁"

Under the delta rule, since `v̄ = W k₃ = W k₂ = v₂`:

    W' k₃ = v₂ + β(v₃ − v₂) = (1−β) v₂ + β v₃   ✓ identical
    W' k₁ = W k₁ = v₁                            ✓ untouched

> "Our update rule thus differs from Peng et al. 2021's one on this property of updating associations
> while keeping other "unrelated" ones intact in an associative memory."

Two consequences for the timeline. First, the delta rule and the gated rule **agree exactly on the
key being written** and differ **only in their collateral damage** — that is a demo, not a
paragraph. Second, `mixers.js`'s `"gated"` branch is *not* Eq. 52; it is `S = gS + (v − gSk)kᵀ`,
delta-on-top-of-decay, which is the DeltaNet-with-forgetting formulation of concept 22's paper
rather than Peng et al.'s. The card should not describe the app's `gated` branch as "the rule the
delta paper argues against". It isn't. It is a later synthesis of both.

### Step 6 — normalisation, and why it is a *sum* (§4.2 and Appendix A.2)

The paper offers two normalisations and **rejects the obvious one**.

*Attention normalisation* is the Eq. 18/19 accumulator carried over from linear attention: keep
`z^(i) = z^(i−1) + φ(k^(i))`, divide the read by `z · φ(q)`.

> "This approach, however, has drawbacks. First, the accumulation of positive values in Eq. 26 always
> grows with the number of steps, and may result in instability. Second, specifically for our update
> rule, this normalisation is not sufficient to balance the weights between write and remove
> operations in Eq. 23."

*Sum normalisation* is the replacement — applied to the **feature vectors themselves**, before they
are ever used:

    φ'(q^(i)) = φ(q^(i)) / Σ_{j=1..d_dot} φ(q^(i))_j                            (29)

> "We divide the effective key and query vectors `φ(k^(i))` and `φ(q^(i))` by the sum of its
> components … before applying Eqs. 20–25."

> "A general consequence of this normalisation is intuitively understood by noticing that the output
> of any matrix-vector operations (like Eq. 25) is a weighted sum of columns of the matrix where
> weights are the components of the vector; thus, **if the vector components sum up to one, the
> operation can be viewed as an attention over the columns of the matrix.**"

Appendix A.2 derives *why the divisor is the sum and not the norm*, and it is worth reproducing
because it is the only place the choice is justified. Expand the state in the Cartesian basis,
`W = Σ_i w^(i) ⊗ e^(i)`, so the columns `w^(i)` are the values stored under the basis keys. One delta
write (with `β` omitted) gives the column-wise update, Eq. 51:

    w'^(i) = w^(i) + k_i v − Σ_{j} k_i k_j w^(j)

> "In Eq. 51, the weight `k_i` on the positive term `v` is in general not equal to the total weights
> on the negative terms `Σ_j k_i k_j`. We can force these weights to be balanced by introducing the
> normalisation: `Σ_j k_i k_j = k_i`. If `k_i` is non zero, we obtain `Σ_j k_j = 1`."

So sum normalisation is precisely the condition that **the amount removed equals the amount
written** — the delta rule's own internal accounting, not a generic scaling convenience. And the
paper is blunt about what happens without it (§6.3): *"The sum normalisation (Sec. 4.2) is used in
all cases: **the models diverged otherwise.**"* Meanwhile the attention normaliser is best switched
*off*: *"better perplexities are obtained when no additional attention normalisation is applied"*,
and for the unbounded-context run, *"It was crucial to remove the attention normalisation for the
Delta Net since the accumulator blows up as indicated in Sec. 4.2."*

**This is the finding with direct consequences for the app, and it is measured below: the app's
current `delta` branch has the rejected normalisation and not the required one.**

### Step 7 — DPFP (§5.4), and the specific complaint against elu+1

The problem statement, §5.2, one sentence:

> "Importantly, as a simple element-wise function, this `φ` function **preserves the dimension of
> the input key vector (`d_key = d_dot`), without modifying the memory capacity** as discussed in
> Sec. 4.1."

`elu(x)+1` buys non-negativity and nothing else. It is a *reparameterisation* of an 8-dimensional
key into an 8-dimensional feature, so the capacity of a `d_k × d_k` state under elu+1 is exactly
`d_k`. Worse, and the paper does not say this but the measurement below does: elu+1 outputs are
**strictly positive in every coordinate**, so every pair of feature vectors sits in the positive
orthant with a large mutual dot product. Non-negativity is bought at the cost of never being near
orthogonal. Capacity `d_dot` is an upper bound that elu+1 has no way of approaching.

FAVOR+ (§5.3) does raise `d_dot` to `2m` but:

> "This sampling process is the main drawback of FAVOR+ as it introduces variance into the model's
> output."

and, on the ceiling still being finite:

> "the model's capacity is still limited, and equals the infinite capacity of the softmax memory only
> when `m` goes to infinity, which is never achieved in practice."

DPFP's stated design goal is orthogonality by construction:

> "It is deterministic and easy to compute like Linear Transformers while increasing the dot product
> dimension without requiring FAVOR+'s random features."

> "**We design `φ` such that it facilitates orthogonality in the projected space**, i.e.
> `φ(k^(i)) · φ(k^(j)) = 0` for `i ≠ j`. Towards this end, we construct `φ` such that if
> `φ_l(x) > 0` then `φ_n(x) = 0` for all `n ≠ l`. Such a constraint can be enforced by limiting the
> domains of the partial functions to be non-overlapping."

The 2-d warm-up (Eqs. 33–36, with `r(a) = max(0, a)`) is the figure the app should steal:

    φ₁(k) = r( k₁) r( k₂)      φ₂(k) = r(−k₁) r( k₂)
    φ₃(k) = r( k₁) r(−k₂)      φ₄(k) = r(−k₁) r(−k₂)

> "each vector in the 2d plane will have a single non-zero component in the 4d space and equally
> splits the input space into four areas which will be orthogonal in the projected space."

The general form (Eq. 37), for `i ∈ [1, 2·d_key]`:

    φ_{iν}(k) = r([k; −k])_i · r([k; −k])_{i+ν}

> "where `ν ∈ {1, 2, …, 2·d_key − 1}` is a capacity controlling hyperparameter. **The codomain
> dimensionality of `φ(k)` is thus `d_dot = 2 · d_key · ν`.** Eq. 37 is highly parallelisable because
> each partial function can be computed independently."

(The upper limit on `ν` renders ambiguously in the ar5iv HTML; `2·d_key − 1` is the reading
consistent with Appendix C, where the roll shift `j` runs `1…ν` over a vector of length `2·d_key`.)

Appendix C is four lines of PyTorch: concat `[relu(x), relu(−x)]`, roll it `ν` times, multiply
elementwise. **For this app `d_key = 8`, so DPFP-1 → `d_dot = 16`, DPFP-2 → 32, DPFP-3 → 48** — i.e.
a state of 8×16, 8×32, 8×48, and a capacity slider that is a real, computable thing rather than a
label.

## Numbers that matter

### From the paper

**Synthetic setting 1 — capacity (§6.1.1).** `d_key = 64` fixed, sequence length equals the number
of unique keys (`L = S`), keys/values sampled **without** replacement, `S` from 20 to 600 in steps
of 20, sum update rule throughout, trained "until the evaluation loss falls below 0.001 or until
lack of progress for 1000 steps".

| model | `d_dot` | where errors start (paper's reading of Fig. 2) |
|---|---|---|
| Linear Attention (elu+1) | 64 | "begins to accumulate errors with **60** or more associations" |
| DPFP-1 | 128 | approaches its limit at **128** |
| DPFP-2 | 256 | approaches its limit at **256** |
| DPFP-3 | 384 | approaches its limit at **384** |
| FAVOR+ (m = 64/128/512) | 2m | "**fails to achieve a loss of 0 in any experiment**" |
| Softmax | — | best of all, "although it struggles to fully converge with more than 500 keys" |

> "The results support our theoretical analysis. Linear-Attention has a capacity of 64 due to the
> choice of `d_key = d_dot = 64`."

**Synthetic setting 2 — update rules (§6.1.2).** Now keys and values are sampled **with**
replacement, `L = 2S`, `S = 20` unique keys, sequence length 40, `φ` = DPFP-1 — deliberately *not*
over capacity:

> "While this setting does not exceed the capacity of DPFP-1, our result is independent of the
> capacity regime."

> "They demonstrate that our new update rule outperforms all other variants. **As expected, the
> baseline sum update rule fails.**"

The two settings are the paper's own separation of the two claims, and the app should copy it:
**setting 1 isolates capacity (and the update rule is irrelevant to it); setting 2 isolates
correction (and the capacity regime is irrelevant to it).**

**WikiText-103 language modelling, Table 2** — the delta-vs-sum comparison, both configs in an
overcapacity regime (`D = 128`, `L = 256`, 40M params, small; `D = 256`, `L = 384`, 90M params,
medium; 16 layers; `H = 8`; `D = H · d_dot`). Perplexity, lower is better:

| model | rule | small valid | small test | medium valid | medium test |
|---|---|---|---|---|---|
| Transformer | — | 33.0 | 34.1 | 27.9 | 29.6 |
| Linear Transformer | sum | 37.1 | 38.3 | 31.1 | 33.0 |
| **Delta Network** | **delta** | **34.1** | **35.5** | **29.7** | **31.5** |
| Performer | sum | 39.0 | 39.6 | 32.2 | 33.8 |
| Performer | delta | 36.1 | 37.2 | 30.0 | 31.8 |

So the delta rule is worth **−2.8 test perplexity** for the Linear Transformer (38.3 → 35.5, small)
and **−1.5** (33.0 → 31.5, medium); **−2.4** and **−2.0** for the Performer. It closes roughly
**two-thirds of the gap** between the linear model and the full Transformer in the small config
(38.3 → 35.5 against a 34.1 target) and about **44%** in the medium (33.0 → 31.5 against 29.6). The
extra parameters are "16 K and 33 K" for small and medium — a rounding error on 40M/90M.

**Table 5 (Appendix D.3) — the same comparison out of overcapacity**, small config, `m = 16` /
`ν = 1` so `d_dot = 256` for both, "The model is thus not necessary in an overcapacity regime":

| model | rule | valid | test |
|---|---|---|---|
| Transformer | — | 33.0 | 34.1 |
| Performer | sum | 38.0 | 38.8 |
| Performer | delta | 36.0 | 37.0 |
| DPFP | sum | 37.7 | 38.8 |
| **DPFP** | **delta** | **33.9** | **35.0** |

> "our update rule improves both variants of linear attention over the sum update-rule baselines
> even in this condition. This indicates the general benefits of our update rule in Fast Weight
> Programmers."

This table is important for the card's honesty: **the delta rule helps even when there is no
capacity pressure at all** (−3.8 test ppl for DPFP). The capacity argument motivates the delta rule
but does not exhaust it. Correction is useful whenever a key recurs, which in language is always.

**Table 3 — the normalisation ablation** (medium, delta rule, sum normalisation always on):

| positional encoding | attention normalisation | valid | test |
|---|---|---|---|
| yes | yes | 30.4 | 32.1 |
| no | yes | 29.2 | 31.2 |
| yes | no | 29.7 | 31.5 |
| **no** | **no** | **28.1** | **31.1** |

Both extras hurt. Learned absolute positions are unnecessary (concept-2 callback), and the
attention normaliser is a net negative for a delta model.

**Table 4 — unbounded context**, state carried across training segments, medium config:

| model | params (M) | state size (M) | valid | test |
|---|---|---|---|---|
| Linear Transformer (sum) | 89.8 | 0.13 | **> 260** | **> 260** |
| Delta Network | 89.9 | 0.13 | **27.8** | **29.4** |
| Transformer-XL | 90.9 | 0.13 | 65.7 | 65.5 |
| Transformer-XL | 90.9 | 1.05 | 29.3 | 30.1 |
| Transformer-XL | 90.9 | 2.10 | 26.4 | 27.4 |
| Transformer-XL | 90.9 | 6.29 | 24.6 | 25.5 |

This is the most dramatic number in the paper and the one to put on the card: **at equal state size
(0.13M), the sum rule scores worse than 260 perplexity and the delta rule scores 29.4.** The
additive state does not merely degrade when run forever — it *breaks*. (For scale: uniform guessing
over a 268K vocabulary is ~268,000 perplexity, so >260 is still "learned something", but it is a
model that has effectively lost its memory.) Transformer-XL beats the Delta Net when allowed 8–48×
the state, and the paper says so plainly: *"Performance of the Delta Net does not yet match the
performance of the Transformer XL when the latter is evaluated with a large state size."*

**WMT14 En-De, Table 1** — this table is about `φ`, not about the update rule; no delta model
appears in it. Bleu, valid/test, "Neither model averaging, nor model specific tuning is done":

| model | `d_dot`=64 | 256 | 512 |
|---|---|---|---|
| Standard | 26.6 / 27.7 | — | — |
| Linear (elu+1) | 25.5 / 26.8 | — | — |
| Performer | 24.2 / 24.4 | 24.9 / 25.3 | 26.7 / 27.7 |
| DPFP (ours) | — | 26.2 / 26.9 | 26.2 / 27.1 |

> "Our DPFP model outperforms the Linear Transformer as well as the Performer when `d_dot` is
> relatively small; providing a good trade-off between simplicity and performance."

At `d_dot = 512` the Performer matches the standard Transformer exactly (27.7) and DPFP does not
(27.1). DPFP's win is at small `d_dot`, and the paper does not overclaim it.

**Cost.** Small LM setting: 63K vs 66K words/sec and 14 vs 13 GB for linear-with-delta vs
linear-without. *"The extra resource requirement is thus marginal."* DPFP 63K vs Performer 57K words/sec —
*"Performers are slower because of the sampling logic, which also motivates our DPFP."* Both beat
the PyTorch Transformer's 33K words/sec — but note the linear models use custom CUDA kernels and the
Transformer does not, which the paper states.

### Measured on this app's model **[measured here — not the paper's numbers]**

Driving `app/model/mixers.js` and `app/model/transformer.js` directly, seed 20260817, default
16-token sentence, `d_k = 8`.

**(a) The app's `delta` branch is numerically divergent as written.** The delta step is a gradient
step with effective learning rate `β · ‖φ(k)‖²`, and it is stable only for `0 < β‖φ(k)‖² < 2`, exact
at `= 1`. On this model's real key vectors, with `elu1` and no normalisation:

    mean ‖φ(k)‖² = 33.9      min 5.4      max 86.6      (128 key vectors, all heads, both blocks)

With `β = 1` (which is what line 88 hard-codes) the effective learning rate is ~34, i.e. every write
overshoots its target by 33×, alternating in sign. `max |S|` per token, head 0, block 0:

| rule | t=1 | t=2 | t=4 | t=8 | t=12 | t=16 |
|---|---|---|---|---|---|---|
| `add` | 1.6e1 | 1.6e1 | 2.6e1 | 4.6e1 | 5.7e1 | 5.5e1 |
| `delta` | 1.6e1 | 3.9e1 | 9.1e3 | 1.1e9 | 1.3e13 | **3.4e18** |

The state grid for the `delta` mechanism is currently rendering a number that has grown by eighteen
orders of magnitude. Applying the paper's sum normalisation (Eq. 29) to `φ(k)` and `φ(q)` fixes it —
final `max |S|` becomes **1.63** for delta and 3.82 for add; L2-normalising instead gives 5.04 and
7.28. This is Eq. 29's *"the models diverged otherwise"*, reproduced at 16 tokens instead of 100M
words. It is a bug, and it is also the most convincing possible demonstration of why §4.2's
normalisation subsection exists.

**(b) The capacity cliff is exactly at `d_k`, and the delta rule does not move it.** Write `n`
distinct L2-unit **orthonormal** keys with random values into an 8×8 state, one write each, then
read every key back. Mean relative recovery error `‖ŵ − v‖ / ‖v‖`, 300 seeds:

| n | 1–8 | 9 | 10 | 12 | 16 |
|---|---|---|---|---|---|
| `add` | **0.000** | 0.398 | 0.617 | 0.908 | 1.257 |
| `delta` | **0.000** | 0.387 | 0.571 | 0.784 | 0.974 |

Both rules are **bit-exact up to `n = d_k = 8` and both break at `n = 9`.** This is Figure 2's claim
reproduced at the app's scale, and it is the number that proves the paper's §4.1: *the error is
identically zero for eight pairs and jumps to 0.4 at the ninth, with nothing changing but the count.*

The same experiment with **random (non-orthogonal) unit keys** — the realistic case — shows
interference from the second pair onward, and here the delta rule does help, by roughly 30%:

| n | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| `add` | 0.310 | 0.622 | 0.971 | 1.253 | 1.458 |
| `delta` | 0.171 | 0.402 | 0.684 | 0.872 | 1.001 |

**(c) The overwrite experiment is where `delta` wins outright.** The paper's setting 2: `S` unique
orthonormal keys, `L` writes sampled **with replacement**, recover the *most recent* value bound to
each key.

| S unique keys, L writes | `add` | `delta` |
|---|---|---|
| 2 keys, 4 writes | 1.012 | **0.0000** |
| 4 keys, 8 writes | 1.004 | **0.0000** |
| 4 keys, 16 writes | 1.782 | **0.0000** |
| 8 keys, 16 writes | 0.990 | **0.0000** |
| 8 keys, 32 writes | 1.772 | **0.0000** |
| 12 keys, 24 writes (over capacity) | 1.710 | 0.651 |
| 16 keys, 32 writes (over capacity) | 2.139 | 0.888 |

**Delta recovers the most recent value exactly — error 0.0000 — for any number of overwrites, as
long as the number of distinct keys stays ≤ `d_k`. Add is wrong by more than 100% of the value's own
magnitude in every row.** An error of 1.0 means the retrieved vector misses by as much as the
answer is worth; 1.78 means the accumulated garbage is nearly twice the signal.

**(d) `β` is a dial, and it behaves.** Same setting (4 keys, 16 writes, orthonormal), delta rule,
sweeping the write strength:

| β | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| recovery error | 0.911 | 0.792 | 0.578 | 0.320 | **0.000** |

`β = 1` is a complete overwrite; `β = 0.1` barely corrects and behaves almost like `add`. The
learned `β^(i) = σ(W_β x^(i))` is choosing a point on this curve per token.

**(e) DPFP raises the ceiling, elu+1 cannot.** Delta rule, L2-normalised features, random keys,
mean recovery error, with `d_key = 8`:

| n pairs | elu+1 (`d_dot`=8) | DPFP-1 (16) | DPFP-2 (32) | DPFP-3 (48) |
|---|---|---|---|---|
| 4 | 0.806 | 0.253 | 0.197 | 0.189 |
| 8 | 1.009 | 0.449 | 0.368 | 0.336 |
| 16 | 1.163 | 0.732 | 0.583 | 0.528 |
| 32 | 1.292 | 1.005 | 0.828 | 0.747 |

elu+1 is the worst feature map at every count by a wide margin — its all-positive outputs are
mutually far from orthogonal, so it never gets near its nominal capacity of 8. DPFP-1 at `n = 8` has
less than half elu+1's error. This is §5.4's motivation, made numeric.

## What the live view must let the reader do

Four interactions, in escalating order of what they prove. Interaction 2 is the one this card
exists for.

### Interaction 1 — the lesson's `40 → 55`, on one cell, with the arithmetic shown

The seam already supports it; this is a scripted two-write scenario, not a new mechanism. One key
`k`, `d_k = 8`, `φ(k)` L2-normalised so the arithmetic is exact and the reader is not asked to
forgive a rounding error.

- Write 1: `v_in = 40`. State empty, so `v_held = 0`, `Δ = 40 − 0 = 40`. Read back: **40**.
- Write 2: `v_in = 55`, same key.
  - **add**: `S += 55·k` → read back **95**. Show the two outer products stacked in the grid and the
    read as their sum. Caption: *the 40 is still there; nothing in Eq. 17 can remove it.*
  - **delta**: `v_held = S φ(k) = 40`, `Δ = 55 − 40 = 15`, `S += β·15·k` → read back **55** at `β=1`.
    Caption: *the state did not learn 55. It learned the correction 15, and 40 + 15 is 55.*
- The `β` slider (interaction 4) makes write 2 read back `40 + 15β`: **47.5 at β = 0.5**, 41.5 at
  β = 0.1. That single interpolating number is the clearest possible statement of what "write
  strength" means, and it is Eq. 22 evaluated on screen.

Show three numbers at all times, labelled with the vocabulary fixed in Step 4: `v_in`, `v_held`,
`Δ = v_in − v_held`. Never the word "new" unqualified.

### Interaction 2 — the capacity lab. This is the card.

A standalone panel, not the sentence pipeline — it needs `n` on a slider, and the sentence is fixed
at 16 tokens. State is `d_k × d_k` = 8×8, matching the model.

**Procedure, exactly:** for a given `n`, generate `n` random values and `n` keys; run the write loop
under both rules; read every key back; report `err(n) = mean_j ‖S φ(k_j) − v_j‖ / ‖v_j‖`.

**Plot:** `n` on the x-axis from 1 to 24, recovery error on the y-axis, **two lines — `add` and
`delta`** — with a vertical rule at `n = d_k = 8` labelled *capacity*. Log y-axis, because the
interesting region spans zero to 1.5 and "zero" must be visibly, categorically different from
"small".

**A key-geometry toggle, which is what makes this honest rather than a stunt:**

- **Orthonormal keys** (the paper's ideal): both lines sit at **exactly 0.000 for n = 1…8** and both
  jump at **n = 9** (0.398 add, 0.387 delta). *This panel proves the capacity claim and simultaneously
  proves the delta rule does not fix it.* Resist the temptation to hide this. It is §4.1 vs §5.1: the
  ceiling is `φ`'s to raise, not the update rule's.
- **Random keys** (the real model's keys): both lines leave zero immediately and `delta` runs ~30%
  below `add` at every `n` (0.684 vs 0.971 at `n = 8`).
- **The model's own keys**, taken from the live sentence's head 0: the honest, messy version.

**The number that proves it:** the headline readout is a single line under the chart —

> **8 pairs: add 0.000, delta 0.000 · 9 pairs: add 0.398, delta 0.387**

Zero, zero, then not-zero, at exactly `d_k + 1`, with nothing changed but the count. That is the
paper's Figure 2 in two numbers, and it is a real computed quantity, not an illustration.

**The `φ` selector on the same panel** (elu+1 / DPFP-1 / DPFP-2 / DPFP-3) moves the vertical
capacity rule to 8 / 16 / 32 / 48 and moves the cliff with it. That is the demonstration that
capacity belongs to the feature map — and DPFP is four lines of code (Appendix C), so it is cheap.

### Interaction 3 — the overwrite lab, where delta actually wins

Same panel, one toggle: **sample keys with replacement**. `S` unique keys on one slider, `L` total
writes on another; recover the *most recent* value per key. This is the paper's setting 2 and it is
where the two rules separate categorically rather than by 30%:

> **4 keys, 16 writes: add 1.782, delta 0.0000**

Exactly zero, for any `L`, as long as `S ≤ d_k`. Interaction 2 shows the wall both rules hit;
interaction 3 shows the thing only one of them can do. Presenting either alone misrepresents the
paper — §6.1.1 and §6.1.2 are two experiments for a reason.

### Interaction 4 — the `β` slider, plus the write/remove decomposition

- **`β` from 0 to 1**, defaulting to 1. At `β = 0` the delta rule *is* the frozen state; at `β = 1`
  it is a complete overwrite; the recovery-error readout traverses 0.911 → 0.000 (measurement (d)).
  Label it **write strength**, the paper's own word, and note underneath that in the real model it
  is `σ(W_β x)` — one learned scalar per token, not a global knob.
- **A "show write / remove" step button** that renders Eq. 23 as two frames on the state grid: first
  subtract `v_held ⊗ φ(k)`, then add `v_new ⊗ φ(k)`. The delta rule's name is "read, then write the
  difference", and the grid should perform all three verbs.

### Two implementation notes the card's demos depend on

1. **`mixers.js` must sum-normalise `φ(k)` and `φ(q)` before the delta branch touches them** (Eq.
   29). Without it the state reaches 3.4e18 by token 16 on the default sentence (measurement (a))
   and the state grid shows nothing. This is the paper's own requirement — *"the models diverged
   otherwise"* — so the fix is faithful, not a patch.
2. **The output division by `z · φ(q)` (line 100) is attention normalisation, which this paper
   evaluates and rejects for delta models** (Table 3: 29.7/31.5 with it, 28.1/31.1 without;
   *"It was crucial to remove the attention normalisation for the Delta Net"*). Eq. 25 is
   `y = W φ(q)`, undivided. Either drop it on the delta path or surface it as a toggle with Table 3
   next to it — but do not present the current divided form as the paper's rule, because it is the
   variant the paper argues against.

For the standalone labs (interactions 2 and 3) use **L2-normalised** keys rather than sum-normalised:
sum normalisation makes `‖φ(k)‖² ≈ 0.18` on this data, so `β = 1` corrects only 18% of the error per
write and "exact" becomes "approximately". L2 normalisation makes the effective learning rate exactly
1 and the demo exactly exact. Say so in the caption — it is a deliberate simplification, and the
running-model path keeps the paper's version.

## What the source does *not* establish

- **The delta rule does not increase capacity.** §4.1 bounds capacity at `d_dot`; §5.1 says `d_dot`
  is `φ`'s to set. The delta rule improves *behaviour under* the bound; it does not move it, and the
  measurement above confirms both rules fail identically at `n = 9`. The paper never claims
  otherwise — but the abstract's adjacency of "memory capacity limitation" and "replace the purely
  additive outer products" invites the misreading, and the card should pre-empt it.
- **The `d_dot` bound is a bound on *exact* retrieval, not a cliff in task performance.** The paper's
  own Appendix D.3 undercuts a strong reading: *"language modelling might be less affected by the
  capacity issue than the synthetic retrieval task, as it might not require the exact retrieval."*
  And doubling `d_dot` there *"was not beneficial for this language modelling setting."* A model can
  be far into overcapacity and still be a good language model.
- **The synthetic tasks are the paper's own construction**, not a community benchmark. Setting 1 and
  setting 2 were designed by these authors to exhibit these two effects. They are clean and the
  conclusions follow, but they are demonstrations, not independent evidence.
- **The delta rule never beats a full Transformer on language modelling** in this paper. Table 2:
  35.5 vs 34.1 (small), 31.5 vs 29.6 (medium). It narrows the gap; it does not close it. And
  Transformer-XL with a 6.29M state reaches 25.5 against the Delta Net's 29.4 — the paper concedes
  this in the text. The delta rule's win is *per unit of state*, and only there.
- **`β` is a single scalar per token per layer**, from `W_β ∈ ℝ^{1×d}`. It cannot express "overwrite
  dimension 3 of the value and leave dimension 5" — the write strength is uniform across the value
  vector. The paper does not discuss this restriction. (Concept 22's literature loosens exactly this
  into a vector-valued gate.)
- **No parallel training form is given.** The paper does not present one, does not claim one exists,
  and does not discuss the cost — it reports wall-clock (63K vs 66K words/sec) using custom CUDA
  kernels and calls the overhead *"marginal"*. That silence is the whole reason concept 21 exists,
  and it is a silence, not a claim: nothing here says the delta rule *can't* be parallelised.
- **DPFP's orthogonality is a design aim, not a theorem.** §5.4 says it "facilitates orthogonality";
  the exact non-overlapping-domain property is demonstrated in the 2-d case (Eqs. 33–36, Figure 1),
  and the generalisation to Eq. 37 is asserted by construction with no proof that arbitrary
  higher-dimensional keys land in disjoint supports. The paper is candid that the supporting
  evidence for the sparsity intuition is Choromanski et al.'s empirical relu-beats-exp result and
  that this *"has not been theoretically justified"*.
- **The identification with 1991 FWPs is exact only up to normalisation**, which the paper flags in
  §1 — *"the formal equivalence of this family of linear Transformers and the Fast Weight Controllers
  … from the '90s (apart from normalisation)"* — and §3.2, *"a Fast Weight Programmer (Sec. 2) with
  normalisation"*. The parenthetical is doing more work than it looks: §4.2 and Appendix A.2 spend
  two pages on exactly that difference.

## Leaves behind

**Backward — concept 9 wrote the state; this card reads it first.**

Concept 9 ended on a promise and named the line: `S += φ(k)vᵀ`, every operation an addition, and
*"the state has no way to express 'not that any more'."* This paper is where that observation comes
from, and it supplies the two things concept 9's card could only gesture at.

The first is a **name for the object**. Concept 9 called `S` "the state" — an accumulator, a
compression of the prefix. This paper says it is a **weight matrix**, and that the tokens streaming
past are *programming* it. That reframing is not cosmetic. A state is something you read; a weight
matrix is something you *train*. Once `S` is a weight matrix, the question "how should a weight
matrix be updated?" has a sixty-year-old answer — Widrow & Hoff, 1960 — and Eq. 24 is that answer
transplanted: a gradient step on `½‖v − Sφ(k)‖²` with learning rate `β`. The delta rule is not a new
idea about attention. It is the oldest idea in the field, applied to an object nobody had noticed was
a weight matrix.

The second is a **number for the ceiling**. Concept 9's card asserted `d_k` from a rank argument and
said "8 in this app, at any sentence length". §4.1 is the citable form of that assertion, and the
capacity lab turns it into a measurement: exactly zero error at 8, 0.398 at 9. Concept 9 could say
the sentence. This card can show the cliff.

And there is a correction to carry back. Concept 9's forward note says the delta rule closes the gap
where a re-bound key returns *"a superposition of both values"*. True, but incomplete in a way that
matters: the delta rule closes it **only while the number of distinct keys stays under `d_k`**. Past
that, delta degrades too (0.651 at 12 keys). The add-only state has two diseases — it cannot correct,
and it runs out of room — and this paper cures the first while proving the second is incurable
without a bigger `φ`. Interaction 2 and interaction 3 exist to keep those separate on screen, because
the paper keeps them separate in §6.1.1 and §6.1.2 and most summaries do not.

**Forward — what read-before-write cost, and who came to collect.**

Look at Eq. 20 again: `v̄^(i) = W^(i−1) φ(k^(i))`. To compute the update at position `i` you must
first *read* `W^(i−1)`, which means `W^(i−1)` must already exist, which means positions
`1 … i − 1` must already have been processed. The additive rule has no such requirement:
`W^(L) = Σ_j v^(j) ⊗ φ(k^(j))` is a sum whose terms are mutually independent, so it can be computed
in any order, in parallel, as one batched matrix product — which is precisely the property
Katharopoulos exploited and precisely why concept 9's mechanism trains as fast as it does.

**The delta rule destroys that.** The recurrence is now genuinely sequential — `W^(i)` depends on
`W^(i−1)` through a *nonlinear-in-the-state* term, not merely through addition — and training
becomes a scan over `L` steps rather than one parallel reduction. The paper does not mention this. It
reports 63K vs 66K words/sec and calls the overhead marginal, which it is *for their custom CUDA
kernel at their sequence lengths*, and moves on. But the structural fact sits there in Eq. 20,
unremarked, and it is the reason the delta rule spent three years as a good idea nobody scaled.

That is exactly the debt **concept 21** (*Parallelizing Linear Transformers with the Delta Rule over
Sequence Length*, arXiv:2406.06484) comes to pay. The observation there is that the delta rule's
recurrence, `W^(i) = W^(i−1)(I − β φ(k)φ(k)ᵀ) + β v φ(k)ᵀ`, is a product of **rank-one-corrected
identity matrices** — a WY-representation, the same trick that makes Householder QR blocked — and a
product of such matrices *can* be chunked and computed in parallel even though a general nonlinear
recurrence cannot. Concept 21 does not change what the delta rule computes. It changes when you are
allowed to compute it. Frame the timeline that way: **2021 finds the right update rule and pays for
it in parallelism; 2024 gets the parallelism back without changing the rule.** That is a rare shape
in this timeline — most cards trade one thing for another, and this pair actually recovers the loss.

**Concept 22** collects the other debt, and Appendix B is where to point. This paper's `β` scales the
*correction only*; nothing in Eq. 24 ever scales `W^(i−1)`, which is exactly the property Appendix B
holds up against Peng et al.'s gated rule (Eq. 52), where `(1 − β)W^(i−1)` fades every association
including the ones the token had no opinion about. The paper wins that argument decisively — *"it
keeps the value associated with `k₁` unmodified"*. But winning it means the Delta Network **can never
forget anything it is not currently overwriting**, and a memory that only ever gets corrected, never
released, is one that fills up and stays full — the state has no way to make room for a new key
except by displacing an old one it happens to overlap with. Table 4's unbounded-context result is
impressive precisely because it stresses that: 89.9M params, 0.13M state, run forever, 29.4
perplexity.

The later synthesis — and it is what `mixers.js`'s `"gated"` branch already implements,
`S = gS + (v − gSk)kᵀ` — is *both*: decay the past **and** correct the present. Concept 22's card
should open by noting that this is not the rule this paper argued against. Appendix B's target was
gating *instead of* the delta rule. What came after was gating *on top of* it, which keeps the
selectivity Appendix B was defending and adds the release valve the Delta Network lacks. Three
operations, arrived at in three papers: **accumulate** (concept 9), **correct** (this card),
**release** (concept 22) — and concept 21 in between, making the second one affordable.
