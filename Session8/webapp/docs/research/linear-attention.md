# Concept 9 — Linear attention, the kernel regrouping
**Card id:** `linear-attention` · **Date:** 2020-06-29 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:2006.16236](https://arxiv.org/abs/2006.16236), Katharopoulos, Vyas, Pappas, Fleuret
  (Idiap / EPFL, University of Washington, University of Geneva) — *Transformers are RNNs: Fast
  Autoregressive Transformers with Linear Attention*. **ICML 2020.**
- Version history from the abstract page: **v1 Mon 29 Jun 2020**, v2 30 Jun 2020, v3 31 Aug 2020.
  The timeline uses v1.
- Full text read from the **LaTeX source** (`arxiv.org/e-print/2006.16236` → `arxiv.tex`, the v3
  tarball), cross-checked against the ar5iv render. Sections used: §3.1 (Transformers), §3.2
  (Linearized Attention), §3.2.1 (Feature Maps and Computational Cost), §3.3 (Causal Masking),
  §3.3.1 (Gradient Computation), §3.3.2 (Training and Inference), §3.4 (Transformers are RNNs),
  §4 (Experiments) with Tables 1–3 and Algorithm 1, plus supplementary §A (Gradient Derivation),
  §B (Training Evolution) and **§C (Image Generation Throughput Discussion)** — which is where the
  honest version of the headline number lives, and which almost no summary reads.
- Every quotation below is the authors' own wording from that source.
- Code released by the authors: <https://linear-transformers.com/>.

## The mechanism, precisely

### Step 0 — attention, generalised away from softmax

They start from ordinary attention, Eq. 2:

    Q = xW_Q,  K = xW_K,  V = xW_V
    A_l(x) = V' = softmax(QKᵀ / √D) V

and then immediately abstract the softmax away. Eq. 3, the **generalised attention equation**, is
the hinge of the whole paper:

> "Equation 2 implements a specific form of self-attention called softmax attention where the
> similarity score is the exponential of the dot product between a query and a key. Given that
> subscripting a matrix with `i` returns the `i`-th row as a vector, we can write a generalized
> attention equation for any similarity function as follows,"

    V'_i = ( Σ_{j=1..N} sim(Q_i, K_j) V_j ) / ( Σ_{j=1..N} sim(Q_i, K_j) )        (Eq. 3)

> "Equation 3 is equivalent to equation 2 if we substitute the similarity function with
> `sim(q, k) = exp( qᵀk / √D )`."

Note what Eq. 3 already contains: **a denominator**. Attention is a weighted *average*, not a
weighted sum, and the normalisation is written explicitly rather than hidden inside the word
"softmax". Every step after this carries the denominator along, and the app must too.

### Step 1 — the only constraint on sim

This is the sentence that licenses everything, and it is a single, narrow requirement:

> "The definition of attention in equation 2 is generic and can be used to define several other
> attention implementations such as polynomial attention or RBF kernel attention. Note that the
> only constraint we need to impose to `sim(·)`, in order for equation 3 to define an attention
> function, is to be **non-negative**. This includes all kernels `k(x, y) : ℝ^{2×F} → ℝ₊`."

Non-negativity is not decoration. If `sim` can go negative the denominator `Σ_j sim(Q_i, K_j)` can
pass through zero, the "weights" stop being a convex combination, and the output is unbounded — it
is not an attention any more. This is the *only* property softmax is being asked to give up its
seat for.

### Step 2 — the kernel form and the exact associativity step

Given a kernel with feature representation `φ(x)`, substitute `sim(q, k) = φ(q)ᵀφ(k)` into Eq. 3:

    V'_i = ( Σ_{j=1..N} φ(Q_i)ᵀ φ(K_j) V_j ) / ( Σ_{j=1..N} φ(Q_i)ᵀ φ(K_j) )      (Eq. 4)

> "and then further simplify it by making use of the associative property of matrix multiplication
> to"

    V'_i = ( φ(Q_i)ᵀ Σ_{j=1..N} φ(K_j) V_jᵀ ) / ( φ(Q_i)ᵀ Σ_{j=1..N} φ(K_j) )      (Eq. 5)

> "The above equation is simpler to follow when the numerator is written in vectorized form as
> follows,"

    ( φ(Q) φ(K)ᵀ ) V  =  φ(Q) ( φ(K)ᵀ V )                                          (Eq. 6)

> "Note that the feature map `φ(·)` is applied rowwise to the matrices `Q` and `K`."

**What actually moved.** Nothing but a pair of brackets. `φ(Q_i)ᵀ` was inside the sum over `j`; now
it is outside. It could move because it does not depend on `j`. That is the entire trick, and it is
worth stating on the card in exactly those words: *the query does not depend on which key you are
looking at, so it can come out of the sum.*

**Why one grouping is O(N²) and the other O(N).** Shapes, with `φ(·)` mapping to `C` dimensions and
`V` having `M`:

| grouping | first product | intermediate | second product | total |
|---|---|---|---|---|
| `(φ(Q) φ(K)ᵀ) V` | `(N×C)(C×N)` = O(N²C) | **N×N** — the attention matrix | `(N×N)(N×M)` = O(N²M) | **O(N² max(C, M))** |
| `φ(Q) (φ(K)ᵀ V)` | `(C×N)(N×M)` = O(NCM) | **C×M** — a fixed-size state | `(N×C)(C×M)` = O(NCM) | **O(N C M)** |

The intermediate object is the whole story. In the left grouping it is `N×N` and grows with the
sentence. In the right grouping it is `C×M` and **does not depend on `N` at all**. The paper's own
accounting:

> "From equation 2, it is evident that the computational cost of softmax attention scales with
> `O(N²)`, where `N` represents the sequence length. The same is true for the memory requirements
> because the full attention matrix must be stored to compute the gradients with respect to the
> queries, keys and values. In contrast, our proposed linear transformer from equation 5 has time
> and memory complexity `O(N)` because we can compute `Σ_{j=1..N} φ(K_j) V_jᵀ` and
> `Σ_{j=1..N} φ(K_j)` once and **reuse them for every query**."

> "For softmax attention, the total cost in terms of multiplications and additions scales as
> `O(N² max(D, M))`, where `D` is the dimensionality of the queries and keys and `M` is the
> dimensionality of the values. On the contrary, for linear attention, we first compute the feature
> maps of dimensionality `C`. Subsequently, computing the new values requires `O(N C M)` additions
> and multiplications."

### The denominator, which most summaries drop

Eq. 5 has **two** regroupings, not one, and the second is the one that gets lost in retellings:

    numerator:    Σ_j φ(Q_i)ᵀ φ(K_j) V_j   →   φ(Q_i)ᵀ ( Σ_j φ(K_j) V_jᵀ )   =  φ(Q_i)ᵀ S
    denominator:  Σ_j φ(Q_i)ᵀ φ(K_j)       →   φ(Q_i)ᵀ ( Σ_j φ(K_j) )        =  φ(Q_i)ᵀ Z

So the model carries **two** running objects: a `C×M` matrix `S` *and* a `C`-vector `Z`. Drop `Z`
and you no longer have an attention — you have an unnormalised accumulator whose output magnitude
grows with sentence length, because every new token adds another `φ(k)vᵀ` term to `S` with nothing
dividing it back down. The normalised weight on key `j` is

    w_ij = φ(Q_i)ᵀφ(K_j) / ( φ(Q_i)ᵀ Z )

which is still per-query — but it is computed from **one shared vector `Z`**, not from `N`
per-query exponentials. That is the precise sense in which the denominator survives the regrouping,
and it is the precise sense in which softmax's does not.

### Step 3 — the feature map they actually use

Two things are ruled out before `elu` arrives, and both matter:

> "The previous analysis does not take into account the choice of kernel and feature function. Note
> that the feature function that corresponds to the **exponential kernel is infinite dimensional,
> which makes the linearization of exact softmax attention infeasible.** On the other hand, the
> **polynomial kernel**, for example, has an exact finite dimensional feature map and has been shown
> to work equally well with the exponential or RBF kernel. The computational cost for a linearized
> polynomial transformer of degree 2 is `O(N D² M)`. This makes the computational complexity
> favorable **when `N > D²`**. Note that this is true in practice since we want to be able to process
> sequences with tens of thousands of elements."

Read that carefully. **`exp` is not rejected as a poor choice; it is rejected as impossible.** There
is no finite `φ` with `φ(q)ᵀφ(k) = exp(qᵀk/√D)`. And the polynomial kernel *does* work exactly — the
degree-2 map is finite — but `C = D²`, so the crossover moves to `N > D²`. The polynomial route is
not wrong, it is just only worth it at length.

Then the actual choice, Eq. 7:

> "For our experiments, that deal with smaller sequences, we employ a feature map that results in a
> **positive similarity function** as defined below,"

    φ(x) = elu(x) + 1                                                              (Eq. 7)

> "where `elu(·)` denotes the exponential linear unit activation function. We **prefer `elu(·)` over
> `relu(·)` to avoid setting the gradients to 0 when `x` is negative.** This feature map results in
> an attention function that requires `O(N D M)` multiplications and additions."

Two reasons, both stated, neither of them "it approximates softmax":

1. **`+1` and the `elu` floor of `−1` make `φ(x) > 0` everywhere** — this is the non-negativity
   requirement from Step 1, satisfied by construction.
2. **`elu` rather than `relu` so the gradient is never exactly zero on the negative side.** `relu(x)+1`
   would also be positive, but flat below zero, and a feature that has died stays dead.

Cost: `C = D`, so `O(NDM)` — the same shape as the value projection. This is the cheapest possible
feature map that still satisfies the constraint.

Their claim for it, stated as a claim and no more: "we show that the feature map of equation 7
performs on par to the full transformer, while significantly reducing the computational and memory
requirements."

### Step 4 — causal masking, and the running sums

Causal masking turns Eq. 3 into a prefix sum (Eq. 8: `Σ_{j=1..i}` in both numerator and
denominator). Linearised, that is Eq. 9:

    V'_i = ( φ(Q_i)ᵀ Σ_{j=1..i} φ(K_j) V_jᵀ ) / ( φ(Q_i)ᵀ Σ_{j=1..i} φ(K_j) )      (Eq. 9)

> "By introducing `S_i` and `Z_i` as follows,"

    S_i = Σ_{j=1..i} φ(K_j) V_jᵀ                                                   (Eq. 10)
    Z_i = Σ_{j=1..i} φ(K_j)                                                        (Eq. 11)

> "we can simplify equation 9 to"

    V'_i = φ(Q_i)ᵀ S_i / ( φ(Q_i)ᵀ Z_i )                                           (Eq. 12)

> "Note that, `S_i` and `Z_i` can be computed from `S_{i−1}` and `Z_{i−1}` in **constant time** hence
> making the computational complexity of linear transformers with causal masking **linear with
> respect to the sequence length**."

`S_i` is `C×M` — for `φ = elu+1`, `D×M`, and with `D = M = d_k` it is a square `d_k × d_k` matrix.
**It is the same size at token 1 as at token 1,000,000.** Causality is free here: the running sum is
*already* a prefix sum, so the causal model is the natural one and the bidirectional model is the
special case, which is the exact inverse of the softmax situation where causality is a mask bolted
on after the fact.

### Step 5 — the transformer is an RNN

> "In literature, transformer models are considered to be a fundamentally different approach to
> recurrent neural networks. However, from the causal masking formulation in §3.3 and the discussion
> in the previous section, it becomes evident that **any transformer layer with causal masking can be
> written as a model that, given an input, modifies an internal state and then predicts an output,
> namely a Recurrent Neural Network (RNN).** Note that, in contrast to Universal Transformers, we
> consider the recurrence **with respect to time and not depth.**"

> "The resulting RNN has two hidden states, namely the **attention memory `s`** and the **normalizer
> memory `z`**."

    s_0 = 0                                                                        (Eq. 13)
    z_0 = 0                                                                        (Eq. 14)
    s_i = s_{i−1} + φ(x_i W_K) (x_i W_V)ᵀ                                          (Eq. 15)
    z_i = z_{i−1} + φ(x_i W_K)                                                     (Eq. 16)
    y_i = f_l( φ(x_i W_Q)ᵀ s_i / ( φ(x_i W_Q)ᵀ z_i ) + x_i )                       (Eq. 17)

Eq. 17 carries the residual `+ x_i` and the feed-forward `f_l` — this is a complete transformer
*layer* rewritten as a recurrence, not just the attention.

And the crucial hedge, in their own words, which the card must not sand off:

> "Note that our formulation does not impose any constraint on the feature function and it can be
> used for representing **any transformer** model, **in theory even the ones using softmax attention.**
> This formulation is a first step towards better understanding the relationship between transformers
> and popular recurrent networks and the processes used for storing and retrieving information."

Softmax attention *is* also expressible as a recurrence — but its state is the growing set of keys
and values, which they build as an explicit baseline (see "stateful-softmax" below) and describe as
**"qualitatively different to our proposed model that has a state with fixed dimensions and computing
the `i`-th state given the previous one has fixed computational cost regardless of `i`."** The RNN
framing is not what makes linear attention fast. *The fixed state size* is.

The inference claim, §3.3.2:

> "when it comes to inference, the **cost per time and memory for one prediction is constant** for our
> model. This means we can simply store the `φ(K_j)V_jᵀ` matrix as an internal state and update it at
> every time step like a recurrent neural network. This results in inference **thousands of times
> faster** than other transformer models."

And the diagnosis of *why* softmax inference is bad, which is the thing the app's counters should
reproduce:

> "during inference the output for timestep `i` is the input for timestep `i+1`. This makes
> autoregressive models impossible to parallelize. Moreover, **the cost per timestep for transformers
> is not constant; instead, it scales with the square of the current sequence length** because
> attention must be computed for all previous timesteps."

### Step 6 — gradient memory, which is a separate contribution

This is the part that gets dropped from every summary, and it is the difference between the idea
working on paper and working on a GPU:

> "**A naive implementation of equation 12, in any deep learning framework, requires storing all
> intermediate values `S_i` in order to compute the gradients. This increases the memory consumption
> by `max(D, M)` times;** thus hindering the applicability of causal linear attention to longer
> sequences or deeper models. To address this, we derive the gradients of the numerator in equation 9
> as cumulative sums. This allows us to compute both the forward and backward pass of causal linear
> attention in **linear time** and **constant memory**."

The gradients, Eqs. 14–16 of the paper (numerator `V̄_i`, loss `L`):

    ∇_{φ(Q_i)} L = ∇_{V̄_i}L · ( Σ_{j=1..i} φ(K_j) V_jᵀ )ᵀ          ← forward cumulative sum
    ∇_{φ(K_i)} L = ( Σ_{j=i..N} φ(Q_j) (∇_{V̄_j}L)ᵀ ) V_i           ← reverse cumulative sum
    ∇_{V_i}    L = ( Σ_{j=i..N} φ(Q_j) (∇_{V̄_j}L)ᵀ )ᵀ φ(K_i)       ← same reverse sum, reused

> "The cumulative sum terms in equations 9, 14-16 are computed in linear time and require constant
> memory with respect to the sequence length. This results in an algorithm with computational
> complexity `O(N C M)` and memory `O(N max(C, M))` for a given feature map of `C` dimensions."

Two precision points on that sentence, because it reads as a contradiction and is not one:

- The **cumulative-sum terms** are constant memory (one `S`, rolled forward then backward).
- The **algorithm** is `O(N max(C, M))` memory, because you still hold `φ(Q)`, `φ(K)`, `V` and the
  output. The win is that the `N×N` attention matrix and the `N` copies of `S` are both gone.

Algorithm 1 makes the mechanism plain: forward is one loop accumulating `S ← S + φ(K_i)V_iᵀ`;
backward is **two** loops, one forward over `i = 1…N` re-accumulating `S` for `∇_{φ(Q)}`, and one
**backward** over `i = N…1` accumulating `S ← S + φ(Q_i)G_iᵀ` for `∇_V` and `∇_{φ(K)}`. `S` is
recomputed rather than stored — the same recomputation trade the Sparse Transformer made, applied to
the state instead of the activations.

And the sentence that says how much engineering this cost:

> "The constant memory gradient computation of equations 14-16 is implemented in approximately
> **200 lines of CUDA code**."

Also worth noting from the supplementary: "The gradient with respect to the denominator and the
fraction are efficiently handled by autograd" — only the numerator needed a custom kernel.

## Numbers that matter

### The worked example the lesson uses, and what it actually shows

Scalars, `d_k = 1`, `φ = identity`. `q = 2`, keys `[0.5, 1.0, 1.5]`, values `[10, 20, 30]`.
Scores `q·k_j = [1, 2, 3]`.

| route | computation | result |
|---|---|---|
| direct — score each key, then weight the values | `1×10 + 2×20 + 3×30` | **140** |
| regrouped — build the state first, then read it | `S = 0.5×10 + 1.0×20 + 1.5×30 = 70`, then `q×S = 2×70` | **140** |

They agree exactly, and they agree for one reason only: `q` is a common factor of every term, so it
can be pulled out. That *is* Eq. 6 at `N = 3`, `C = M = 1`.

**But 140 is the numerator, not the attention output.** The card must say this out loud, because the
whole point of the concept is that the normaliser rides along:

| | numerator | denominator | `V'` |
|---|---|---|---|
| direct | `1×10 + 2×20 + 3×30 = 140` | `1 + 2 + 3 = 6` | **23.33** |
| regrouped | `φ(q)ᵀS = 2 × 70 = 140` | `φ(q)ᵀZ = 2 × (0.5+1.0+1.5) = 2 × 3 = 6` | **23.33** |

Both routes agree on the numerator **and** on the denominator, because `Z` regroups exactly the way
`S` does. The lesson's 140 is right; it is just half the equation, and the missing half is the half
that softmax cannot give up.

With the paper's actual feature map, `φ(x) = elu(x) + 1 = x + 1` for positive `x`:
`φ(q) = 3`, `φ(k) = [1.5, 2.0, 2.5]`. Numerator `3×1.5×10 + 3×2×20 + 3×2.5×30 = 390`; regrouped
`S = 1.5×10 + 2×20 + 2.5×30 = 130`, `φ(q)×S = 3×130 = 390` — agreeing again. Denominator
`3 × 6 = 18`, so `V' = 21.67`. The routes agree under *any* `φ`; the answer changes, the agreement
does not.

**Turn softmax on and both facts break.** `softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]`, output
`0.0900×10 + 0.2447×20 + 0.6652×30 = 25.75`.

| | weight on k=0.5 | on k=1.0 | on k=1.5 | output |
|---|---|---|---|---|
| linear (`φ = id`), normalised | 0.167 | 0.333 | **0.500** | 23.33 |
| softmax | 0.090 | 0.245 | **0.665** | 25.75 |

There is now **no `S` that works**. The precomputed `S = 70` is still a perfectly good number, but
`q × 70` is not 25.75 and no rescaling of it will be, because the softmax weights are not
proportional to `k_j` — they are proportional to `exp(q·k_j)`, and *that* depends on `q` inside the
exponential, so `q` cannot come out of the sum. The two routes agreeing at 140 and disagreeing at
25.75 is the concept, in four numbers.

### The paper's reported results

**Synthetic (§4.1).** Sequence-duplication copy task, max length 128, 10 symbols, 4 layers, 8 heads,
batch 64, RAdam at `1e-3` → `1e-4` after 3000 updates. "linear converges smoothly and reaches a
lower loss than lsh due to the lack of noise introduced by hashing. In particular, **it reaches the
same loss as softmax.**"

**Benchmark (§4.1.2).** `N ∈ {2⁹ … 2¹⁶}`, peak GPU memory and time for attention + gradients, batch
size scaled inversely with `N`, reported per sample. **NVidia GTX 1080 Ti, 11GB.** The number that
frames everything: "This results in a **maximum sequence length of 4,096 elements for softmax** and
**16,384 for lsh-4 and lsh-8**." Linear is not length-capped in that sentence. "Our method is faster
and requires less memory than the baselines for every configuration."

**Image generation — MNIST (Table 1).** 8 layers, 8 heads, embedding 256 (**32 dims per head**),
FFN 4×, mixture of 10 logistics, RAdam `1e-4`, 250 epochs, **batch size 10 for all methods**, 784
pixels.

| method | bits/dim | images/sec | speedup |
|---|---|---|---|
| Softmax | **0.621** | 0.45 | 1× |
| LSH-1 | 0.745 | 0.68 | 1.5× |
| LSH-4 | 0.676 | 0.27 | 0.6× |
| **Linear (ours)** | 0.644 | **142.8** | **317×** |

"linear transformers achieve almost the same performance, in terms of final perplexity, as softmax
transformers while being able to generate images more than 300 times faster. This is achieved due to
the low memory requirements of our model, which is able to **simultaneously generate 10,000 MNIST
images with a single GPU**. In particular, **the memory is constant with respect to the sequence
length because the only thing that needs to be stored between pixels are the `s_i` and `z_i` values**
as described in equations 15 and 16."

**Image generation — CIFAR-10 (Table 2).** 16 layers, same per-layer config, **7 days of training on
one GPU for every model**, NVidia P40 24GB. Batch size **1 for softmax** (it does not fit otherwise)
and **4 for linear and Reformer**.

| method | bits/dim | images/sec | speedup |
|---|---|---|---|
| Softmax | 3.47 | 0.004 | 1× |
| LSH-1 | 3.39 | 0.015 | 3.75× |
| LSH-4 | 3.51 | 0.005 | 1.25× |
| **Linear (ours)** | **3.40** | **17.85** | **4,462×** |

"**for every image generated by the softmax transformer, our method can generate 4,460 images.**"
The bits/dim win is explicitly a budget artefact, not a modelling one: "Our linear transformer
**completes 3 times more epochs than softmax**, which results in better perplexity."

**Speech recognition (Table 3).** 80-hour WSJ, 40-dim mel filterbanks, CTC loss, **non**-autoregressive
(a distribution over phonemes per frame), ~800 frames average and 2,400 max, 9 layers × 6 heads.

| method | validation PER | time/epoch (s) |
|---|---|---|
| Bi-LSTM | 10.94 | 1047 |
| **Softmax** | **5.12** | 2711 |
| LSH-4 | 9.33 | 2250 |
| **Linear (ours)** | 8.08 | **824** |

"linear outperforms the recurrent network baseline and Reformer both in terms of performance and
speed by a large margin… Note that the **softmax transformer achieves lower phone error rate in
comparison to all baselines**, but is significantly slower. In particular, linear transformer is more
than **3× faster per epoch**." And in the supplementary, flatly: "**softmax outperforms significantly
both Reformer and linear in terms of convergence** … **Even though softmax attention is better in
this task**, we observe that linear transformers significantly outperform Reformer."

**The honest baseline, supplementary §C.1 — "stateful-softmax".** They build the fair comparison
themselves: a softmax transformer that caches keys and values instead of recomputing them.

| | MNIST img/s | CIFAR-10 img/s |
|---|---|---|
| Softmax (naive, recompute) | 0.45 | 0.004 |
| **Stateful-softmax (KV cache)** | **7.56** (16.8×) | **0.32** (80×) |
| Linear (ours) | 142.8 (317×) | 17.85 (4,462×) |

"stateful-softmax is significantly faster than vanilla transformers. However, its complexity is still
quadratic with respect to the sequence length and **our formulation is more than 50× faster for
CIFAR-10.**" So the real, like-for-like autoregressive speedup on CIFAR-10 is **≈56×**, not 4,462×.
The 4,462× is measured against a baseline that recomputes the entire prefix at every pixel.

**Latency, supplementary §C.2, batch size 1.** Throughput and latency are different questions:

| CIFAR-10, one image | CPU (s) | GPU (s) |
|---|---|---|
| Softmax | 8651.4 | 300.1 |
| Stateful-softmax | 71.9 | 70.4 |
| LSH-1 | 2318.9 | 221.6 |
| **Linear (ours)** | **45.1** | 61.3 |

"all methods underutilize the GPU… The proposed linear transformer is faster than all the methods and
in particular it is **almost 6.6× faster than softmax transformers for generating an image on
CIFAR-10**" — that 6.6× is each method's *best* time (softmax 300.1 s on GPU vs linear 45.1 s on
CPU). Against stateful-softmax at batch 1 it is ~1.6×. And the delightful detail: "our linear
autoregressive transformer is **the only method that is faster on the CPU than on the GPU in every
case**. This is due to the fact that computing the attention as an RNN has such a low cost that the
main computational bottleneck becomes the inevitable outer loop over the sequence."

**The spread the card must print together: 4,462× / 56× / 6.6× / 1.6×, all from the same paper, on
the same task.** Which one is true depends entirely on what you compare against and whether you are
measuring throughput or latency.

### Translated to this app — `d_k = 8`, `4` heads, `2` blocks, `T = 16`

Computed from the shapes, not estimated. Per head, per block:

| | at `T` tokens | at `T = 16` |
|---|---|---|
| softmax KV cache | `2·T·d_k` = `16T` numbers | **256** |
| linear state `S` + `Z` | `d_k² + d_k` = `64 + 8` | **72**, constant |

Across all 8 head-instances (4 heads × 2 blocks): cache **2,048** vs state **576**.
Crossover: `16T = 72` → **`T = 4.5`**. From the fifth token onward the fixed state is already
smaller, and the gap widens forever after.

Compute, multiplications in the mixing step:

| | formula | at `T = 16` |
|---|---|---|
| softmax (`QKᵀ` + `AV` over causal cells) | `2 · T(T+1)/2 · d_k` = `T(T+1)d_k` | **2,176** |
| linear (write `d_k²` + read `d_k²` per token) | `2 · T · d_k²` | **2,048** |

Crossover: `T(T+1)d_k = 2T d_k²` → **`T = 2d_k − 1 = 15`**. At this app's sentence length the two are
within 6% of each other — **linear attention buys the reader almost nothing in FLOPs here**, and would
lose outright on a 10-token sentence. Memory crosses at token 5; compute crosses at token 15. Both
numbers should be on screen, because the asymmetry is the honest lesson: *the state is small long
before it is fast*, and this is the same `N > D²` caveat the paper raises for the polynomial kernel,
in miniature.

### The seam, as it stands

`app/model/mixers.js`. `stateMixer({ write, decay, phi })` is already this concept when
`write: "add"`:

- line 79 — `S[a][b] = S[a][b]*g + target*k[b]` with `target = V[i][a]` for `"add"`, `g = 1`. That is
  Eq. 15, `s_i = s_{i−1} + φ(k_i) v_iᵀ`, exactly.
- line 81 — `norm[b] = norm[b]*g + k[b]`. That is Eq. 16, `z_i = z_{i−1} + φ(k_i)`. The `Z` term is
  already implemented; the app just does not draw it.
- line 86 — `o[a] = dot(S[a], q) / (Math.abs(z) + 1e-6)`. That is Eq. 12, with a guard.
- line 89 — `snapshots.push(...)` already stores `S` after every token. **The `d_k × d_k` heatmap is
  free; the data is there.**
- line 92 — `reads: T`. The counter already reports one read per token rather than a growing list.

**Two deviations from the paper to fix or to state.** (1) The default `phi` is
`x => Math.max(x, 0) + 0.01` — that is `relu(x) + 0.01`, not `elu(x) + 1`. It satisfies
non-negativity but it is precisely the choice the paper rejects, "to avoid setting the gradients to 0
when `x` is negative". For this card, pass `phi: x => (x > 0 ? x : Math.exp(x) - 1) + 1`. (2) The
`Math.abs(z)` on line 86 silently rescues a negative denominator. With a valid `φ` that case cannot
arise; with an invalid one it should be *shown*, not hidden. See interaction 5.

## What the live view must let the reader do

Five interactions. Each one changes a number the model actually computes.

**1. The two routes, side by side, with a softmax toggle.**

Three columns on the lesson's own numbers (`q = 2`, `k = [0.5, 1.0, 1.5]`, `v = [10, 20, 30]`),
editable by the reader:

| | direct route | regrouped route |
|---|---|---|
| numerator | `1×10 + 2×20 + 3×30 = 140` | `S = 70`, `q×S = 140` |
| denominator | `1 + 2 + 3 = 6` | `Z = 3`, `q×Z = 6` |
| `V'` | **23.33** | **23.33** |

with a badge reading **`agree — difference 0.00`**. Flip the softmax toggle and the same panel
becomes:

| | softmax route | regrouped route |
|---|---|---|
| `V'` | **25.75** | **no S exists** |

The displayed number that proves it: **`|direct − regrouped| = 0.000000` with softmax off, and the
regrouped column going blank with softmax on** — not showing a wrong value, showing *nothing*, because
the operation is undefined. Underneath, the two weight vectors so the reader sees where the answer
moved: `[0.167, 0.333, 0.500]` linear versus `[0.090, 0.245, 0.665]` softmax. The last key's share
going from 50% to 66.5% is what "softmax is sharper" means numerically.

Let the reader edit `q`. The agreement must hold at every value they type — that is what "exact"
means, and one worked example does not demonstrate it.

**2. The state matrix as a live `d_k × d_k` heatmap, with a size counter that never moves.**

Drive it from `snapshots[i]`, which the mixer already returns. A token slider `i = 1…16` steps the
heatmap through `S_1 … S_16`; each step visibly adds the rank-1 outer product `φ(k_i) v_iᵀ` — one
row/column pattern superimposed on what is already there. Draw `Z_i` as an 8-cell strip beside it,
since the denominator is a first-class part of the state and the app currently hides it.

Two counters directly above the heatmap, updating as the slider moves:

    tokens seen           1  2  3  …  16
    state size          72 72 72  …  72       ← never changes
    softmax KV cache    16 32 48  …  256      ← 16 per token

The proof number is the **72 that does not move while the sentence grows to 16 tokens and the cache
reaches 256**. Let the reader lengthen the sentence and watch the second row climb past the first at
**token 5** while the first stays put. This is the single most important picture on the card.

**3. Softmax output versus linear output, on the reader's own sentence, quantified.**

Run `softmaxMixer()` and `stateMixer({ write: "add", phi: elu1 })` on the *same* seeded weights and
the *same* sentence — the seam is designed for exactly this — and display, per token:

- **Cosine similarity and L2 distance** between the two 32-dim outputs, per token and averaged. One
  headline number: *"the two models disagree by X on average"*.
- **Which words each attends to.** For softmax, the top-3 of `weights[i]` directly. For the linear
  path, compute the implied weights `w_ij = φ(q_i)ᵀφ(k_j) / (φ(q_i)ᵀ Z_i)` — these are recoverable
  even though the model never forms them, and showing them is the honest way to compare. Display both
  top-3 lists side by side over the reader's words.
- **Rank agreement**: how many of softmax's top-3 keys appear in linear's top-3. This is the number
  that carries the concept — expect it to be high for the top key and to degrade below it, because
  linear attention's weights are a smooth positive function of the query-key dot product with no
  exponential to sharpen them.
- **Attention entropy** for each row, both paths. Linear's entropy will be systematically higher.
  That single scalar is "linear attention cannot be as peaky", measured rather than asserted.

Hard caveat to print in the panel: the weights are seeded and **untrained**, so *which* words come out
on top is noise. What is not noise is the **shape** — how spread out the two distributions are, and
how much the outputs differ. Read the entropy, not the words.

**4. The two counters that decide whether any of this was worth it.**

A live pair, recomputed as the reader adds or deletes words:

    memory   softmax  2·T·d_k = 256 numbers/head     linear  d_k² + d_k = 72     crossover at T = 4.5
    compute  softmax  T(T+1)·d_k = 2,176 mults       linear  2T·d_k² = 2,048     crossover at T = 15

with a mini-plot of both curves against `T` out to a few hundred, the reader's `T` marked. At `T = 16`
the compute bars are **nearly equal** — and the card should say so plainly rather than hide it: *at
your sentence length linear attention is not faster. It becomes faster at 15 tokens and stays faster
forever; the paper measured it at 3,072.* Add a second row of the same counters at CIFAR-10's
`N = 3,072`, `d_k = 32`: softmax ≈ 302M mults against linear ≈ 6.3M, a **48× gap**. The reader's ~1×
and the paper's 48× side by side is the same honest contrast the Sparse Transformer card draws, and
for the same reason — asymptotics only pay at length.

**5. The `φ` selector, and the non-negativity requirement made visible.**

A dropdown over four feature maps, each recomputing the model:

| `φ` | positive? | gradient below 0? | what happens |
|---|---|---|---|
| `identity` | **no** | yes | denominator can hit or cross zero |
| `relu(x)` | ≥ 0 | **no** | dead features, and `φ(k) = 0` writes nothing |
| `relu(x) + 0.01` (the seam's current default) | yes | **no** | works, but the paper's rejected choice |
| **`elu(x) + 1`** (the paper) | yes | yes | Eq. 7 |

The displayed number that proves the constraint: **a count of `(i, head, block)` positions where the
denominator `φ(q_i)ᵀ Z_i ≤ 0`**, plus the minimum `|z|` observed. With `elu+1` that count is **0** by
construction. With `identity` it is nonzero, and the output at those positions is garbage — which the
seam currently launders through `Math.abs(z) + 1e-6` on line 86. For this view, bypass the guard and
show the raw value going through zero. That is §3.2's "the only constraint … is to be non-negative",
turned into an integer the reader can watch change.

## What the source does *not* establish

- **It does not claim parity with softmax attention.** The abstract's word is "**similar**", and the
  body is more careful than that. On MNIST linear is **worse** (0.644 vs 0.621 bits/dim). On WSJ ASR
  it is **much worse** — 8.08 vs 5.12 PER, a 58% relative increase in error, and the authors say so:
  "softmax outperforms significantly both Reformer and linear in terms of convergence… Even though
  softmax attention is better in this task". The one place linear wins on quality, CIFAR-10 (3.40 vs
  3.47), is a **fixed 7-day wall-clock budget** in which linear completed **3× more epochs** at a
  batch size of 4 against softmax's batch size of 1. That is a throughput result wearing a perplexity
  result's clothes. **No equal-epoch, equal-batch quality comparison at scale appears in the paper.**
- **It does not approximate softmax, and does not claim to.** "the feature function that corresponds
  to the exponential kernel is infinite dimensional, which makes the linearization of exact softmax
  attention **infeasible**." `elu(x)+1` is chosen for positivity and non-vanishing gradients, not for
  fidelity to `exp`. Anyone reading this card as "a fast approximation to softmax" has the
  relationship backwards: it is a **different attention function** that happens to be cheap.
- **The 4,000× is one number from a family that spans four orders of magnitude.** Same task, same
  paper: **4,462×** against naive recompute-everything softmax at unequal batch sizes; **≈56×**
  against their own stateful-softmax KV-cache baseline; **6.6×** on single-image latency comparing
  each method's best device; **~1.6×** on single-image latency against stateful-softmax. The headline
  compares against a baseline nobody deploys. The 56× is the real number and is still excellent.
- **No language modelling experiment.** The tasks are: a synthetic copy task (128 symbols), MNIST
  pixel generation (784), CIFAR-10 pixel generation (3,072), and WSJ phoneme recognition (~800
  frames). There is **no enwik8, no WikiText, no perplexity on text of any kind**. Every claim about
  this mechanism on language is extrapolation from images and speech — and the one non-image task in
  the paper is the one where it loses to softmax by the widest margin.
- **The sequences are short by the paper's own standard.** They say so: "For our experiments, that
  deal with **smaller sequences**, we employ a feature map…". The regime where `O(N)` decisively beats
  `O(N²)` is barely entered; the benchmark plot goes to `2¹⁶` but no *trained model* does.
- **"Constant memory" needs its qualifier.** Inference state is genuinely constant. Training memory is
  `O(N max(C, M))` — linear, not constant; it is the *cumulative-sum terms* that are constant, and
  only after ~200 lines of hand-written CUDA. In plain PyTorch the naive version costs `max(D, M)`
  times more memory, which the authors state is enough to hinder "the applicability of causal linear
  attention to longer sequences or deeper models". **The idea is `O(N)`; the implementation is a
  kernel.**
- **"Transformers are RNNs" is weaker than the title.** The formulation "can be used for representing
  *any transformer* model, in theory even the ones using softmax attention" — so being an RNN is not
  what buys the speed. Their own stateful-softmax is an RNN too, with a state that grows. The property
  that matters is the **fixed-size** state, and the title does not say that.
- **The app proves none of the speed.** JavaScript in a browser at `T = 16` with `d_k = 8` is on the
  wrong side of the compute crossover by a hair and would be on the wrong side outright at `T = 10`.
  The app demonstrates **exact algebraic agreement**, **fixed state size**, and **the shape of the
  divergence from softmax**. It does not and cannot demonstrate 4,000×.
- **The model is untrained.** Seeded weights shaped by a rule. Read the geometry and the counters;
  never read the predictions.

## Leaves behind

**Backward — why softmax's denominator is what blocks the regrouping.**

Every card before this one accepted the `N×N` score matrix and argued about which cells to compute
(Sparse Transformer: skip most of them; Transformer-XL: reuse them across segments). This is the
first card that argues the matrix **should never be built**. The obstacle is precisely the softmax,
and it is worth being exact about *which part* of the softmax:

Eq. 5 works because `φ(Q_i)ᵀ` is a factor of every term in the sum over `j`, so it slides out. Softmax
attention has `sim(q, k) = exp(qᵀk/√D)`, and **`exp(qᵀk)` does not factor into a `q`-only term times a
`k`-only term.** If the exponent were `q + k` it would — `exp(q+k) = exp(q)·exp(k)` — but it is an
*inner product*, and the exponential of an inner product has no finite factorisation. The paper's
statement of this is the infinite-dimensionality remark in §3.2.1.

The denominator is the same obstacle, seen from the other side, and it is the sharper way to say it.
In softmax the normaliser is

    Σ_{j=1..N} exp(q_iᵀk_j / √D)

— a **different scalar for every query**, each one requiring all `N` keys, each one impossible to
share, because `q_i` is trapped inside the exponential. That single shared quantity is what makes the
weights on a row sum to 1 and compete with each other: raising one key's score *lowers* every other
key's weight, through the denominator. **Softmax's expressive power and softmax's quadratic cost are
the same fact.** The competition between keys is implemented by a per-query normaliser, and a
per-query normaliser cannot be precomputed.

In the kernel form, the denominator becomes `φ(Q_i)ᵀ Z` — the query comes *out*, and one shared
`Z = Σ_j φ(K_j)` serves every query in the sentence. The regrouping is not a trick applied to
attention; it is the definition of what makes attention linearisable. Give up the exponential and you
give up per-query competition, and in exchange the whole prefix collapses into one fixed-size object.
The `140 = 140` / `25.75 ≠ anything` pair in interaction 1 is that trade, in four numbers, on the
reader's screen.

There is a second backward thread. The Sparse Transformer bought `O(N√N)` by legislating *which* keys
to read, from the indices alone, before seeing the data. Linear attention reads **every** key and pays
`O(N)`, which is asymptotically better than both. The price is not paid in coverage; it is paid in
**resolution** — every key is read, but they are read through a `d_k × d_k` bottleneck that can no
longer tell them apart sharply.

**Forward — an add-only state cannot take anything back.**

Eq. 15 is `s_i = s_{i−1} + φ(k_i) v_iᵀ`. Every operation is an addition. There is no term anywhere in
the recurrence that can subtract, overwrite, or forget. Two consequences, and they are the next two
cards:

1. **The state cannot correct an association it already wrote.** Suppose token 3 writes `(k, v)` and
   token 11 arrives wanting to bind that same `k` to a different `v'`. The add rule writes
   `φ(k)v'ᵀ` on top, and a read at `φ(k)` now returns a **superposition of both values**, weighted by
   `‖φ(k)‖²` each — not `v'`. The state has no way to express "not that any more". This is exactly the
   gap the **delta rule** closes two concepts later: `S += (v − S φ(k)) φ(k)ᵀ` — read what the state
   currently returns for this key, and write only the *difference*. Line 78 of `mixers.js` already has
   it (`target = V[i][a] − cur[a]`); this card is the one that explains what the `add` branch is
   missing. And the gated rule after that adds the other missing operation, decay — a way for the past
   to fade rather than pile up.
2. **Exact retrieval is gone, and the capacity is countable.** `S` is `d_k × d_k`, so it has rank at
   most `d_k`. It can hold at most `d_k` linearly independent key→value associations without crosstalk
   — **8 in this app, at any sentence length.** Softmax attention with sharp scores does something a
   fixed-rank state cannot: it performs a near-exact lookup by *selecting* one key from a list it has
   physically kept. Linear attention threw the list away at the first token. Everything the model will
   ever know about token 3 has already been summed into an 8×8 matrix, and the only question left is
   whether it can be read back out. **The paper does not make this capacity claim** — it is the
   pressure the next cards respond to, and the reason the timeline does not end here.

The pressure this card names is **compressing the past**. The Sparse Transformer's pressure was *how
many comparisons*; this one is *how much do you keep*. Softmax keeps everything and pays `O(N)` per
step forever. Linear attention keeps a fixed `d_k × d_k` summary and pays `O(1)`. Every mechanism
after this one accepts the fixed budget and argues about the **write rule** — what to put in the
matrix, and what to take back out.
