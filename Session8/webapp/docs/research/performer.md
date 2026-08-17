# Concept 10 — Performer, FAVOR+

**Card id:** `performer` · **Date:** 2020-09-30 (arXiv v1) · **Pressure:** how many comparisons

## What was read

- [arXiv:2009.14794](https://arxiv.org/abs/2009.14794), Choromanski, Likhosherstov, Dohan, Song, Gane,
  Sarlos, Hawkins, Davis, Mohiuddin, Kaiser, Belanger, Colwell, Weller (Google / Cambridge / DeepMind /
  Alan Turing Institute) — *Rethinking Attention with Performers*. ICLR 2021.
- Abstract page for the version history, [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.14794) for a
  first pass, and then **the LaTeX source pulled from `arxiv.org/e-print/2009.14794`** — directory
  `Performers - Fast Transformers V5/`, files `intro_related_work.tex`, `algorithm.tex`, `theory.tex`,
  `experiments.tex`, `appendix.tex`, `main.bbl`. Every quotation below is the authors' own wording out
  of that source, not a summariser's paraphrase, and not recall.
- Version history: **v1 30 Sep 2020**, v2 16 Feb 2021, v3 9 Mar 2021, v4 19 Nov 2022. The timeline uses
  v1. **Caveat worth printing:** the e-print tarball serves the *latest* version, so the quotations are
  v4 text. The v1 abstract was fetched separately and is substantively identical — FAVOR+ is named and
  expanded as "Fast Attention Via positive Orthogonal Random features" already in v1, so the naming and
  the central claim are not a later retrofit. Section-level wording in v1 was not diffed; if a quote
  ever becomes load-bearing for a priority argument, diff it first.
- `main.bbl` confirms the backward link: the citation key `trans-rnns` used in the introduction's
  criticism resolves to **Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs", abs/2006.16236** —
  the plain-linear-attention concept immediately upstream on this timeline. The criticism is aimed at it
  by name.

## The mechanism, precisely

### The gap it names in plain linear attention — the exact framing

This is the part most summaries get backwards, so it is quoted in full. Two consecutive paragraphs of
§1, the first ending on the citation to `trans-rnns` (= arXiv:2006.16236):

> "There is also a long line of research on using dense attention matrices, but defined by low-rank
> kernels substituting softmax (Katharopoulos et al., 2020; Shen et al., 2018). Those methods critically
> rely on kernels admitting explicit representations as dot-products of finite positive-feature vectors."

> "**The approaches above do not aim to approximate regular attention, but rather propose simpler and
> more tractable attention mechanisms**, often by incorporating additional constraints (e.g. identical
> query and key sets as in Kitaev et al.), or by trading regular with sparse attention using more layers.
> Unfortunately, **there is a lack of rigorous guarantees for the representation power produced by such
> methods**, and sometimes the validity of sparsity patterns can only be verified empirically through
> trial and error…"

That is the whole thesis in two sentences. Linear attention picks a feature map `φ` — `elu(x)+1` in
Katharopoulos et al. — writes `A(i,j) = φ(q_i)ᵀφ(k_j)`, and the resulting kernel is *whatever that
expression happens to be*. It is not softmax and no one claimed it was. **Performer inverts the
question: fix the target kernel to be softmax, and go find a `φ` whose expectation reproduces it.**

The response paragraph, verbatim:

> "In response, we introduce the first Transformer architectures, *Performers*, capable of **provably**
> accurate and practical estimation of regular (softmax) full-rank attention, but of only linear space
> and time complexity and **not relying on any priors** such as sparsity or low-rankness."

> "Consequently, Performers are the first linear architectures **fully compatible** (via small amounts of
> fine-tuning) with regular Transformers, providing strong theoretical guarantees: unbiased or
> nearly-unbiased estimation of the attention matrix, uniform convergence and lower variance of the
> approximation."

**FAVOR+ = Fast Attention Via positive Orthogonal Random features.** The `+` is part of the acronym as
printed; the paper also parses the name piecewise in §2 — "The above scheme constitutes the FA-part of
the FAVOR+ mechanism. The remaining OR+ part…", then "The above constitutes the R+ part of the FAVOR+
method. It remains to explain the O-part." So: **FA** = the fast attention reordering, **R+** = positive
random features, **O** = orthogonality.

### The FA part — kernelizable attention and the associativity reorder

Baseline, §2.1 (`d` here is the dimension of the rows of `Q, K, V`):

    Att↔(Q, K, V) = D⁻¹ A V,    A = exp(QKᵀ / √d),    D = diag(A 1_L)

> "Time and space complexity of computing (1) are `O(L²d)` and `O(L² + Ld)` respectively, **because A has
> to be stored explicitly**."

Causal is the same with `Ã = tril(A)`, `D̃ = diag(Ã 1_L)`, where "`tril(·)` returns the lower-triangular
part of the argument matrix including the diagonal" — i.e. exactly this app's causal mask.

The generalization, §2.2. FAVOR+ works for any `A(i,j) = K(q_iᵀ, k_jᵀ)` with a kernel
`K : R^d × R^d → R₊` defined for a "(usually randomized) mapping" `φ : R^d → R₊^r`:

    K(x, y) = E[ φ(x)ᵀ φ(y) ]                                        (Eq. 3)

Note two things the paper is deliberate about and that carry the whole design: the kernel maps into
**`R₊`** (non-negative), and `φ` maps into **`R₊^r`** (non-negative). Both are stated as codomains, not
as a hope.

Then with `Q′, K′ ∈ R^{L×r}` whose rows are `φ(q_iᵀ)ᵀ` and `φ(k_iᵀ)ᵀ`:

    Ât↔(Q, K, V) = D̂⁻¹ ( Q′ ( (K′)ᵀ V ) ),    D̂ = diag( Q′ ( (K′)ᵀ 1_L ) )

> "Here `Ât↔` stands for the approximate attention and **brackets indicate the order of computations**.
> It is easy to see that such a mechanism is characterized by space complexity `O(Lr + Ld + rd)` and time
> complexity `O(Lrd)` as opposed to `O(L² + Ld)` and `O(L²d)` of the regular attention."

The brackets *are* the trick, and it is the same trick as plain linear attention: multiply `K′ᵀV` first
(an `r × d` object, no `L` in it), then apply `Q′`. The `L × L` matrix is never formed. What is new is
everything downstream of that.

The paper then poses its own two open questions, which is a clean way to structure the card:

> "**(1)** How expressive is the attention model defined in Equation 3, and in particular, can we use it
> in principle to approximate regular softmax attention? **(2)** How do we implement it robustly in
> practice, and in particular, can we choose `r ≪ L` for `L ≫ d` to obtain desired space and time
> complexity gains?"

### The general feature-map template

§2.3. For functions `f₁,…,f_l : R → R`, a function `h : R^d → R`, and vectors
`ω₁,…,ω_m ~iid D` for some `D ∈ P(R^d)`:

    φ(x) = h(x)/√m · ( f₁(ω₁ᵀx), …, f₁(ω_mᵀx), …, f_l(ω₁ᵀx), …, f_l(ω_mᵀx) )        (Eq. 5)

So the feature vector has length `r = l · m`: `m` random projections, each pushed through `l` functions.
The paper's catalogue of instantiations:

| `h(x)` | `l` | `f`s | `D` | kernel |
|---|---|---|---|---|
| `1` | 1 | `f₁ = sgn` | `N(0, I_d)` | angular / PNG kernels |
| `1` | 2 | `sin, cos` | `N(0, I_d)` | Gaussian kernel `K_gauss` (Rahimi–Recht) |
| `exp(+‖x‖²/2)` | 2 | `sin, cos` | `N(0, I_d)` | **softmax, trigonometric — `ŜM_m^trig`** |
| `exp(−‖x‖²/2)` | 1 | `exp` | `N(0, I_d)` | **softmax, positive — `ŜM_m^+`** |
| `(1/√2)·exp(−‖x‖²/2)` | 2 | `exp(u), exp(−u)` | `N(0, I_d)` | **softmax, hyperbolic — `ŜM_m^hyp+`** |

The softmax kernel is defined without the scale factor:

    SM(x, y) ≝ exp(xᵀy)

> "In the above, **without loss of generality, we omit `√d`-renormalization since we can equivalently
> renormalize input keys and queries**."

That sentence is an instruction for the implementation, not a throwaway — see the live-view section.

### How *not* to do it — the trigonometric features and why they were rejected

The trig route exists and is derived first, because it is the obvious one. Since

    SM(x, y) = exp(‖x‖²/2) · K_gauss(x, y) · exp(‖y‖²/2)

you get an unbiased softmax estimator for free by taking the classical Rahimi–Recht random Fourier
features for the Gaussian kernel and multiplying by the two `exp(+‖·‖²/2)` factors:
`h(x) = exp(‖x‖²/2)`, `l = 2`, `f₁ = sin`, `f₂ = cos`. This is `ŜM_m^trig`. **It is unbiased.** It is
also the thing the paper spends its central paragraph destroying:

> "There is however a caveat there. The attention module from (1) constructs for each token, **a convex
> combination of value-vectors** with coefficients given as corresponding renormalized kernel scores.
> **That is why kernels producing non-negative scores are used.** Applying random feature maps with
> potentially negative dimension-values (sin/cos) leads to unstable behaviours, especially when kernel
> scores close to `0` (which is the case for many entries of `A` corresponding to low relevance tokens)
> are approximated by estimators with large variance in such regions. This results in abnormal
> behaviours, e.g. **negative-diagonal-values renormalizers `D⁻¹`**, and consequently either completely
> prevents training or leads to sub-optimal models."

And the historical claim they attach to it:

> "We demonstrate empirically that this is what happens for `ŜM_m^trig` and provide detailed theoretical
> explanations showing that **the variance of `ŜM_m^trig` is large as approximated values tend to `0`**…
> **This is one of the main reasons why the robust random feature map mechanism for approximating regular
> softmax attention was never proposed.**"
>
> "We propose a robust mechanism in this paper. Furthermore, **the variance of our new unbiased positive
> random feature map estimator tends to `0` as approximated values tend to `0`**."

Two distinct failures, and the card must keep them apart because they compound:

1. **Sign.** `sin`/`cos` take negative values, so an individual estimate of `A(i,j)` can come out
   negative. Attention then stops being a convex combination, and the row normalizer `D` — a *sum* of
   those estimates — can come out negative or near zero. Dividing by it is the "abnormal behaviour".
2. **Variance where it hurts.** The relative error of `ŜM^trig` explodes exactly in the regime that
   dominates the matrix — the many near-zero entries for irrelevant token pairs. This is the failure
   that survives even if you got lucky on signs.

Unbiasedness is therefore *not sufficient*, and that is the single sharpest technical point on this
card. Both estimators are unbiased. One of them is unusable.

### Lemma 1 — the positive random features

The construction, verbatim (Lemma 1, "Positive Random Features (PRFs) for Softmax"). For
`x, y ∈ R^d`, `z = x + y`:

    SM(x, y) = E_{ω ~ N(0, I_d)} [ exp(ωᵀx − ‖x‖²/2) · exp(ωᵀy − ‖y‖²/2) ]
             = Λ · E_{ω ~ N(0, I_d)} [ cosh(ωᵀz) ],      Λ = exp( −(‖x‖² + ‖y‖²)/2 )

> "Consequently, softmax-kernel admits a positive random feature map unbiased approximation with
> `h(x) = exp(−‖x‖²/2)`, `l = 1`, `f₁ = exp` and `D = N(0, I_d)` **or**:
> `h(x) = (1/√2)·exp(−‖x‖²/2)`, `l = 2`, `f₁(u) = exp(u)`, `f₂(u) = exp(−u)` and the same `D`
> (**the latter for further variance reduction**). We call related estimators: `ŜM_m^+` and `ŜM_m^hyp+`."

Substituting into Eq. 5, the map to implement is:

    φ⁺(x) = exp(−‖x‖²/2)/√m · ( exp(ω₁ᵀx), exp(ω₂ᵀx), …, exp(ω_mᵀx) ) ∈ R_{>0}^m

Every coordinate is `exp` of something, hence **strictly positive**, hence every estimate
`φ⁺(x)ᵀφ⁺(y)` is strictly positive and every row sum is strictly positive. Failure mode (1) is gone by
construction, not by clipping.

Note the sign flip on `h` against the trig version: **`exp(−‖x‖²/2)`, not `exp(+‖x‖²/2)`**. This is the
detail summaries most often mangle. The trig estimator *multiplies up* by `exp(+‖x‖²/2)`, amplifying
whatever error the Gaussian-kernel estimate carried; the positive estimator *damps down* by
`exp(−‖x‖²/2)` and recovers the growth from `exp(ωᵀx)` inside the features. Same expectation, opposite
error behaviour.

The hyperbolic variant `φ^hyp+(x) = (1/√(2m))·exp(−‖x‖²/2)·(exp(ω_iᵀx), exp(−ω_iᵀx))` costs `2m` features
and is the `cosh` form of the same identity.

There is also a **regularized** variant, `SMREG`:

> "If we replace in (7) `ω` with `√d · ω/‖ω‖`, we obtain the so-called **regularized softmax-kernel**
> `SMREG` which we can approximate in a similar manner, simply changing `D = N(0, I_d)` to
> `D = Unif(√d · S^{d−1})`, a distribution corresponding to Haar measure on the sphere of radius `√d`…"

i.e. every `ω` is rescaled to exactly length `√d` instead of being `χ`-distributed around it. `SMREG` is
**not** the softmax kernel — it is a different kernel that Theorem 2 shows is a close lower bound.

### The R+ payoff — the variance is a closed form

§3, Lemma 2 ("positive (hyperbolic) versus trigonometric random features"), for independent `ω_i`.
These are exact, not bounds, and every term is computable from `x` and `y` alone:

    MSE( ŜM_m^trig(x,y) ) = (1/2m) · exp(‖x+y‖²) · SM⁻²(x,y) · (1 − exp(−‖x−y‖²))²

    MSE( ŜM_m^+(x,y) )    = (1/m)  · exp(‖x+y‖²) · SM²(x,y)  · (1 − exp(−‖x+y‖²))

    MSE( ŜM_m^hyp+(x,y) ) = ½ · (1 − exp(−‖x+y‖²)) · MSE( ŜM_m^+(x,y) )

And the conclusion drawn from them, verbatim:

> "Thus, for `SM(x,y) → 0` we have: `MSE(ŜM_m^trig(x,y)) → ∞` and `MSE(ŜM_m^+(x,y)) → 0`. Furthermore,
> the hyperbolic estimator provides additional accuracy improvements that are **strictly better than
> those from `ŜM_{2m}^+(x,y)` with twice as many random features**."

The `SM⁻²` versus `SM²` is the entire story in two symbols. Both MSEs carry `1/m` and
`exp(‖x+y‖²)`; the trig one is divided by the squared kernel value and the positive one is multiplied by
it. As a token pair becomes irrelevant, one estimator's error goes to infinity and the other's goes to
zero.

Figure 3's caption frames this as a ratio and names the regime:

> "Symmetrized (around origin) utility function `r` (defined as the ratio of the mean squared errors
> (MSEs) of estimators built on: trigonometric and positive random features) as a function of the angle
> `φ` (in radians) between input feature vectors and their lengths `l`… **We see that for critical regions
> with `φ` large enough (small enough softmax-kernel values) our method is arbitrarily more accurate than
> trigonometric random features.** Plot presented for domain `[−π, π] × [−2, 2]`."

### The O part — orthogonal random features, and precisely what they buy

> "To further reduce the variance of the estimator (so that we can use an even smaller number of random
> features `r`), **we entangle different random samples `ω₁,…,ω_m` to be exactly orthogonal**. This can be
> done **while maintaining unbiasedness** whenever isotropic distributions `D` are used… by the standard
> Gram-Schmidt orthogonalization procedure."

The novelty claim is carefully scoped — ORFs are old, the *theorem for this kernel* is not:

> "ORFs is a well-known method, yet it turns out that it works particularly well with our introduced PRFs
> for softmax. This leads to the **first theoretical results** showing that ORFs can be applied to reduce
> the variance of softmax/Gaussian kernel estimators **for any** dimensionality `d` rather than just
> asymptotically for large enough `d`… and leads to the **first exponentially small bounds** on large
> deviations probabilities that are strictly smaller than for non-orthogonal methods. **Positivity of
> random features plays a key role in these bounds.** The ORF mechanism requires `m ≤ d`, but this will be
> the case in all our experiments."

Orthogonalization is applied to the `ω` vectors: draw a Gaussian matrix `W ∈ R^{m×d}`, Gram-Schmidt the
rows, then rescale each row to keep the marginal `χ_d` norm — "these maintain the marginal distributions
of samples `ω_i` while enforcing that different samples are orthogonal". The marginal preservation is why
unbiasedness survives.

The `m > d` escape hatch, from Appendix B.2:

> "**If we need `m > d`, ORFs still can be used locally within each `d × d` block of `W`.**"

Three ORF flavours, with their costs (Appendix B.2):

- **R-ORFs** (Gaussian orthogonal): `O(md)` space, `O(md)` time for `Wx`, **unbiased**, one-time
  `O(md²)` Gram-Schmidt preprocessing.
- **H/G-ORFs** (Hadamard / Givens): `O(m)` or `O(m log d)` space, `O(m log d)` time for `Wx`, "gives
  **small bias** (tending to `0` with `d → ∞`)".

That is where the abstract's hedge "unbiased **or nearly-unbiased**" comes from. Plain FAVOR+ with
R-ORFs is exactly unbiased; the fast-transform variants are not.

### The theoretical guarantees, quoted

Four results, and they guarantee genuinely different things. Read the scoping carefully — the paper does
not oversell them, and the card must not either.

**Unbiasedness** — Lemma 1, an exact identity: `SM(x,y) = E[φ⁺(x)ᵀφ⁺(y)]`. Not asymptotic, not a bound.
Holds for every `m ≥ 1`. It is a statement about the *un-normalized* kernel entry, not about the
row-normalized attention weight, and not about the layer output.

**Theorem 2 (regularized vs softmax kernel).** Assuming `‖A‖_∞ ≤ C` for some `C ≥ 1`:

    inf_{i,j} A^reg(i,j) / A(i,j) ≥ 1 − 2/d^{1/3} + o(1/d^{1/3}),     sup_{i,j} A^reg(i,j) / A(i,j) ≤ 1

> "Furthermore, the latter holds for `d ≥ 2` even if the `L_∞`-norm condition is not satisfied, i.e. **the
> regularized softmax-kernel is a universal lower bound for the softmax-kernel**."

**Theorem 3 (orthogonality reduces MSE, with an explicit gap).** The variance-reduction claim, stated
exactly. `ŜM_m^ort+` is `ŜM_m^+` with orthogonal features, so `m ≤ d`:

> "Our next result shows that orthogonality **provably reduces mean squared error** of the estimation with
> positive random features **for any dimensionality `d > 0`** and we explicitly provide the gap."

    MSE(ŜM_m^ort+(x,y)) ≤ MSE(ŜM_m^+(x,y)) − (2(m−1) / (m(d+2))) · ( SM(x,y) − exp(−(‖x‖²+‖y‖²)/2) )²

This is a strict improvement whenever `m > 1` and `SM(x,y) ≠ exp(−(‖x‖²+‖y‖²)/2)`, and the subtracted
term is fully computable per cell. "Furthermore, completely analogous result holds for the regularized
softmax-kernel `SMREG`."

**Theorem 4 (concentration / large deviations).** For `SMREG`, `a > SMREG(x,y)`, `θ > 0`, `m ≤ d`:

    P[ ŜMREG_m^+(x,y)   > a ] ≤ exp(−θma) · M_Z(θ)^m
    P[ ŜMREG_m^ort+(x,y) > a ] ≤ exp(−θma) · ( M_Z(θ)^m
                                  − exp(−(m/2)(‖x‖²+‖y‖²)) · (θ⁴ m(m−1) / (4(d+2))) · ‖x+y‖⁴ )

> "We see that ORFs provide **exponentially small and sharper bounds for critical regions where the
> softmax-kernel is small**."

Note the scoping: this concentration bound is stated for **`SMREG`**, the regularized kernel, not for
`SM` itself, and it is a **one-sided upper-tail** bound.

**Theorem 5 (uniform convergence for attention approximation).** This is the one that produces the
`d log d` figure, and — the detail that gets lost — **it is proved for the trigonometric mechanism**:

> "Below we show that **even for the `SM^trig` mechanism with ORFs**, it suffices to take
> `m = Θ(d log(d))` random projections to accurately approximate the attention matrix (**thus if not
> attention renormalization, PRFs would not be needed**). In general, `m` depends on the dimensionality
> `d` of the embeddings, radius `R` of the ball where all queries/keys live and precision parameter `ε`…,
> **but does not depend on input sequence length `L`**."

> "Assume that `L₂`-norms of queries/keys are upper-bounded by `R > 0`. Define `l = R d^{−1/4}` and take
> `h* = exp(l²/2)`. Then for any `ε > 0`, `δ = ε/(h*)²` and the number of random projections
> `m = Θ( (d/δ²) · log( 4 d^{3/4} R / δ ) )` the following holds for the attention approximation mechanism
> leveraging estimators `ŜM^trig` with ORFs: **`‖Â − A‖_∞ ≤ ε` with any constant probability**, where `Â`
> approximates the attention matrix `A`."

The parenthesis is a remarkable admission and belongs on the card: *if attention did not renormalize, the
positive features would not be needed.* The whole PRF contribution exists because of the `D⁻¹` division.

Appendix D.6 gives the headline form:

> "we prove that if we take `m_opt = Θ(d log(d))`, then with **`O(L d² log(d))`-time**, we can approximate
> `A` up to any precision, **regardless of the number of tokens `L`**."

And the honest reading of what `R` costs you (Appendix D.7):

> "The dependence on `R` means that **the length of queries and keys cannot grow at a fixed `m` if we want
> to retain the quality of the approximation**."

### Complexity, exactly as stated

| | time | space |
|---|---|---|
| regular attention (§2.1) | `O(L² d)` | `O(L² + Ld)` |
| kernelizable attention, general `r` (§2.2) | `O(L r d)` | `O(Lr + Ld + rd)` |
| FAVOR+ Algorithm 1 (App. B.3) | `O(L m d)` | `O(md + Ld + mL)` |
| FAVOR+ with `m = Θ(d log d)` (App. D.6) | **`O(L d² log d)`** | — |
| Reformer / LSH, for comparison (§2.1) | `O(L d² log L)` | — |

The comparison the paper itself draws:

> "We will show that attention matrix `A` can be approximated up to any precision in time `O(L d² log(d))`.
> For comparison, popular methods leveraging sparsity via Locality-Sensitive Hashing (LSH) techniques have
> `O(L d² log L)` time complexity."

**`log d` versus `log L` is the entire point of that sentence** — the cost stops depending on sequence
length altogether.

Space, in more detail (App. B.3):

> "a variant of bidirectional FAVOR+ using iid samples or R-ORFs has `O(md + Ld + mL)` space complexity as
> opposed to `Θ(L² + Ld)` space complexity of the baseline. **Unidirectional FAVOR+ using fast prefix-sum
> pre-computation in parallel has `O(mLd)` space complexity** to store `G^PS` which can be reduced to
> `O(md + Ld + mL)` by running a simple (though non-parallel in `L`) aggregation… With G-ORFs space is
> `O(m log(d) + Ld + mL)`, with H-ORFs `O(m + Ld + mL) = O(Ld + mL)`."

Causal attention is *not* free here: the parallel prefix-sum form costs `O(mLd)` **space**, and getting
back to `O(md + Ld + mL)` costs you the parallelism in `L`. That trade is a real cost of the causal case
and is stated only in the appendix.

The causal mechanism itself (App. B.1) is a running prefix sum of outer products:

    [ tril(Q′(K′)ᵀ) C ]_i = G^PS_{i,:,:} × Q′_i,   G^PS_{i,:,:} = Σ_{j=1..i} G_{j,:,:},   G_{j,:,:} = K′_j C_jᵀ

with `C = [V | 1_L]` — the all-ones column is how the normalizer `D̂` is carried along in the same pass.
"An efficient algorithm to compute the prefix-sum of `L` elements takes `O(L)` total steps and `O(log L)`
time when computed in parallel." **This is precisely a `stateMixer` with `write: "add"`** — a fixed-size
state `S ∈ R^{m×(d+1)}` accumulating `φ(k_j) ⊗ [v_j, 1]`, read by `φ(q_i)`.

### Initialising from a pretrained softmax Transformer

The claim in the abstract is "**fully compatible** with regular Transformers"; §1 qualifies it in
parentheses as "(**via small amounts of fine-tuning**)". What the experiments actually show:

> "**1.** Backwards compatibility with pretrained models is available as a benefit from softmax
> approximation, **via small finetuning (required due to error propagation)** even for trigonometric
> features (Fig. 5, left) on the LM1B dataset. However, when on larger dataset PG-19, **2.** Positive (POS)
> softmax features (with redrawing) **become crucial** for achieving performance matching regular
> Transformers."

Figure 5's caption is the number that matters:

> "We transferred the original pretrained Transformer's weights into the Performer, which produces an
> **initial non-zero 0.07 accuracy** (dotted orange line), but **quickly recovers accuracy in a small
> fraction of the original number of gradient steps**. However on PG-19, Trigonometric (TRIG) softmax
> approximation **becomes highly unstable**…, while positive features (POS) (without redrawing) and
> Linformer (which also approximates softmax) *even with redrawn projections*, **plateau at the same
> perplexity**. **Positive softmax with feature redrawing is necessary to match the Transformer**, with
> SMREG allowing faster convergence."

So: weight transfer works, zero-shot transfer does **not**. 0.07 accuracy is near-useless output. And the
reason is named:

> "Even if the approximation of the attention mechanism is tight, **small errors can easily propagate
> throughout multiple Transformer layers** (e.g. MLPs, multiple heads)… In other words, **the model's
> *Lipschitz constant* can easily scale up small attention approximation error, which means that very
> tight approximations may sometimes be needed**."

One more operational requirement, easy to miss and load-bearing:

> "To further improve overall approximation of attention blocks across multiple iterations which further
> improves training, **random samples should be periodically redrawn**."

> "If redrawing is not used, an **'unlucky' set of random features may cause training degradation**, shown
> by the early-stopped curve with Seed 1, while a 'lucky' set of random features may cause no issue, shown
> by the curve with Seed 2. Redrawing allows the training to correct itself."

A Performer is therefore not a fixed function of its weights. It is a randomized algorithm whose random
seed is resampled during training, and one fixed seed can kill a run.

## Numbers that matter

**Defaults from the released implementation (App. A.3–A.4).**

- Approximate softmax attention: `renormalize_attention = True`, numerical stabilizer `10⁻⁶`,
  **number of features = 256**, `ortho_features = True`, `ortho_scaling = 0.0`.
- Generalized (non-softmax) attention: numerical stabilizer `0.0`, number of features = 256,
  **kernel = ReLU**, `kernel_epsilon = 10⁻³`.
- Standard model: `(n_heads, n_layers, d_ff, d) = (8, 6, 2048, 512)`.
- Training: 0.5 grad clip, 0.1 weight decay, 0.1 dropout, `10⁻³` fixed LR, Adam `(0.9, 0.98, 10⁻⁹)`.
  "**Note that Performers are using the same training hyperparameters as Transformers**… this shows that
  FAVOR can act as a simple drop-in without needing much tuning."

**Approximation-error experiment (Fig. 4).** `L = 4096`, `d = 16`, `m` varied, 15 samples, on
"appropriately normalized random matrix input data". Two findings, stated as the empirical validation of
the two letters of the acronym:

> "**1.** Orthogonal features produce lower error than unstructured (IID) features, **2.** Positive
> features produce lower error than trigonometric sin/cos features. These two empirically validate the
> PORF mechanism."

Note what this experiment is *not*: it is random matrix input, not a trained model, and the y-axis is MSE
of the approximation **output**.

**Protein modelling, `L = 1024`, 36 layers, 16×16 TPU-v2 (Table 5).** The only head-to-head accuracy
table against an exact-softmax Transformer in the paper:

| dir | set | model | accuracy | perplexity |
|---|---|---|---|---|
| UNI | Test | Empirical baseline | 9.92 | 17.80 |
| UNI | Test | Transformer | 30.80 | 9.37 |
| UNI | Test | Performer (generalized, ReLU) | **31.58** | **9.17** |
| UNI | OOD | Empirical baseline | 9.07 | 17.93 |
| UNI | OOD | Transformer | **19.70** | **13.20** |
| UNI | OOD | Performer (generalized) | 18.44 | 13.63 |
| BID | Test | Transformer | 33.32 | 9.22 |
| BID | Test | Performer (generalized) | **36.09** | **8.36** |
| BID | Test | **Performer (softmax)** | **33.00** | **9.24** |
| BID | OOD | Transformer | **25.07** | **12.09** |
| BID | OOD | Performer (generalized) | 24.10 | 12.26 |
| BID | OOD | Performer (softmax) | 23.48 | 12.41 |

**Read the softmax row, not the generalized row.** The FAVOR+ softmax approximation — the thing this card
is about — scores **33.00 vs the Transformer's 33.32** in-distribution and **23.48 vs 25.07** out of
distribution. It loses both, narrowly in-distribution and by **1.59 accuracy points** OOD. The row that
beats the Transformer (36.09) is `Performer-RELU`, which is **not an approximation of softmax at all** —
it is generalized attention with an arbitrary ReLU feature map, i.e. the very thing §1 criticized
Katharopoulos et al. for. The paper is candid about this: "the usefulness of generalized attention is
evidenced by Performer-RELU… achieving the highest accuracy in both (U) and (B) cases."

The paper's own summary sentence for the softmax row is the softer one:

> "Our proposed softmax approximation is also shown to be tight, **achieving the same accuracy as the
> exact-softmax Transformer** and confirming our theoretical claims from Section 3."

"The same accuracy" is 33.00 vs 33.32. That is a fair description of a plot; it is a generous description
of the table.

**ImageNet64, `L = 12288`, unidirectional, after 100K steps (App. C.3):** Performer-ReLU **3.67**,
Performer-Softmax **3.69**, Performer-Softmax (SMREG) **3.67** BPD. No exact Transformer number — at
`L = 12288` "which is unfeasible for regular Transformers". Against Reformer: "Performer/6-layers matches
the Reformer/12-layers, while the Performer/12-layers matches the Reformer/24-layers" and "the Performer
can be 2x faster than the Reformer via Jax optimizations for the (U) setting".

**Concatenated proteins, `L = 8192`:** a regular Transformer "overloads memory even at a batch size of 1
per chip, by a wide margin"; the reduced 3-layer baseline "is quickly bounded at ≈ 19%, while the
Performer is able to train continuously to ≈ 24%". Note this compares a full Performer against a
crippled Transformer, which the paper states plainly.

**Long Range Arena (App. C.6).** The scoped claim, verbatim:

> "Performers obtain the largest LRA (Long Range Arena) score **among all tested scalable Transformers
> methods (which we define by having speed of > 100 examples/sec)**."

The actual LRA table is reproduced as an image (`lra-all-3.pdf`), so no per-task numbers exist in the
source text — **do not put LRA numbers on the card from memory.** Also note the qualifier: largest among
*scalable* methods, on a benchmark where "all models do not learn anything on Path-X task (denoted by
FAIL)".

**Linear Transformer head-to-head (App. C.5)** — the backward edge, measured. Using
`φ(x) = elu(x) + 1` from Katharopoulos et al., same hyperparameters as Performer-ReLU, same ProGen setting:

> "we empirically found that the Linear Transformer possessed numerical instability during training via
> unstable training curves, **ultimately stopping training by producing exploding gradients (NaNs)**."

> "In the unidirectional 36-ProGen setting, we ran **3 seeds** of the Linear Transformer, and found that
> **all 3 seeds produced exploding gradients very early on**, stopping the training run. The Linear
> Transformer in the bidirectional setting also produced an exploding gradient in the middle of training,
> near **125K steps**."

And the same failure mode for trig features: the kernel-sweep figures are log-scaled "to emphasize the
highest accuracy runs but also show the **NaN issues with certain kernels which caused runs to stop
early**", with cosine being "(original softmax approximation)".

**Derived arithmetic — the crossover, clearly labelled as mine, not the paper's.** Exact causal attention
costs `≈ L²d` multiplies (scores plus the value sum); FAVOR+ costs `≈ 4Lmd` (building `Q′` and `K′`, then
`K′ᵀC`, then `Q′·`). Setting them equal gives **`L ≈ 4m`**. At the paper's default `m = 256` that is
`L ≈ 1024` — below roughly a thousand tokens the linear method is doing *more* arithmetic than the exact
one, and it wins on memory long before it wins on FLOPs. The paper never states a crossover length; Fig. 2
shows it empirically and the constants there are implementation-specific. **Treat `L ≈ 4m` as an
order-of-magnitude sanity check, not a quoted result.**

**Derived for this app.** `T = 16`, causal ⇒ `T(T+1)/2 = 136` live cells (same count the sparse-transformer
card uses). `d_k = 8`. Exact scores: `136 × 8 = 1088` multiplies. FAVOR+ at `m = 32`:
`2 × 16 × 32 × 8 = 8192` just to build the features. **At this size the approximation costs about 8× more
than the exact answer it is approximating**, and that number should be on screen.

## What the live view must let the reader do

The seam is `stateMixer({ write, decay, phi })` next to `softmaxMixer`. **One blocking implementation
fact, up front:** as `app/model/mixers.js` stands today, `phi` is applied *elementwise*
(`Array.from(K[i], phi)`) and the state `S` is `dh × dh`. A random feature map is neither — it is a map
`R^{d_k} → R^m` that changes the dimension, so the state must become `m × dh` and the normalizer vector
must become length `m`. **The seam needs a vector-valued `phi` before this card can be built.** That is a
small change (`phi` takes and returns an array; `S` is sized from `phi(K[0]).length`), and it is the same
change every later feature-map card will need, but it must not be papered over — pretending an elementwise
`phi` is a random feature map would make every number on the card wrong.

Second implementation fact, from the paper's own "without loss of generality" sentence. `softmaxMixer`
computes `q·k/√d_k`. FAVOR+ defines `SM(x,y) = exp(xᵀy)` with no scale factor and instructs you to fold it
into the inputs. So feed the feature map **`q̃ = q / d_k^{1/4}`** and **`k̃ = k / d_k^{1/4}`**, giving
`q̃ᵀk̃ = q·k/√d_k` — exactly the app's existing score. Get this wrong and the approximation converges to
the wrong matrix, which looks identical to a bug in the estimator.

Third: numerical stabilization. Subtract a **single global constant** `c = max over all i,j of ω_jᵀ·(that
vector)` inside every exponent, for queries and keys alike. A global constant scales the whole
un-normalized `Â` by `e^{−2c}` and cancels in the row normalization. A *per-row* constant does not cancel
for keys and will silently bias the result — the reference implementation instead uses an additive
stabilizer of `10⁻⁶`.

**The computation, once, shared by all five interactions.** Per head, per block:

1. Seeded `ω₁…ω_m ~ N(0, I_8)` (Box–Muller over the existing seeded generator; reuse the app's seed so it
   is reproducible).
2. `φ⁺(x) = exp(−‖x‖²/2 − c)/√m · (exp(ω₁ᵀx), …, exp(ω_mᵀx))`.
3. `Â_raw(i,j) = φ⁺(q̃_i)ᵀφ⁺(k̃_j)` for `j ≤ i`, zero above the diagonal.
4. `Ŵ(i,j) = Â_raw(i,j) / Σ_{j′≤i} Â_raw(i,j′)` — the row-normalized weights, the thing that is
   comparable to `softmaxMixer`'s `weights`.
5. Exact: `A_raw(i,j) = exp(q_i·k_j/√d_k)`, `W = softmax` rows — already computed by `softmaxMixer`.

**The two error metrics, defined so they can be checked:**

    maxAbs  = max over the 136 causal cells of |Ŵ(i,j) − W(i,j)|
    meanAbs = ( Σ over the 136 causal cells of |Ŵ(i,j) − W(i,j)| ) / 136

Display both, always, to 4 decimal places. **And display a third, separately labelled:**
`‖Â_raw − A_raw‖_∞` over the same cells — because *that* is the quantity Theorem 5 bounds. The reader sees
the normalized picture; the theorem is about the un-normalized one. Showing only one of them would let the
card claim a guarantee it is not displaying.

---

**1. The `m` slider — the headline demo.** `m = 1, 2, 4, 8, 16, 32, 64, 128, 256`, log-spaced, at fixed
seed. Two heatmaps side by side (exact `W`, approximate `Ŵ`) plus a difference map, and the three numbers
above.

- **The number that proves it:** `meanAbs` falling roughly as `1/√m`. Show the measured ratio between
  consecutive stops next to the predicted `1/√2 ≈ 0.707`. Four doublings of `m` should cut the error by
  about `4×`, not `16×` — and the reader watching it fall *slower than they expect* is the real lesson.
- **The number that proves the second half:** `maxAbs` at `m = 256` is small but **is not zero**, and
  never becomes zero. Print it. Add one line of text: *`m = 256` is the paper's own default, and there is
  no setting of this slider that makes the error vanish.*
- Anchor the cost against it: a live counter showing `exact: 1088 multiplies` vs `FAVOR+: 2·T·m·d_k`, so
  that as the reader drags `m` up to shrink the error they watch the cost pass the exact method (at
  `T = 16` it passes almost immediately — see the honesty note below).

**2. Unbiasedness, shown rather than asserted — a "redraws" slider.** `R = 1, 2, 5, 10, 50, 200`
independent `ω`-draws at fixed `m`; average `Â_raw` over the `R` draws before normalizing, and plot
`meanAbs` for both the single-draw and the averaged estimate.

- **The number that proves it:** averaged `meanAbs → 0` as `R` grows at *fixed* `m`, while the single-draw
  error sits flat. If the estimator had any bias, the averaged curve would flatten at a positive floor.
  Print the averaged error at `R = 200` next to the single-draw error at the same `m`.
- This is the one interaction that distinguishes *unbiased* from *low-variance*, which is the distinction
  the whole card rests on. It also doubles as the honest demonstration of "redrawing": the paper redraws
  during training for exactly this reason, and the reader can see why one unlucky draw is a real risk.

**3. Trigonometric vs positive, with the pathology made visible.** A toggle `positive (exp)` /
`trigonometric (sin, cos)` / `hyperbolic (exp, −exp)`, plus a **query/key norm slider `s`** that scales
`q̃, k̃ ← s·q̃, s·k̃`. Same `ω`, same seed, same `m`.

- The norm slider is not decoration and must be justified on screen: with 32-dim untrained seeded weights
  and `d_k = 8`, `‖q̃‖` and `‖k̃‖` are small, so `SM(q̃,k̃) ≈ 1` everywhere and **the pathological regime does
  not occur naturally.** The reader has to be able to reach it. This is exactly the paper's own Fig. 3,
  which sweeps the angle `φ` *and the length `l`* for the same reason.
- **Three numbers prove the point, all computed:**
  - **count of negative entries in `Â_raw`** — structurally `0` for positive features, and climbing for
    trig as `s` rises. Highlight those cells in the difference map.
  - **count of rows whose normalizer `Σ_j Â_raw(i,j)` is ≤ 0** — the paper's "negative-diagonal-values
    renormalizers `D⁻¹`", quoted underneath. When this is nonzero the approximate attention row is not a
    convex combination and the app should say so in words, not just colour a cell.
  - **the closed-form MSE ratio `r`** from Lemma 2, evaluated per cell with `x = q̃_i`, `y = k̃_j`:

            r(i,j) = MSE_trig / MSE_+
                   = [ (1/2m)·exp(‖x+y‖²)·SM⁻²·(1−exp(−‖x−y‖²))² ]
                     / [ (1/m)·exp(‖x+y‖²)·SM²·(1−exp(−‖x+y‖²)) ]

    Pick the cell with the **smallest exact attention weight** in the reader's own sentence — the least
    relevant token pair — and print `SM(x,y)`, `r`, and the *measured* squared error of both estimators
    over the redraws from interaction 2. Predicted and measured should track. `r → ∞` as `SM → 0` is the
    key technical contribution of the paper, and this is it on the reader's own sentence.
- The hyperbolic option is nearly free (it is the same `ω`, with `exp(−ω ᵀx)` appended) and lets the card
  check the paper's sharper claim: `ŜM^hyp+` at `m` should beat `ŜM⁺` at `2m`, not merely match it.
  Display both errors side by side. If it does not reproduce, say so — that is a finding, not a bug to
  hide.

**4. Orthogonal vs IID — the O in FAVOR+.** A toggle at fixed `m` and fixed seed. Build `W ∈ R^{m×8}`,
Gram-Schmidt the rows **in blocks of `d_k = 8`** (the paper's own `m > d` instruction), and renormalize each
row to the norm it had before orthogonalization, so the marginal is preserved and unbiasedness survives —
show that as a checked invariant, e.g. print `mean ‖ω_i‖` for both modes and confirm they match.

- **The number that proves it:** `meanAbs(ORF)` vs `meanAbs(IID)` at identical `m`, plus the *guaranteed*
  per-cell gap from Theorem 3:

            gap(i,j) = (2(m−1) / (m(d+2))) · ( SM(x,y) − exp(−(‖x‖²+‖y‖²)/2) )²

  With `d = d_k = 8` and `m = 8` the prefactor is `2·7/(8·10) = 0.175`. Print the summed predicted gap
  next to the measured MSE reduction (averaged over the redraws from interaction 2 — with a single draw
  the comparison is noise and the card would be lying).
- Print `m ≤ d` as a live constraint: at `d_k = 8`, a *single* orthogonal block holds only 8 vectors.
  Above `m = 8` the app is in the block-wise regime, which is the appendix's fallback, not the theorem's
  setting — Theorem 3 is stated for `m ≤ d`. Say which regime the current slider position is in.

**5. Where it actually costs you — the cost panel, always visible.** Three rows:
`your sentence, T = 16, m = 32: exact 1088 multiplies vs FAVOR+ ≈ 8192 (7.5× worse)`;
`crossover at L ≈ 4m — for m = 32 that is L ≈ 128`;
`paper default m = 256 ⇒ crossover near L ≈ 1024; the paper's wins are at L = 4096, 8192, 12288`.

- **The number that proves it:** the ratio in row 1 being **greater than 1**. The reader must not leave
  thinking they have watched something get faster. They have watched an estimator converge. Below it, one
  line: *the `L²` term the Performer removes is invisible at 16 tokens; what you are measuring here is
  accuracy, and accuracy is the thing you pay for the speed you cannot see.*

*Optional, if there is room and it can be done honestly:* a **layer-depth** readout. Run the full 2-block
model with `softmaxMixer` and with FAVOR+ at the current `m`, and display the relative error of the
**final hidden states**, not just the attention rows. It will be visibly larger than the attention-matrix
error. That is the paper's own Fig. 8 and its Lipschitz-constant paragraph — "small errors can easily
propagate throughout multiple Transformer layers" — and it is the mechanism behind the `0.07` zero-shot
transfer accuracy. Two numbers, one lesson: a tight attention approximation is not a tight model
approximation.

## What the source does *not* establish

- **It does not claim the approximation is exact, ever.** `m` buys precision at a price and the paper says
  so plainly: "The number of random features `m` allows a trade-off between computational complexity and
  the level of approximation: bigger `m` results in higher computation costs, but also in a lower variance
  of the estimate of `A`." There is no finite `m` at which `Â = A`.
- **The `O(L d² log d)` result is proved for the *trigonometric* mechanism, not for PRFs.** Theorem 5 and
  Appendix D.6 are explicit: "even for the `SM^trig` mechanism with ORFs, it suffices to take
  `m = Θ(d log(d))`". The paper's own aside — "thus if not attention renormalization, PRFs would not be
  needed" — concedes that the headline complexity does not require the headline contribution. Do not
  attribute the `d log d` bound to positive features.
- **The concentration bounds (Theorem 4) are for `SMREG`, not `SM`,** and they are one-sided upper-tail
  bounds. `SMREG` is a *different kernel*; Theorem 2 bounds how close it is (`≥ 1 − 2/d^{1/3}` ratio, and
  a universal lower bound), which is a good bound for large `d` and a weak one for small `d`. At this app's
  `d_k = 8`, `2/d^{1/3} = 1.0` — the guarantee is vacuous. Say so if the card mentions SMREG at all.
- **`‖Â − A‖_∞ ≤ ε` is about the un-normalized matrix.** No stated guarantee covers the row-normalized
  weights, the layer output, or the model's predictions — and the paper's own error-propagation figure and
  Lipschitz remark exist precisely because that gap is real.
- **Unbiasedness is not preserved by every variant they recommend.** R-ORFs are unbiased; H/G-ORFs "give
  small bias (tending to 0 with `d → ∞`)". Hence "unbiased **or nearly-unbiased**" in the abstract.
- **The `m ≤ d` condition is quietly violated by the defaults, and the paper does not reconcile it.** §2.4
  asserts "The ORF mechanism requires `m ≤ d`, but this will be the case in all our experiments", while
  App. A.3 gives `number of features = 256` and the standard model is `d = 512` with `n_heads = 8` — a
  per-head dimension of 64. Random features act on the per-head query/key rows, so `m = 256 > 64`.
  Either `d` in that sentence means the model dimension (in which case the theorems, which are stated over
  `R^d` for the vectors being kernelized, are being applied at the wrong dimension), or the experiments
  used the appendix's block-wise fallback (in which case "this will be the case in all our experiments" is
  wrong). **This is an unresolved tension in the source, not a settled fact — flag it, do not pick a
  side.**
- **The strongest accuracy result is not a softmax result.** Performer-RELU wins the protein tables;
  Performer-**softmax** loses to the exact Transformer on all four protein comparisons where both are
  reported (33.00 vs 33.32; 23.48 vs 25.07 OOD). "Achieving the same accuracy as the exact-softmax
  Transformer" is the paper's gloss on a table where it is behind. The card should print the table.
- **Zero-shot weight transfer does not work.** 0.07 accuracy on transfer, recovered only "in a small
  fraction of the original number of gradient steps" of fine-tuning. "Fully compatible" in the abstract is
  qualified in §1 as "(via small amounts of fine-tuning)" and the qualification is doing real work.
- **Stability requires redrawing, and one seed can kill a run.** Not an implementation detail: "an
  'unlucky' set of random features may cause training degradation… Redrawing allows the training to
  correct itself."
- **The LRA claim is scoped to "scalable" methods** (>100 examples/sec), on a benchmark where every model
  failed Path-X. The per-task numbers exist only as a reproduced image in the appendix and are not in the
  source text.
- **The paper says nothing about whether Performers displaced exact attention.** It could not — it is from
  September 2020. Its claims are: linear time and space, provable unbiasedness, competitive results, best
  LRA score among scalable methods. It never claims to match exact attention on large-scale language
  modelling, and it presents no such experiment. Any statement about what happened afterwards is outside
  this source and must be labelled as such on the card.
- **The app proves none of the speed claims.** At `T = 16` FAVOR+ does several times more arithmetic than
  exact attention. The app demonstrates *convergence of an estimator*, which is the concept; it cannot
  demonstrate *linear scaling*, which needs length. Same discipline as the sparse-transformer card, and
  the same honest sentence belongs on screen. And the model is 32-dim, 2-block, untrained: the attention
  matrices are seeded noise, so read the error curves, never the predictions.

## Leaves behind

**Backward — what this answers.** The previous card swapped `exp(qᵀk)` for `φ(q)ᵀφ(k)` with
`φ = elu(·)+1`, got a fixed-size state and linear time, and asked no further questions. The kernel it
ended up computing was whatever that expression computes. Performer is the card that asks the question
that was skipped: **how far is the cheap thing from the exact thing?** — and it is the first one able to
answer, because it runs the derivation in the other direction. Instead of picking `φ` and accepting the
kernel, it fixes the kernel to `exp(xᵀy)` and *solves* for a `φ` whose expectation reproduces it. The
paper's charge against the prior work is not that it was slow or inaccurate; it is that it never made a
claim you could check: "The approaches above **do not aim to approximate regular attention**… there is a
lack of rigorous guarantees for the representation power produced by such methods."

Two things follow that the earlier card cannot supply. First, an **error bar**: `‖Â − A‖_∞ ≤ ε` with
`m = Θ(d log d)` features, independent of `L`. Second, **compatibility** — because the target is the real
softmax kernel, pretrained softmax weights mean something in the new model (0.07 accuracy and a short
fine-tune, versus meaningless with an arbitrary `φ`). And the appendix supplies the empirical
counterpunch: the Linear Transformer's `elu(x)+1` blew up with exploding gradients in 3 of 3 seeds
unidirectionally and again at 125K steps bidirectionally, on the same setting where Performer trained
fine. Choosing the feature map arbitrarily was not merely unprincipled; it broke.

But the win is narrower than it sounds, and the card must draw the line in the right place. Both the
trigonometric and the positive estimators are **unbiased**. Unbiasedness was never the hard part. The
contribution is entirely about **where the variance lives**: `MSE^trig ∝ SM⁻²` and `MSE⁺ ∝ SM²`, so as a
token pair becomes irrelevant — which is most pairs, in most rows — one estimator's error diverges and the
other's vanishes. Attention divides by a row sum, and a row sum of near-zero entries estimated with
diverging relative error is a division by garbage. **Positivity is not an aesthetic preference for
non-negative attention weights; it is the fix for the denominator.** The paper's own aside makes this
exact: "thus if not attention renormalization, PRFs would not be needed."

**Forward — the weakness, stated precisely.** Three limits, each written into the paper itself.

1. **It is still an estimate, and the error is where you cannot see it.** No finite `m` gives the exact
   answer; the guarantee is on the un-normalized matrix, not on the model output; and the error compounds
   through depth — "the model's Lipschitz constant can easily scale up small attention approximation
   error, which means that **very tight approximations may sometimes be needed**". A method whose accuracy
   knob must sometimes be turned all the way up is a method whose cost knob goes with it.
2. **Enough random features erodes the advantage.** Cost is `O(Lmd)` against `O(L²d)`; the ratio is
   `L/m` up to constants. Wanting a tighter approximation means raising `m`, and raising `m` walks the
   crossover length up with it. The default `m = 256` puts the break-even somewhere around a thousand
   tokens. The linear-time claim is real, but it is a claim about the *asymptote*, and the constant is a
   dial the accuracy requirement controls.
3. **It is a randomized algorithm, and the randomness has to be managed.** Features must be periodically
   redrawn or an unlucky seed degrades training. The model is not a fixed function of its weights.

The paper does not claim Performers replaced exact attention, and nothing in it speaks to that; the
question is outside the source and the card must say so rather than smuggle in hindsight. What the paper
*does* claim is bounded and defensible: linear complexity, an unbiased estimator with a real variance
analysis, competitive accuracy, the best LRA score among scalable methods, and compatibility with
pretrained softmax weights after fine-tuning. What it also, honestly, records is that its own best
protein result came from **`Performer-RELU`** — an arbitrary feature map with no approximation guarantee
at all — beating its own principled softmax approximation. That is the crack the timeline widens next.

Two roads lead out, and they disagree about the premise rather than the technique. One says the
approximation was never the problem and the memory traffic was: stop estimating softmax, compute it
exactly, and fix the `O(L²)` *materialization* instead of the `O(L²)` arithmetic — which is
FlashAttention, and which makes the exact answer cheap enough that an unbiased estimate of it stops being
worth the variance. The other reads Performer's own ReLU result as the real finding and gives up on
softmax as a target altogether: if a hand-picked feature map beats a provably-unbiased softmax
approximation, then softmax was never sacred, and the right move is to design the recurrence directly
rather than approximate an inherited one — which is where the gated linear and state-space family goes.
Performer is the pivot between them, and it is the card that earns the pivot, because it is the only one
that measured how far the cheap thing actually was from the exact thing.
