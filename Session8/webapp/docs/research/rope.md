# Concept 12 — RoPE, rotary position embedding
**Card id:** `rope` · **Date:** 20 April 2021 (arXiv v1) · **Pressure:** where a token sits

## What was read

- [arXiv:2104.09864](https://arxiv.org/abs/2104.09864), Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed
  Murtadha, Bo Wen, Yunfeng Liu (Zhuiyi Technology, Shenzhen) — *RoFormer: Enhanced Transformer with
  Rotary Position Embedding*. Abstract pages for v1/v2/v3/v5 for the version history and the
  per-version abstracts.
- [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2104.09864) — fetched raw and read end to end
  (§2 background, §3.1 formulation, §3.2 RoPE, §3.3 properties, §3.4.1–3.4.3 theoretical explanation,
  §4.1–4.5 experiments, §4.5.5 limitations, §5). ar5iv renders the **latest** version, so the section
  numbering and all quotes below are v5 text unless marked.
- The **v1 PDF**, text-extracted locally, to answer the version question directly rather than by
  inference.

**Version history.** v1 **Tue 20 Apr 2021**; v2 Sat 9 Oct 2021; v3 Fri 5 Aug 2022; v4 Tue 9 Aug 2022;
v5 Wed 8 Nov 2023. Comment fields: v1 *"Preprint. English experiments are coming"*, v2 *"more
experiments"*, v5 *"fixed some typos"*.

**Did the mechanism change? No.** v1 already contains, verbatim and in the same order: the
requirement `⟨f_q(x_m,m), f_k(x_n,n)⟩ = g(x_m, x_n, m−n)` (its eq. 11, same number as v5); the 2D
complex solution; the block-diagonal rotary matrix with `Θ = {θ_i = 10000^(−2(i−1)/d), i ∈ [1,…,d/2]}`
(its eq. 15); the efficient elementwise realization (v1 Appendix B, promoted to §3.4.2 in v5); the
long-term-decay Abel-transformation derivation (v1 Appendix C, promoted to §3.4.3); and the
linear-attention variant. What changed across versions is the **evidence**, not the mechanism: v1
reports Chinese results only (CAIL2019-SCM, Table 4, the same 64.13/68.29 and 66.07/69.79 numbers that
survive into v5) and says so in its own abstract — *"We release the theoretical analysis along with
some preliminary experiment results on Chinese data. The undergoing experiment for English benchmark
will soon be updated."* v2 delivers those: WMT14 En-De, BERT-vs-RoFormer MLM pretraining, GLUE, and
Performer+RoPE on Enwik8. An author (Ahmed Murtadha) is added at v3. The timeline uses **v1** —
the idea, the derivation, and the decay analysis were all complete on day one.

## The mechanism, precisely

**The starting point is a requirement, not a formula.** This is the sentence the whole card hangs on
(§3.1):

> *"In order to incorporate relative position information, we require the inner product of query
> **q**_m and key **k**_n to be formulated by a function g, which takes only the word embeddings
> **x**_m, **x**_n, and their relative position m−n as input variables. In other words, we hope that
> the inner product encodes position information only in the relative form:"*

    ⟨ f_q(x_m, m), f_k(x_n, n) ⟩ = g(x_m, x_n, m − n)        (11)

> *"The ultimate goal is to find an equivalent encoding mechanism to solve the functions f_q(x_m, m)
> and f_k(x_n, n) to conform the aforementioned relation."*

Read the shape of that. Every scheme on this timeline so far picked a mechanism and then argued about
what it bought. This paper writes down the **property first** and solves for the mechanism. §2 ends
by saying so explicitly: *"all these approaches attempt to modify Equation 6 based on the
decomposition of Equation 3 … Unlikely, our approach aims to derive the relative position encoding
from Equation 1 under some constraints."* Equation 1 is just `q_m = f_q(x_m, m)`, `k_n = f_k(x_n, n)`,
`v_n = f_v(x_n, n)` — the bare statement that position enters through *some* function. Everything else
is a consequence.

**The 2D solution.** For `d = 2`, treating the vector as a complex number (§3.2.1, proved in §3.4.1
under an *"initial condition"* `q = f_q(x_q, 0)`, `k = f_k(x_k, 0)`, *"which can be read as the vectors
with empty position information encoded"*):

    f_q(x_m, m) = (W_q x_m) e^(imθ)
    f_k(x_n, n) = (W_k x_n) e^(inθ)                          (12)
    g(x_m, x_n, m−n) = Re[ (W_q x_m)(W_k x_n)* e^(i(m−n)θ) ]

> *"θ ∈ ℝ is a preset non-zero constant."*

In matrix form (eq. 13) that is a 2×2 rotation applied **after** the projection:

    f_{q,k}(x_m, m) = [ cos mθ  −sin mθ ] [ W^(11) W^(12) ] [ x^(1)_m ]
                      [ sin mθ   cos mθ ] [ W^(21) W^(22) ] [ x^(2)_m ]

> *"Specifically, incorporating the relative position embedding is straightforward: simply rotate the
> affine-transformed word embedding vector by amount of angle multiples of its position index and thus
> interprets the intuition behind Rotary Position Embedding."*

**General d.** §3.2.2: *"we divide the d-dimension space into d/2 sub-spaces and combine them in the
merit of the linearity of the inner product"* —

    f_{q,k}(x_m, m) = R^d_{Θ,m} W_{q,k} x_m                  (14)

`R^d_{Θ,m}` (eq. 15) is block-diagonal: `d/2` independent 2×2 rotation blocks on **consecutive
dimension pairs** `(1,2), (3,4), …, (d−1, d)`, block `i` rotating by angle `m·θ_i`, everything
off-block zero.

> *"is the rotary matrix with pre-defined parameters Θ = {θ_i = 10000^(−2(i−1)/d), i ∈ [1,2,...,d/2]}."*

Applied to the score (eq. 16):

    q_m^⊺ k_n = (R^d_{Θ,m} W_q x_m)^⊺ (R^d_{Θ,n} W_k x_n) = x^⊺ W_q R^d_{Θ,n−m} W_k x_n

> *"where R^d_{Θ,n−m} = (R^d_{Θ,m})^⊺ R^d_{Θ,n}. Note that R^d_Θ is an orthogonal matrix, which
> ensures stability during the process of encoding position information."*

That single line `R_m^⊺ R_n = R_{n−m}` **is** the whole trick: two absolute rotations compose into one
relative rotation inside the ordinary dot product. Nothing is added to anything; no extra term appears
in the expanded score; `QK^⊺` is still one matrix multiply.

> *"In contrast to the additive nature of position embedding method adopted in the previous works …
> our approach is multiplicative. Moreover, RoPE naturally incorporates relative position information
> through rotation matrix product instead of altering terms in the expanded formulation of additive
> position encoding when applied with self-attention."*

**The efficient realization** (§3.4.2) — the form every implementation actually ships, including this
app's:

> *"Taking the advantage of the sparsity of R^d_{Θ,m} in Equation 15, a more computational efficient
> realization of a multiplication of R^d_Θ and x ∈ ℝ^d is:"*

    R^d_{Θ,m} x = (x_1, x_2, x_3, x_4, …, x_{d−1}, x_d) ⊗ (cos mθ_1, cos mθ_1, cos mθ_2, cos mθ_2, …, cos mθ_{d/2}, cos mθ_{d/2})
                + (−x_2, x_1, −x_4, x_3, …, −x_d, x_{d−1}) ⊗ (sin mθ_1, sin mθ_1, sin mθ_2, sin mθ_2, …, sin mθ_{d/2}, sin mθ_{d/2})
                                                             (34)

`⊗` is elementwise. The second vector is the "rotate half" trick in its **interleaved** form: swap
each adjacent pair and negate the first of the two. (Llama-style implementations use a *half-split*
permutation — first `d/2` dims against second `d/2` — which is a different but equivalent basis
ordering, and the paper never mentions it. This app's `rope()` matches the paper's interleaved layout,
`out[2i] = x·cos − y·sin`, `out[2i+1] = x·sin + y·cos`.)

**Long-term decay** (§3.3, one paragraph, then §3.4.3):

> *"Following Vaswani et al. 2017, we set θ_i = 10000^(−2i/d). One can prove that this setting
> provides a long-term decay property (refer to Section 3.4.3 for more details), which means the
> inner-product will decay when the relative position increase. This property coincides with the
> intuition that a pair of tokens with a long relative distance should have less connection."*

§3.4.3 rewrites the score as a complex sum (eq. 35), sets `h_i = q_[2i:2i+1] k*_[2i:2i+1]`,
`S_j = Σ_{i=0}^{j−1} e^{i(m−n)θ_i}`, `h_{d/2} = 0`, `S_0 = 0`, and applies *"Abel transformation"*
(summation by parts):

    Σ_i q_[2i:2i+1] k*_[2i:2i+1] e^{i(m−n)θ_i} = Σ_i h_i (S_{i+1} − S_i) = −Σ_i S_{i+1} (h_{i+1} − h_i)   (36)

    | Σ_i q k* e^{i(m−n)θ_i} | ≤ Σ_i |S_{i+1}| |h_{i+1} − h_i| ≤ ( max_i |h_{i+1} − h_i| ) · Σ_i |S_{i+1}|  (37)

and then stops:

> *"Note that the value of (1/(d/2)) Σ_{i=1}^{d/2} |S_i| decay with the relative distance m−n increases
> by setting θ_i = 10000^(−2i/d), as shown in Figure 2."*

**So the honest reading is: the bound is proved, the decay is not.** Eq. 37 is a rigorous inequality.
That the *right-hand side* falls with `m−n` is asserted and shown in a plot — *"as shown in Figure 2"*
— not derived. And the bound carries a factor `max_i |h_{i+1} − h_i|` that depends entirely on the
query and key content, is nowhere bounded, and can be arbitrarily large. An upper bound that decays
does not force the quantity beneath it to decay. §3.3's *"One can prove"* is a promise §3.4.3 does not
keep. The paper's own §4.5.5 then leans on the word anyway: *"Although we have proved that our model
has favourable property of long-term decay for intern-token products … we have not come up with a
faithful explanation."*

**Linear attention** (§3.3, second property). The general form (eq. 17) `Attention(Q,K,V)_m =
Σ_n sim(q_m,k_n) v_n / Σ_n sim(q_m,k_n)`; linear attention (eq. 18, Katharopoulos et al.) replaces
`sim` with `φ(q_m)^⊺ φ(k_n)`. Then:

> *"Since RoPE injects position information by rotation, which keeps the norm of hidden representations
> unchanged, we can combine RoPE with linear attention by multiplying the rotation matrix with the
> outputs of the non-negative functions."*

    Attention(Q,K,V)_m = Σ_n (R^d_{Θ,m} φ(q_m))^⊺ (R^d_{Θ,n} φ(k_n)) v_n / Σ_n φ(q_m)^⊺ φ(k_n)   (19)

Note what happened to the denominator — it is **not** rotated, and the paper says why, and concedes
the consequence:

> *"It is noteworthy that we keep the denominator unchanged to avoid the risk of dividing zero, and the
> summation in the numerator could contain negative terms. Although the weights for each value v_i in
> Equation 19 are not strictly probabilistic normalized, we kindly argue that the computation can still
> model the importance of values."*

This is the paper's weakest link and it is stated in the paper's own voice. Numerator and denominator
no longer compute the same similarity; the attention weights are not a distribution. The claim in the
abstract — *"the capability of equipping the linear self-attention with relative position encoding"* —
is a claim about *feasibility*, and the construction that delivers it breaks normalization.

**What is not touched.** Eq. 14 defines `f_{q,k}` only. `f_v` never receives `R^d_Θ`, anywhere in the
paper. `Θ` is *"pre-defined"*; nothing in eq. 14/15 is differentiated. The input embedding `x_m` is
never modified — RoPE enters *after* `W_q`/`W_k`, inside the head, at every layer, and touches nothing
in the residual stream. See "What the source does *not* establish" for how much of that is stated
versus how much is true-by-omission, because the difference matters.

## Numbers that matter

**The frequency ladder in the paper's units.** `θ_i = base^(−2i/d)` with `base = 10000`, `i` over
pairs. Wavelength `2π/θ_i`.

**In this app, RoPE runs on the head dimension** — `position.js` builds `rope({dims = DH})` with
`DH = d_k = 8`, so there are **4 pairs per head**, not 16. That changes the picture sharply:

| pair `i` | `θ_i = 10000^(−i/4)` | rad/token | wavelength (tokens) | turn over a 16-token sentence |
|---|---|---|---|---|
| 0 | 1 | 1.0 | **6.283** | 16 rad ≈ 2.5 full turns |
| 1 | 0.1 | 0.1 | 62.83 | 1.6 rad ≈ 92° |
| 2 | 0.01 | 0.01 | 628.3 | 0.16 rad ≈ 9.2° |
| 3 | 0.001 | 0.001 | 6283 | 0.016 rad ≈ 0.9° |

| quantity | paper | this app | note |
|---|---|---|---|
| rotated width | `d` (generic; BERT-base head dim 64 in §4.2) | `d_k = 8` per head | 4 pairs |
| base constant | 10000 | 10000 | same, then a slider |
| pairs completing a cycle in a 16-token sentence | — | **1 of 4** | only `i=0`; at base 10 it is 2 of 4 (λ = 6.28, 11.2, 19.9, 35.3) |
| learned parameters added | **0** | **0** | vs. learned table 12×32 = 384, vs. Shaw `(2k+1)·d_z` = 72/head at k=4 |
| extra score terms | **0** | **0** | vs. Shaw's per-pair operand, which cost 7% throughput |
| `‖R_m x‖ − ‖x‖` | 0 (orthogonal, eq. 16) | ~1e-16 | float check the reader can run |
| embedding budget consumed | none | **none** | `cos(x_in, x_used) = 1.000000` exactly |
| pieces of V rotated | none | none | rotation is on q and k only |

**Experimental results, all of them.**

| experiment | setting | baseline | RoFormer |
|---|---|---|---|
| WMT14 En-De (Table 1) | fairseq, beam 4, avg last 5 ckpts | Transformer-base **27.3** BLEU | **27.5** BLEU |
| MLM pretraining (Fig. 3 left) | BookCorpus+Wikipedia, bs 64, len 512, 100k steps | BERT-base-uncased | *"RoFormer experiences faster convergence"* (curve only, no final number) |
| Performer + RoPE (Fig. 3 right) | Enwik8, 12-layer char Performer, 768d/12h, len 1024 | Performer | *"rapid convergence and lower loss"* (curve only) |
| GLUE (Table 2) | 3 epochs, len 512, bs 32 | BERT | see below |
| CAIL2019-SCM (Table 5) | Chinese legal triplets, 8964, 6:2:2 split | BERT-512 / WoBERT-512 | see below |

GLUE, **complete** (the paper reports only the wins in prose):

| | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
|---|---|---|---|---|---|---|
| BERT | 88.9 | **93.5** | **90.5** | 85.8 | 71.2 | **84.6/83.4** |
| RoFormer | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

> *"As can be seen, RoFormer can significantly outperform BERT in three out of six datasets, and the
> improvements are considerable."*

It loses the other three, by 2.8 (SST-2), 2.5 (QNLI) and 4.4/3.6 (MNLI) — never mentioned. The +15.2
on QQP is far outside the spread of every other cell and is much more plausibly a broken baseline than
a position-encoding effect.

CAIL2019-SCM (validation / test accuracy):

| model | valid | test |
|---|---|---|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | **68.29%** |
| RoFormer-1024 | **66.07%** | **69.79%** |

> *"With short text cut-offs, i.e., 512, the result from RoFormer is comparable to WoBERT and is
> slightly better than the BERT implementation. However, when increasing the maximum input text length
> to 1024, RoFormer outperforms WoBERT by an absolute improvement of 1.5%."*

**The long-sequence statement, and the number that undercuts it.** §4.5.2:

> *"As shown in Table 4, the accuracy of RoFormer elevates with an increasing upper bound of sequence
> length, which demonstrates the ability of RoFormer in dealing with long texts. We claim that this is
> the attribute to the excellent generalizability of the proposed RoPE."*

Table 4 is the **pre-training** schedule, and its max sequence lengths are 512, **1536**, 256, 128,
**1536**, 512 across six stages. RoFormer was therefore *trained* at 1536 before being *evaluated* at
1024. The headline long-text result is inside the training distribution. There is no evaluation
anywhere in this paper at a length the model was not trained on.

## What the live view must let the reader do

The toy: 32 dims, 4 heads (`d_k = 8`), 2 blocks, causal mask, seeded untrained weights, editable
~16-token sentence. `position.js` already exposes `rope({base, stretch, dims})` returning
`rotate(v, pos)`, and the mixer applies it to q and k before the dot product. Every interaction below
changes a real computed quantity through that path. None of them repeats concept 2's shuffle test,
offset curve, or base slider — those were about a *vector added to the input*; these are about a
*rotation applied inside the head*.

1. **Slide the whole sentence down the number line — the identity, to floating point.**
   A `Δ` control (0 → 4096) that offsets every token's position by the same amount, leaving the
   sentence untouched. Recompute the full pre-mask score matrix through a real head and display
   `max_{i,j} |A_Δ[i,j] − A_0[i,j]|`. With RoPE it reads **~1e-15**, at Δ = 1, Δ = 100, Δ = 4096
   alike — the score genuinely depends only on `m−n`, and the reader has just verified eq. 11 on their
   own sentence rather than being told it. Now flip the scheme selector to *sinusoidal* and re-run the
   identical test: the number is large and grows with Δ, because the additive encoding changes the
   content vector that the projections see. Two schemes, one test, one number each. Show eq. 11 and
   `R_m^⊺ R_n = R_{n−m}` beside the readout so it is clear which line of the paper just got checked.

2. **The budget meter — position that costs the embedding nothing.**
   Three readouts, live, for the reader's sentence: (a) `cos(x_in, x_used)` per token and
   `‖x_used‖ − ‖x_in‖` — with RoPE both read **1.000000** and **0.000000** exactly, because the
   residual stream is never touched; switch to sinusoidal and they move visibly (concept 2 measured
   that cost, this card measures its absence). (b) `‖rotate(q,m)‖ − ‖q‖` ≈ **1e-16** for every token
   and head — eq. 16's *"R^d_Θ is an orthogonal matrix"*, checked. (c) a parameter counter next to the
   scheme selector: RoPE **+0**, learned table **+384**, Shaw at k=4 **+72/head**. And yet the score
   matrix and the model's top-1 prediction both change when RoPE is toggled — position is doing work
   while consuming zero channels and zero parameters. On the dataflow picture, light the `rotate` hook
   between `W_q`/`W_k` and the dot product, and leave the V path grey. Add a **"rotate V too"** toggle:
   it is one line of the same hook, it is *not* what the paper does, and turning it on kills the
   invariant from interaction 1 — the Δ readout jumps from 1e-15 to large. That is the cleanest
   possible demonstration of why values are excluded.

3. **The decay claim, plotted honestly — two curves, one bound.**
   On the curve view, against relative distance `Δ = m−n` from 0 to 64 (past the sentence, on purpose),
   plot: (a) the paper's own quantity `(1/(d/2)) Σ_{i=1}^{d/2} |S_i|` with `S_j = Σ_{i<j} e^{iΔθ_i}` —
   literally Figure 2, at `d = 8`; (b) the **actual** `|q_m · k_n|` for a chosen head and query token
   from the reader's sentence, extended past the sentence end. Print the third quantity the bound
   depends on: `max_i |h_{i+1} − h_i|` as a live number, and the resulting slack
   `bound / |q_m·k_n|`. The point that lands: with 4 pairs curve (a) is a jagged staircase, not a
   smooth decay; curve (b) does not track it; and the slack factor is large and content-dependent. The
   paper says *"One can prove"* — this view shows exactly which half was proved. A base slider
   (10 → 100000) re-derives both curves and prints `pairs whose wavelength ≤ sentence length: 1 of 4 at
   base 10000, 2 of 4 at base 10`, so the reader can see the decay claim is a statement about the
   *frequency schedule*, not about rotation.

4. **Run past the end of training — the setup for concepts 16–18.**
   Two things happen when Δ from interaction 1 goes large, and both are numbers. (a) **Nothing breaks.**
   `rotate(v, 4096)` computes; there is no table to fall off, unlike the learned scheme, which returns
   the token unmodified past its last row. Show them side by side: learned table → "no row for position
   4096"; RoPE → a perfectly valid vector of the same norm. (b) **But the angles have wrapped.** With
   `θ_0 = 1` rad/token the fastest pair repeats every 2π ≈ 6.283 tokens of *distance*. Plot
   `q_m · k_n` against `Δ` out to 4096 and print `|score(Δ) − score(Δ + 6.283)|` alongside the score's
   own magnitude — the fast pair's contribution is near-identical, so distances 8 apart and distances
   14 apart are partly indistinguishable to it. Defined everywhere, discriminative only where the slow
   pairs still separate. Then turn the **`stretch`** knob that `position.js` already has, 1 → 0.25:
   every angle scales, so position 4096 is rotated as if it were 1024, back inside the trained range.
   Display the max change to the attention rows. That knob **is** position interpolation, and naming it
   here is what makes concepts 16–18 land as answers instead of new topics.

5. **The linear-attention variant, including the part the paper concedes.**
   If the mixer's kernel/linear path is available, apply `rotate` to `φ(q)` and `φ(k)` per eq. 19 while
   leaving the denominator un-rotated exactly as written. Display two numbers: the fraction of `(m,n)`
   pairs whose numerator term is **negative**, and `max_m |Σ_n w_{m,n} − 1|`. Both are nonzero. Put the
   paper's own sentence under them — *"the summation in the numerator could contain negative terms …
   not strictly probabilistic normalized … we kindly argue"* — and the reader sees a concession
   measured rather than quoted. This is the one interaction that reaches back to concept 9, and it is
   the reason RoPE, not Shaw's scheme, is what linear-attention models ended up using.

## What the source does *not* establish

- **It does not prove long-term decay, despite saying it does.** §3.3 says *"One can prove that this
  setting provides a long-term decay property"*. §3.4.3 proves an inequality (eq. 37) and then asserts
  the bound's decay *"as shown in Figure 2"* — numerically, for one plot, at one width. The bound's
  leading factor `max_i |h_{i+1} − h_i|` is query/key-dependent and never bounded, so even a genuinely
  decaying `Σ|S_i|` does not constrain the actual inner product. The correct summary: *the frequency
  schedule makes an upper bound decay, empirically.* Nothing more. This is the single most-repeated
  overclaim about RoPE and the paper is its source.
- **It never says "no learned parameters".** The phrase does not appear. `Θ` is *"pre-defined
  parameters"* (§3.2.2) and `θ` is *"a preset non-zero constant"* (§3.2.1). Zero-parameter is true —
  read eq. 14 and 15 and there is nothing to differentiate — but it is a property of the equations the
  card derives, not a sentence the card can quote.
- **It never states that RoPE is withheld from the values.** Eq. 14 defines `f_{q,k}` and only
  `f_{q,k}`; `f_v` from eq. 1 is silently dropped and never reappears. The one sentence in the paper
  about position in the value term is about *prior* work (§2.3, on Shaw/Transformer-XL): *"It is
  noteworthy that the position information in the value term is removed by setting f_v(x_j) :=
  W_v x_j. Later work … followed these settings by only encoding the relative position information into
  the attention weights."* RoPE inherits that convention by omission. True, load-bearing, and
  unargued — which is exactly why interaction 2's "rotate V too" toggle earns its place.
- **It does not demonstrate length extrapolation.** The abstract claims *"the flexibility of sequence
  length"* (v1/v2: *"flexibility of being expand to any sequence lengths"*), and the longest evaluation
  is 1024 on CAIL2019-SCM — while Table 4 shows the model was pre-trained at max length **1536**. No
  experiment anywhere runs the model past a length it was trained on. "Defined at any position" is a
  property of eq. 15; "works at any position" is not tested here and, as concepts 16–18 exist to say,
  is false in practice.
- **It cannot explain its own results, and says so.** §4.5.5: *"there lacks of thorough explanations on
  why it converges faster than baseline models"*, and *"our model shows superior performance on long
  texts than peer models, we have not come up with a faithful explanation."*
- **The empirical margins are thin or mixed.** +0.2 BLEU, single run, no variance reported anywhere in
  the paper. GLUE is 3 wins and 3 losses, with the prose naming only the wins and the largest win
  (QQP +15.2 F1) sitting so far outside every other margin that the baseline is the more likely
  explanation. The two pretraining results are loss *curves* with no final numbers.
- **The linear-attention combination is not a clean drop-in.** Eq. 19 rotates the numerator only; the
  paper acknowledges negative terms and lost normalization and argues by assertion (*"we kindly
  argue"*). No downstream evaluation of that variant exists beyond one Enwik8 loss curve.
- **Internal inconsistencies to not propagate.** (i) The frequency schedule is written twice with
  different indexing: `θ_i = 10000^(−2(i−1)/d), i ∈ [1,…,d/2]` in eq. 15, and `θ_i = 10000^(−2i/d)` in
  §3.3 and §3.4.3 — the same ladder, 1-based vs 0-based. This app uses the 0-based form. (ii) The sign
  convention flips: eq. 11 requires `g(x_m, x_n, m−n)`, eq. 21 in the derivation writes
  `g(x_m, x_n, n−m)`, eq. 16 produces `R_{n−m}` and eq. 35 produces `e^{i(m−n)θ}`. It is not
  cosmetic — the score contains `Im[h_i] sin((m−n)θ_i)`, which is **odd** in `m−n`, so the score is not
  symmetric under swapping the two tokens. (iii) Eq. 16 drops the transpose on `W_q` and the subscript
  on the leading `x`.
- **Nothing here says why 10000.** *"Following Vaswani et al."* is the entire justification, inherited
  from a paper that also gave none. That the constant survived four years and a change of mechanism is
  a fact about convention, not about evidence — and it is why the base slider belongs on the card.
- **The app establishes nothing empirical.** Seeded untrained weights, no training. Every number the
  live view produces is geometry — which is precisely why interaction 1 is worth doing: an exact
  algebraic identity is the one class of claim a toy with random weights can settle completely.

## Leaves behind

**Backward, this closes two open bills at once.** Shaw (concept 3) got the Toeplitz property by adding
a per-pair operand inside the dot product, and paid for it: attention stopped being one matrix
multiply, and the rescue cost 7% throughput plus `O(n²d_a)` memory plus learned parameters per unit.
RoPE gets the same property with **zero** extra terms — `R_m^⊺ R_n = R_{n−m}` means the relative
dependence is already inside a plain `QK^⊺`. Shaw also needed `clip(m−n, ±k)`, a distance ceiling
justified by *"the hypothesis that precise relative position information is not useful beyond a certain
distance"*; RoPE has no clip and no ceiling, and replaces that hypothesis with the (partly proved)
decay of §3.4.3. And sinusoidal (concept 2) put position into the residual stream, summed with content
under one `d_model` budget and one dropout — concept 2's interaction 5 measured that competition
directly. RoPE never touches the embedding: `cos(x_in, x_used) = 1` exactly, forever. The seam moved
from the bottom of the stack to inside every head, which is also why RoPE applies at **every** layer
rather than once.

**Forward, it opens exactly one thing, and opens it wide.** `R^d_{Θ,m}` is defined for every real `m`.
That is a property of the formula, and the paper's *"flexibility of sequence length"* leans on it. But
the model is only ever *trained* at the positions it saw — here, at most 1536 — and the fastest pair
turns a full circle every 6.283 tokens, so angles far outside the trained range are not new, they are
recycled. Defined everywhere, calibrated nowhere. Every method on the back half of this timeline is a
response to that one sentence: position interpolation scales `m` down so long contexts land back
inside the trained arc; NTK-aware scaling raises the base instead so the fast pairs stretch and the
slow pairs are left alone; YaRN separates the two by wavelength and treats each band differently. All
three are edits to the *same two constants this card already exposes* — the base and the stretch — and
`position.js` has both knobs today. The card should end by having the reader turn `stretch` and watch
a score move, so that when concepts 16–18 arrive, they are recognized as answers to a question the
reader already asked with their own hands.
