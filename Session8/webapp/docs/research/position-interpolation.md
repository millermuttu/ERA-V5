# Concept 16 — Position Interpolation

**Card id:** `position-interpolation` · **Date:** 2023-06-27 (arXiv v1) · **Pressure:** where a token sits

## What was read

- [arXiv:2306.15595](https://arxiv.org/abs/2306.15595), Shouyuan Chen, Sherman Wong, Liangjian Chen,
  Yuandong Tian (Meta Platforms Inc.) — *Extending Context Window of Large Language Models via
  Position Interpolation*. Abstract page for the version history.
- [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2306.15595), read end to end: §1 Introduction,
  §2.1 Background (RoPE), §2.2 Direct Extrapolation, §2.3 Proposed approach, §3.1–3.5 Experiments
  (Tables 1–6), §4 Related Work, §5 Conclusions, **Appendix A (the proof of Theorem 2.1)**,
  **Appendix B (the plot of `B(s)/d`)**, and **Appendix C (the source code for Figures 2 and 5)** —
  Appendix C is where the paper's central figure turns out to be a synthetic least-squares fit rather
  than a measurement, which is the single most load-bearing thing in this note.
- Both **PDFs (v1 and v2)**, text-extracted locally and diffed, because ar5iv mangles two sentences
  that matter (the step-0 perplexity claim, and the proof's inequality condition).

**Version history, verified.** v1 **Tue, 27 Jun 2023 16:26:26 UTC**; v2 Wed, 28 Jun 2023 04:26:05 UTC.
The timeline's `"2023-06-27"` is **correct**. There are only two versions, seventeen hours apart, and
the v2 comment field says *"Fix template issues"*. A line-by-line diff of the two extracted PDFs shows
exactly that and nothing else: v1 carried a stray running header, **"Published as a conference paper
at ICLR 2021"**, on every page — a leftover from the LaTeX template — and v2 removes it. Every other
diff line is whitespace reflow from that removal. **No text, no number, no table, no equation changed
between v1 and v2.** Cite either; the timeline's v1 date is the right one.

**Provenance note the card should carry.** §1 concedes a concurrent, earlier, non-academic origin:

> *"Right before our release, we are informed with a concurrent blogpost (SuperHOT kaiokendev 2023)
> that also interpolates positional encoding in RoPE to extend the context window from 2K to 8K.
> Recently, open source community picks it up in Reddit post and Github Issues, which shows that
> fine-tuning with LoRA (Hu et al. 2021) also seems to work well."*

The idea was in the wild first. What this paper adds, by its own accounting, is scale (*"a full
fine-tuning with up to 65B model"*) and *"theoretical explanations why interpolation achieves much
more stable results than extrapolation"*. Both of those claims need auditing, and both get it below.

## The mechanism, precisely

**The problem, stated as the thing concept 12's card left open.** §1:

> *"While certain techniques such as ALiBi (Press et al. 2022) and LeX (Sun et al. 2022) enable length
> extrapolation of Transformers, i.e. train on short context windows and inference on longer ones,
> many existing pre-trained LLMs, including LLaMA (Touvron et al. 2023), use positional encodings that
> have weak extrapolation properties (e.g., RoPE (Su et al. 2021)). Therefore, the applicability of
> these techniques for extending the context window sizes of such LLMs remains limited."*

That is the fork in the road the ALiBi card predicted, taken from the other side: the weights already
exist, they already have RoPE, and swapping the position scheme means retraining. §2.2 puts the
failure in one number: *"when directly extending to larger context windows unseen in the training, the
perplexity may shoot up to very high numbers (i.e., `>10³`), comparable to untrained models."*

**The modification, complete.** §2.1 restates RoPE as a complex function (eq. 1), with
`θ_j = 10000^(−2j/d)` and `d` the head dimension, and the score `a(m,n) = Re⟨f(q,m), f(k,n)⟩ = a(m−n)`
(eq. 2). Then §2.3, in its entirety, is:

> *"instead of extrapolate the attention score in Eqn. 3 to `s > L`, how about we define an attention
> score `ã(s) = a(Ls/L′)` where `L′` is the longer context window? Formally, we replace RoPE **f** by
> **f′** defined as follows"*

    f′(x, m) = f( x , m·L/L′ )                                    (4)

> *"We call this transformation on the position encoding Position Interpolation. In this step, we
> reduce position indices from `[0, L′)` to `[0, L)` to match the original range of indices before
> computing RoPE. Consequently, as inputs to RoPE, the maximum relative distance between any two
> tokens has been reduced from `L′` to `L`."*

Read where it sits. The scale factor `L/L′` multiplies the **position index**, *before* it enters the
rotation, and nothing else in the model is touched. Not the frequencies `θ_j`, not the base 10000, not
the query, not the key, not the value, not a weight anywhere:

> *"Notably, our method of rescaling of position indices does not introduce extra weight, or modify the
> model architecture in any way. This makes it attractive in practical applications, since most
> infrastructure and optimization for the original model can be reused after the extension."*

Because `m·θ_j·(L/L′)` is what actually reaches the cosine, scaling the index and scaling every
frequency are the same operation — which is why **this app's `rope({stretch})` parameter already *is*
Position Interpolation, exactly**, with `stretch = L/L′`. That equivalence is verified to
bit-for-bit equality in "Numbers that matter" below, and it is the reason this card is cheap to build.

**The enabling fact, stated once and never dwelt on** (§1):

> *"to accommodate more input tokens, we interpolate the position encodings at neighboring integer
> positions, utilizing the fact that position encodings can be applied on non-integer positions, as
> opposed to extrapolating outside the trained positions, which may lead to catastrophic values."*

RoPE's rotation is a function of a real number. A learned table (concept 4) has no row for position
2.5; RoPE has an angle for it. Extending a 2048-window model to 4096 by PI means every odd token now
sits at a half-integer position the model has literally never seen. **This is where the whole cost of
the method lives**, and it is what Theorem 2.1 is a bound on.

**Why extrapolation fails, in the paper's own diagnosis** (§2.2). This section starts by demolishing
the property concept 12's card already flagged as over-claimed:

> *"How could this happen if the attention score `a_{m−n}` decays as the relative distance `|m−n|`
> increases, according to Section 3.4.3 of (Su et al. 2021), and content from very far distances should
> not matter that much? It turns out that the upper bound derived in Section 3.4.3 of (Su et al. 2021)
> may be too loose: while it indeed decays with respect to `|m−n|`, the bound can still be quite large
> (i.e., the bound can be critically depends on the magnitude of `v_j`) and thus vacuous."*

(`v_j` is undefined in this paper; from context it means Su et al.'s `h_j`. A typo, but note it — the
sentence is the one everybody quotes.) This is an independent paper reaching exactly the conclusion the
RoPE card's research reached: the long-term decay bound is real, and it is vacuous because its leading
factor is content-dependent and unbounded. Two sources, one verdict; the card can now say it without
hedging.

Then the positive account, which is a statement about **function classes, not about models**:

    a(s) = Re[ Σ_{j=0}^{d/2−1} h_j e^{isθ_j} ],   h_j := (q_2j + i q_2j+1)(k_2j − i k_2j+1)   (3)

> *"The underlying reason is that the trigonometric family `{φ_j}` (with sufficiently large `d`) is a
> universal approximator and can fit any arbitrary functions. Therefore, for `a_s`, there **always
> exist** coefficients `{h_j}` (i.e. key and query) that corresponds to small function values in
> [0, 2048] but much larger in regions beyond."* (emphasis added)

"There always exist" is an existence claim. Hold onto it; Figure 2 is what stands in for a measurement,
and Appendix C.1 shows Figure 2 is `torch.randn` fitted by least squares.

**Theorem 2.1 (Interpolation bound), verbatim.** For `a(s) = Re[Σ_j h_j e^{isθ_j}]` with
`θ_j = c^(−2j/d)`, and `s ∈ [s₁, s₂]`:

    | a(s) − a_linear(s) |  ≤  d ( max_j |h_j| ) (s − s₁)(s₂ − s) / (8 ln c)          (5)

    a_linear(s) := (1 − λ(s)) a(s₁) + λ(s) a(s₂),   λ(s) := (s − s₁)/(s₂ − s₁)        (6)

> *"where `a_linear(s)` is the linear interpolation of two grid point `a(s₁)` and `a(s₂)` that are
> known to behave well, enforced by LLM pre-training … Intuitively, in LLM pre-training, we know that
> the attention score `a(s)` behaves well on integer grid `s₁` and `s₂`. Therefore, for any
> interpolation `s ∈ [s₁, s₂]`, we have `(s − s₁)(s₂ − s) ≤ 1/4`. Note that `c = 10000`, the bound
> becomes:"*

    | a(s) − a_linear(s) | ≤ d/(32 ln c) · max_j |h_j| ≈ d · max_j |h_j| / 294.73      (7)

**The comparison that produces the headline.** Eq. 8 is Su et al.'s §3.4.3 bound, restated:

    |a(s)| ≤ (max_j |h_j − h_{j+1}|) Σ_k |A_{k+1}(s)| ≤ 2(max_j |h_j|) Σ_k |A_{k+1}(s)|,
        A_k(s) := Σ_{j=0}^{k−1} e^{isθ_j}                                              (8)

> *"While there is no close form for `B(s) := Σ_{k=0}^{d/2−1} |A_{k+1}(s)|`, numerically it is at least
> larger than `d`, and for many positional difference `s`, `B(s)` is much larger than `d` (check
> Appendix B for the plot). Therefore, the interpolation bound is at least `2 · 294.73 ∼ 600×` smaller
> than the extrapolation bound, and thus the interpolated attention score is much more stable than
> extrapolated one."*

So `600 = 2 × 32 ln(10000)`, times `B(s)/d ≥ 1`. **Three things about that number, all of which the
card must carry:**

1. **`B(s)/d ≥ 1` is asserted numerically at one head width and never proved.** Appendix B: *"We use
   `θ_j = c^(−2j/d)` with `c = 10000` and `d = 4096/32 = 128` (LLaMA-7B setting), and Fig. 5 shows that
   `B(s)/d` almost always larger than 1"*, and Figure 5's caption concedes the direction of travel:
   *"The bound `B(s)/d` decays with `s`. While the bounds goes down with large positional difference
   `s`, numerically `B(s)/d ≥ 1` and at many `s` much larger than 1."* Decaying toward the floor that
   the whole 600× rests on. And it is **d-dependent**: at `d = 8` it fails outright (measured below).
2. **The two bounds bound different quantities.** Eq. 5 bounds the *deviation of `a(s)` from the linear
   chord between two integer grid points* — conditional on those grid values being *"known to behave
   well, enforced by LLM pre-training"*. Eq. 8 bounds `|a(s)|` itself. Calling one "600× smaller than"
   the other compares an interpolation *error* against an absolute *magnitude*. They are not the same
   units of claim, and the abstract's *"the upper bound of interpolation is at least ∼600× smaller than
   that of extrapolation"* elides that.
3. **The proof of eq. 5 contains an inequality that runs the wrong way.** Appendix A is four lines of
   Taylor remainder plus a bound on `|a″(s)|`. Eq. 13 is exact and correct:
   `|a″(s)| ≤ (max_j|h_j|) Σ_j c^(−4j/d) = (max_j|h_j|)/(1 − c^(−4/d))`. Eq. 14 then replaces
   `1/(1 − c^(−4/d))` by `d/(4 ln c)`, justified by *"Note that when x<1, `c^x ≤ 1 + x ln c`"*. That
   inequality is **false**: `c^x` is convex, so `c^x ≥ 1 + x ln c` for every real `x` — the tangent line
   lies below the exponential, always. Checked numerically: at `d = 128`, `c^(−4/d) = 0.74995` while
   `1 + x ln c = 0.71218`, so `0.74995 ≤ 0.71218` is false. The consequence is that `d/(4 ln c)` is a
   **lower** bound on `1/(1 − c^(−4/d))`, not an upper one, and eq. 5's constant is therefore *too
   small*. The correct constant is `max_j|h_j| / (8(1 − c^(−4/d)))`. See "Numbers that matter" for how
   much that costs: **15% at `d = 128` (headline survives), 4.65× at `d = 8` (headline does not)**.

**Fine-tuning, exactly as specified** (§2.3 and §3.1):

> *"We can further fine-tune the interpolated model using the next token prediction task with
> interpolated position encodings on the extended context window size using a pre-training corpus such
> as the Pile (Gao et al. 2020). In the next section, we show that our fine-tuning process only needs
> tens to hundreds thousands of examples. We also find that the result of the fine-tuning is not
> sensitive to the choice of examples. The reason may be that the model is only adapting to the new
> context window during the fine-tuning phase, starting from a good initialization, as opposed to
> acquiring new knowledge."*

§3.1: AdamW, `β₁ = 0.9`, `β₂ = 0.95`, linear warmup of 20 steps from 10% of max LR, LR `2×10⁻⁵` for
7B/13B and `10⁻⁵` for 33B/65B, weight decay zero. *"For extending 7B, 13B and 33B models to the 8192
context window size, we use 32 A100 GPUs and 64 global batch size. For all other cases we use 128 A100
GPUs and 128 global batch size."* PyTorch + FSDP + FlashAttention. **1000 steps for PI, 10000 steps for
the direct-fine-tuning baseline** — the baseline gets ten times the step budget and still loses.

**One more idea the paper raises and drops**, worth a footnote because concepts 17–18 do not take it up
either:

> *"a common term is `max_j |h_j|`, which is the maximal magnitude of query/key products. If we enforce
> a regularization on `|h_j|` during LLM training, it is possible that the catastrophic extrapolation
> error can be mitigated or even resolved. In fact, if we apply ridge regression with proper
> regularization to fit a curve in Fig. 2, the magnitude of extrapolated `a(s)` when `s > L` can be
> comparable to that within `[0, L]`. To our knowledge, we are not aware of existing LLM pre-training
> techniques that leverage this regularization and will leave it for future work."*

Note what that admits: the blow-up in Figure 2 is removable by changing the *fitting procedure*. That
is a strong hint about what Figure 2 actually measures.

## Numbers that matter

### Fine-tuning cost

| quantity | paper | derived here |
|---|---|---|
| PI steps | **1000** (§3.1; passkey saturates at **200**, Table 4) | — |
| direct-FT steps | **10000** (§3.1), *"more than 10000 batches"* (§1) | 10× PI's budget |
| examples | *"tens to hundreds thousands of examples"* | 64k (batch 64) or 128k (batch 128) |
| hardware | 32 A100s → 8192; 128 A100s otherwise | — |
| tokens, → 8192 | not stated | **0.52 B** = 1000 × 64 × 8192 **[derived]** |
| tokens, → 16384 | not stated | **2.10 B** = 1000 × 128 × 16384 **[derived]** |
| tokens, → 32768 | not stated | **4.19 B** = 1000 × 128 × 32768 **[derived]** |
| direct-FT → 8192 | not stated | **5.24 B** tokens **[derived]** |

**What "1000 steps" is a fraction of.** The paper says only *"The cost of fine-tuning is negligible
compared to the pre-training costs"* (§1) and gives **no number at all** — no GPU-hours, no wall-clock,
no token count, no comparison figure anywhere. So the fraction has to be assembled from outside: LLaMA
7B/13B were pre-trained on **1.0 T tokens** and 33B/65B on **1.4 T** (Touvron et al. 2023, *not* this
paper). Against 1.0 T, the 8192 extension's 0.52 B tokens is **≈ 0.05%** of pre-training, and even the
32768 extension's 4.19 B is **≈ 0.42%** **[derived, using an external figure — flag it as such on the
card]**. That is the honest version of "negligible", and it is a three-order-of-magnitude claim the
paper leaves the reader to compute.

### The theory, recomputed

`32 ln 10000 = 294.7309` **[measured here]** — the paper's `294.73` checks out, and `2 × 294.73 ≈ 589` is
where `∼600×` comes from.

`B(s)/d`, computed over `s ∈ [0, 4096]` at integer `s`, for the paper's `d = 128` and the app's `d = 8`
**[measured here]**:

| `d` | `B(0)/d` | max `B(s)/d` | **min** `B(s)/d` | fraction of `s` with `B(s)/d < 1` | implied ratio `2·294.73·B(s)/d` |
|---|---|---|---|---|---|
| **128** (LLaMA 7B) | 16.250 | 16.250 | **1.1159** (at `s`=3643) | **0.0%** | **658× … 9579×** |
| **8** (this app) | 1.250 | 1.250 | **0.2595** (at `s`=2440) | **87.0%** | **153× … 737×** |

Two readings. First, at `d = 128` the paper's numerics are **reproduced and confirmed**: `B(0)/d =
16.25` is exactly the peak of Figure 5's y-axis, `B(s)/d ≥ 1` holds at every integer `s` in the range,
and the minimum 1.1159 gives 658× — so *"at least ∼600×"* is honest at the width they checked. (Closed
form for the peak, derived here: `A_k(0) = k`, so `B(0)/d = (d/2)(d/2+1)/(2d) = (d+2)/8`; at `d=128`
that is 16.25 exactly.) Second, at `d = 8` the floor **fails 87% of the time** and the ratio falls to
153× at worst. `B(s)/d ≥ 1` is a fact about wide heads, not a fact about RoPE, and the paper checks
precisely one width.

**The proof's constant, corrected** **[derived here]**. Using eq. 13's exact geometric sum rather than
eq. 14's reversed inequality, the interpolation bound with `(s−s₁)(s₂−s) ≤ 1/4` is
`max_j|h_j| / (8(1 − c^(−4/d)))`:

| `d` | paper's eq. 7 coefficient `d/(32 ln c)` | correct coefficient `1/(8(1−c^(−4/d)))` | understatement |
|---|---|---|---|
| 128 | 0.43429 | 0.49990 | **1.15×** |
| 8 | 0.02714 | 0.12626 | **4.65×** |

At LLaMA's width the error is 15% and the headline is unaffected; re-running the ratio with the
corrected constant gives `16(1 − c^(−4/d))·B(s) ≥ 571×` at `d = 128` — still "about 600". At `d = 8`
the corrected ratio is **33× … 158×**. **The theorem's shape is right and its stated constant is not
established by its own proof.** Say exactly that; do not repeat "600×" as if it were derived.

**How tight is Theorem 2.1 on real vectors?** Driving the app's own model — all 2 blocks × 4 heads ×
all causal `(m,n)` pairs, `h_j` formed from the model's real `Q`/`K`, excursion measured on every unit
interval `s₁ ∈ [0, 32)` at step 0.02 **[measured here]**:

| quantity | value |
|---|---|
| mean actual `\|a(s) − a_linear(s)\|` | **0.4755** |
| worst actual | **3.6577** |
| mean of eq. 7's bound with the same `max_j\|h_j\|` | **0.3043** |
| median slack (bound / actual) | **0.8** |

The measured excursion **exceeds the bound as eq. 7 states it** — mean 0.4755 against 0.3043, median
slack below 1. That is not a bug in the measurement; it is the eq. 14 error showing up, and `d = 8` is
where it bites hardest (4.65× understatement). Against the *corrected* coefficient the mean is
comfortable (0.126 × mean `max|h_j|` 11.53 = 1.456 vs 0.476 measured); a per-pair check of the
corrected bound was **not** run, so do not claim it holds everywhere. Also measured: eq. 8's chaining
step `max_j|h_j − h_{j+1}| ≤ 2 max_j|h_j|` gives away a mean factor of **0.619** (max 0.984) on real
vectors — so the extrapolation side of the comparison is itself ~1.6× looser than it needs to be.

### Figure 2, reproduced — and what it actually shows

Appendix C.1's recipe, re-implemented and run **[measured here]**: sample `L` i.i.d. standard-normal
targets, least-squares fit them with the `d` trig basis functions `{sin(sθ_j), cos(sθ_j)}`, then
evaluate the fitted curve on `[L, 2L)`.

| `d` | `L` | max `\|a\|` on `[0,L)` | max `\|a\|` on `[L,2L)` | ratio | max `\|coef\|` |
|---|---|---|---|---|---|
| **128** | **2048** (paper's setting) | **1.100** | **17776** | **16155×** | 1531 |
| 8 | 16 (app scale) | 1.267 | 154.8 | 122× | 14542 |
| **8** | **2048** | **0.138** | **0.1** | **1.0×** | **0.1** |

Row 1 reproduces the paper: their caption says the fit is *"approximately within [−1,1]"* inside the
window (measured 1.100) and *"out of this region it may goes beyond 8000"* (measured 17776 on a
different seed — same order, same phenomenon). The figure is real and replicable.

Row 3 is the control the paper does not run, and it changes the interpretation. **Give the same basis
only 8 functions against 2048 constraints and the blow-up vanishes entirely — ratio 1.0×, coefficients
of order 0.1.** The catastrophe is a function of how *rich* the basis is relative to the range being
fitted, i.e. of over-parameterisation driving `|h_j|` to enormous values (1531 and 14542 above). It is
not a property of extrapolating a rotation.

And the universal envelope makes this exact: from eq. 3, `|a(s)| ≤ Σ_j |h_j|` for **every** `s`, real or
integer, in-window or out. A score can only be "small inside `[0,L]` and huge outside" if it is
achieving massive cancellation inside — which is what least-squares against white noise manufactures.
Measured on the app's real (untrained) `Q`/`K` across every block, head and causal pair, scanning
`s ∈ [16, 4096]` at step 0.25 **[measured here]**: the **worst** ratio of out-of-window peak to
in-window peak is **9.32×** (block 1, head 1, `m=10`, `n=7`), with `max_j|h_j| = 12.83` and
`Σ_j|h_j| = 20.16` — the envelope, respected. Mean `Σ_j|h_j| = 25.28`, mean `max_j|h_j| = 11.53` in
block 0.

Caveat, stated plainly: this app is **untrained**, so 9.32× is not evidence about LLaMA. The
load-bearing part is the inequality `|a(s)| ≤ Σ_j|h_j|`, which is exact and universal, plus the fact
that **the paper never measures `h_j` from any real model, anywhere.**

### Quality — every number

**Table 1, PG-19 perplexity**, sliding window, stride `S = 256`, following Press et al.:

| size | window | method | @2048 | @4096 | @8192 | @16384 | @32768 |
|---|---|---|---|---|---|---|---|
| 7B | 2048 | None | 7.20 | >10³ | >10³ | >10³ | >10³ |
| 7B | 8192 | FT | 7.21 | 7.34 | 7.69 | — | — |
| 7B | 8192 | PI | 7.13 | 6.96 | 6.95 | — | — |
| 7B | 16384 | PI | 7.11 | 6.93 | 6.82 | 6.83 | — |
| 7B | 32768 | PI | 7.23 | 7.04 | 6.91 | 6.80 | **6.77** |
| 13B | 2048 | None | 6.59 | — | — | — | — |
| 13B | 8192 | FT | 6.56 | 6.57 | 6.69 | — | — |
| 13B | 8192 | PI | 6.55 | 6.42 | 6.42 | — | — |
| 13B | 16384 | PI | 6.56 | 6.42 | 6.31 | 6.32 | — |
| 13B | 32768 | PI | 6.54 | 6.40 | 6.28 | 6.18 | **6.09** |
| 33B | 2048 | None | 5.82 | — | — | — | — |
| 33B | 8192 | FT | 5.88 | 5.99 | 6.21 | — | — |
| 33B | 8192 | PI | 5.82 | 5.69 | 5.71 | — | — |
| 33B | 16384 | PI | 5.87 | 5.74 | 5.67 | 5.68 | — |
| 65B | 2048 | None | 5.49 | — | — | — | — |
| 65B | 8192 | PI | 5.42 | 5.32 | 5.37 | — | — |

**Table 2, Arxiv Math Proof-pile perplexity** (128 documents, ≥32768 tokens, truncated to 32768):

| size | window | method | @2048 | @4096 | @8192 | @16384 | @32768 |
|---|---|---|---|---|---|---|---|
| 7B | 2048 | None | 2.77 | — | — | — | — |
| 7B | 8192 | FT | 2.85 | 2.74 | 2.73 | — | — |
| 7B | 8192 | PI | 2.79 | 2.57 | 2.39 | — | — |
| 7B | 16384 | PI | 2.79 | 2.57 | 2.37 | 2.25 | — |
| 7B | 32768 | PI | 2.82 | 2.59 | 2.39 | 2.24 | **2.48** |
| 13B | 2048 | None | 2.66 | — | — | — | — |
| 13B | 8192 | FT | 2.71 | 2.56 | 2.50 | — | — |
| 13B | 8192 | PI | 2.67 | 2.47 | 2.30 | — | — |
| 13B | 16384 | PI | 2.68 | 2.47 | 2.29 | 2.18 | — |
| 13B | 32768 | PI | 2.68 | 2.46 | 2.28 | 2.15 | **2.35** |
| 33B | 2048 | None | 2.49 | — | — | — | — |
| 33B | 8192 | FT | 2.56 | 2.48 | 2.47 | — | — |
| 33B | 8192 | PI | 2.50 | 2.32 | 2.18 | — | — |
| 33B | 16384 | PI | 2.53 | 2.34 | 2.18 | 2.07 | — |
| 65B | 2048 | None | 2.42 | — | — | — | — |
| 65B | 8192 | PI | 2.43 | 2.26 | 2.12 | — | — |

The paper's own summary of the gains: *"By increasing the context window size from 2048 to 16384, we
observed -0.28 and -0.5 reductions of perplexity for extending LLaMA 7B models on both datasets, -0.27
and -0.48 reductions for extending LLaMA 13B models, and -0.14 and -0.42 reductions for extending LLaMA
33B models. For LLaMA 65B models, we observed -0.12 and -0.3 reductions of perplexity by extending to
the 8192 context window size."* And on the baseline: *"models extended via the direct fine-tuning
method has shown regression (up to +0.48) or minor improvement (up to -0.12) on the perplexity at
longer context windows."*

**Two things in those tables the prose does not name.** (i) The 32768 models **turn back up at 32768**
on proof-pile: 7B goes 2.24 at 16384 → **2.48** at 32768, and 13B goes 2.15 → **2.35** — the worst
value in either row after 2048. The paper only says *"we found this trend extends to 32768 window size
without diminishing on the PG19 dataset"*, which is true and carefully scoped; the proof-pile column
does diminish, and nothing in the text mentions it. (ii) On PG-19 the 33B/8192 PI model reads 5.71 at
8192, *worse* than its own 5.69 at 4096 — the "graceful" monotone improvement is not universal.

**The degradation at the original context window — the cost everyone forgets.** §3.2:

> *"We saw a minor degradation of the perplexity on the original context window of 2048 for our extended
> models in some cases. For example, on the Proof-pile dataset, we saw a degradation ranging from 0.01
> to 0.05 across all models with extended with Position Interpolation. A small degradation of
> performance within original evaluation context window is expected since Position Interpolation forces
> position encodings in original context window to reside in a much narrower region, which may
> negatively affect the language model's performance."*

The cells, at `@2048`, PI minus baseline: proof-pile 7B **+0.02 / +0.02 / +0.05**, 13B **+0.01 / +0.02 /
+0.02**, 33B **+0.01 / +0.04**, 65B **+0.01** — the claimed 0.01–0.05 range is exact. On PG-19 it is
mixed and sometimes *negative*: 7B **−0.07 / −0.09 / +0.03**, 13B **−0.03 / −0.03 / −0.05**, 33B **0.00 /
+0.05**, 65B **−0.07**. So on PG-19, PI's 2048 perplexity is often *better* than the unmodified model's,
which the paper does not point out either — worth noting because it means the perplexity story is not a
clean tax. The benchmark story below is.

**Table 4 — passkey retrieval, the effective-context measurement.** Protocol: 32 values of `k`
uniformly spaced in `L′`, 10 trials each, a fresh random 5-digit passkey per trial, and

> *"`k_max` is defined as the maximum `k` such that, for all `k′ ≤ k`, the model has a success rate of
> at least 20% on `k′`."*

| size | window | method | 200 | 400 | 600 | 800 | 1000 | 10000 |
|---|---|---|---|---|---|---|---|---|
| 7B | 8192 | FT | 1792 | 2048 | 2048 | 2048 | 2304 | **2560** |
| 33B | 8192 | FT | 1792 | 2048 | 1792 | 2048 | 2304 | — |
| 7B | 8192 | PI | **8192** | 8192 | 8192 | 8192 | 8192 | — |
| 7B | 16384 | PI | **16384** | 16384 | 16384 | 16384 | 16384 | — |
| 7B | 32768 | PI | **32768** | 32768 | **18432** | 32768 | 32768 | — |
| 33B | 8192 | PI | **8192** | 8192 | 8192 | 8192 | 8192 | — |
| 33B | 16384 | PI | **16384** | 16384 | 16384 | 16384 | 16384 | — |

This is the paper's strongest table and it is genuinely striking: **200 steps to saturate a 16× extension**,
against direct fine-tuning creeping 2048 → 2560 in 10000 steps. Two caveats the card must print
alongside it. The **20% success threshold** is a very low bar — 2 of 10 trials on a 5-digit passkey
counts as "effective". And the 7B/32768 row is not clean: it reads **18432 at 600 steps**, a drop to
56% of the target in the middle of an otherwise perfect row, unmentioned in the text. Non-monotone
behaviour in the one metric that is supposed to demonstrate saturation.

**Table 3 — no fine-tuning at all, and naive extrapolation instead.** PG-19, 7B, PI applied:

| window | 0 steps | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|
| 8192 | **16.10** | 7.12 | 7.10 | 7.02 | 6.99 | 6.95 |
| 16384 | **112.13** | 7.05 | 6.93 | 6.88 | 6.84 | 6.83 |

The paper's sentence, taken from the PDF because ar5iv mangles it:

> *"We can see without fine-tuning (at step 0) the model can exhibit certain language modeling
> capability, as indicated by < 20 perplexity for extending to 8192 context window (in contrast, the
> direct extrapolation method leads to > 10³ perplexity). With fine-tuning, we observed that the
> perplexity improves quickly. At 200 steps the models surpassed the original model's perplexity on
> 2048 context window size."*

So the three-way comparison the card needs is all in one place: baseline at 2048 is **7.20**; PI with
**zero** fine-tuning at 8192 is **16.10** (2.2× worse — degraded, not destroyed); PI with zero
fine-tuning at 16384 is **112.13** (15× worse, and the paper's *"< 20"* claim quietly covers only the
8192 row); naive extrapolation with zero fine-tuning is **>10³** (Table 1, row 1 — untrained-model
territory, and the paper's own §2.2 wording is *"comparable to untrained models"*). PI without
fine-tuning is not usable, but it is not catastrophic, and the gap between 16.10 and >1000 is the
entire argument of the paper made in one column.

**Table 5 — the benchmark tax on the original window**, zero-shot, evaluated at 2048:

| size | window | fine-tuned on | BoolQ | PIQA | Race-M | Race-H | WinoGrande |
|---|---|---|---|---|---|---|---|
| 7B | 2048 | None | **76.1** | **78.9** | **55.7** | **42.2** | **69.6** |
| 7B | 8192 | Pile | 73.2 | 78.2 | 53.8 | 41.7 | 69.0 |
| 7B | 16384 | Pile | 69.8 | 77.6 | 53.3 | 40.9 | 67.8 |
| 7B | 32768 | Pile | **64.7** | 77.2 | 50.1 | 39.6 | 66.9 |
| 7B | 8192 | RedPajama | 75.5 | 77.4 | 54.5 | 41.5 | 68.1 |
| 33B | 2048 | None | **81.6** | 80.2 | **61.1** | **45.9** | **76.2** |
| 33B | 8192 | Pile | 80.2 | **80.7** | 60.2 | 45.7 | 75.9 |

> *"we saw that models extended to 8192 produce comparable results on the original benchmark which is
> designed for a much smaller context window, with a degradation of up to 2% on the benchmark tasks,
> for both 7B and 33B model sizes. Models extended to longer context windows regressed more on the
> benchmarks, but still in reasonable ranges for most tasks."*

**"Up to 2%" does not survive its own table.** 7B → 8192 loses **2.9 points on BoolQ** (76.1 → 73.2), a
3.8% relative drop; Race-M loses 1.9. The 33B row does obey the claim (worst −1.4). And the tax is
severe at 32768: BoolQ **76.1 → 64.7, −11.4 points, −15.0% relative**, and Race-M −5.6. The table's own
caption concedes the pattern rather than the prose: *"comparable performance as the original models,
**except for BoolQ dataset** that may require models to pay close attention to word ordering in a short
reference paragraph."* That parenthetical is the most interesting sentence in the section — it is a
hypothesis that PI blurs *fine-grained local order*, which is exactly what the resolution measurement
below shows, and the paper never follows it up.

The dataset ablation, in full: *"we also note that the choice of fine-tuning datasets does not seem to
lead significant difference in the benchmark performances, which may be due to the limited number of
fine-tuning steps used in our method."* Pile vs RedPajama at 7B/8192: BoolQ 73.2 vs 75.5, WinoGrande
69.0 vs 68.1 — a 2.3-point swing on BoolQ, which is the same size as the effect being measured. One run
each, no variance anywhere in the paper.

**Table 6 — long document summarization**, GovReport, inputs truncated to 15000 tokens, PI model at
16384 context, fine-tuned 10 epochs, temperature 0.5, top-p 0.95, output truncated at 1000 tokens:

| model | window | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| CoLT5 Base | 16K | 58.7 | 29.6 | 31.4 |
| CoLT5 XL | 16K | **61.3** | **32.2** | **33.8** |
| LLaMA-7B Extended | 16K | 60.0 | 28.0 | 29.5 |

> *"In general, we have obtained competitive R1 score among other models with minimal tuning of
> hyper-parameters."*

Read the hedge: **R1 only**. On ROUGE-2 and ROUGE-L the extended LLaMA loses to *both* baselines,
including CoLT5 **Base**. The claim made is precisely the claim the table supports, which is honest —
but "competitive on long summarization" is not what this table says.

### What the app measures about the geometry

The app: 32 dims, 4 heads (`d_k = DH = 8`), 2 blocks, seed 20260817, causal, untrained, 16-token
default sentence. `θ_j = 10000^(−2j/8) = {1, 0.1, 0.01, 0.001}` rad/token — 4 pairs **[measured here]**.

**`stretch` is `L/L′`, exactly.** For `g ∈ {4, 16, 128}` and `m ∈ {1, 7, 15, 63, 4095}`,
`max |rope({stretch: 1/g}).rotate(v, m) − rope({stretch: 1}).rotate(v, m/g)| = **0.00e+0**` — not small,
*zero*, in every case **[measured here]**. Eq. 4 is already implemented. `rope({}).rotate(v, 0.25)` is
finite and length-preserving to `0.00e+0`, so non-integer positions work today.

**The identity that is the card's centrepiece** **[measured here]**. Take the same 16 tokens and place
them at absolute positions `0, 64, 128, …, 960` — the sentence as it would sit spread through a long
document. Then:

| arrangement | max `\|ΔA\|` vs plain RoPE at positions 0…15 |
|---|---|
| PI with `g = 64` applied to positions 0, 64, …, 960 | **0.00e+0** |
| raw RoPE at positions 0, 64, …, 960 | **0.9983** |

Attention weights are probabilities, so 0.9983 is very nearly the maximum possible change: the raw
model's attention matrix is unrecognisable, while PI reproduces the trained-range matrix **bit for
bit**. That is Figure 1 turned into a checkable equality on the reader's own sentence.

**And the price, in the same units.** PI does not leave the original window alone. Comparing the block-0
head-0 attention matrix under `stretch = 1/g` against `stretch = 1` on the plain 0…15 positions
**[measured here]**:

| `g` | max `\|ΔA\|` | max `\|Δscore\|` | fastest pair, rad between adjacent tokens | pairs completing a full turn in 16 tokens | model's top-1 |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 1.0000 | 1 of 4 | `<end>` (p 0.824) |
| 2 | 0.9397 | 7.53 | 0.5000 | 1 of 4 | `<end>` (p 0.191) |
| 4 | 0.9304 | 10.58 | 0.2500 | **0 of 4** | **`her`** (p 0.502) |
| 16 | 0.8912 | 11.27 | 0.0625 | 0 of 4 | `her` (p 0.234) |
| 128 | 0.9148 | 12.10 | 0.0078 | 0 of 4 | `<end>` (p 0.207) |

Two facts to put on screen together. **Resolution collapses**: at `g = 4` no dimension pair completes a
turn inside the sentence any more, and adjacent tokens are separated by 0.25 rad on the fastest pair
instead of 1.0 — `cos(0.25) = 0.9689` against `cos(1) = 0.5403`, so neighbours that were nearly
orthogonal on that pair are now nearly parallel. This is the geometric content of the paper's *"forces
position encodings in original context window to reside in a much narrower region"*, and of the BoolQ
caption's *"pay close attention to word ordering"*. And **the model's output changes**: at `g = 4` the
top-1 prediction flips. In LLaMA that is what 1000 steps of fine-tuning repairs; here nothing repairs
it, which is the boundary this card has to be honest about.

| quantity | paper | this app |
|---|---|---|
| head dim `d` | 128 (7B), up to 8192/64 at 65B | **8** — 4 frequency pairs |
| `L → L′` | 2048 → 8192 / 16384 / 32768 (`g` = 4 / 8 / 16) | 16 → whatever `g` says |
| scale factor | `L/L′` on the position index | `stretch`, already present |
| `B(s)/d ≥ 1` | holds (measured, 0% violations) | **fails, 87% of `s`** |
| eq. 7 coefficient error | 1.15× understated | **4.65× understated** |
| fine-tuning | 1000 steps, 0.52–4.19 B tokens | **impossible** |
| perplexity | the paper's only quality metric | **cannot be computed** |

## What the live view must let the reader do

The seam needs **no change at all**. `rope({base, stretch, dims})` already implements eq. 4 exactly, and
`softmaxMixer({rotate})` already routes it to `q` and `k` only. Concept 12's card already owns the
shift-invariance identity, the zero-parameter budget meter, the decay-bound critique, and a `stretch`
slider used as a teaser — that card's final line is literally *"the rate slider you just moved is
position interpolation"*. **This card must therefore not repeat the stretch slider as a curiosity; it
must make it the subject, and everything below is something concept 12 does not do.**

**The boundary, stated on the card in the reader's first screen.** The paper's every quality claim is a
perplexity or a benchmark score obtained *after fine-tuning*. This model is untrained and cannot be
fine-tuned. So the card can demonstrate: the geometry of the rescaling, the exactness of the mapping
back onto the trained grid, the resolution the rescaling destroys, the shape of the theory, and the
un-fine-tuned behaviour. It **cannot** demonstrate: that fine-tuning recovers quality, any perplexity
number, or that PI beats direct fine-tuning. Put that in a fixed note, not a footnote — half the card's
honesty is in it.

1. **The document slider — PI as an exact return to the trained grid.**
   One control: *"this sentence sits inside a document, with `g` tokens between each of its words"*,
   `g` from 1 to 128 (i.e. the sentence occupies positions `0, g, 2g, …, 15g`). Two toggles: *rescale
   positions (PI)* on/off. Show the block-0 head-0 attention heatmap plus one readout,
   `max_{i,j}|A[i,j] − A_ref[i,j]|` against the plain 0…15 reference. With PI on it reads **0.00e+0** at
   every `g`; with PI off it reads **0.9983** at `g = 64` — nearly the maximum a probability matrix can
   move. Show the two heatmaps side by side. This is the paper's Figure 1 as an equality the reader
   verifies on their own sentence, and it is the single clearest thing the app can say about this
   method. Print eq. 4 beneath it, and the sentence *"we reduce position indices from `[0, L′)` to
   `[0, L)` to match the original range of indices before computing RoPE."*

2. **The half-integer positions — where the cost actually lives, and what Theorem 2.1 bounds.**
   Interaction 1 is deliberately rigged: with the sentence's tokens exactly `g` apart, PI lands every
   one of them back on an *integer* trained position. Real PI does not. Add a second control, *token
   spacing* = 1, so the sentence occupies `0…15` inside a window extended by `g`; now PI maps it to
   `0, 1/g, 2/g, …, 15/g` and **every position except the first is one the model never saw**. On the
   curve view, for a reader-chosen `(m, n)` pair and head, plot `a(s)` over one unit interval
   `s ∈ [s₁, s₁+1]` at fine resolution, with: the **linear chord** `a_linear(s)` as the reference curve,
   the **bound envelope** `±d·max_j|h_j|·(s−s₁)(s₂−s)/(8 ln c)` as the shaded band, and a marker at the
   actual interpolated position the reader's `g` produces. Three readouts: the actual excursion
   `max|a(s) − a_linear(s)|`, the paper's bound with the same `max_j|h_j|`, and the slack. On this
   model's real vectors the mean excursion is **0.4755** against a mean bound of **0.3043** — **the band
   is visibly too narrow, and the curve leaves it.** Put the reason underneath, because it is a real
   finding and not a rendering artifact: the proof's eq. 14 asserts `c^x ≤ 1 + x ln c`, which is the
   tangent-line inequality reversed and false for every `x`; the correct coefficient is
   `1/(8(1−c^(−4/d)))`, which is **4.65× larger at `d = 8`** and 1.15× larger at LLaMA's `d = 128`. Offer
   a *"use the corrected constant"* toggle that widens the band, and let the reader see the curve fit
   inside it. A card that draws a published bound and shows the data escaping it — then shows exactly
   which line of the appendix is responsible — is worth more than five that restate the theorem.

3. **The resolution meter — what the original window pays.**
   `g` slider again, but now measuring the *near* end instead of the far end. Three live numbers plus a
   bar per frequency pair: (a) radians between adjacent tokens on pair 0, `θ_0/g` — **1.0 → 0.25 → 0.0625**
   at `g` = 1, 4, 16; (b) `cos` of that angle — **0.5403 → 0.9689 → 0.9980**, i.e. how nearly parallel two
   neighbours' fastest pair becomes; (c) pairs completing a full turn inside the sentence: **1 of 4 at
   `g = 1`, 0 of 4 from `g = 4` onward**. Beside them, the model's top-1 prediction and its probability,
   which flips from `<end>` (p 0.824) to `her` (p 0.502) at `g = 4` on the default sentence. Quote the
   paper's own concession — *"Position Interpolation forces position encodings in original context
   window to reside in a much narrower region, which may negatively affect the language model's
   performance"* — and Table 5's BoolQ column beside it: **76.1 → 73.2 → 69.8 → 64.7** as `g` goes
   1 → 4 → 8 → 16. The app supplies the *mechanism* (angles collapsing), the paper supplies the
   *consequence* (benchmark points lost); neither can supply the other, and showing them adjacent is
   the point. Label the paper's column clearly as measured after fine-tuning, on a model 200 million
   times larger.

4. **Extrapolation's catastrophe, reproduced and then debunked in the same panel.**
   Two curves on the curve view over `s ∈ [0, 2L)` with a dead region past `L` and the `L` marker:
   (a) **the paper's Figure 2 recipe** — fit random targets on `[0, L)` by least squares over the trig
   basis and extend. At the app's `d = 8`, `L = 16` this gives max `|a|` of **1.267** inside and **154.8**
   outside, a **122×** blow-up, with fitted coefficients reaching **14542**. The paper's own `d = 128`,
   `L = 2048` setting reproduces at **1.100 → 17776** (their caption: *"approximately within [−1,1]"* and
   *"may goes beyond 8000"*). (b) **the same plot using the model's real `q`, `k`** for a reader-chosen
   pair — a bounded ripple whose worst out-of-window blow-up across every block, head and pair is
   **9.32×**, because `|a(s)| ≤ Σ_j |h_j|` always, and that envelope should be drawn as a horizontal
   line. Then the control that lands the argument: an *"`L` = 2048 with `d` = 8"* preset, where the same
   fitting recipe gives **ratio 1.0×** and coefficients of order **0.1** — no blow-up at all. The
   catastrophe is what happens when a rich basis is fitted to noise on a short range, and the paper says
   as much in the sentence it buries in §2.3: *"if we apply ridge regression with proper
   regularization to fit a curve in Fig. 2, the magnitude of extrapolated `a(s)` when `s > L` can be
   comparable to that within `[0, L]`."* The failure of extrapolation is **real** — Table 1's `>10³`
   column is a measurement of an actual LLaMA — but Figure 2 is not the evidence for it, and this panel
   is how a reader finds that out.

5. **The 600×, recomputed at the reader's width.**
   A `d` selector (8, 16, 32, 64, 128) driving a single plot of `B(s)/d` against `s ∈ [0, 4096]` with the
   `B/d = 1` line the paper's Figure 5 draws, plus three readouts: `B(0)/d = (d+2)/8` (exact, derived
   here — **16.25** at `d = 128`, matching Figure 5's peak, **1.25** at `d = 8`), the minimum over the
   range, and the resulting ratio `2·(32 ln c)·B(s)/d`. At `d = 128` it reproduces the paper:
   **658× … 9579×**, `B(s)/d < 1` at **0%** of `s`, so *"at least ∼600×"* is fair. At `d = 8`:
   **153× … 737×**, and `B(s)/d < 1` at **87%** of `s` — the floor the whole constant rests on is gone.
   One sentence beneath: *"numerically it is at least larger than `d`"* was checked at one head width,
   and it is a fact about wide heads. Add a *"corrected constant"* toggle here too (`16(1−c^(−4/d))B(s)`),
   which reads **571×** at `d = 128` — so the arithmetic error costs the headline nothing at LLaMA scale,
   and everything at toy scale. Both halves matter: the card should neither swallow 600× nor pretend
   the paper's conclusion collapses.

6. **The three-way column the paper's Table 3 makes, drawn as bars.**
   Not a live computation — the app cannot produce a perplexity — but a `barList` of the paper's own
   numbers with the app's fixed note attached, because it is the comparison that defines the method and
   nothing else on the card can carry it: baseline @2048 **7.20**; PI, **zero** fine-tuning, @8192
   **16.10**; PI, zero fine-tuning, @16384 **112.13**; raw extrapolation, zero fine-tuning, @4096+
   **>10³**; PI @ 200 steps **7.12**; PI @ 1000 steps **6.95**. Alongside it the passkey row: direct
   fine-tuning reaches **2560** after **10000** steps; PI reaches **8192** after **200**. Mark every bar
   *paper, LLaMA 7B, PG-19* and mark the whole panel *not reproducible here*. If the deck has a
   convention for paper-sourced panels, use it; if not, this card should establish one, because concepts
   17 and 18 will need the same.

## What the source does *not* establish

- **It does not prove Theorem 2.1 as stated.** Appendix A's eq. 14 rests on *"when x<1, `c^x ≤ 1 + x ln c`"*,
  which is the tangent-line inequality for a convex function written backwards; `c^x ≥ 1 + x ln c` for
  all real `x`. Checked at the paper's own setting: `c^(−4/128) = 0.74995` and `1 + x ln c = 0.71218`.
  The bound's *form* — a second-derivative interpolation-error bound — is right, and eq. 13 (the exact
  geometric sum) is right; only the substitution of `d/(4 ln c)` is invalid, and it makes eq. 5 and
  eq. 7 too tight by `1/((1−c^(−4/d))·d/(4 ln c))`: **1.15× at `d = 128`, 4.65× at `d = 8`**. On this
  app's real vectors the measured excursion exceeds eq. 7's stated bound (mean 0.4755 vs 0.3043).
  Nothing about the paper's *conclusions* changes at LLaMA scale; the theorem as a quotable inequality
  does.
- **The `∼600×` is a ratio between two *bounds*, one of which the paper has just called vacuous.** §2.2
  dismisses Su et al.'s §3.4.3 bound as *"too loose … and thus vacuous"*, and §2.3 then uses that same
  bound as the extrapolation side of the 600× comparison. Beating a bound you have declared vacuous by
  600× is a statement about bound quality, not about attention scores. Nowhere does the paper measure
  the actual interpolated and extrapolated attention scores of a real model and compare them.
- **The two bounds are not commensurable.** Eq. 5 bounds `|a(s) − a_linear(s)|` — deviation from a chord
  between grid points *assumed* to be well-behaved (*"known to behave well, enforced by LLM
  pre-training"*, an assumption never tested). Eq. 8 bounds `|a(s)|` — an absolute magnitude. The
  abstract's *"the upper bound of interpolation is at least ∼600× smaller than that of extrapolation"*
  reads as though they measure the same thing.
- **`B(s)/d ≥ 1` is a numerical observation at one head width.** Appendix B checks `d = 128` only, and
  Figure 5's caption concedes the quantity *"decays with `s`"*. Measured here: at `d = 8` it is below 1
  for **87%** of `s ∈ [0, 4096]`, bottoming at **0.2595**. The constant is not universal.
- **Figure 2 measures a least-squares fit to random noise, not a language model.** Appendix C.1 is
  eleven lines of PyTorch: `y = torch.randn(...)`, `torch.linalg.solve` on the normal equations, plot.
  No LLaMA weights, no real `q`, no real `k`, `eps = 0.000`. §2.2's supporting claim is explicitly an
  existence statement — *"there always exist coefficients `{h_j}`"* — and §2.3 admits the effect is
  removable by regularising the fit. Reproduced here at both widths, plus the control the paper omits:
  at `d = 8, L = 2048` the same recipe blows up by **1.0×**. **The paper contains no measurement of
  `max_j|h_j|` or of `a(s)` from any trained model.** The catastrophic-extrapolation *fact* is well
  established by Table 1's `>10³`; the *explanation* offered for it is a synthetic construction.
- **"Up to 2% degradation" is contradicted by the paper's own Table 5.** 7B → 8192 loses **2.9 points**
  on BoolQ (3.8% relative). The table's caption concedes BoolQ as an exception the prose does not. At
  32768 the loss is **11.4 points, −15.0% relative**, described only as *"regressed more … but still in
  reasonable ranges for most tasks"*, with no threshold given for "reasonable".
- **The perplexity gains do not extend monotonically to 32768, on one of the two datasets.** Proof-pile
  7B: 2.24 @16384 → **2.48** @32768; 13B: 2.15 → **2.35**. The paper's *"this trend extends to 32768
  window size without diminishing"* is scoped to PG-19 and is accurate there; the proof-pile reversal is
  never mentioned. Nor is 33B/8192's PG-19 uptick, 5.69 @4096 → 5.71 @8192.
- **The passkey metric is weak and its own table is non-monotone.** `k_max` requires only a *"success
  rate of at least 20%"* — 2 of 10 trials — at every `k′ ≤ k`. And 7B/32768/PI reads **18432** at 600
  steps, inside a row that is 32768 everywhere else. No explanation, no error bars, no repeats.
- **There are no repeated runs and no variance anywhere.** Every perplexity, every benchmark cell, every
  ROUGE score is a single number from a single run. The Pile-vs-RedPajama comparison swings BoolQ by
  **2.3 points**, which is larger than the effect the section is claiming to measure, and is dismissed
  as *"does not seem to lead significant difference"*.
- **The summarization result is competitive on ROUGE-1 only.** 60.0 R1 beats CoLT5 Base's 58.7, but
  R2 (28.0) and RL (29.5) lose to CoLT5 **Base** (29.6 / 31.4) as well as XL. The paper's own wording is
  scrupulously narrow — *"competitive R1 score"* — and should be quoted that way, not paraphrased.
- **It never explains why fine-tuning is needed at all**, given that PI's whole argument is that
  interpolated scores are already well-behaved. Table 3 shows 16.10 → 6.95, a 2.3× perplexity
  improvement from a procedure the theory says should barely be necessary. The offered reason is a
  guess: *"The reason **may be** that the model is only adapting to the new context window during the
  fine-tuning phase, starting from a good initialization"*. There is no ablation isolating what the 1000
  steps change.
- **It does not compare against ALiBi, or against any length-extrapolation method, empirically.** §4
  dismisses that literature on applicability grounds — *"these methods have not been applied in some of
  the largest language models such as LLaMA … or OPT"* — which is a deployment argument, not a result.
  No head-to-head anywhere.
- **It does not ablate the scale factor, or consider anything but a uniform one.** Every frequency is
  divided by the same `L/L′`. There is no experiment on scaling low frequencies differently from high
  ones, no discussion of wavelength bands, and no mention of the fact that the fast pairs are the ones
  carrying local order and therefore the ones the rescaling damages most. That gap **is** concepts
  17–18, and the card should name it as unfinished business rather than as a later improvement.
- **It says nothing about a model needing to serve short prompts.** The rescaling is baked in: an
  extended model applies `L/L′` to *every* input, so a 100-token prompt is also compressed into
  `100·L/L′` of arc. Table 5's regression is the measurement of that, but the paper never frames it as
  the structural cost it is, and never discusses turning the scaling off for short inputs.
- **§2.2's central claim carries a broken citation.** *"its extrapolation performance is not great ."* —
  the space before the period is a failed `\citep` in **both** v1 and v2. Minor, but this is the one
  sentence the whole paper's motivation rests on, and it is unsourced in the published text. (Table 1's
  `>10³` row supplies the evidence a few pages later, so the claim stands; the citation does not.)
- **The app establishes nothing empirical about quality.** Untrained weights, 16 tokens, no fine-tuning,
  no perplexity, `d = 8` where several of the paper's numerical constants do not hold. What it can
  settle completely is the algebra: `stretch = L/L′` reproduces eq. 4 to **0.00e+0**, PI returns a
  spread-out sentence to the trained grid to **0.00e+0**, `|a(s)| ≤ Σ_j|h_j|` is exact, and Theorem 2.1's
  stated bound is violated at `d = 8`. Exact identities and counterexamples are the class of claim a toy
  with random weights can settle; everything else on this card is quoted, not measured.
- **Folklore to refuse.** PI is routinely described as *"free"* or *"training-free"* — it is not: 1000
  steps and 0.5–4.2 B tokens, and Table 3 without them reads 16.10 and 112.13. It is described as
  *"lossless on short contexts"* — Table 5 says −11.4 BoolQ points at 32×. It is credited with
  *"proving"* the 600× — it asserts a numerical floor at one width and mis-signs an inequality on the way
  to the other side. And it is often cited as *originating* the idea, which its own §1 declines to claim
  (kaiokendev's SuperHOT blogpost, concurrent and earlier to publication).

## Leaves behind

**Backward, this is the bill concept 12 wrote coming due, and it is paid in exactly the currency that
card predicted.** RoPE's card ends: *"Defined everywhere, calibrated nowhere"*, and its final
interaction has the reader turn a `stretch` knob from 1 to 0.25 so that *"position 4096 is rotated as if
it were 1024, back inside the trained range"*. That knob, unchanged, is eq. 4. The card should open by
saying so and then immediately do the thing concept 12 could not: put a real extension factor on it and
measure both ends — the exact return to the trained grid at the far end (`0.00e+0`), and the collapse of
neighbour resolution at the near end (`cos` of the adjacent-token angle going 0.5403 → 0.9689 → 0.9980).

It also settles two arguments the timeline has been running. Concept 12's research concluded that RoPE's
long-term decay is a bound, not a proof, and that the bound's leading factor is content-dependent and
unbounded; §2.2 of this paper reaches the identical conclusion independently and states it as *"too
loose … and thus vacuous"* — the card can now retire the hedging. And concept 13's ALiBi established
that RoPE extrapolates poorly (19.33 → 106.99), which this paper confirms far more brutally at
LLaMA scale (`7.20 → >10³`) while rejecting ALiBi's *remedy* on grounds ALiBi never had to answer: the
weights already exist and already have RoPE, so a method requiring a different position scheme requires
a new pre-training run. ALiBi's branch was *train differently*; this is the first card on the branch
*ship what you have, then bend it*. The timeline should show the fork explicitly here.

**Forward, it opens exactly one crack, and concepts 17–18 are both squeezed through it.** PI divides
*every* frequency by the same `L/L′`. But the frequency ladder is not homogeneous: the slow pairs
(`θ = 0.001`, wavelength 6283 tokens) never came close to completing a turn even at the original length,
so compressing them costs nothing, while the fast pairs (`θ = 1`, wavelength 6.28 tokens) are precisely
the ones encoding local word order — and compressing *those* is what turns 1.0 rad between neighbours
into 0.0625. The app can show the asymmetry directly: at `g = 4`, **0 of 4** pairs still complete a turn
inside the sentence, where at `g = 1` one did. The paper never separates the two populations, never
ablates a non-uniform scale, and files the entire question under future work. Its own Table 5 caption
even names the symptom — BoolQ *"may require models to pay close attention to word ordering in a short
reference paragraph"* — and walks past it.

That is the whole of what comes next. NTK-aware scaling changes the **base** instead of the index, which
stretches the fast pairs while leaving the slow ones nearly alone; YaRN splits the ladder by wavelength
and applies a different rule per band, plus a temperature on the softmax. Both are edits to the same two
constants this app has exposed since concept 12 — `base` and `stretch` — and `position.js` needs no new
seam for either. So the card should end where concept 12's ended, but one turn further in: the reader has
now *seen* the uniform scale work perfectly at the far end and do real damage at the near end, and the
obvious question — *why is the slowest pair paying the same tax as the fastest?* — is the question the
next two cards answer. Ask it on this card, in the reader's own numbers, and concepts 17 and 18 arrive as
answers rather than as further methods.
