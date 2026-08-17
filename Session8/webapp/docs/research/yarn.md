# Concept 18 — YaRN, wavelength-targeted interpolation plus an attention temperature

**Card id:** `yarn` · **Date:** 2023-08-31 (arXiv v1) · **Pressure:** where a token sits

## What was read

- [arXiv:2309.00071](https://arxiv.org/abs/2309.00071), Bowen Peng, Jeffrey Quesnelle, Honglu Fan,
  Enrico Shippole (Nous Research, EleutherAI, University of Geneva) — *YaRN: Efficient Context
  Window Extension of Large Language Models*. Abstract page for version history; the v1 abstract
  page separately for the v1-only "this version" line.
- **Three renderings, read end to end and diffed against each other**, because they are not the same
  paper:
  - `arxiv.org/html/2309.00071v1` — the version this card is dated to.
  - [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.00071) — which serves **v2**, not v1 (it contains
    an appendix on Mistral 7B, released seven weeks *after* v1). Anyone who reads "the YaRN paper"
    through ar5iv is reading v2. This note flags every place that matters.
  - `arxiv.org/html/2309.00071v3` — the current version, which is where the **ablation** lives.
- All numbers below are cell-by-cell from the tables of the version named beside them. Where a
  quantity exists only as a figure with no printed values, it says so rather than guessing.

**Version history, verified.** v1 **Thu, 31 Aug 2023 18:18:07 UTC** (42 KB); v2 Wed, 1 Nov 2023
17:28:26 UTC (354 KB); v3 Fri, 6 Feb 2026 19:40:50 UTC (587 KB). The app's record of `2023-08-31` is
**correct**.

**And the 2309 identifier against an August date is not an error.** arXiv's announcement cycle closes
at 14:00 US Eastern; v1 was stamped 18:18 UTC on 31 August, which is *past* that day's cutoff, so the
submission fell into the next cycle and drew a September (`2309.*`) number. The submission date and
the identifier month legitimately disagree by one day. The v1 HTML's own banner reads
`arXiv:2309.00071v1 [cs.CL] 31 Aug 2023`. Use 31 August; do not "correct" it to September.

**What changed between versions — this matters more than usual, because v2 deleted the paper's only
explanation of its own headline trick.**

| | v1 (31 Aug 2023) | v2 (1 Nov 2023) | v3 (6 Feb 2026) |
|---|---|---|---|
| §3.4 title | *"Increase in Average Minimum Cosine Similarity for Long Distances - YaRN"* | *"YaRN"* | *"YaRN"* (renumbered §3.3) |
| why a temperature | an entropy argument, with a footnote deriving `L/(N²−1)` | **deleted**; replaced by *"we also observe that introducing a temperature t … has a uniform impact on perplexity"* | as v2 |
| the fit | `√t ≈ 0.1 ln(s) + 1` (eq. 27) | `√(1/t) = 0.1 ln(s) + 1` (eq. 22) — **reciprocal symbol** | as v2 (eq. 15) |
| appendices | **none at all** | A.1–A.2, B.1–B.4 | A.1–A.3, B.1–B.7 |
| passkey retrieval | **absent** | Table 5 | Figure 3 + Appendix B.5 |
| Dynamic Scaling experiment | **absent** | Appendix B.3 | Appendix B.7 |
| ablation of the four methods | **absent** | **absent** | **Table 5 + Table 2** |
| PI's rescaling direction | **inverted** (see below) | corrected | corrected |

**v1 ships the position-interpolation formula backwards, twice.** v1 eq. 9 writes
`f'_q(x_m, m, θ_d) = f_q(x_m, mL'/L, θ_d)` and v1 eq. 12 writes `g(m) = s·m`. Both *multiply* the
position by the scale factor — that is extrapolation, the exact thing PI exists to avoid. v2 fixes
both (`mL/L'`, `g(m) = m/s`). This is a transcription error, not a different method — everything
downstream in v1 uses the corrected sense — but a card built by reading v1's equations literally
would implement the opposite of concept 16.

## The mechanism, precisely

**The frame the whole card hangs on: blind versus targeted.** §2.3 (v2) introduces the wavelength and
then draws the line that names what concepts 16 and 17 both got wrong:

> *"Additionally, we define λ_d as the wavelength of the RoPE embedding at d-th hidden dimension:"*

    λ_d = 2π/θ_d = 2π·b^(2d/|D|)                                   (v2 eq. 13)

> *"The wavelength describes the length of tokens needed in order for the RoPE embedding at dimension
> d to perform a full rotation (2π). Given that some interpolation methods (eg. PI) do not care about
> the wavelength of the dimensions, we will refer to those methods as "blind" interpolation methods,
> while others do (eg. YaRN), which we will classify as "targeted" interpolation methods."*

That is the synthesis in one sentence. PI (concept 16) scales every dimension by `1/s`; NTK-aware
(concept 17) scales every dimension by a smoothly varying amount determined by a base change. Neither
one *looks* at whether a given dimension's wavelength is longer or shorter than the context. YaRN's
claim is that the right treatment is a function of that comparison and of nothing else.

The common notation, §2.3 eq. 12, is the seam of the whole family:

    f'_W(x_m, m, θ_d) = f_W(x_m, g(m), h(θ_d))

> *"where g(m), h(θ_d) are method-dependent functions. For PI, we have g(m) = m/s, h(θ_d) = θ_d."*

Every method on concepts 16–18 is a choice of `(g, h)`. PI moves `g`. NTK-aware and NTK-by-parts both
leave `g(m) = m` and move `h`. **This is exactly the app's `rope({base, stretch})`:** `stretch` is a
multiplier on the angle, so it can express any `h` that is a per-pair scalar, and `base` expresses the
NTK-aware `h` specifically.

---

### 1. NTK-by-parts — the wavelength criterion

**The observation** (§3.2, v2):

> *"One interesting observation of RoPE embeddings is that given a context size L, there are some
> dimensions d where the wavelength is longer than the maximum context length seen during pretraining
> (λ > L), this suggests that some dimensions' embeddings might not be distributed evenly in the
> rotational domain. In such cases, we presume having all unique position pairs implies that the
> absolute positional information remains intact. On the contrary, when the wavelength is short, only
> relative positional information is accessible to the network."*

**The cost of ignoring it:**

> *"Moreover, when we stretch all the RoPE dimensions either by a scale s or using a base change b′,
> all tokens become closer to each other, as the dot product of two vectors rotated by a lesser amount
> is bigger. This scaling severely impairs a LLM's ability to understand small and local relationships
> between its internal embeddings. We hypothesize that such compression leads to the model being
> confused on the positional order of close-by tokens, and consequently harming the model's abilities."*

**The rule, in the authors' three bullets:**

> - *"if the wavelength λ is much smaller than the context size L, we do not interpolate;"*
> - *"if the wavelength λ is equal to or bigger than the context size L, we want to only interpolate and
>   avoid any extrapolation (unlike the previous "NTK-aware" method);"*
> - *"dimensions in-between can have a bit of both, similar to the "NTK-aware" interpolation."*

**The ratio and the two thresholds.**

    r(d) = L/λ_d = L / (2π·b^(2d/|D|))                              (v2 eq. 17)

`r(d)` is the number of full rotations dimension `d` completes inside the original context. v3 spells
it out: *"This ratio represents the number of rotations a certain RoPE dimension makes given a fixed
pretrained context length L."*

> *"In order to define the boundary of the different interpolation strategies as above, we introduce
> two extra parameters α, β. All hidden dimensions d where r(d) < α are those where we interpolate by
> a scale s (exactly like PI, avoiding any extrapolation), and the d where r(d) > β are those where we
> do not interpolate at all."*

**The ramp** (v2 eq. 18):

    γ(r) = 0                if r < α
         = 1                if r > β
         = (r − α)/(β − α)   otherwise

**And the method** (Definition 2, v2 eqs. 19–20):

    g(m) = m
    h(θ_d) = (1 − γ(r(d)))·θ_d/s  +  γ(r(d))·θ_d

Read the endpoints. `γ = 0` → `h = θ_d/s`, which is **PI applied to that dimension alone**. `γ = 1` →
`h = θ_d`, the dimension is **untouched**. So NTK-by-parts is not a new interpolation at all: it is
a per-dimension switch between PI and identity, with a linear crossfade in the band. Nothing here is
NTK-aware — the base is never changed. The name is inherited from the pull request it shipped in
(reference [7], `jquesnelle/scaled-rope` PR #1, titled *"Add NTK-Aware interpolation 'by parts'
correction"*) and is misleading about the mechanism.

**The values:**

> *"The values of α and β should be tuned on a case-by-case basis. For example, we have found
> experimentally that for the Llama family of models, good values for α and β are α = 1 and β = 32."*

`α = 1` is the wavelength-equals-context boundary exactly (`r = 1` ⟺ `λ = L`), so the first bullet's
"λ equal to or bigger than L" and the threshold agree. `β = 32` — thirty-two full rotations inside the
context before a dimension is considered safe to leave alone — is a bare empirical number with no
derivation anywhere in any version.

**A discrepancy in eq. 17 worth not propagating.** v2/v3 write `r(d) = L / (2π·b′^(2d/|D|))` — with
**b′**, the NTK-*aware* base. NTK-by-parts never performs a base change; v1's equivalent (v1 eqs.
19–20) correctly uses `b`. Treat the primed `b` in v2/v3 eq. 17 as a typo and use the original base.

**A second one: v1 defines the ramp in the wrong space, and v2 quietly demotes it to "an
alternative".** v1 first interpolates *wavelengths* (v1 eq. 23):

    λ'_d = (1 − γ_d)·s·λ_d + γ_d·λ_d

then says *"After converting λ_d to θ_d, the method can be described as:"* and prints the same
`h(θ_d) = (1−γ)θ_d/s + γθ_d` as v2. **That is not the conversion.** Inverting the wavelength form gives
`θ' = θ_d / ((1−γ)s + γ)` — a *harmonic* mean of `θ_d/s` and `θ_d` — not the arithmetic mean actually
printed. v2 notices and adds footnote 1:

> *"The interpolation by linear ramp on h may have alternatives, such as a harmonic mean over θ_d/s and
> θ_d converted from a linear interpolation on wavelengths. The choice of h here was for the simplicity
> of implementation, but both would work."*

*"Both would work"* is asserted, never tested. They are not close: **[measured here]** at `γ = 0.5`,
the arithmetic form gives `0.53125·θ` and the harmonic form `0.11765·θ` at `s = 16` — a factor of
**4.5** — and a factor of **8.5** at `s = 32`. The two forms agree only at `γ ∈ {0, 1}`. Every shipped
implementation uses the arithmetic one.

---

### 2. The attention temperature — and why it is free

v2 §3.4 states it as a bare observation, having deleted v1's explanation:

> *"In addition to the previous interpolation techniques, we also observe that introducing a
> temperature t on the logits before the attention softmax has a uniform impact on perplexity
> regardless of the data sample and the token position over the extended context window (See Appendix
> A.2). More precisely, instead of Eq. 2, we modify the computation of attention weights into"*

    softmax( q_m^T k_n / (t·√|D|) )                                (v2 eq. 21)

**The part most summaries drop — it folds into the rotation, so it costs nothing:**

> *"The reparametrization of RoPE as a set of 2D matrices has a clear benefit on the implementation of
> this attention scaling: we can instead use a "length scaling" trick which scales both q_m and k_n by
> a constant factor √(1/t) by simply scaling the complex RoPE embeddings by the same amount. With this,
> YaRN can effectively alter the attention mechanism without modifying its code. Furthermore, it has
> zero overhead during both inference and training, as RoPE embeddings are generated in advance and are
> reused for all forward passes."*

The mechanism in one line: RoPE multiplies `q` and `k` by unit-modulus complex numbers `e^{imθ}`. Give
those numbers modulus `√(1/t)` instead of 1 and the dot product `q^T k` comes out scaled by `1/t`,
which is exactly eq. 21 — with no touch to the attention kernel. The rotation table is precomputed
once, so the multiply is absorbed into a constant that was already being materialised. **This is the
one place a "temperature" is genuinely free, and it is free only because the position encoding is
multiplicative.** Concept 12's card already makes the point that rotation preserves norm; YaRN is the
method that deliberately breaks that norm preservation by a constant factor and collects the softmax
temperature as change. ALiBi (concept 13) could not do this — an additive bias cannot be folded into a
scale.

**The fit** (Definition 3 / v2 eq. 22):

    √(1/t) = 0.1·ln(s) + 1

> *"For LLaMA and Llama 2 models, we recommend the following values … The equation above is found by
> fitting √(1/t) at the lowest perplexity against the scale extension by various factors s using the
> "NTK-by-parts" method (Section 3.2) on LLaMA 7b, 13b, 33b and 65b models without fine-tuning. We note
> that the same values of t also apply fairly well to Llama 2 models (7b, 13b and 70b). It suggests
> that the property of increased entropy and the temperature constant t may have certain degree of
> "universality" and may be generalizable across some models and training data."*

**The reciprocal trap.** v1's eq. 27 is `√t ≈ 0.1 ln(s) + 1`; v2/v3's eq. 22/15 is
`√(1/t) = 0.1 ln(s) + 1`. Same right-hand side, reciprocal left-hand side — so **`t` in v1 is `1/t` in
v2**. The *arithmetic is identical in both*: q and k are each multiplied by `0.1 ln(s) + 1`, so the
logits are multiplied by `(0.1 ln(s) + 1)²`. Only the symbol flipped. Implementations expose this as a
single `attn_factor` = `0.1·ln(s) + 1` and never name `t` at all, which is why the flip went unnoticed.

**And the direction is the opposite of what the word "temperature" implies.** Because
`0.1 ln(s) + 1 > 1` for any `s > 1`, the logits are multiplied by a number **greater than one** —
attention gets **sharper**, entropy goes **down**. v1's prose says the goal is *"to reverse the
decrease of entropy (i.e. increase the 'temperature' of the attention logits)"* and prescribes
*"multiplying the intermediate attention matrix by a temperature t > 1 before applying the softmax"* —
which *decreases* entropy, the same direction as the diagnosis, not the opposite. v1's diagnosis
paragraph is itself self-contradictory in one sentence: interpolation makes the distribution
*"'spikier' (i.e. decreases the average entropy of the attention softmax)"* and then, immediately,
*"the network 'pays more attention' to more tokens"* — which is higher entropy. The arithmetic is
consistent and checkable; the story attached to it is not, which is presumably why v2 deleted the
story and kept the arithmetic. §Numbers has the measured entropies.

---

### 3. Why "YaRN" is the combination

> **Definition 3.** *"By the "YaRN method", we refer to a combination of the attention scaling in
> Eq. 21 and the "NTK-by-parts" interpolation introduced in Section 3.2."*

That is the whole definition. YaRN contributes **no new interpolation of its own** — its `(g, h)` pair
is NTK-by-parts', unchanged. Its only original component is the temperature, and its only original
claim is that the two together beat either alone. §Numbers has the ablation, which is more interesting
than the claim.

---

### 4. Dynamic scaling

§3.3 (v2) frames it as a choice between two ways of applying *any* scale-factor method:

> *"1. Throughout the whole inference cycle, the embedding layer is fixed including the scale factor
> s = L′/L … 2. In each forward-pass, the position embedding updates the scale factor
> s = max(1, l′/L) where l′ is the sequence length of the current sequence."*
>
> *"The problem of (1) is that the model may experience a performance discount at a length less than L
> and an abrupt degradation when the sequence length is longer than L′. But by doing Dynamic Scaling as
> (2), it allows the model to gracefully degrade instead of immediately breaking when hitting the
> trained context limit L′. We call this inference-time method the Dynamic Scaling method."*

The `max(1, ·)` is the load-bearing part: below the original context the method is a no-op and the
model runs exactly as pre-trained, so there is no short-context tax at all. Note that **Dynamic Scaling
is orthogonal to YaRN** — it applies to PI, NTK-aware, NTK-by-parts and YaRN alike. "Dynamic NTK" is
dynamic scaling + NTK-aware (reddit, ref [14]); "Dynamic-YaRN" is dynamic scaling + YaRN.

The implementation warning is real and worth a line on the card:

> *"The correct implementation should cache the kv-embeddings before applying RoPE, as the RoPE
> embedding of every token changes when s changes."*

Every previously cached key is invalidated the moment `s` moves, so a KV cache holding post-rotation
keys is silently wrong under dynamic scaling.

## Numbers that matter

### The wavelength ladder, in this app and at a realistic width

**[measured here]** — driving `rope()` from `app/model/position.js` with `λ_i = 2π·b^(2i/dims)`,
using the app's 0-based pair index (`freqs[i] = base^(−2i/dims)`).

At `d_k = 8` — **4 pairs per head**, base 10000:

| pair `i` | `θ_i` | wavelength `λ_i` (tokens) |
|---|---|---|
| 0 | 1.0 | **6.283** |
| 1 | 0.1 | 62.83 |
| 2 | 0.01 | 628.3 |
| 3 | 0.001 | 6283.2 |

At `d = 128` (a realistic head width), base 10000: `λ_0 = 6.283`, `λ_63 = 54,410`.

### Which dimensions fall each side of the thresholds

**[measured here]**, with `α = 1`, `β = 32`, `r = L/λ`:

At `d_k = 8`:

| `L` | `i=0` | `i=1` | `i=2` | `i=3` | verdict |
|---|---|---|---|---|---|
| **16** (this app's sentence) | r=2.546, γ=0.050 | r=0.255, γ=0 | r=0.025, γ=0 | r=0.003, γ=0 | **nothing is left alone** |
| **2048** (LLaMA) | r=325.9, γ=1 | r=32.59, γ=1 | r=3.259, γ=0.073 | r=0.326, γ=0 | 2 untouched, 1 ramp, 1 full PI |
| **4096** (Llama 2) | r=651.9, γ=1 | r=65.19, γ=1 | r=6.519, γ=0.178 | r=0.652, γ=0 | 2 untouched, 1 ramp, 1 full PI |

The `L = 16` row is the app's honest baseline and it is instructive rather than embarrassing: at a
16-token context *every* pair completes fewer than 32 rotations, so `γ ≈ 0` almost everywhere and
YaRN degenerates to plain PI. The card must let the reader move `L` (and the base) to escape it —
that is the demonstration, not a defect.

At `d = 128`, `L = 4096`, `α = 1`, `β = 32` — **the partition the method actually produces**:

| band | count of 64 pairs | which |
|---|---|---|
| untouched, `γ = 1` (`r > 32`) | **21** | `d = 0…20` |
| the ramp, `0 < γ < 1` | **25** | `d = 21…45` |
| fully interpolated, `γ = 0` (`λ ≥ L`) | **18** | `d = 46…63` |

Boundaries: the first pair with `r ≤ 32` is `d = 21` (`r = 31.75`, `γ = 0.992`); the first with
`λ ≥ L` is `d = 46`. So on a real model **a third of the dimensions are left completely alone, a
quarter get the full PI squeeze, and the ramp is 39% of the width** — the crossfade is not a thin
seam, it is most of the picture. That is the number that makes "NTK-by-parts" concrete, and it is
printed nowhere in the paper.

### NTK-aware's base, for the comparison the card needs

`b′ = b·s^(|D|/(|D|−2))` (v2 eq. 16 / v1 eq. 16). **[measured here]:**

| `\|D\|` | exponent `\|D\|/(\|D\|−2)` | `s=2` | `s=8` | `s=16` | `s=32` |
|---|---|---|---|---|---|
| 8 (this app) | 1.3333 | 2.52e4 | 1.60e5 | 4.03e5 | 1.02e6 |
| 128 | 1.0159 | 2.02e4 | 8.27e4 | 1.67e5 | 3.38e5 |

At `|D| = 8` the exponent is 1.333 rather than ~1.016, so the app's toy exaggerates NTK-aware's base
change by a third. Say so on the card rather than letting the reader read the toy as the real thing.

**An inconsistency inside Appendix A.1's derivation.** The text says *"the last dimension d ∈ D is
|D|−2"*, but eq. 23 then writes `b′^((|D|−2)/|D|) = s·b^((|D|−2)/|D|)` — the exponent you get if `d`
indexes *pairs* and the last pair is `|D|/2 − 1`, not `|D|−2`. Taking the prose literally would give
`b′ = b·s^(|D|/(2(|D|−2)))`, half the exponent that eq. 24 and every implementation use. The algebra is
the correct half; the sentence justifying it is off by a factor of two.

### The temperature, in numbers

**[measured here]**, from `√(1/t) = 0.1·ln(s) + 1`:

| `s` | `√(1/t)` = the factor on the RoPE embedding | logit multiplier `1/t` | `t` (v2 sense) |
|---|---|---|---|
| 2 | 1.0693 | 1.1434 | 0.8746 |
| 4 | 1.1386 | 1.2965 | 0.7713 |
| 8 | **1.2079** | 1.4591 | 0.6853 |
| 16 | 1.2773 | 1.6314 | 0.6130 |
| 32 | 1.3466 | 1.8133 | 0.5515 |

The `s = 8` row reproduces the paper's own worked value: Appendix A.2 prints
*"√(1/t) = 0.1 ln(s) + 1 ≈ 1.208"*. It is a **very** weak function of `s` — over a 16× range of scale
factor the multiplier moves from 1.14 to 1.81. A logarithm with a hand-set coefficient of 0.1.

### What the temperature does to a real attention row

**[measured here]** — `app/model/transformer.js` (32 dims, 4 heads, `d_k = 8`, 2 blocks, seed
20260817), the 16-token sentence *"the cat sat on the mat and the dog sat on the log by the door"*,
RoPE at base 10000, scaling the last query's pre-softmax logits by `(0.1 ln s + 1)²` and measuring
Shannon entropy in nats:

| `s` | logit multiplier | entropy, block 0 head 0 | mean over all 8 (block, head) rows |
|---|---|---|---|
| 1 (off) | 1.0000 | 0.7534 | 0.8604 |
| 2 | 1.1434 | 0.6118 | 0.7351 |
| 8 | 1.4591 | 0.3949 | 0.5512 |
| 16 | 1.6314 | 0.3141 | 0.4842 |
| 32 | 1.8133 | 0.2479 | 0.4292 |

Uniform attention over 16 keys would be `ln 16 = 2.7726` nats. **The recommended factor lowers entropy
monotonically — at `s = 16` it cuts the head-0 row's entropy by 58%.** So the operation labelled a
"temperature" and motivated (in v1) as *"increasing the temperature"* measurably **sharpens** the
distribution. The card can put that number on screen and let the reader check the paper's wording
against it. (Untrained weights, so treat this as arithmetic about softmax, not evidence about
language models — see the boundary section.)

### Perplexity: the ablation, which only exists in v3

**Table 5 (v3), LLaMA 7B (2k pretrained context), sliding-window perplexity `S = 256`, ten 128k
Proof-pile documents.** Non-fine-tuned:

| method | scale | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|
| none | — | 4.05 | — | — | — | — |
| PI | 2k×2 | 4.36 | 3.90 | — | — | — |
| NTK-aware | 2k×2 | 4.08 | 5.97 | — | — | — |
| NTK-by-parts | 2k×2 | 4.12 | 3.71 | — | — | — |
| **YaRN** | 2k×2 | 4.07 | **3.67** | — | — | — |
| PI | 2k×8 | >10¹ | >10¹ | >10¹ | >10¹ | — |
| NTK-aware | 2k×8 | 4.64 | 4.27 | 4.24 | >10¹ | — |
| NTK-by-parts | 2k×8 | 4.98 | 4.91 | 5.33 | 5.79 | — |
| **YaRN** | 2k×8 | **4.37** | **3.95** | **3.81** | **3.33** | — |
| PI | 2k×16 | >10² | >10² | >10² | >10² | >10² |
| NTK-aware | 2k×16 | 5.23 | 5.02 | 5.22 | 6.85 | >10¹ |
| NTK-by-parts | 2k×16 | 6.04 | 7.54 | >10¹ | >10¹ | >10¹ |
| **YaRN** | 2k×16 | **4.61** | **4.24** | **4.18** | **3.66** | **3.45** |

**Read the NTK-by-parts rows.** Without the temperature, at `s = 8` it is *worse than NTK-aware at
every length*, and at `s = 16` it **collapses** (6.04, 7.54, >10¹) while YaRN with the same
interpolation is fine (4.61 … 3.45). In the non-fine-tuned regime the temperature is not a garnish —
it is carrying the entire result, and the interpolation half alone is the second-worst method tested.
This inverts the usual telling, in which NTK-by-parts is the substance and the temperature a tweak.

Fine-tuned, 400 steps, `s = 16`, same table:

| method | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| PI | 5.70 | 4.95 | 4.64 | 3.97 | 3.57 |
| NTK-aware | 4.39 | 3.92 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts | **4.14** | **3.75** | 3.62 | 3.12 | 2.81 |
| YaRN | 4.19 | 3.77 | **3.30** | **3.09** | **2.77** |

**Now it reverses.** After fine-tuning, NTK-by-parts *alone* beats full YaRN at 2048 (4.14 vs 4.19) and
4096 (3.75 vs 3.77) — i.e. the temperature costs a little at and near the original context — and YaRN
wins at 8192 and beyond. So the honest decomposition is: **the interpolation is what survives
fine-tuning; the temperature is what makes the method work without it, and it is a small tax at short
lengths.** Note also NTK-aware's 8.49 at 32768 — it holds up to 16k and then breaks, which is the
"out-of-bound extrapolation" failure §3.1 predicts for it.

### Perplexity: the headline tables

**Table 1 (v2) / Table 6 (v3) — Llama 2 7B, 4096 → 8192, and the source of "2.5×".** v2's version
carries a *Trained Tokens* column that v3 drops:

| method | trained tokens (v2) | steps (v3) | 2048 | 4096 | 6144 | 8192 |
|---|---|---|---|---|---|---|
| none (v3 only) | — | — | 4.00 | 3.58 | — | — |
| PI (`s=2`) | 1B | 1000 | 3.92 | 3.51 | 3.51 | **3.34** |
| NTK (`θ=20k`) | 1B | — | 4.20 | 3.75 | 3.74 | 3.59 |
| **YaRN (`s=2`)** | **400M** | **400** | **3.91** | **3.50** | 3.51 | 3.35 |

v3's own summary of this table is notably more modest than the abstract:
*"Even if YaRN was only fine-tuned for 400 steps compared to PI's 1000 steps, we obtain similar results
to PI."* **Similar**, not better — and PI actually wins the 8192 column, 3.34 vs 3.35. The 2.5× claim is
a *cost* claim at matched quality, and the paper says so in its own words.

**Table 2 (v2) / Table 7 (v3) — Llama 2 at 64k and 128k, against open models:**

| size | model | window | method | 8192 | 32768 | 65536 | 98304 | 131072 |
|---|---|---|---|---|---|---|---|---|
| 7B | Together | 32k | PI | 3.50 | 2.64 | >10² | >10³ | >10⁴ |
| 7B | Code Llama | 100k | NTK | 3.71 | 2.74 | 2.55 | 2.54 | 2.71 |
| 7B | YaRN (`s=16`) | 64k | YaRN | 3.51 | 2.65 | 2.42 | **>10¹** | **>10¹** |
| 7B | YaRN (`s=32`) | 128k | YaRN | 3.56 | 2.70 | 2.45 | 2.36 | 2.37 |
| 13B | Code Llama | 100k | NTK | 3.54 | 2.63 | 2.41 | 2.37 | 2.54 |
| 13B | YaRN (`s=16`) | 64k | YaRN | 3.25 | 2.50 | 2.29 | **>10¹** | **>10¹** |
| 13B | YaRN (`s=32`) | 128k | YaRN | 3.29 | 2.53 | 2.31 | 2.23 | 2.24 |

Two things the prose does not dwell on. (i) The `s=16` models **blow up past their window** exactly as
hard as PI does — `>10¹` at 98k. Graceful degradation past `L′` is *not* a property of YaRN; it is a
property of Dynamic Scaling, which these fixed-`s` rows do not use. (ii) `s=32` is *worse than* `s=16`
at every length where both work (3.56 vs 3.51, 2.70 vs 2.65, 2.45 vs 2.42). Extending further costs
quality everywhere below.

The claim resting on the 128k column:

> *"Of particular note are the YaRN (s=32) models, which show continued declining perplexity through
> 128k, despite the fine-tuning data being limited to 64k tokens in length, demonstrating that the
> model is able to generalize to unseen context lengths."*

Check that against the cells: 7B goes 2.36 at 98304 → **2.37** at 131072, and 13B 2.23 → **2.24**. Both
tick *up* in the last column. "Continued declining perplexity through 128k" is true from 8192 to 98304
and false in the final step, by 0.01. Small, but the sentence is the headline extrapolation claim.

**GovReport (Table 4 v2 / Table 8 v3), 50 documents, fixed 32k window:** 7B — Together (PI) 3.67,
Code Llama (NTK) 4.44, YaRN `s=16` **3.59**, YaRN `s=32` 3.64; 13B — Code Llama 4.22, YaRN `s=16`
**3.35**, YaRN `s=32` 3.39. Again `s=32` loses to `s=16`.

### Passkey

**Not in v1.** Table 5 (v2):

| size | model | `s` | window | training ctx | method | passkey ctx | accuracy |
|---|---|---|---|---|---|---|---|
| 7B | Together | 4 | 32k | 32k | PI | 32k | 100% |
| 7B | Code Llama | 88.6 | 100k | 16k | NTK | 112k | 94.3% |
| 7B | YaRN | 16 | 64k | 64k | YaRN | 64k | 96.3% |
| 7B | YaRN | 32 | 128k | 64k | YaRN | 128k | **99.4%** |
| 13B | Code Llama | 88.6 | 100k | 16k | NTK | 128k | 99.4% |
| 13B | YaRN | 16 | 64k | 64k | YaRN | 64k | 97.5% |
| 13B | YaRN | 32 | 128k | 64k | YaRN | 128k | **99.4%** |

§4.3.2: *"Both 7b and 13b models fine-tuned using YaRN at 128k context size passes the passkey
retrieval task with very high accuracy (>99%) within the entire context window size."* Note Code Llama
13B ties YaRN's 99.4% at the same 128k, using NTK-aware and 16k training data — on this benchmark YaRN
does not beat it. The paper's own caveat, from Appendix B.2:

> *"as YaRN with s = 32 was trained for 200 more steps than YaRN with s = 16 while having a higher
> passkey accuracy with similar perplexity, we hypothesize that perplexity may not be a great indicator
> of whether an LLM is able to attend to all tokens and does not exhaustively determine long context
> performance. This also suggests that the YaRN models with s = 16 might be relatively undertrained for
> the passkey retrieval task."*

That is the paper conceding that its own primary metric may not measure the thing it claims, and that
the `s=16`/`s=32` comparison is confounded by training steps.

### Standardised benchmarks — degradation on the original context

**Table 3 (v1/v2), Hugging Face Open LLM Leaderboard** (25-shot ARC-c, 10-shot HellaSwag, 5-shot MMLU,
0-shot TruthfulQA):

| size | model | window | method | ARC-c | HellaSwag | MMLU | TruthfulQA |
|---|---|---|---|---|---|---|---|
| 7B | Llama 2 | 4k | none | **53.1** | 77.8 | **43.8** | 39.0 |
| 7B | Together | 32k | PI | 47.6 | 76.1 | 43.3 | **39.2** |
| 7B | Code Llama | 100k | NTK | 39.9 | 60.8 | 31.1 | 37.8 |
| 7B | YaRN (`s=16`) | 64k | YaRN | 52.3 | **78.8** | 42.5 | 38.2 |
| 7B | YaRN (`s=32`) | 128k | YaRN | 52.1 | 78.4 | 41.7 | 37.3 |
| 13B | Llama 2 | 4k | none | **59.4** | 82.1 | **55.8** | 37.4 |
| 13B | Code Llama | 100k | NTK | 40.9 | 63.4 | 32.8 | **43.8** |
| 13B | YaRN (`s=16`) | 64k | YaRN | 58.1 | **82.3** | 52.8 | 37.8 |
| 13B | YaRN (`s=32`) | 128k | YaRN | 58.0 | 82.2 | 51.9 | 37.3 |

> *"We observe that there is minimal performance degradation between the YaRN models and their
> respective Llama 2 baselines. We also observe that there was on average a 0.49% drop in scores between
> the YaRN s=16 and s=32 models. From this we conclude that the the iterative extension from 64k to 128k
> results in negligible performance loss."* (the doubled *"the the"* is in the source, in all versions)

The degradation is real but small — worst cell is 13B MMLU, 55.8 → 51.9, a 3.9-point / 7% drop at
`s=32`. Set against Code Llama's NTK-aware collapse (7B MMLU 43.8 → 31.1, HellaSwag 77.8 → 60.8) it is
excellent, and that contrast is the strongest thing in this table. v3 adds *"Some variance is to be
expected as the PG19 dataset we used for fine-tuning is very different from the original pre-training
datased used for LLaMA and Llama 2 models"* (typo in source) — an honest confound: some of the drop is
domain shift from fine-tuning on books, not from the interpolation.

**Table 2 (v3 only) — the benchmark ablation, LLaMA 7B, all fine-tuned 400 steps at 2k×16:**

| method | ARC-c | HellaSwag | MMLU | TruthfulQA |
|---|---|---|---|---|
| none (baseline) | **51.0** | **77.8** | **35.7** | 34.3 |
| PI | 44.8 | 70.2 | 25.9 | 34.1 |
| NTK-aware | 47.4 | 73.9 | 27.7 | 32.6 |
| **NTK-by-parts** | **48.5** | 76.6 | **32.7** | 33.4 |
| **YaRN** | 48.1 | **77.2** | 30.0 | **35.1** |

**The two halves split the wins.** NTK-by-parts alone beats full YaRN on ARC-c (48.5 vs 48.1) and by
2.7 points on MMLU (32.7 vs 30.0); YaRN wins HellaSwag and TruthfulQA. On the benchmark suite the
temperature is not free — it is a trade. No version of the paper comments on this table's YaRN-vs-
NTK-by-parts comparison at all.

### Training cost — and exactly what "10× / 2.5×" is comparing

The abstract, all three versions: *"requiring 10x less tokens and 2.5x less training steps than
previous methods."* §4 attributes the two halves to **two different baselines**:

> *"this result is achieved with only 400 training steps, representing approximately 0.1% of the model's
> original pre-training corpus, a 10x reduction from Rozière et al. 2023 and 2.5x reduction in training
> steps from Chen et al. 2023, making it highly compute-efficient for training with no additional
> inference costs."*

10× tokens is versus **Code Llama**; 2.5× steps is versus **PI**. "Than previous methods" in the
abstract merges two comparisons against two different papers into one clause.

Training recipe (§4.1, all versions): Llama 2 7B and 13B; lr `2e-5`, no weight decay, 20-step linear
warmup, AdamW `β₁=0.9, β₂=0.95`; `s=16` for **400 steps** at **global batch 64** on **PG19** chunked to
**64k**; `s=32` continues from the `s=16` checkpoint for **200 more steps**, still on 64k data.

**[measured here]** — the arithmetic behind "≈0.1%": `400 × 64 × 65,536 = 1.678B` tokens, which against
Llama 2's 2T-token pre-training is **0.084%**. The full `s=32` model at 600 steps is `2.517B` =
**0.126%**. So "approximately 0.1%" is right, and the *token* count is ~1.7B — note this is **not** the
"400M" in v2's Table 1, which describes a different, `s=2`, 8k model whose training procedure is
described nowhere in the paper.

**Table 4 (v3 only) — GPU-hours,** the clearest cost statement in any version:

| model | method | scale | effective context | A100-hours |
|---|---|---|---|---|
| LLaMA 7B YaRN | YaRN | 2k×16 | 32k | **128** |
| Llama 2 7B YaRN | YaRN | 4k×16 | 64k | **256** |
| Llama 2 7B YaRN | YaRN | 4k×32 | 128k | **256 + 128** |
| PI (Chen et al.) | PI | 2k×8 | 16k | 640 |
| Together | PI | 4k×8 | 32k | ? |
| NTK-aware (ref 38) | NTK-aware | 4k×44.2 | ≈50k | **64,000** |
| Code Llama | NTK-aware | 4k×88.6 | ≈100k | 6,400 |

384 A100-hours for a 128k model against 6,400 for Code Llama's 100k — a **16.7×** gap, which is the
real efficiency headline and it appears only in a version published two and a half years after v1.

## What the live view must let the reader do

The toy: 32 dims, 4 heads (`d_k = 8`), 2 blocks, causal mask, seeded untrained weights, editable
~16-token sentence. `position.js` already exposes `rope({base, stretch, dims})`; `mixers.js` takes
`rotate(v, pos)`; `flow.js` draws a per-token rotation dial via `opts.rotate: {angle(pos), label}`;
`curve.js` gives a line plot with a reference curve, a shaded band, a dead region and a marker;
`bars.js` gives `barList` and `readout`.

**The boundary, stated first and on the card itself.** This app has an untrained model, 16 tokens, and
no training loop. It **cannot** fine-tune, **cannot** measure perplexity, and **cannot** reproduce a
single quality number in §Numbers. What it *can* do is exactly the three things YaRN is actually made
of, and all three are pure arithmetic that a toy settles completely: the per-dimension partition, the
wavelength criterion as a live computation, and the temperature's effect on a real softmax row.
Everything below is one of those three. Where a panel shows a paper number, it is displayed as a
quotation, never as something the app measured.

A seam note: `rope()` currently applies one scalar `stretch` to all pairs. NTK-by-parts needs a
**per-pair** factor. The smallest honest change is to let `stretch` be a number *or* a function
`(i) => factor` — a one-line change at `position.js:107` (`const a = pos * freqs[i] * (typeof stretch
=== "function" ? stretch(i) : stretch)`). The temperature needs a modulus on the rotation, which is a
second one-line change (scale `out[2i]`/`out[2i+1]` by a constant). Both are the paper's own
implementation story — *"YaRN can effectively alter the attention mechanism without modifying its
code"* — so making them one-line edits to `rope()` and nothing else **is** the demonstration.

1. **The partition, live — which dimensions get interpolated, which are left alone, which are in the
   ramp.** A `barList` of the head's pairs (4 at `d_k = 8`), each showing `λ_i`, `r(d) = L/λ_i`, and
   `γ`, coloured in three bands: `γ = 1` untouched, `0 < γ < 1` ramp, `γ = 0` full PI. Give the reader
   `L` (the "original context", 16 → 8192), `s` (1 → 32), and `α`/`β` (defaulting to 1 and 32, quoting
   *"good values for α and β are α = 1 and β = 32"*). At the app's own `L = 16` the readout says
   **0 untouched, 1 in ramp, 3 fully interpolated** — YaRN has degenerated to PI, and the card should
   say so out loud rather than hide it: with only 4 pairs and a 16-token context, no dimension
   completes 32 rotations. Push `L` to 2048 and it becomes **2 untouched, 1 ramp, 1 full**. Beside the
   live bars, print the `d = 128, L = 4096` partition as a static quoted-arithmetic panel —
   **21 untouched / 25 ramp / 18 full** — labelled as the app's computation for a realistic width, so
   the reader learns that the ramp is 39% of a real model and not a hairline.

2. **The criterion itself: unique angles versus repeated ones.** This is what `r(d)` *means*, and
   `curve.js` can show it directly. For a chosen pair `i`, plot the rotation angle `pos·θ_i mod 2π`
   against `pos` over `0…L`. For a slow pair (`r < 1`) the line is a single rising ramp that never
   wraps — every position has a unique angle, which is the paper's *"having all unique position pairs
   implies that the absolute positional information remains intact"*. For a fast pair (`r > 32`) it is
   a sawtooth with dozens of teeth — *"when the wavelength is short, only relative positional
   information is accessible to the network"*. Mark `λ_i` on the x-axis and shade the region past the
   first wrap as the dead region. Then flip on the `s`-scaling and show what PI does to each: the fast
   pair's teeth get `s` times wider (the local structure the paper says is *"severely impaired"*), the
   slow pair merely rises more gently. **One picture, two pairs, and the reader has derived why the
   two cases deserve different treatment** — which is the entire argument of §3.2.

3. **The three schemes on one axis — this is the synthesis, and it is what the card is for.** On the
   curve view, plot the *effective per-pair frequency multiplier* `h(θ_i)/θ_i` against pair index `i`,
   with a selector for PI / NTK-aware / NTK-by-parts, all at the same `s`. PI is a flat line at `1/s`
   (blind: every pair identically). NTK-aware is a smooth curve from ~1 down to ~`1/s` (blind to
   wavelength, but graded — a base change, `b′ = b·s^(|D|/(|D|−2))`, computed live). NTK-by-parts is a
   **step-with-a-ramp**: exactly 1 on the untouched pairs, exactly `1/s` on the interpolated ones, the
   crossfade between. Overlay all three and the card's thesis is one image: PI is a constant,
   NTK-aware is a smooth function of the *index*, and only NTK-by-parts is a function of the
   *wavelength relative to the context*. Print `b′` beside the NTK-aware curve and flag the app's own
   distortion — **[measured here]** the exponent `|D|/(|D|−2)` is 1.333 at `d_k = 8` versus 1.016 at
   `d = 128`, so the toy overstates the base change by a third.

4. **The temperature: fold it into the rotation and watch the entropy move.** Two panels sharing one
   control, the `s` slider, with `√(1/t) = 0.1 ln(s) + 1` printed live (1.2079 at `s = 8`, matching the
   paper's *"≈ 1.208"*). (a) On the dataflow picture, the existing `opts.rotate` dial: keep the angle
   as-is but draw the dial's **radius** scaled by `√(1/t)`, with the label *"turned by position, and
   lengthened by √(1/t)"*. That is the whole trick, drawn — the temperature is a radius, not a code
   path. Print `‖rotate(q,m)‖ / ‖q‖`, which reads exactly `√(1/t)` and is the one place on this
   timeline where RoPE's norm preservation is deliberately given up. (b) Beneath it, the last query's
   attention weights as a `barList` at `t = 1` and at the recommended `t`, with the **entropy in nats**
   for both. **[measured here]** on the default sentence, block 0 head 0: **0.7534 → 0.3141 nats at
   `s = 16`**, against `ln 16 = 2.7726` for uniform. Then the honest label, which is the card's sharpest
   moment: the paper calls this a temperature and v1 motivated it as *"increase the 'temperature' of
   the attention logits"*, and the measured effect is that entropy goes **down** by 58%. Show the two
   version strings side by side (`√t = 0.1 ln s + 1` v1, `√(1/t) = 0.1 ln s + 1` v2) and the identical
   number they produce, so the reader sees a symbol flip that changed nothing in any implementation.

5. **The ablation, as a readout the app can honestly host: what each half does alone.** Four toggle
   states — off / interpolation only / temperature only / both — each recomputing the model's real
   attention and showing two app-measured quantities (the partition from panel 1, the entropy from
   panel 4) plus, quoted beside them, the paper's v3 Table 5 row for the same combination at `s = 16`.
   The point to land is the one the paper never states: **without fine-tuning, NTK-by-parts alone
   collapses (6.04 → >10¹) and the temperature rescues it; with fine-tuning, NTK-by-parts alone is
   *better* than YaRN at the original context (4.14 vs 4.19) and worse at long range (3.62 vs 3.30).**
   Two numbers, one reversal. Label the perplexities unambiguously as quoted from Table 5 of **v3**,
   not measured here, and note that no ablation existed in the version this card is dated to.

6. **Dynamic scaling, which the app can show completely.** `s = max(1, l′/L)` recomputed as the reader
   grows the sentence, with `L` set low (say 8) so a 16-token sentence actually crosses it. Display
   `s`, `γ` per pair, and the attention matrix, all live. Two facts fall out with no training required:
   below `L` the scheme is **bit-for-bit identical** to plain RoPE (`s = 1`, so `h(θ) = θ` for every
   pair regardless of `α`, `β`) — check it with a max-abs difference readout reading `0.0e0` exactly —
   and above `L` the treatment changes *every* token's rotation, not just the new one. Put the paper's
   warning next to that second readout: *"The correct implementation should cache the kv-embeddings
   before applying RoPE, as the RoPE embedding of every token changes when s changes."* A "reuse the
   cached rotated keys" toggle that shows the resulting corruption makes the warning a measurement.
   This is the one interaction where the app's inability to train costs nothing at all, because dynamic
   scaling is an inference-time method by definition.

## What the source does *not* establish

- **v1 — the version this card is dated to — has no appendices, no ablation, no passkey experiment, no
  dynamic-scaling experiment, and no table supporting its own abstract.** The "10× tokens / 2.5× steps"
  claim appears in v1's abstract and §4 with **no supporting cells anywhere in v1**; the 1B-vs-400M
  comparison first appears in v2's Table 1, and the 1000-vs-400-steps framing in v3's Table 6. A card
  dated 31 Aug 2023 is describing a paper whose evidence arrived in two later instalments, the second
  of them in 2026. Say which version each number comes from, always.
- **The temperature has no derivation, and the paper withdrew the one it had.** v1 offered an entropy
  argument (average minimum distance `L/(N²−1)`, a "spikier" softmax) and it is internally
  contradictory in a single sentence — *"decreases the average entropy"* alongside *"the network 'pays
  more attention' to more tokens"*. v2 deletes it and replaces it with a bare observation. What remains
  is a curve fit: *"found by fitting √(1/t) at the lowest perplexity against the scale extension by
  various factors s"*. There is no theory of why `ln`, why 0.1, or why the offset is 1.
- **The values behind that fit are printed nowhere, in any version.** Its entire empirical support is
  Appendix A.2/A.3's Figures 2, 3 and 4 (v2 numbering) — a perplexity-versus-`1/√t` sweep, a
  per-position perplexity-change plot, and a histogram of best-`1/√t` by position segment. **Not one
  numeric value from any of them appears in the text.** There is no table of `(s, best √(1/t))` pairs —
  the very data the recommended equation was fitted to. The claim *"the best value of t is mostly
  consistent across different samples and different positions"* and *"this finding is consistent for
  different values of s"* is supported by figures only. This is the single largest unprinted-figure
  gap in the paper, and it sits directly under the equation everyone implements.
- **Other figures whose values are not in the text:** Figure 1 (the sliding-window perplexity curve, in
  every version — v1's Figure 1 is only partly recoverable from v1's Table 1), Figure 5 (v2) /
  Figure 8 (v3), the RoPE-vs-Dynamic-PI-vs-Dynamic-YaRN comparison — **so the entire Dynamic-YaRN
  result is a single unlabelled chart on a single GovReport sample**, and its conclusions
  (*"Dynamic Scaling effectively prevents the blow-up of perplexity score beyond pretrained context
  window"*, *"Dynamic-YaRN outperforms Dynamic-PI"*) carry no numbers at all — and v3's Figure 2
  (training-loss curves behind *"YaRN converges faster"*) and Figure 3's passkey half.
- **α = 32 — sorry, β = 32 — is unjustified, and α, β are never swept.** *"should be tuned on a
  case-by-case basis"* and *"we have found experimentally that … good values … are α = 1 and β = 32"*
  is the complete treatment. No sensitivity analysis, no table of alternatives, no argument for why 32
  rotations rather than 8 or 128. Given that **[measured here]** those two numbers determine a 21/25/18
  split of a 128-dimensional head, that is a lot of behaviour resting on one unexamined pair of
  constants.
- **The two halves of YaRN are never compared to each other in the version that introduced them.** The
  ablation is v3, February 2026 — two and a half years later. And when it arrives it does not support
  the simple story: v3 Table 5 shows NTK-by-parts alone *beating* full YaRN at 2048 and 4096 after
  fine-tuning, and v3 Table 2 shows it beating YaRN on ARC-c and MMLU. Neither is discussed. "YaRN
  surpasses all previous methods" (§3.4, unchanged since v1) is not what these two tables say cell by
  cell; what they say is that YaRN wins at long range and NTK-by-parts alone is competitive-to-better
  at the original context.
- **"With no downsides" is contradicted by the paper's own tables.** The conclusion, all versions:
  *"YaRN improves upon all existing RoPE interpolation methods and can act as a drop-in replacement to
  PI, with no downsides and minimal implementation effort."* The downsides in-paper: 7B MMLU 43.8 →
  41.7 at `s=32`; `s=32` worse than `s=16` at every jointly-evaluated length in Tables 2 and 4; and PI
  beating YaRN at 8192 in Table 1 (3.34 vs 3.35).
- **The fixed-`s` models break past their window exactly like PI.** YaRN `s=16` reads `>10¹` at 98304
  and 131072. Graceful degradation belongs to **Dynamic Scaling**, a separate and orthogonal
  technique. Any claim that "YaRN degrades gracefully" is attributing Dynamic Scaling's property to
  YaRN.
- **"Continued declining perplexity through 128k" is contradicted by its own final cell** — 2.36 → 2.37
  (7B) and 2.23 → 2.24 (13B) from 98304 to 131072.
- **The transfer-learning claim is asserted, not measured.** *"the s=32 model is equivalent to the s=16
  model across the entire context size, despite only being trained on s=32 for 200 steps"* —
  "equivalent" is not defined, no equivalence metric is computed, and Table 2 in fact shows `s=32`
  slightly *worse* everywhere the two overlap. The defensible claim is "200 extra steps sufficed";
  "equivalent" is not established.
- **The `s=2` / 8k model in v2's Table 1 has no described training procedure.** §4.1 describes only
  `s=16` and `s=32` runs on PG19 at 64k. The 400M-token figure that anchors the token-efficiency claim
  belongs to a model the paper never says how it trained.
- **No comparison against ALiBi, xPos, ReRoPE or LM-Infinite.** §2.4 explicitly declines the last two
  because they *"modify the attention mechanism"* and are *"not immediately compatible with Flash
  Attention 2"* — a deployment argument, not an empirical one. The concept-13 branch is never
  measured against.
- **Everything is Llama-family.** LLaMA 7/13/33/65B and Llama 2 7/13/70B, plus Mistral 7B in v2's
  Appendix B.4 (which *"broadly follows the Llama architecture"*). The universality claim is hedged in
  the authors' own words — *"may have certain degree of 'universality' and may be generalizable across
  some models and training data"* — with two "may"s and a "some".
- **Claims commonly attributed to YaRN that are not in it.** (i) That YaRN "needs no fine-tuning" — the
  headline results are all fine-tuned; the non-fine-tuned results are the *ablation* (v3 Table 5) and
  Dynamic-YaRN (an unlabelled figure). (ii) That the temperature "increases entropy" or "flattens
  attention" — **[measured here]** the recommended factor multiplies logits by >1 and *sharpens*
  (0.7534 → 0.3141 nats at `s=16`). (iii) That YaRN introduces a new interpolation — Definition 3 is
  explicit that it is NTK-by-parts plus the temperature. (iv) That "NTK-by-parts" involves the NTK base
  change — it does not; `g(m) = m`, `h` is a ramp between `θ/s` and `θ`, base untouched. (v) That YaRN
  is 10× cheaper than PI — the 10× is versus Code Llama; versus PI it is 2.5×, and v3 describes the
  outcome as *"similar results"*.
- **Two transcription errors to not inherit.** v1's inverted PI direction (`g(m) = s·m`, `mL'/L`), fixed
  in v2. And v2/v3 eq. 17's `b′` where `b` is meant. Also the doubled *"the the"* in §4.3.3 and *"the
  entirety of of the code"* in §6, both present in every version, and *"surpassing previous the
  state-of-the-art"* in the abstract of all three — useful as a fingerprint that a quotation is
  genuine.
- **The app establishes nothing empirical.** Untrained seeded weights, 16 tokens, 4 pairs per head, no
  training and no perplexity. Every app-side number in this note is arithmetic — wavelengths, ramp
  memberships, base changes, softmax entropies — which is the honest scope: YaRN's *mechanism* is
  entirely arithmetic, and only its *justification* is empirical. The card can own the first completely
  and must quote the second.

## Leaves behind

**Backward, this is the card that names what concepts 16 and 17 each got wrong, and it does so with one
word: blind.** PI (16) divides every position by `s`, which preserves the trained angular range exactly
but squeezes the fast pairs — the ones whose whole job is telling adjacent tokens apart — until, in the
paper's words, *"all tokens become closer to each other"* and the model *"is confused on the positional
order of close-by tokens"*. NTK-aware (17) fixes that by changing the base so high frequencies move
less, but it is still blind: it grades by *dimension index*, not by whether that dimension's wavelength
actually fits inside the context, and it pays for the grading by pushing the slowest dimensions to
angles the model never saw — the paper's *"some dimensions are slightly extrapolated to 'out-of-bound'
values"*, which is exactly why *"fine-tuning with 'NTK-aware' interpolation yields inferior results to
PI"* and why v3's Table 5 shows NTK-aware fine at 16k and broken (8.49) at 32k. YaRN's answer is that
both were answering the wrong question. The question is not *how much* to compress; it is *which
dimensions have a compression problem at all*. Compute `r(d) = L/λ_d`, and the ones that already
complete many rotations inside the trained context are done — they encode relative offset, they wrap,
they saw every angle in training, leave them alone. The ones whose wavelength exceeds the context never
completed a cycle, hold absolute position, and must be interpolated or they extrapolate into angles
that do not exist in the training distribution. **[measured here]** on a real 128-wide head at
`L = 4096` that splits 21 / 25 / 18 — a third untouched, a third interpolated, and a ramp in between
that is bigger than either.

The second half is a different kind of lesson and it belongs to concept 12. RoPE's rotation matrix is
orthogonal — the card for concept 12 makes `‖R_m x‖ − ‖x‖ ≈ 1e-16` one of its readouts. YaRN's
temperature works by *breaking that on purpose*: give the rotation a modulus of `√(1/t)` and the
softmax temperature comes along free, because the table was precomputed anyway. That is only available
to a multiplicative position encoding. Sinusoidal (concept 2) adds, so a scale on the encoding is not a
scale on the logit. ALiBi (concept 13) adds a bias, and adding a constant to logits does nothing to a
softmax. **The one method on this timeline that can rescale attention for free is the one that encodes
position as a rotation** — a payoff for concept 12's design decision that arrives two years later and
that the RoPE paper did not anticipate.

**Forward, it closes the extension arc and leaves two open bills.** The first is the α/β constants: two
unswept numbers that determine how a third of a model's dimensions are treated, with *"tuned on a
case-by-case basis"* as their entire methodology. The second is bigger and the paper half-admits it in
Appendix B.2 — *"perplexity may not be a great indicator of whether an LLM is able to attend to all
tokens and does not exhaustively determine long context performance"*. Every headline number in this
paper is a perplexity; the passkey table exists because the authors suspected perplexity of not
measuring the thing. That suspicion is the bridge out of the position-encoding arc entirely: once you
can *define* attention at 128k, the question stops being "does the encoding hold up" and becomes "can
the model use what it can now reach" — which is where the timeline's efficiency and retrieval cards
take over. And note what YaRN never touches: `g(m) = m`. After three concepts of argument, the
position index itself is left alone and all the work happens in `h(θ_d)` — the frequency ladder, the
same two constants concept 12's card already put sliders on.
