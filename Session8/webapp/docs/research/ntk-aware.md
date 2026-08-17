# Concept 17 — NTK-aware scaled RoPE

**Card id:** `ntk-aware` · **Date:** 2023-06-29 (see below — the record says 2023-06-30 and that is
the *follow-up* post's date) · **Pressure:** where a token sits

**There is no paper.** This is the one entry on the timeline whose source is a Reddit post, and the
card has to render it as what it is: a result that was published to a forum, adopted by half the
open-source ecosystem inside three weeks, and only written up — by its own author, two months later,
as a section of somebody else's method — in YaRN. The chronology record already carries
`kind: "post"`, `verified: false`. Everything below is organised around keeping the difference
between *derived*, *measured*, *claimed* and *repeated* visible, because for this concept those four
categories genuinely came apart.

## What was read

**Reached, and read in full:**

- **The primary post.** bloc97, *"NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+)
  context size without any fine-tuning and minimal perplexity degradation."*, r/LocalLLaMA.
  `reddit.com` and `old.reddit.com` are both unreachable from this environment (WebFetch refuses the
  host; `curl` and `r.jina.ai` both get a 403 "blocked by network security" page; the `.json` API is
  403 as well). The post was recovered instead from the **Wayback Machine**, snapshot
  `20230629143320`, fetched raw and parsed out of the `<shreddit-post>` element. That element carries
  the post's own metadata, which is how the date below is verified. The body text is complete in that
  snapshot; the **comments are not** (the snapshot renders the post only), so bloc97's replies were
  not read.
- **The follow-up post.** emozilla, *"Dynamically Scaled RoPE further increases performance of long
  context LLaMA with zero fine-tuning"*, r/LocalLLaMA — same route, Wayback snapshot `20231011075140`
  of `/comments/14mrgpr/`. Body complete, comments not.
- **YaRN**, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071), Bowen Peng (= /u/bloc97), Jeffrey
  Quesnelle (= /u/emozilla), Honglu Fan, Enrico Shippole — *YaRN: Efficient Context Window Extension
  of Large Language Models*. [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2309.00071) fetched
  raw and read end to end: §1, §2.1–2.4, §3.1 (the "NTK-aware" section), §3.2, §3.3 (Dynamic NTK),
  §3.4, §4 with Tables 1–3, and **Appendix A.1**, which is the only published derivation of the
  exponent anywhere. The author list is the load-bearing fact here: YaRN is bloc97 and emozilla
  writing up their own forum posts, and it says so — the paper's own footnotes give their Reddit
  handles. So YaRN's criticism of "NTK-aware" is self-criticism, which makes it more credible as a
  negative finding and less useful as an independent check.
- **The reference implementation it spread through.** `huggingface/transformers` v4.32.0,
  `src/transformers/models/llama/modeling_llama.py`, read from raw GitHub — classes
  `LlamaLinearScalingRotaryEmbedding` (position interpolation) and
  `LlamaDynamicNTKScalingRotaryEmbedding`, whose docstring is the provenance record in code:
  *"Credits to the Reddit users /u/bloc97 and /u/emozilla"*.

**Not reached, and not reconstructed:**

- The **colab notebook** the post links (`NTKAwareScaledRotaryEmbedding.ipynb`) — the post says
  *"the changes to the RoPE code is only 3 lines"* and points there rather than inlining them. Not
  fetched. The three lines quoted everywhere (`max_position_embeddings = 16384`, `a = 8 #Alpha
  value`, `base = base * a ** (dim / (dim-2)) #Base change formula`) are *not* quoted here from the
  post text, because they are not in the post text I read.
- The **three graphs**, which are the post's entire empirical content. They are images; the snapshot
  preserves their captions, not their contents. **Every perplexity claim in the post therefore rests
  on evidence this note could not inspect**, and the card must say so rather than repeating the
  numbers people have since attached to it.
- The **comment threads** under both posts, `jquesnelle/scaled-rope` PR #1 ("NTK-by-parts"), and
  **llama.cpp**. llama.cpp's `--rope-freq-base` is real and widely used, but nothing about it was
  verified here, so the card should not assert what llama.cpp does.

**Date, verified.** The archived `<shreddit-post>` element gives
`created-timestamp="2023-06-29T08:21:29.413000+0000"`, `author="bloc97"`, and at capture time
`score="114"`, `comment-count="19"`. So the post is **29 June 2023, 08:21 UTC** — no timezone puts it
on the 30th. The follow-up, emozilla's Dynamic NTK post, is
`created-timestamp="2023-06-30T05:34:06.771000+0000"` — **30 June 2023**. The record's `2023-06-30`
is the date of the *second* post. Either fix the record to `2023-06-29` or relabel it; it is
currently 21 hours and one author out. (For the timeline's ordering it does not matter — position
interpolation, arXiv:2306.15595, is 2023-06-27, and the post is explicitly a response to it, *"the
paper from Meta which uses RoPE interpolation"*, posted **two days later**. That two-day gap is the
most interesting thing about the date and it survives either correction.)

## The mechanism, precisely

**The one-sentence difference.** Position interpolation changes **where the token is**; this changes
**how fast each dimension turns**. In YaRN's notation (§2.3, Eq. 12) every method in this family is a
pair of functions applied to RoPE — `f'(x_m, m, θ_d) = f(x_m, g(m), h(θ_d))` — and the two methods
differ by which of the two they touch:

    PI          g(m) = m/s        h(θ_d) = θ_d          (Eq. 10 / §2.3)
    NTK-aware   g(m) = m          h(θ_d) = b'^(−2d/|D|) (Definition 1, Eqs. 14–15)

That is the whole taxonomy. PI leaves the frequency ladder alone and divides the position; NTK-aware
leaves the position alone and rebuilds the ladder from a bigger base.

**The base.** YaRN, Definition 1, Eq. 16:

    b' = b · s^(|D|/(|D|−2))

with `s = L'/L` the scale factor (Eq. 11) and `b = 10000` the original base. Every angle in the model
is then `m · b'^(−2d/|D|)` instead of `m · b^(−2d/|D|)`. Nothing else changes. No position is
rescaled, no dimension is special-cased, no term is added to the score. It is a different constant in
one `pow`.

**Why that exponent — the derivation, in the only place it was ever written down.** YaRN Appendix
A.1, quoted in full because it is three sentences and the card's central argument depends on all
three:

> *"Recall that our goal is to spread out the interpolation pressure across the hidden dimensions
> using a base-change instead of scaling the frequencies by a fixed factor s. The property we want to
> guarantee is that: The lowest frequency needs to be scaled as much as linear positional scaling and
> the highest frequency to stay constant."*
>
> *"We introduce a new base b' such that the last dimension matches the wavelength of linear
> interpolation with a scale factor s. Since the original RoPE method skips odd dimensions in order to
> concatenate both cos(2πx/λ) and sin(2πx/λ) components into a single embedding, the last dimension
> d ∈ D is |D|−2."*

    b'^((|D|−2)/|D|) = s · b^((|D|−2)/|D|)        (23)
    b' = b · s^(|D|/(|D|−2))                       (24)

Read Eq. 23 rather than Eq. 24 — 23 is the *specification* and 24 is just algebra. The specification
is a **boundary condition on the slowest pair**: make its new wavelength exactly `s` times its old
one, i.e. make the slowest pair behave exactly as PI would have made it behave. The fastest pair
(`d = 0`) is untouched automatically, because `b'^0 = b^0 = 1` whatever the base — the exponent
`d/(d−2)` is not chosen to fix the fast end, it is chosen to fix the **slow** end, and the fast end
is fixed for free by the shape of the exponential ladder. Everything between the two ends is
interpolated geometrically: pair `i` (0-based over pairs, which is the app's indexing and matches
Eq. 23's `(|D|−2)/|D|` for the last pair) receives an **effective scale**

    s_i = θ_i / θ'_i = (b'/b)^(2i/|D|) = s^(2i/(|D|−2))          [derived here]

which is `1` at `i = 0` and exactly `s` at `i = |D|/2 − 1`. That single line is the mechanism, the
argument for it, and — as §3.1 goes on to say — the defect, all at once.

**The `−2` is not cosmetic.** It is there because RoPE's dimensions come in pairs, so with `|D|`
channels there are `|D|/2` frequencies and the last one sits at exponent `(|D|−2)/|D|`, not at 1. The
correction is tiny at real widths (`128/126 = 1.015873`) and large at toy widths (`8/6 = 1.3333`),
which is exactly why the app is a good place to see it: at `d_k = 8` the `−2` moves `b'` by a factor
of two, at `d = 128` by 3%.

**The post's own account of why base and not scale** — bloc97, verbatim, and note that it is an
intuition plus a citation, not a derivation:

> *"Basically if you apply Neural Tangent Kernel (NTK) theory to this problem, it becomes clear that
> simply interpolating the RoPE's fourier space 'linearly' is very sub-optimal, as it prevents the
> network to distinguish the order and positions of tokens that are very close by. Borrowing from NTK
> literature, scaling down the fourier features too much will eventually even prevent succesful
> finetunes (this is corroborated by the recent paper by Meta that suggests an upper bound of ~600x)"*
>
> *"Instead of the simple linear interpolation scheme, I've tried to design a nonlinear interpolation
> scheme using tools from NTK literature. Basically this interpolation scheme changes the base of the
> RoPE instead of the scale, which intuitively changes the 'spinning' speed which each of the RoPE's
> dimension vectors compared to the next. Because it does not scale the fourier features directly, all
> the positions are perfectly distinguishable from eachother, even when taken to the extreme (eg.
> streched 1million times, which is effectively a context size of 2 Billion)"*

YaRN §3.1 rewrites the same argument with a citation to Tancik et al.'s Fourier features:

> *"it was shown in [36], using Neural Tangent Kernel (NTK) theory, that deep neural networks have
> trouble learning high frequency information if the input dimension is low and the corresponding
> embeddings lack high frequency components. Here we can see the similarities: a token's positional
> information is one-dimensional, and RoPE expands it to an n-dimensional complex vector embedding."*
>
> *"Stretching the RoPE embeddings indiscriminately results in the loss of important high frequency
> details which the network needs in order to resolve tokens that are both very similar and very close
> together (the rotation describing the smallest distance needs to not be too small for the network to
> be able to detect it)."*

That parenthesis is the honest core of the whole method, and it is a statement about **the fastest
pair's angle between adjacent tokens**. Under PI that angle is divided by `s`; under NTK-aware it is
untouched. Everything else is bookkeeping.

**"NTK-aware" is a name, not a result.** Nothing in the post or in YaRN computes a neural tangent
kernel, states an NTK theorem, or derives Eq. 24 from NTK theory. The NTK reference is an *analogy*
to Tancik et al.'s finding about Fourier features in low-dimensional coordinate networks; Eq. 23 is
derived from a wavelength boundary condition and nothing else. The card should use the name because
the field does, and should not let it imply a derivation that does not exist.

**Dynamic NTK — recompute the scale from the current length.** This is emozilla's post, one day
later, and it is a different kind of idea: not a new schedule but a rule for choosing `s` at
inference time. From the post, verbatim:

> *"My idea was to use the exact position values for the first 2k context (after all, why mess with a
> good thing?) and then re-calculate the position vector for every new sequence length as the model
> generates token by token. Essentially, set scale to original model context length / current sequence
> length."*
>
> *"The main hyperparamter of NTK-Aware is α. Like static linear scaling, it represents a tradeoff
> between short/long sequence performance. So I thought, why not use the same dynamic scaling method
> with NTK-Aware? For Dynamic NTK, the scaling of α is set to
> (α * current sequence length / original model context length) - (α - 1)."*

YaRN §3.3 formalises it as the choice between two ways of applying any scale factor —

> *"1. Throughout the whole inference cycle, the embedding layer is fixed including the scale factor
> s = L'/L where L' is the fixed number of extended context size. 2. In each forward-pass, the
> position embedding updates the scale factor s = max(1, l'/L) where l' is the sequence length of the
> current sequence."*
>
> *"The problem of (1) is that the model may experience a performance discount at a length less than L
> and an abrupt degradation when the sequence length is longer than L'. But by doing Dynamic Scaling as
> (2), it allows the model to gracefully degrade instead of immediately breaking when hitting the
> trained context limit L'."*

— and adds the implementation trap, which is the sort of thing that only shows up when a method
ships:

> *"in some implementations when the RoPE embeddings are cached, some care has to be taken in order to
> modify it for Dynamic Scaling with kv-caching. The correct implementation should cache the
> kv-embeddings before applying RoPE, as the RoPE embedding of every token changes when s changes."*

**What Dynamic NTK fixes, stated exactly:** static scaling taxes *every* sequence, including short
ones, with the compression needed for the longest one. Dynamic scaling makes the tax proportional to
the length actually being processed, and is the identity below `L`. That is why it is the variant
that works on an un-fine-tuned model: at `l' ≤ L` it *is* the original model, bit for bit.

**The shipped formula, from `transformers` v4.32.0** — `LlamaDynamicNTKScalingRotaryEmbedding`:

    if seq_len > self.max_position_embeddings:
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

Two things to notice, because they are the difference between the idea and the code. First, the ramp
is emozilla's `(α·l'/L) − (α−1)`, not the plain `l'/L` of YaRN §3.3's clause 2 — it is built so that
at `l' = L` the bracket is exactly `1` and the base is exactly unchanged. Second, and less discussed:
at `l' = αL` the bracket is `α² − α + 1`, **not** `α`. Asking for `α = 8` at 16k on a 2k model gives
an effective alpha of **57** at the tail, not 8 [derived here]. Whether that is a bug or a deliberate
over-scaling consistent with YaRN's *"the scale value s has to be set higher than the expected scale"*
is not resolved by anything read here — but a card that shows the dynamic curve must show the ramp
that actually shipped, and label the discrepancy.

## Numbers that matter

**The card's central picture, at the app's own width.** `d_k = 8`, base 10000, four pairs, a 16-token
sentence treated as the trained length `L = 16`, extended 4× to `N = 64`. `b' = 10000 · 4^(8/6) =
63496.04`. Every column below was computed by driving `app/model/position.js` from node
**[measured here]**:

| pair `i` | `θ_i` (plain) | λ plain | λ under PI `s=4` | λ under NTK `s=4` | effective scale `s_i` | angle at pos 64: plain / PI / NTK | biggest angle seen in training (`L·θ_i`) | overshoot |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 6.28 | 25.13 | **6.28** | **1.00** | 64.0 / 16.0 / **64.0** | 16.0 | **4.00×** |
| 1 | 0.1 | 62.83 | 251.33 | 99.74 | 1.587 | 6.40 / 1.60 / 4.03 | 1.60 | 2.52× |
| 2 | 0.01 | 628.3 | 2513.3 | 1583.3 | 2.520 | 0.640 / 0.160 / 0.254 | 0.160 | 1.587× |
| 3 | 0.001 | 6283.2 | 25132.7 | **25132.7** | **4.00** | 0.064 / 0.016 / **0.016** | 0.016 | **1.00×** |

Read the two bold rows first: **pair 0 is untouched** (identical to plain RoPE) and **pair 3 lands
exactly on PI's wavelength**. That is Eq. 23's boundary condition, checked. The middle rows are the
geometric blend, `s_i = 4^(i/3) = 1, 1.587, 2.520, 4`. And the last column is the defect: the
overshoot factor is `s^(1 − 2i/(|D|−2))`, so **three of four pairs finish at an angle larger than any
angle they saw in training**, the fastest by the full factor `s`.

**The single number the fast pair buys.** Angle between *adjacent tokens* on the fastest pair
**[measured here]**:

| | fastest pair | slowest pair |
|---|---|---|
| plain RoPE | **1.000000 rad** | 1.000e−3 |
| PI, `s = 4` | **0.250000 rad** | 2.500e−4 |
| NTK-aware, `s = 4` | **1.000000 rad** | 2.500e−4 |

Neighbouring tokens are told apart by 1 radian under RoPE and NTK-aware, and by a quarter of that
under PI. The slow pair is compressed identically by both. That two-row table *is* YaRN's *"the
rotation describing the smallest distance needs to not be too small for the network to be able to
detect it"*, and it is the entire reason the method needs no fine-tuning.

**What that costs the model, measured on the real toy.** Switching a live 2-block head from plain
RoPE to each scheme, same seeded weights, same 16-token sentence, largest change in any attention
weight **[measured here]**:

| scheme | max |Δ attention weight| vs plain RoPE | shift-invariance still exact? |
|---|---|---|---|
| PI, `s = 4` | **0.930** | yes, 5.2e−15 |
| NTK-aware, `s = 4` | **0.345** | yes, 2.6e−14 |

Both remain exactly relative (shifting the whole sentence by 4096 changes nothing beyond float
noise — the base change does not break `R_m^⊺ R_n = R_{n−m}`, because it is still one rotation
schedule). But for the same nominal 4×, PI rearranges the existing attention pattern nearly
completely and NTK-aware disturbs it about a third as much. On a *trained* model that difference is
the difference between "needs fine-tuning" and "does not" — the toy cannot show that, only the
disturbance, and the card must say so in those words.

**At a realistic width.** `|D| = 128`, base 10000, `L = 2048`, `s = 8`, evaluated at `N = 16384`.
Exponent `128/126 = 1.015873`, so `b' = 10000 · 8^1.015873 = **82684.6**` — the famous "raise the base
to about 8×" **[measured here]**:

| pair `i` | λ plain | λ NTK | effective scale `s_i` | angle at 16384: plain / PI / NTK | overshoot | full turns during training |
|---|---|---|---|---|---|---|
| 0 | 6.3 | 6.3 | 1.000 | 16384 / 2048 / 16384 | 8.00× | 325.9 |
| 8 | 19.9 | 25.9 | 1.302 | 5181 / 648 / 3979 | 6.14× | 103.1 |
| 16 | 62.8 | 106.5 | 1.696 | 1638 / 205 / 966 | 4.72× | 32.6 |
| 32 | 628.3 | 1806.7 | 2.875 | 163.8 / 20.5 / 57.0 | 2.78× | 3.26 |
| 41 | 2294.5 | 8879.8 | 3.870 | 44.9 / 5.6 / 11.6 | 2.07× | **0.89** |
| 48 | 6283.2 | 30637 | 4.876 | 16.4 / 2.0 / 3.4 | 1.64× | 0.33 |
| 56 | 19869 | 126161 | 6.350 | 5.2 / 0.6 / 0.8 | 1.26× | 0.10 |
| 63 | 54410 | 435281 | **8.000** | 1.9 / 0.2 / **0.2** | **1.00×** | 0.04 |

**Which dimensions are extrapolating, exactly** [all measured here]:

- **63 of 64 pairs** end past their trained maximum angle. Only the last one does not.
- **23 of 64 pairs** (`i = 41…63`) have `λ > L = 2048` under plain RoPE — they never complete a full
  rotation during pre-training, so their angle range is genuinely unexplored rather than merely
  wrapped. Under the NTK base that count rises to **31 of 64** (`i = 33…63`).
- The pairs where both things are true — never completed a turn in training **and** pushed past their
  trained maximum angle — are **22 of 64**, `i = 41…62`. Those are the dimensions YaRN means by
  *"out-of-bound"*. The fast pairs also "overshoot" by the arithmetic, but they wrapped hundreds of
  times during training, so their angles are all familiar; the overshoot that matters is at the slow
  end, and it is bounded by about 2× there, not 8×.

**The exponent across widths** [measured here] — how much the `−2` actually matters:

| `\|D\|` | `\|D\|/(\|D\|−2)` | `b'/b` at `s = 8` |
|---|---|---|
| 8 (this app) | 1.3333 | **16.00** |
| 64 | 1.0323 | 8.555 |
| 128 | 1.0159 | 8.268 |
| 256 | 1.0079 | 8.132 |

At `s = 8`, `d = 8` the formula gives `8^(4/3) = 16` exactly — the toy's base doubles relative to the
naive `b·s`, which is a clean thing to show a reader who wants to know why the exponent is written
that way.

**YaRN Table 1 is the only side-by-side published measurement of this method, and it is small.**
Llama-2 7B extended 4096 → 8192, sliding-window perplexity (`S = 256`) on ten 128k Proof-pile
documents, lower better:

| method | tokens | window | 2048 | 4096 | 6144 | 8192 | 10240 |
|---|---|---|---|---|---|---|---|
| PI (`s = 2`) | 1B | 8k | **3.92** | **3.51** | 3.51 | **3.34** | 8.07 |
| NTK (`θ = 20k`) | 1B | 8k | 4.20 | 3.75 | 3.74 | 3.59 | **6.24** |
| YaRN (`s = 2`) | 400M | 8k | 3.91 | 3.50 | 3.51 | 3.35 | **6.04** |

That `θ = 20k` is not a hyperparameter someone tuned: it is Eq. 16 at `s = 2`, `|D| = 128`, which
gives `10000 · 2^1.015873 = 20221` [measured here]. So the row is the formula, rounded.

The reading is unambiguous and it is the shape of the whole card: **inside the extended window NTK
loses to PI at every single length** (4.20 vs 3.92, 3.75 vs 3.51, 3.74 vs 3.51, 3.59 vs 3.34 — about
0.25 perplexity, consistently), and **past the extended window it wins by a lot** (6.24 vs 8.07). The
overshoot that YaRN calls a defect is also what keeps it from falling off a cliff at 10240. Both
models here were fine-tuned; this is the fine-tuned comparison.

**Table 3 — the benchmark column people quote as NTK's cost, and the confound.** Hugging Face Open
LLM suite:

| model | ext. | ARC-c | HellaSwag | MMLU | TruthfulQA |
|---|---|---|---|---|---|
| Llama 2 7B (4k) | none | 53.1 | 77.8 | 43.8 | 39.0 |
| Together 7B (32k) | PI | 47.6 | 76.1 | 43.3 | 39.2 |
| **Code Llama 7B (100k)** | **NTK** | **39.9** | **60.8** | **31.1** | 37.8 |
| YaRN 7B `s=16` (64k) | YaRN | 52.3 | 78.8 | 42.5 | 38.2 |

Those Code Llama numbers are catastrophic — and they are **not clean evidence against NTK scaling**.
Code Llama is a code-specialised continued-pretrain of Llama 2 on 500B tokens of code; a 17-point
HellaSwag drop is what code specialisation does to a general benchmark. YaRN's own §4.2 records that
Code Llama used *"a scale factor set to s ≈ 88.6, which corresponds to a context size of 355k"* while
training on 16k data, and §3.1 notes it *"uses 'NTK-aware' scaling by manually scaling the base b to
1M"* — a hand-set base, not Eq. 16. The card can show this table as *what shipped*; it must not show
it as *what base scaling costs*.

| quantity | source | this app |
|---|---|---|
| `\|D\|` the exponent is computed from | 128 (Llama head dim) | `d_k = 8`, so `8/6` not `128/126` |
| base `b` | 10000 | 10000 (`position.js` default) |
| `b'` at `s = 4` | 40889.94 (at d=128) | **63496.04** (at d=8) |
| trained length `L` | 2048 / 4096 | **16 tokens, and untrained** |
| perplexity | the post's whole claim | **cannot be computed** |
| fine-tuning | not done by the post; done by YaRN | **cannot be done** |
| code change required | 1 line (`base = base * a ** (dim/(dim-2))`) | 1 argument: `rope({ base: b })` |

## What the live view must let the reader do

The toy: 32 dims, 4 heads (`d_k = 8`), 2 blocks, causal mask, seeded **untrained** weights, editable
16-token sentence. `position.js` already exposes `rope({ base, stretch, dims })`, and that is the
lucky part of this concept: **PI is the `stretch` knob and NTK-aware is the `base` knob, and both
already exist**. Concept 12's card ended by having the reader turn `stretch`; concept 16 is that knob
named; this card is the other knob, and the comparison between them needs no new model code at all.

The hard boundary, stated once and repeated on the card itself: **this app cannot fine-tune and cannot
measure perplexity.** Every claim this method is famous for is a perplexity claim on a trained 7B
model. What the app can settle completely is the *geometry* — wavelengths, angles, which pairs land
outside their trained arc, and how much the attention pattern moves — and geometry is where the actual
difference between PI and base scaling lives. Design for that and say so.

None of the five below repeats concept 12's panels (shift invariance to floating point, the budget
meter and "rotate V too", the decay bound, the wrap-around/`stretch` demo).

1. **One dial, two knobs — the frequency-by-frequency picture.**
   A single `s` control (1 → 32) driving three schedules at once: plain RoPE, PI (`stretch = 1/s`),
   NTK-aware (`base = 10000 · s^(8/6)`). Draw the four pairs as four horizontal wavelength bars on a
   log axis, plain in grey behind, the active scheme in front. At `s = 4` the reader sees pair 0's
   bar **not move at all** under NTK while it quadruples under PI, and pair 3's bar land in exactly
   the same place under both. Print `b'` live (63496.04 at `s = 4`), and print the effective per-pair
   scale row `1, 1.587, 2.520, 4.000` beside PI's `4, 4, 4, 4`. That contrast — one flat row against
   one ramp — is the mechanism and should be the first thing on the card. Put YaRN Eq. 23 under it,
   because the ramp's right-hand end being pinned to PI *is* Eq. 23.

2. **The neighbour test — why no fine-tuning is needed.**
   Two numbers, live: the rotation angle between **adjacent** tokens on the fastest pair, and on the
   slowest. Plain 1.000 rad / 1.0e−3; PI at `s = 4` **0.250 rad** / 2.5e−4; NTK **1.000 rad** /
   2.5e−4. Then show what the model does with it: max change in any attention weight against plain
   RoPE, on the reader's own sentence — **0.930 for PI, 0.345 for NTK** at the same nominal 4×. The
   sentence to put beneath it is YaRN's: *"the rotation describing the smallest distance needs to not
   be too small for the network to be able to detect it"*. Then the honesty line, in the app's own
   voice: this shows the *disturbance* is smaller, not that perplexity is better — these weights were
   never trained, and no number this app produces can close that gap.

3. **Which dimensions are out of bounds — the defect, drawn.**
   Four rows, one per pair, each a bar running from angle 0 to the largest angle that pair reaches at
   the reader's chosen position `N`, with the **trained arc** (`0 … L·θ_i`, `L` = sentence length)
   shaded green behind it and the excess in the warn colour. At `s = 4`, `N = 64`: pair 0 overshoots
   4.00×, pair 1 2.52×, pair 2 1.587×, pair 3 exactly 1.00× — *"some dimensions are slightly
   extrapolated to 'out-of-bound' values"*, as a picture with a number on each row. `curve.js`
   already has `band` and `deadFrom`, which is exactly this. Add the qualifier the arithmetic
   demands and that YaRN skips: overshoot on a pair that already wrapped 2.5 times in training is not
   the same kind of unseen as overshoot on a pair that never completed a turn, so also print **full
   turns during training** per pair (2.546, 0.255, 0.025, 0.003 at `L = 16`). Then a width selector —
   `d_k = 8` (this app) vs `d = 128` (a real head) — which switches the same panel to the 64-pair
   version and prints the counts: **23 of 64 pairs never complete a rotation in training; 22 of 64
   both never wrap and end past their trained angle**. This is the panel that earns the card its
   "community result, examined" framing.

4. **The dynamic variant, as a curve over generation.**
   `s` recomputed per forward pass from the current length. Plot the effective base against sequence
   length `l'` from 1 to `4L`, with the flat segment below `L` (base exactly 10000 — the model is
   untouched) and the ramp above it, for both rules: YaRN §3.3's `s = max(1, l'/L)` and the ramp that
   actually shipped, `(α·l'/L) − (α−1)`. Print the divergence: at `l' = αL` with `α = 4`, the shipped
   ramp's effective alpha is **13**, not 4. Beneath the curve, the two consequences of the flat
   segment as live readouts on the reader's 16-token sentence: with dynamic scaling the attention
   matrix is **bit-identical** to plain RoPE (max |Δ| = 0.00e+0, because `s = 1` below `L`), whereas
   static NTK at `s = 4` has already moved it by 0.345. That is precisely what "graceful degradation"
   buys and precisely what it costs nothing. Then the kv-cache trap, quoted from §3.3, as a note —
   caching post-RoPE keys is *wrong* here because every token's rotation changes when `s` changes. It
   is the only implementation detail on this card and it is a real bug people shipped.

5. **The provenance panel — this one is not decoration.**
   The card's last panel should be the evidence itself, laid out as a table the reader can audit:
   *claim* / *who claimed it* / *what was measured* / *by whom*. Rows: the base formula (bloc97, post,
   derivation not shown; written up by the same author in YaRN Appendix A.1 two months later —
   **reproducible arithmetic, and this app recomputes it**); "minimal perplexity degradation" (bloc97;
   three graphs, 40 documents from a gov_report subset, LLaMA 7B, no table, no variance, images this
   note could not fetch); "no fine-tuning needed" (bloc97 — and his own words, *"I did not test
   fine-tuning performance as I do not have the resources or the time to fine tune an LLM, I just
   derived this formula during lunch"*); "worse than PI after fine-tuning" (YaRN Table 1, ~0.25 ppl
   across four lengths, one model pair, self-reported by the same authors); "some dimensions go
   out-of-bounds" (YaRN §3.1 — **and this app measures exactly which ones**). Two of those five rows
   the app can verify itself; three it can only attribute. Marking which is which, on the card, is
   what makes this a timeline entry rather than a rumour with a formula attached.

## What the source does *not* establish

- **The post reports no numbers at all.** Its entire empirical claim is three images:
  *"a graph showing the average perplexity of LLaMA 7b on a set of 40 very long prompts (12k+ context
  size)"*, a second with *"more scale and alpha factors"*, and a zoom of the second. There is no
  table, no perplexity value in the text, no baseline number, no variance, no seed, no second model,
  no second dataset — the data is *"a subset of gov_report"* from tau/scrolls, 40 prompts. Everything
  the field repeats as "minimal perplexity degradation" traces to those three pictures. This note
  could not fetch the images, so it cannot even report what they show.
- **The author did not test the thing that turned out to matter, and says so.**
  *"I did not test fine-tuning performance as I do not have the resources or the time to fine tune an
  LLM, I just derived this formula during lunch and experimented with it. However, I think that this
  method will do even better with fine tuning."* That guess is the one claim in the post that later
  turned out **backwards** — YaRN §3.1: *"fine-tuning with 'NTK-aware' interpolation yields inferior
  results to PI"*, and Table 1 shows it losing at every in-window length. The card should put those
  two sentences next to each other; it is the cleanest example on this timeline of a plausible
  extrapolation from a real result being wrong.
- **The derivation was never published as a derivation.** The post: *"I might give a more detailed
  explanation on how I derived the formula used to calculate the base if enough people are
  interested."* The only account is YaRN Appendix A.1, two months later, three sentences long,
  described by the paper itself as *"a short note on its mathematical deduction"*. It is correct and
  it is checkable — but for the two months in which the method was adopted into `transformers`,
  Code Llama and Qwen, **no derivation existed in public at all**.
- **"NTK-aware" does not mean an NTK result was used.** The reference is to Tancik et al.'s Fourier
  features paper, by analogy (*"RoPE closely resembles Fourier Features in many aspects"*). No kernel
  is computed, no theorem is invoked, and Eq. 23 is a wavelength boundary condition. The name has done
  a great deal of unearned persuasive work.
- **The exponent is a choice, not an optimum.** YaRN §3.1: *"One can obtain such a transformation in
  many ways, but the simplest would be to perform a base change on the value of θ."* The specification
  — pin the slowest pair to PI, leave the fastest alone — is a reasonable pair of endpoints, and the
  geometric interpolation between them is whatever a base change happens to give. Nothing anywhere
  compares it against another curve with the same endpoints. "Simplest", in the authors' own word.
- **The known defect, in YaRN's exact words**, which the card should quote rather than paraphrase:
  > *"However, one major disadvantage of this method is that given it is not just an interpolation
  > scheme, some dimensions are slightly extrapolated to 'out-of-bound' values, thus fine-tuning with
  > 'NTK-aware' interpolation yields inferior results to PI. Furthermore, due to the 'out-of-bound'
  > values, the theoretical scale factor s does not accurately describe the true context extension
  > scale. In practice, the scale value s has to be set higher than the expected scale for a given
  > context length extension."*

  Note what that last sentence actually says: **`s` under-delivers**. The arithmetic explains why —
  every pair except the last receives `s_i = s^(2i/(|D|−2)) < s`, so the *achieved* compression is
  less than requested and you must ask for more to get the length you want. Anyone stating the defect
  as "the effective scale is larger than requested" has the sign of the practical consequence
  backwards, even though both descriptions point at the same asymmetry: what is larger than requested
  is the *rotation angle* at the extended positions (up to `s`× past the trained maximum on the fast
  pairs), and what is smaller than requested is the *effective context extension*. The card must get
  this right, because it is the single most-garbled sentence about this method.
- **The "out-of-bound" claim is asserted, not measured, in YaRN.** §3.1 gives no count, no plot, no
  threshold, no per-dimension breakdown, and no ablation isolating the overshoot from any other
  difference. The counts in this note (63 of 64 pairs past their trained angle; 22 of 64 both unwrapped
  and overshooting) are **this app's arithmetic from the definitions**, not the paper's, and the card
  must label them so.
- **"Fine-tuning with NTK yields inferior results to PI" rests on Table 1 and Table 1 only.** One
  model family (Llama-2 7B), one extension (4096 → 8192, `s = 2`), one dataset (Proof-pile, ten
  documents), one run per arm, no variance, no seeds — and the two arms were fine-tuned by different
  people at different times (the PI arm is LLongMA-2 7b, a third-party checkpoint on RedPajama). The
  finding is consistent across four lengths, which is real evidence; it is not a controlled experiment,
  and it is self-reported by the method's own authors in a paper proposing its replacement.
- **The Code Llama benchmark collapse is not attributable to base scaling.** See Table 3 above: the
  arm is confounded by 500B tokens of code specialisation, by a hand-set base of 1M rather than Eq. 16,
  and by `s ≈ 88.6`. Three differences, one number.
- **Nothing establishes it works on non-Llama models.** Every measurement, in both the post and YaRN,
  is on LLaMA/Llama-2 (plus Code Llama and Qwen as adoption anecdotes). The base is a property of the
  pre-trained model's frequency schedule; there is no evidence here about a model trained at a
  different base to begin with.
- **The extrapolation claim in the post is unmeasured hyperbole.** *"all the positions are perfectly
  distinguishable from eachother, even when taken to the extreme (eg. streched 1million times, which
  is effectively a context size of 2 Billion)"* — distinguishable is a statement about the *formula*
  producing distinct vectors, exactly the "defined everywhere" property concept 12's card already
  dismantled. Nothing at 2 billion, or at 1 million, was run.
- **The app establishes nothing empirical, and less than usual here.** Untrained seeded weights mean
  there is no "trained range" in any real sense — the app's `L = 16` is a stand-in the reader chooses,
  not a fact about the model. Every geometric number above is exact and checkable; every number about
  *quality* is out of reach. This is the card where the boundary between the two must be printed on
  screen, not just written in this note.

## Leaves behind

**Backward, it answers concept 16 with a smaller edit than concept 16 made.** Position interpolation
divides every position by `s`, which is correct, provable (Chen et al.'s interpolation bound), and
indiscriminate: it takes the same factor out of the pair that distinguishes adjacent tokens as out of
the pair that measures thousands. This post's observation is that the ladder already encodes which
pairs can afford it — a pair with `λ = 6.3` at `d_k = 8` has wrapped hundreds of times and cannot
afford to lose resolution, a pair with `λ = 54410` at `d = 128` has never completed a turn and can
afford all of it — and one constant, the base, moves the whole ladder in exactly that graded way. It
is the first method on this timeline that treats RoPE's frequencies as *different from each other*
rather than as one uniform schedule, and everything after it inherits that idea. It also closes, with
one measured line, the question concept 12 left hanging: the `stretch` knob and the `base` knob were
both sitting in `position.js` from the start, and they are the two competing answers to the same
question.

**Forward, it is wrong in a specific and productive way.** The graded schedule is right; the
particular grading — geometric, pinned at both ends, derived from *"the simplest"* transformation —
is not principled, and its consequence is that almost every dimension ends up a little outside where
it was trained. YaRN's next two sections are both direct repairs. §3.2's "NTK-by-parts" replaces the
smooth ramp with a decision per dimension, using the ratio `r(d) = L/λ_d`: *"if the wavelength λ is
much smaller than the context size L, we do not interpolate; if the wavelength λ is equal to or bigger
than the context size L, we want to only interpolate and avoid any extrapolation (unlike the previous
'NTK-aware' method)"*, with a ramp only in between (`α = 1`, `β = 32` for Llama). That is the
overshoot column of this card's table 3 being driven to exactly zero by construction. And §3.3's
Dynamic Scaling — which arrived one day after this post, from the other author — fixes the orthogonal
problem, that a static `s` taxes short sequences for the sake of long ones. YaRN is the two repairs
plus an attention temperature, and its abstract's claim to *"10x less tokens and 2.5x less training
steps"* is measured against PI, not against this.

**And it leaves the timeline a different kind of question.** Concepts 1–16 are all papers; this one is
a forum post that reached `transformers`, Code Llama and Qwen before anybody wrote down why it worked,
and whose author's own follow-up paper is the source of both its best explanation and its sharpest
criticism. The card's job is not to be skeptical *about* it — the arithmetic is sound and this app
recomputes it from scratch — but to be exact about which parts were ever checked. Two rows of the
provenance table the app can verify itself. Three it can only attribute. That ratio is the entry.
