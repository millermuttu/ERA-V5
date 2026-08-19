# Concept 25 — DeepSeek sparse attention
**Card id:** `dsa` · **Date:** 2025-09-29 (release) · **Pressure:** how many comparisons

## What was read

- **The primary source is a release, not a paper.** *DeepSeek-V3.2-Exp: Boosting Long-Context
  Efficiency with DeepSeek Sparse Attention*, DeepSeek-AI, `research@deepseek.com` — the tech report
  shipped inside the model repository as
  [`DeepSeek_V3_2.pdf`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf).
  Six pages plus one appendix, no arXiv identifier, no conference, no submission history. **Read in
  full**, locally text-extracted from the PDF: §1 Architecture (prototype of DSA, Eq. 1–2, Figure 1,
  instantiation under MLA), §2.1 Continued Pre-Training (Eq. 3–4, both stages), §2.2 Post-Training,
  §3 Evaluations (Table 1, Figures 2–3), the "Future Validation in Real World" paragraph, the
  reference list, and Appendix A (the MHA and MQA modes of MLA).
- **The date is verified**, and against a dated primary announcement rather than a guess:
  DeepSeek's own release note at `api-docs.deepseek.com/news/news250929` is stamped
  **September 29, 2025** and says *"Introducing DeepSeek-V3.2-Exp — our latest experimental model"*,
  *"it debuts DeepSeek Sparse Attention (DSA)"*, and **"API prices cut by 50%+!"** effective with the
  release. **The app's record of `2025-09-29` is correct.**
- **A record correction that is not a date correction.** When this record was written there was no
  paper. There is one now: **arXiv:2512.02556, *DeepSeek-V3.2: Pushing the Frontier of Open Large
  Language Models*, v1 Tue, 2 Dec 2025, 09:25:14 UTC, 593 KB, `cs.CL`**, whose abstract lists DSA
  as its first contribution. It is **two months later and about a different model** — the full V3.2,
  including a `Speciale` variant and an RL story — and it does not mention V3.2-Exp or the lightning
  indexer in its abstract. So the **timeline keeps the 2025-09-29 release as the first public
  appearance of this mechanism**, which is the deck's rule, and the card and README name the later
  paper as further reading. The record stays `kind: "release"`, so the card keeps its unverified
  badge: the deck marks an entry verified only when a paper backs it.
- Also read: the model repository's own changelog, whose only entry is dated **2025.11.17** and
  reports that *"previous versions of the inference demo code contained an implementation discrepancy
  in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded
  model performance"*. Worth carrying onto the card — the published code for this mechanism had a
  bug in the very component the mechanism is named for, found seven weeks after release.
- **What does not exist to be read**: no ablation of `k`, no comparison against NSA or any other
  sparse method, no table behind either figure, no statement of the indexer's head count or width.

---

## The one-sentence version

The previous card made selection cheap by making it coarse — score blocks, not tokens, because
scoring every token is what costs. This one keeps per-token selection and makes the *scorer* cheap
instead: a small, separate, ReLU-based index network with its own heads, trained by KL divergence to
imitate what the real attention pays attention to, and then used to pick the top 2,048 tokens the
real attention is allowed to see.

## The mechanism, exactly

Two components, in the report's own words: *"a lightning indexer and a fine-grained token selection
mechanism."*

### The lightning indexer, Eq. 1

> **I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )**

*"where H^I denotes the number of indexer heads; q^I_{t,j} ∈ R^{d^I} and w^I_{t,j} ∈ R are derived
from the query token h_t; and k^I_s ∈ R^{d^I} is derived from the preceding token h_s."*

Four things are worth pulling out of that line:

- **It reads the residual stream, not the attention's own queries and keys.** `h_t` and `h_s` are
  block inputs. The indexer is a separate small network living beside attention, with its own
  projections and its own heads.
- **ReLU, and the report says why**: *"We choose ReLU as the activation function for throughput
  consideration."* Not for accuracy — for speed. What it does to the score is subtler than it first
  looks, and getting it wrong is easy: the rectifier is inside the sum, but **`w^I_{t,j} ∈ R` is
  unconstrained**, so `I_{t,s}` itself can be negative. What the rectifier buys is that **a head is
  silent about any key its own query does not align with** — the term is exactly zero, not a small
  negative — so each head speaks only about the half-space it points into, and a key that no head
  points at scores exactly zero. Measured below: with two heads on this model, **a quarter of all
  query/key pairs score exactly zero**, which means the ranking Eq. 2 depends on contains ties broken
  arbitrarily.
- **No softmax.** `I_{t,s}` is a score used for ranking, and its scale never matters. The softmax
  appears only in the training loss.
- **It is cheap, and the report is explicit about why**: *"Given that the lightning indexer has a
  small number of heads and can be implemented in FP8, its computational efficiency is remarkable."*
  **How small is not stated.** No `H^I`, no `d^I` anywhere in the report. The card must not print
  either.

### Fine-grained token selection, Eq. 2

> **u_t = Attn( h_t, { c_s | I_{t,s} ∈ Top-k(I_{t,:}) } )**

Top-k over individual tokens — the block structure the previous card argued was necessary is gone.
**k = 2048**, stated in §2.1 for the sparse training stage, against a context extended to **128K**:
each query is allowed to see **1.5625%** of what is behind it.

### Instantiated under MLA, in MQA mode — and the reason is the previous card

> "At the kernel level, each key-value entry must be shared across multiple queries for computational
> efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA (Shazeer, 2019) mode
> of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query
> heads of the query token."

`Yuan et al., 2025` is **concept 24**, and `Shazeer, 2019` is **concept 7**. This is not a card
reaching backwards for a connection; the report cites both, and it cites NSA specifically for the
constraint that forces the design. The consequence is structural and the card must show it: **the
selection is per token, not per head.** One index score row picks one set of keys, and every
attention head of that token reads the same set. Concept 21's latent cache is what makes that
affordable, and Appendix A is there to explain which of its two modes is being used.

## Training — and this is where the mechanism actually lives

Starting from a V3.1-Terminus base already extended to 128K, two stages:

**Dense warm-up.** *"we keep dense attention and freeze all model parameters except for the lightning
indexer."* The target is built from the real attention: *"for the t-th query token, we first
aggregate the main attention scores by summing across all attention heads. This sum is then
L1-normalized along the sequence dimension to produce a target distribution p_{t,:}"*, and the
indexer is fitted to it (Eq. 3):

> **L_I = Σ_t D_KL( p_{t,:} ‖ Softmax( I_{t,:} ) )**

Learning rate **1e-3**, **1000 steps**, **16 sequences of 128K per step** — *"resulting in a total of
2.1B tokens."*

**Sparse stage.** Selection switched on, everything unfrozen, and the same KL restricted to the
selected set (Eq. 4):

> **L_I = Σ_t D_KL( p_{t,S_t} ‖ Softmax( I_{t,S_t} ) )**, with **S_t = { s | I_{t,s} ∈ Top-k(I_{t,:}) }**

Learning rate **7.3e-6**, **k = 2048**, **15000 steps × 480 sequences of 128K** — *"a total of 943.7B
tokens."* And a detail that matters more than its one sentence suggests: *"we detach the indexer
input from the computational graph for separate optimization. The training signal of the indexer is
from only L_I, while the optimization of the main model is according to only the language modeling
loss."*

**Read that against the previous card.** NSA's whole claim was end-to-end differentiability — the
selection is trained by the language-modelling loss like everything else. DSA does the opposite: two
losses, a detached graph, and the indexer never sees the language-modelling objective at all. It is
trained to *imitate the attention it is replacing*. That is a distillation, not an end-to-end
learned sparsity, and the two papers are four months apart from overlapping groups.

## The evidence, quoted with its conditions

Table 1, all fourteen rows — V3.1-Terminus against V3.2-Exp, the same post-training pipeline, data
and algorithm on both so that the comparison isolates DSA:

| | benchmark (metric) | V3.1-Terminus | V3.2-Exp |
|---|---|---|---|
| General | MMLU-Pro (EM) | 85.0 | 85.0 |
| General | GPQA-Diamond (Pass@1) | 80.7 | **79.9** |
| General | Humanity's Last Exam (Pass@1) | 21.7 | **19.8** |
| Search agent | BrowseComp (Acc.) | 38.5 | 40.1 |
| Search agent | BrowseComp_zh (Acc.) | 45.0 | 47.9 |
| Search agent | SimpleQA (Acc.) | 96.8 | 97.1 |
| Code | LiveCodeBench 2408-2505 (Pass@1) | 74.9 | **74.1** |
| Code | Codeforces-Div1 (Rating) | 2046 | 2121 |
| Code | Aider-Polyglot (Acc.) | 76.1 | **74.5** |
| Code agent | SWE Verified (Agent mode) | 68.4 | **67.8** |
| Code agent | SWE-bench Multilingual (Agent) | 57.8 | 57.9 |
| Code agent | Terminal-bench (Terminus 1) | 36.7 | 37.7 |
| Math | AIME 2025 (Pass@1) | 88.4 | 89.3 |
| Math | HMMT 2025 (Pass@1) | 86.1 | **83.6** |

Seven up, six down, one level. The report's own reading, quoted rather than paraphrased:

> "Overall, DeepSeek-V3.2-Exp does not show substantial performance degradation compared with
> DeepSeek-V3.1-Terminus. The performance of DeepSeek-V3.2-Exp on GPQA, HLE, and HMMT 2025 is lower
> than that of DeepSeek-V3.1-Terminus because DeepSeek-V3.2-Exp generates fewer reasoning tokens.
> However, this performance gap closes when using intermediate checkpoints that produce a comparable
> number of tokens."

**That explanation is offered, not demonstrated** — the intermediate checkpoints are not shown, and
the claim is about output length rather than about attention. It belongs on the card as the
authors' account, labelled as such.

**Cost.** *"DSA reduces the core attention complexity of the main model from O(L²) to O(Lk), where
k (≪ L) is the number of selected tokens. Although the lightning indexer still has a complexity of
O(L²), it requires much less computation compared with MLA."* The quadratic term does not disappear;
it gets a smaller constant. The cost evidence itself is **Figure 3**, cost per million tokens against
token position for prefill and decode, *"estimated from benchmarking the actual service deployed on
H800 GPUs, at a rental price of 2 USD per GPU hour"* — **a figure with no table, so no number from
it may be printed.** The commercial number that is quotable comes from the release note: **API prices
cut by more than 50%.**

One more line from the same paragraph, easy to miss and worth the card's space: *"for short-sequence
prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher
efficiency under short-context conditions."* **Below some unstated length they do not run the
mechanism at all** — they simulate it with a mask over dense attention, because dense is faster
there. The previous card's break-even, conceded in a subordinate clause.

### What the source does not establish

- **No limitations section.** The nearest thing, quoted in full: *"Although our internal evaluations
  show promising results of DeepSeek-V3.2-Exp, we are actively pursuing further large-scale testing
  in real-world scenarios to uncover potential limitations of the sparse attention architecture."*
  The model's name carries the same message: `Exp`.
- **No ablation of any kind.** Not on `k`, not on the indexer's size, not on ReLU against anything
  else, not on the detached graph, not against NSA or any other sparse method. The paper's one
  comparison is against the model it was continued from.
- **No indexer dimensions**, so nobody can compute what "much less computation" amounts to.
- **The comparison is one model pair**, and V3.2-Exp is a continuation of V3.1-Terminus rather than a
  model trained sparse from scratch — the opposite of the previous card's "natively trainable".

---

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

DSA written from Eq. 1–4 and run on the app's own model, both blocks, all four heads, the default
sentence (16 tokens). The indexer reads the block input, exactly as the report specifies, through the
existing `latent` hook — **no seam change was needed**, because DSA's core attention is ordinary
attention over a top-k subset and that is what `softmaxMixer({ readable })` already expresses. The
novelty is entirely in how the subset is chosen. The indexer's projections are **declared seeded
stand-ins** with `H^I = 2` and `d^I = 8`, both of which the report leaves unstated and the card
labels as this app's choice rather than the paper's.

### 1. The corner is exact

`k = T` — every token selected — reproduces plain causal attention to **0.0%** deviation across the
whole model, and returns the baseline's own next word at the baseline's own probability. It has to,
and it does.

### 2. An untrained indexer is worth exactly nothing, and that is the finding

Share of the main attention's own weight that lands inside the top-k the indexer chose, averaged over
every query and every head in both blocks:

| k | stand-in indexer | **oracle — the KL target** | random |
|---|---|---|---|
| 1 | 0.209 | **0.454** | 0.206 |
| 2 | 0.342 | **0.666** | 0.350 |
| 4 | 0.569 | **0.876** | 0.586 |
| 8 | 0.874 | **0.984** | 0.792 |
| 12 | 0.959 | **0.999** | 0.942 |
| 16 | 1.000 | 1.000 | 1.000 |

**The stand-in indexer is indistinguishable from random** — at `k = 2` and `k = 4` it is fractionally
*worse*. The oracle column is `p_{t,:}` itself, the head-summed L1-normalised attention that Eq. 3
fits the indexer to, and it is far above both: at `k = 4` of up to 16 keys it captures **87.6%** of
the weight that reading everything would have found.

The same thing in the report's own loss: **KL(p ‖ softmax(I)) = 1.6556 nats** on the stand-in, against
**0.5402 nats** for a *uniform* indexer. An untrained indexer is three times worse than no indexer at
all, and Eq. 3's whole job is to drive that number down.

**This is the honest centre of the card.** Every other mechanism in this deck does something visible
on untrained weights — a rotation is a rotation, a mask is a mask, a state decays whether or not it
was trained. DSA does nothing at all until its indexer is trained, and this page can prove it by
bracketing the mechanism between its floor (random, which is where an untrained indexer sits) and its
ceiling (the oracle it is fitted to). **The gap between those two columns is what the 943.7B tokens
buy**, and it is the only card in the deck whose entire value is measurable only as an absence.

### 3. What the rectifier actually does to the scores

Over both blocks and every query row on the default sentence — 272 query/key pairs:

- **153 of them, 56.3%, are negative.** The rectifier is inside the sum and `w^I` is not constrained,
  so the score is not a non-negative quantity. Anyone who reads Eq. 1 as "ReLU, therefore
  non-negative" has misread it, and the self-check for this card exists because that is the mistake
  the note made first.
- **70 of them, 25.7%, are exactly zero** — every indexer head silent, because no head's query points
  into the same half-space as that key. The ranking then contains a tie a quarter of the way across,
  and which of those keys the top-k keeps is decided by nothing at all.

That second figure is a real mechanism-level observation rather than a curiosity: at small `k` on a
poorly-fitted indexer, part of the selection is not a decision. It is also `H^I`-dependent — with
more indexer heads, exact zeros get rarer — and `H^I` is the number the report does not give.

### 4. What the sparsity costs the output, at this scale

Relative deviation from full attention over the whole model, with the stand-in indexer choosing:

| k | deviation | top word |
|---|---|---|
| 1 | 119.3% | keeps 0.3112 |
| 2 | 102.5% | softmax 0.2798 |
| 4 | 83.9% | memory 0.2561 |
| 8 | 45.1% | sea 0.7730 |
| 12 | 24.9% | sea 0.3720 |
| 16 | 0.0% | her 0.2046 |

(baseline: her 0.2046). These are large because 16 keys is not 128K keys and because the selector is
random in effect — the card shows the same table under the oracle selector so the reader can see how
much of that damage is the sparsity and how much is the untrained scorer.

### 5. The report's own arithmetic reproduces

- warm-up: `1000 × 16 × 131072` = **2.097B**, and the report says **2.1B**.
- sparse: `15000 × 480 × 131072` = **943.7B**, and the report says **943.7B** — exactly.
- `k = 2048` of a 128K context = **1.5625%** of the keys.

Small checks, but they are the only numbers in this source a reader can verify without a cluster, and
two of them are the training bill: **just under a trillion tokens of continued pre-training to teach
one small network to imitate the attention it is standing in front of.**

---

## What the live view must let the reader do

1. **See the indexer as a separate network.** Eq. 1 on screen, the per-key index scores for a chosen
   query, and the fact that they are non-negative and never normalised.
2. **Switch the selector** between the stand-in indexer, the oracle target, and random — and watch
   the mass-captured number and the output deviation move. The stand-in and random landing on top of
   each other is the point, not a bug to be hidden.
3. **Move k**, from 1 to the whole sentence, and reach the corner where the deviation is exactly zero.
4. **See the KL target being built** — the real attention's weights summed across heads and
   L1-normalised — beside the indexer's own distribution, with the KL number the report's Eq. 3
   minimises.
5. **Read the training bill and the selection rate**: 2.1B + 943.7B tokens, k = 2048 of 128K = 1.56%,
   both recomputed here.

Not to be drawn: a trained indexer, any head count or width for it (unstated), any cost per million
tokens (Figure 3 has no table), the MoE model, and the RL training curves.

## What it leaves behind

Forward, **nothing in this deck**. The record's `leaves.to` is null: what this concept leaves — a
scorer that only works once trained, and an indexer that is still quadratic with no stated size — is
picked up by no later entry in the timeline. The final card is about positions, not about how many
comparisons, and pretending otherwise would be a link invented for the sake of symmetry.

Backwards, it answers **concept 24 (`nsa`)** directly and by name. NSA fixed a hierarchy — block
length, stride, selection block size, how many, window, three blocks always on — and made the
selection coarse so the scoring could be free. DSA drops the hierarchy: per-token top-k, one number
`k`, no blocks, no window branch, no gate. The cost of that is a scorer that has to be *trained to be
worth anything*, which NSA's derived score never did, and which this app can only show as the space
between random and an oracle.
