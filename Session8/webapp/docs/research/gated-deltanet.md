# Concept 23 — Gated DeltaNet
**Card id:** `gated-deltanet` · **Date:** 2024-12-09 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), Songlin Yang, Jan Kautz, Ali Hatamizadeh —
  *Gated Delta Networks: Improving Mamba2 with Delta Rule*. Categories `cs.CL`, `cs.LG`. The
  abstract page carries the comment line **"ICLR 2025 camera ready"**.
- **Version history, read off the abstract page — three versions:**
  - **v1 — Mon, 9 Dec 2024, 13:09:04 UTC**, 261 KB.
  - v2 — Wed, 5 Mar 2025, 08:47:27 UTC, 266 KB.
  - v3 — Thu, 6 Mar 2025, 06:57:34 UTC, 130 KB (the camera-ready).

  **The app's record of `2024-12-09` is verified correct.** The timeline dates v1, per the repo
  convention; the text below was read from the **v3 HTML render** (`arxiv.org/html/2412.06464v3`),
  which is the camera-ready, not the version the timeline dates. Same situation as `delta-rule` and
  `mamba`, and stated rather than hidden. Nothing quoted here is a v1-only claim — but nothing here
  has been diffed against v1 either, and the card must not attribute a number to "the December 2024
  paper" as though the two were checked to be the same.
- Sections used: §1 (Introduction), §2.1–2.2 (Mamba2 and DeltaNet as online learners), §3.1
  (Formulation: Gated Delta Rule, Eq. 10 and Table 1), §3.2 (the S-NIAH case study and Table 2),
  §3.3 (chunkwise parallel training for the gated rule), §3.4 (Neural Architecture), §4 (experiments,
  Tables 3–5 and Figures 2–3), Appendix A (the extended WY representation) and Appendix B.1
  (training configuration).
- **What could not be reached.** Three things this note wanted and does not have:
  1. the **α parameterization**. §3.4 says only *"α, β use linear projection only"*, and a footnote
     says *"We use Mamba2's parameterization for α"* without reproducing it. So the exponent, the
     `A` matrix and the softplus are inherited from a different paper and are not in this one.
  2. the **ablation table in Appendix B.2**. The section is referenced; the table did not come back
     in any fetch of the render. The case study in §3.2 is used instead, and is labelled as such.
  3. the **hyperparameter table** beyond what B.1 states in prose (below). No head dimension, no
     head count, no chunk size, no layer count came back. **The card must not print a state size or
     a chunk size for this paper.** Concept 22's chunk sizes are concept 22's.
- Every quotation below is the authors' own wording from that render.

---

## The one-sentence version

DeltaNet can correct what it already wrote but has no way to forget; Mamba2 can forget but writes
without looking at what is already there. Gated DeltaNet does both in one line — decay the state,
then correct it — and the whole paper is the argument that these two are not alternatives.

## The mechanism, exactly

Table 1 is the paper's own framing and it is the spine of this card. Every rule in the family is the
**closed-form minimiser of a per-token online objective**, and the two mechanisms are two separate
terms in that objective:

| Method | Online learning objective | Online update |
|---|---|---|
| LA | ‖**S**_t − **S**_{t−1}‖²_F − 2⟨**S**_t **k**_t, **v**_t⟩ | **S**_t = **S**_{t−1} + **v**_t **k**_tᵀ |
| Mamba2 | ‖**S**_t − α_t**S**_{t−1}‖²_F − 2⟨**S**_t **k**_t, **v**_t⟩ | **S**_t = α_t**S**_{t−1} + **v**_t **k**_tᵀ |
| Longhorn | ‖**S**_t − **S**_{t−1}‖²_F − β_t‖**S**_t **k**_t − **v**_t‖² | **S**_t = **S**_{t−1}(**I** − ε**k**_t**k**_tᵀ) + ε_t**v**_t**k**_tᵀ |
| DeltaNet | ‖**S**_t − **S**_{t−1}‖²_F − 2⟨**S**_t **k**_t, β_t(**v**_t − **S**_{t−1}**k**_t)⟩ | **S**_t = **S**_{t−1}(**I** − β_t**k**_t**k**_tᵀ) + β_t**v**_t**k**_tᵀ |
| **Gated DeltaNet** | ‖**S**_t − α_t**S**_{t−1}‖²_F − 2⟨**S**_t **k**_t, β_t(**v**_t − α_t**S**_{t−1}**k**_t)⟩ | **S**_t = **S**_{t−1}(α_t(**I** − β_t**k**_t**k**_tᵀ)) + β_t**v**_t**k**_tᵀ |

Read the objective column downward and the whole timeline segment falls out: **α_t moves the anchor**
(what the state is pulled back towards — a decayed copy of itself instead of itself), and **β_t
changes the target** (the residual **v**_t − α_t**S**_{t−1}**k**_t instead of **v**_t itself). Two
independent edits to one objective. That is the card's central picture and it needs no new
apparatus to draw.

The update, Eq. 10, expanded — this is the form the app already implements:

> **S**_t = **S**_{t−1}(α_t(**I** − β_t**k**_t**k**_tᵀ)) + β_t**v**_t**k**_tᵀ
> = α_t**S**_{t−1} + β_t(**v**_t − α_t**S**_{t−1}**k**_t)**k**_tᵀ

The paper's definitions: **α_t ∈ (0,1)** *"controls state decay"*, **β_t ∈ (0,1)** is the
*"writing strength"*. The transition is a **decayed generalised Householder**: the rank-1
`(I − βkkᵀ)` factor of DeltaNet with a scalar in front of it.

### The degenerate settings — not stated by the paper, derived here

The paper never writes down what happens at the corners. They matter for the card, so they are
derived here and labelled as derived, not quoted:

- **α_t = 1 for all t** → DeltaNet exactly. Verified numerically below, to zero difference.
- **β_t = 0 for all t** → **S**_t = α_t**S**_{t−1}: nothing is ever written, the state decays from
  zero and stays zero. Verified below.
- **β_t = 1 and ‖k_t‖ = 1** → **S**_t **k**_t = α**S**k − α**S**k + **v**_t = **v**_t. The current
  token's pair is stored *exactly*, whatever α was and whatever else is in the state. Note the
  condition: it needs the key to be a unit vector, which is exactly what §3.4's **L2 normalisation
  of q and k** provides. This is the sharpest statement the card can make about what the delta term
  buys, and it is a consequence of the paper's own architecture choices rather than an extra
  assumption.
- **Mamba2 is not reachable from Eq. 10.** No α, β makes `S_{t-1}(α(I − βkkᵀ)) + βvkᵀ` equal
  `αS_{t-1} + vkᵀ`, because the delta form's write is always tied to the read of the same key. The
  comparison panel therefore needs the add rule as a separate branch, which is a seam change (below)
  and not a configuration.

### The architecture, §3.4

> "queries, keys and values {q, k, v} are generated through linear projection, short convolution and
> SiLU, with L2 normalization applied to q, k. α, β use linear projection only."

Output gating before the final projection. The hybrids: **Gated DeltaNet-H1** interleaves Gated
DeltaNet layers with sliding window attention; **Gated DeltaNet-H2** interleaves Mamba2, Gated
DeltaNet and sliding window attention. The paper does not give the ratio in any text this note could
reach, so the card must not draw a layer stack for either hybrid.

### Chunkwise training, §3.3 — what carries over from concept 22 and what does not

The gated rule keeps concept 22's chunkwise WY form, with the decay folded in as a **cumulative
product inside the chunk**: γ^r_[t] = ∏_{i≤r} α^i_[t], used to rescale the keys and values before
the same UT-transform inverse concept 22 already implements:

> **Ũ**_[t] = [**I** + strictLower(diag(β_[t])(**Γ**_[t] ⊙ **K**_[t]**K**_[t]ᵀ))]^{−1} diag(β_[t])**V**_[t]

so the training cost story is concept 22's story unchanged — *"preserves chunkwise parallelism …
while maintaining linear-time training complexity"*. **The card should not re-tell it.** Concept 22
is the card that demonstrates chunking, with an exact equivalence result; this card links back and
spends its space on the rule instead. One thing worth saying and nothing more: the decay enters the
chunk as a **product of α over the chunk**, so a very small α makes that product underflow — the
paper says nothing about this, and neither should the card beyond noting the shape.

---

## The evidence, quoted with its conditions

All of it at **1.3B parameters, 100B tokens of FineWeb-Edu**, peak LR 4e−4, weight decay 0.1,
gradient clip 1.0, cosine schedule with 1B warmup tokens, batch 0.5M tokens, **training length 4K**,
vocabulary 32,000 (Appendix B.1). Every number below is **[quoted]**.

### Table 2 — S-NIAH, and it is the only place the two mechanisms are separated

| Model | S-NIAH-1 (1K/2K/4K/8K) | S-NIAH-2 (1K/2K/4K/8K) | S-NIAH-3 (1K/2K/4K) |
|---|---|---|---|
| DeltaNet | 97.4 / 96.8 / 99.0 / 98.8 | 98.4 / 45.6 / 18.6 / 14.4 | 85.2 / 47.0 / 22.4 |
| Mamba2 | 99.2 / 98.8 / 65.4 / 30.4 | 99.4 / 98.8 / 56.2 / 17.0 | 64.4 / 47.6 / 4.6 |
| Gated DeltaNet | 98.4 / 88.4 / 91.4 / 91.8 | 100.0 / 99.8 / 92.2 / 29.6 | 86.6 / 84.2 / 27.6 |

The paper's three readings, verbatim:

> **Decay hurts memory retention.** … DeltaNet achieves near-perfect performance across all sequence
> lengths. Mamba2 degrades significantly beyond 2K sequences since it decays historical information
> too quickly, while Gated DeltaNet's degradation is less severe thanks to the use of delta rule.

> **Gating facilitates filtering.** … With fixed state size, lack of clearance causes memory
> collision—information becomes superimposed and indistinguishable. DeltaNet's performance drops
> significantly at longer sequences due to poor memory clearance.

> **Delta rule helps memorization.** In S-NIAH-3, values change from numbers to UUIDs … Mamba2's
> performance drops quickly, while Gated DeltaNet performs better.

**This table is the card's quoted panel and it must be printed in full**, because the honest reading
is not "Gated DeltaNet wins": it is **97.4→98.8 for DeltaNet against 98.4→91.8 for Gated DeltaNet on
S-NIAH-1** — the gate costs retention — beside **98.4→14.4 against 100.0→29.6 on S-NIAH-2** — the
gate buys clearance. Two columns, two directions. A card that prints only the second is quoting
selectively.

Note also the honest detail that **every model collapses on S-NIAH-2 at 8K** (14.4 / 17.0 / 29.6),
and that S-NIAH-3 stops at 4K.

### Table 3 — language modelling and commonsense, 1.3B

Wiki ppl ↓ / LMB ppl ↓ / avg accuracy ↑ over 8 tasks:
DeltaNet **17.71 / 16.88 / 52.14** · Mamba2 **16.56 / 12.56 / 54.89** · Gated DeltaNet
**16.42 / 12.17 / 55.32** · Transformer++ **18.53 / 18.32 / 52.25** · Samba **16.13 / 13.29 / 54.00** ·
Gated DeltaNet-H1 **16.07 / 12.12 / 56.40** · Gated DeltaNet-H2 **15.91 / 12.55 / 56.18**.

The margin over Mamba2 on perplexity is **0.14 / 0.39** — small, and the card should say so in the
same breath as the S-NIAH gap, which is enormous. The two mechanisms buy recall, not perplexity.

### Table 4 — real-world recall, 2K context, average over 6 tasks

DeltaNet **26.2** · Mamba2 **29.8** · Gated DeltaNet **30.6** · Transformer++ **37.0** ·
Gated DeltaNet-H1 **39.0** · Gated DeltaNet-H2 **40.1**.

**The full-attention baseline is still 6.4 points ahead of the best pure recurrent model here**, and
the hybrids only pass it by adding attention back. That belongs in "gives up".

### Table 5 — LongBench, 14 tasks, average

DeltaNet **13.6** · Mamba2 **13.5** · Gated DeltaNet **16.6** · Transformer++ **11.0** ·
Samba **15.9** · H1 **17.8** · H2 **18.4**.

### Figure 3 — throughput

Training throughput on a **single H100**, 1.3B models, at several sequence-length × batch settings.
**It is a figure with no table**, so the only honest statement is the shape: Gated DeltaNet sits
close to DeltaNet and slightly below Mamba2 and Transformer++ at 2K; the hybrids are the fastest at
the longer settings. **Do not print a Kt/s number on the card.** Reading one off a chart and
printing it to three significant figures is exactly the failure this deck exists to avoid.

### What the source does not establish

- **No limitations section.** The paper has none. The nearest thing is §4's *"We observe mixed
  results in length extrapolation"* and its attribution of weak real-data recall gains to
  *"instruction-unaligned small language models being prone to repetition errors"* — an explanation
  offered, not demonstrated.
- **No ablation isolating α from β** was reachable (B.2 unavailable). §3.2's case study compares
  three *models*, not three *terms*; the claim "these two mechanisms are complementary" is supported
  by a model comparison, not by a term-by-term ablation. The card says this.
- **No head dimension, state size, chunk size or layer count** for this paper.
- **No claim about α at inference**: nothing in what was read says whether a trained α actually
  concentrates near 1, near 0, or spreads. So no card panel may assert what a learned gate does.

---

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

Produced by driving the app's own model from node on the default sentence ("The lighthouse keeper
wrote the code in a notebook and hid it under the third stair", 16 tokens), with the four rules of
Table 1 written out from the paper directly rather than through `mixers.js` — so that the seam
reproducing them afterwards is a check and not a tautology. Queries and keys are SiLU'd then
L2-normalised, per §3.4.

### 1. The corners, and both are exact

| claim | measured |
|---|---|
| gated with α_t = 1 ≡ DeltaNet, per-token head outputs | max abs difference **0.00e+0** (bit-identical) |
| gated with β_t = 0, final state | ‖S‖_F = **0.00e+0** — nothing is ever written |

These are the two assertions the self-check gets. They are the card's claim that the gate and the
write strength are genuinely independent knobs, and they are exact rather than approximate.

### 2. What the gate buys, and what it costs, in one table

A standalone associative memory at this app's own head dimension — **d = 8 state, 64 random unit
key/value pairs written in sequence, β = 1** — probed at both ends of the same run. Cosine between
what the state returns for a key and the value that was written with it, averaged over 40 seeds:

**Add rule (LA when α = 1, Mamba2's row of Table 1 otherwise):**

| α | half-life (writes) | first pair | last pair | last 4 | all 64 |
|---|---|---|---|---|---|
| 1 | ∞ | 0.364 | 0.341 | 0.339 | 0.333 |
| 0.99 | 69.0 | 0.279 | 0.439 | 0.432 | 0.327 |
| 0.95 | 13.5 | 0.065 | 0.696 | 0.660 | 0.245 |
| 0.9 | 6.6 | 0.005 | 0.828 | 0.765 | 0.172 |
| 0.8 | 3.1 | −0.024 | **0.920** | 0.814 | 0.112 |
| 0.5 | 1.0 | 0.001 | 0.985 | 0.705 | 0.061 |

**Delta rule (DeltaNet when α = 1, Gated DeltaNet otherwise):**

| α | half-life | first pair | last pair | last 4 | all 64 |
|---|---|---|---|---|---|
| 1 | ∞ | 0.030 | **1.000** | 0.824 | 0.133 |
| 0.99 | 69.0 | 0.027 | 1.000 | 0.827 | 0.130 |
| 0.95 | 13.5 | 0.013 | 1.000 | 0.831 | 0.118 |
| 0.9 | 6.6 | −0.002 | 1.000 | 0.825 | 0.105 |
| 0.8 | 3.1 | −0.020 | 1.000 | 0.798 | 0.086 |
| 0.5 | 1.0 | −0.007 | 1.000 | 0.656 | 0.054 |

Three findings, and the third is the uncomfortable one:

1. **The gate is a bounded memory, not a filter.** α multiplies the *whole* state, so it cannot
   spare one pair: the add rule's most recent pair goes 0.341 → 0.920 as α falls to 0.8, and its
   first pair goes 0.364 → −0.024 in the same column. Both directions of Table 2 in one sweep,
   measured rather than quoted. The half-life column is the readable form: at α = 0.8 the state
   remembers about three writes.
2. **The delta rule's last pair is exact, at every α — 1.000 to three decimals.** That is the
   ‖k‖ = 1 identity above, showing up as a measurement. It is why the gate costs the delta rule so
   much less than it costs the add rule: the newest pair cannot be damaged by decay.
3. **On this workload the gate adds almost nothing on top of the delta rule.** Last-4 recall moves
   0.824 → 0.831 at α = 0.95 and then falls; the 64-pair average falls monotonically, 0.133 → 0.086.
   **The paper's gain does not reproduce at this scale, and the card must say so.** The reason is
   not mysterious: these are random near-orthogonal keys, so there is no correlated junk for a gate
   to clear, and there is no training, so no α is learned. What this page can honestly show is the
   *mechanism* — that α and β are independent, what each does to a state, and what each costs. The
   *benefit* is a claim about 1.3B parameters trained on 100B tokens and it stays in the quoted
   panel where it belongs.

**A note on seeds.** The tables above average **40 seeds**; the card's panel averages **16**, which
is what keeps a slider drag at about 26 ms per render. The third decimal therefore differs between
this note and the screen — 0.341 here against 0.338 there for the add rule's newest pair, 0.133
against 0.125 for the delta rule's 64-pair average. The card's prose quotes the screen, since that is
what a reader can check.

### 3. β, at fixed α = 0.9, delta rule, same lab

| β | last 4 | first pair | all 64 |
|---|---|---|---|
| 1 | 0.825 | −0.002 | 0.105 |
| 0.8 | 0.843 | −0.004 | 0.115 |
| 0.5 | 0.847 | −0.006 | 0.131 |
| 0.2 | 0.815 | −0.002 | 0.153 |
| 0.05 | 0.780 | 0.003 | 0.167 |

β trades the newest pair against the average: a partial write leaves the state closer to what it
already held. **The two knobs are not redundant** — α moves recall from old to new, β moves it from
the newest to the many — and that is the strongest thing this app can say for the paper's
"complementary" claim on its own evidence.

### 4. On the model's own head (block 0, head 0), with a declared stand-in gate

α and β are read off the key by a seeded projection, because **the mixer never sees x_t** and,
more importantly, because **this model is untrained** — a learned α does not exist here. With
α_t = exp(−softplus(⟨k_t, P_α⟩)) and β_t = σ(⟨k_t, P_β⟩), seed 4242: α spans **0.0033–0.9983**
(mean 0.3953), β spans **0.0003–0.9039** (mean 0.3109), and the resulting state is nearly annihilated
(‖S‖_F = 0.262 against DeltaNet's 8.082 and LA's 23.742).

**That is a finding about the stand-in, not about Gated DeltaNet**, and it decides a design point:
the card exposes a **gate-strength knob** and says in as many words that on an untrained model the
*direction* of the effect is real and the *value* is arbitrary. Whole-model top word under each rule
on the default sentence: LA "where" 0.216703, Mamba2 "&lt;end&gt;" 0.439634, DeltaNet "&lt;end&gt;"
0.156459, gated "and" 0.187591, softmax baseline "her" 0.204610 — printed only to show the rules do
different things, never as a quality comparison, because untrained.

---

## The seam — two one-line changes, declared before any code

`app/model/mixers.js` already implements Eq. 10: `write: "gated"` is
`S = gS + β(v − gSk)kᵀ`, which is the expanded gated delta rule exactly. Concept 20 put it there.
Two things it cannot currently express, both needed by this card's comparison panel:

1. **β cannot read the token.** The call is `beta(i, g)`. Concept 20 only needed β tied to the gate
   (Theorem 1's B̄ = 1 − Ā); this paper's β is an independent projection of x_t.
   → `beta(i, g, K[i], V[i], at)` — same arguments `decay` already receives. Backwards compatible:
   concept 20's `(i, g) => 1 - g` is unaffected.
2. **The add rule cannot decay.** `const g = write === "gated" ? … : 1` hard-codes α = 1 for
   `write: "add"`, so Mamba2's row of Table 1 — `S = αS + vkᵀ` — is unreachable, and Table 1 is this
   card's spine.
   → drop the condition and let `decay` apply in every mode. It defaults to 1, so `"add"` and
   `"delta"` are unchanged unless a decay is passed.

Both are one line. Nothing else in the seam moves. The self-check must re-confirm the existing
degenerate reductions after the change, not merely add new ones.

## What the live view must let the reader do

1. **Table 1, as the card's spine.** Four rules — LA, Mamba2, DeltaNet, Gated DeltaNet — selectable,
   showing the objective and the update side by side with the *changed term highlighted*, and the
   same sentence run through the selected rule on the model's own head. The point is that the four
   are one family with two switches.
2. **The two switches, live.** α and β as sliders over the standalone memory lab, with the
   two-ended probe of measurement 2 drawn as it moves: recall of the first pair and of the most
   recent, plus the half-life in writes. The reader must be able to reach every corner and see
   α = 1 land exactly on the DeltaNet card's behaviour, β = 0 empty the state, and β = 1 pin the
   newest pair at exactly 1.000.
3. **Add rule against delta rule at the same α.** The same lab with the delta term switched off —
   this is where the gate's effect is dramatic (0.341 → 0.920) and it is the honest way to show
   what gating is for without pretending it does the same thing on top of the delta rule.
4. **The quoted panel: Table 2 in full**, both directions, with the 1.3B / 100B conditions attached,
   and the sentence that this page reproduces the *mechanism* and not the *gain*.
5. **The negative result gets its own space**, per house style: at d = 8 with random keys the gate
   does not improve on the delta rule here, and the card says why (no training, no correlated junk)
   rather than hiding a flat number.

Not to be drawn: a layer stack for H1 or H2 (ratio unknown), a chunkwise animation (concept 22's
job, and its equivalence result is stronger), a throughput number (Figure 3 is a chart), any claim
about what a *learned* gate concentrates on (unestablished).

## What it leaves behind

Forward, to **concept 24 (`nsa`)**: the state is still one fixed-size object. The gate chooses *when*
to forget but not *what*, and no setting of α and β makes a d × d state hold more than d pairs — the
64-pair average never rises above 0.133 in any row of any table above. Sparse attention answers the
same pressure from the other side: keep every key and value, read a chosen few. That is the link.

Backwards, it closes concept 22's own stated gap, in that paper's words:

> "the length generalization of DeltaNet was limited … We speculate that this is because DeltaNet
> lacks explicit decay factors. This could be improved through incorporating a gating term in the
> recurrence, as demonstrated in a recent work by Yang et al. 2024."

Same first author, six months later. The link text on the timeline comes from there rather than
being invented.
