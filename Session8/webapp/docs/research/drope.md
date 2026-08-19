# Concept 26 — DroPE
**Card id:** `drope` · **Date:** 2025-12-13 (arXiv v1) · **Pressure:** where a token sits

## The record this note corrects

This entry was created from the course material with **no public source and no date**. Its date field
held `2026-01-01`, a placeholder sort key whose only job was to put the entry last, and the plan for
the card said in as many words that it must present that value as a sort key rather than as a date.
The lesson's own instruction was stricter still: *"The widget deliberately does not simulate an
unverified DroPE mechanism."*

**All of that is now out of date.** The research pass for this card found a paper:

> **arXiv:2512.12167** — *Extending the Context of Pretrained LLMs by Dropping Their Positional
> Embeddings*, **Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin**. Categories `cs.CL`,
> `cs.AI`, licence CC BY 4.0. **One version: v1 — Sat, 13 Dec 2025, 04:23:47 UTC, 2,135 KB.**
> Project page: <https://pub.sakana.ai/DroPE/>. Also on OpenReview.

So the record changes in three ways, and the card says so on its face:

1. **The date is real**: `2026-01-01` → **`2025-12-13`**, the arXiv v1 submission date, verified on the
   abstract page. It still sorts last, now for a reason rather than by construction.
2. **The source is a paper**, so the entry loses its unverified badge — the deck marks an entry
   verified exactly when a paper backs it.
3. **The mechanism is knowable**, so the card can show it. The lesson's boundary — *"any more detailed
   mechanism is a hypothesis until it is checked against the reference implementation"* — has been
   resolved by publication rather than by guessing.

### One thing that is still not established: whether these are the same DroPE

The course material's DroPE is **LightningLM V4's**, recorded as *"positional recalibration: DroPE,
applied before annealing"*, with **8K trained → 256K reported, a 32× extension**. The paper's DroPE
is **removing RoPE from a pretrained model and briefly recalibrating**. Same name, same shape of
claim — a recalibration step that buys length — and the paper's procedure would fit the cookbook's
one-line description without contradicting it. **That is not identity.** The cookbook gives no
algorithm, so there is nothing to compare against, and the card must present them as two records
about a name rather than one mechanism with two write-ups. Nothing in the paper mentions LightningLM,
and nothing in the course material mentions Sakana AI.

## What was read

- The paper, from the **v1 HTML render** (`arxiv.org/html/2512.12167v1`) and its abstract page:
  the procedure, the argument for why a decoder-only model can carry position without an embedding,
  the recalibration budgets, Tables 1–3, and the attention-pattern analysis (Figures 7–8).
- The **course material**, which is the other primary source and the reason this entry exists:
  `reference/session8-lesson.md` §9 in full, and `reference/widgets/widget_8_drope.html`, whose own
  text carries the evidence boundary this card inherits.
- **What could not be reached**: the exact learning-rate schedules (*"adjusted … accordingly"* is all
  the render gave), and the per-task columns behind the Table 2 and 3 averages.

---

## The mechanism, exactly

**Remove the positional embedding from a trained model, then briefly continue training at the
original context length.** That is the whole method: *"removing positional embeddings from every
layer"*, then continued pretraining on the same data, the same hyperparameters and the same context
length. There is no new positional scheme and no equation — the recalibration objective is the
ordinary language-modelling loss applied to the modified architecture. What comes out is a **NoPE**
transformer: no rotation, no table, no bias, nothing added for position at all.

The argument for why that can possibly work is architectural, and the app can test it:

> "The first attention layer in a NoPE transformer can perfectly reconstruct sequence positions"

with later layers able to *"emulate the effects of relative or absolute positional embeddings"*, and
the causal mask supplying the structure through *"uniform mixing of tokens"*.

And the argument for why you would use RoPE in the first place, given that: the paper finds
NoPE-from-scratch *"significantly underperforms RoPE during training"*, so RoPE is kept for its
inductive bias during pretraining and dropped afterwards. **Positional embeddings as scaffolding** —
necessary to build the thing, removable once it stands.

Why dropping beats stretching, in the paper's own diagnosis of what concepts 16–18 do:

> "Scaling warps low-frequency phases, shifting long-range attention in precisely the subspaces most
> used for semantic matching"

and, on YaRN specifically, that it produces attention patterns *"closely matching that of simply
cropping the sequence length"* while perplexity looks fine. Figure 7 reports that the high-frequency
positional heads are largely unaffected by removing RoPE; Figure 8 that the semantic heads are the
ones YaRN moves.

### The numbers, quoted with their conditions

Models: 500M from scratch, then **SmolLM-360M, SmolLM-1.7B, Llama2-7B**. Recalibration at the
original context length — 2,048 tokens for SmolLM, 4,096 for Llama2 — and evaluation at **2× that
length zero-shot**, up to **80×** on LongBench knowledge-extraction tasks.

**Recalibration is cheap, which is the practical claim**: SmolLM-360M recovered 95% of performance in
*"less than 5B tokens, representing a minuscule 0.8% of SmolLM's original budget"*; the larger models
used **20B tokens, 0.5%–2% of pretraining**. From scratch, DroPE matched RoPE's perplexity
*"within 2K steps"* and ended below the NoPE-from-scratch baseline. **QKNorm was added after dropping
the embeddings** for longer recalibration runs, to mitigate *"training instabilities"* — a real
caveat, not a footnote.

**Needle in a haystack at 2× the training context (Table 1):**

| method | multi-query | multi-key | multi-value |
|---|---|---|---|
| RoPE (unmodified) | **0.0%** | — | — |
| YaRN | 17.8% | — | — |
| **DroPE** | **28.0%** | **41.6%** | **23.3%** |

**LongBench averages:** SmolLM-360M base **2.98**, YaRN **19.94**, **DroPE 30.52** (Table 2);
SmolLM-1.7B **DroPE 21.49** against YaRN **16.23**; Llama2-7B **DroPE 26.08** against YaRN **19.14**
(Table 3).

The first row of that table is the one to keep: **an unmodified RoPE model scores zero** at twice its
training length. Concepts 16–18 exist because of that zero, and this paper's answer is to delete the
thing they were all repairing.

### What the source does not establish

- **No dedicated limitations section.** The nearest statement is about *other* approaches:
  *"alternative architectures and positional embedding schemes have shown early promise… Yet, these
  parallel efforts are still far from challenging established pipelines, introducing notable
  performance and stability trade-offs that prevent wide adoption."*
- **Recalibration is required**, so this is not an inference-time switch — the same caveat the course
  material insists on for V4's version, arrived at independently.
- **Largest model is 7B**, and the zero-shot extension demonstrated is **2×**; the 80× figure is
  LongBench knowledge extraction, a different kind of task, and the two should not be quoted as one
  number.
- **Nothing about massive activations or attention sinks** (concept 19), despite that being an obvious
  question for a model with no positional signal.

---

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

**The deck's own baseline is already a NoPE transformer.** `forward(tokens)` with no `position` and no
`rotate` adds nothing for position anywhere: concept 1's card is this paper's endpoint. So the thing to
measure is not the mechanism — there is nothing to switch on — but the **premise**: that a causal
transformer with no positional embedding is nevertheless position-aware.

### 1. The mask is the position signal, and the control makes it exact

Swap two tokens in the middle of the sentence (2 and 9) and look at the **last** position, whose own
token was not touched:

| model | largest change in the final hidden state |
|---|---|
| causal, no positional embedding — the deck's baseline | **4.46** |
| the same, **causal mask removed** | **7.11e-15** |

Zero, to floating-point noise, once the mask is gone. With no positional embedding and no mask, the
model literally cannot tell the two orderings apart — the last position's output is a function of the
*set* of earlier tokens. Restore the mask and the same swap moves the state by 4.46. **The causal mask
is not a detail of the implementation; in a NoPE transformer it is the entire positional mechanism**,
and this is the exact form of the paper's premise.

Swept over every distinct two-token swap that leaves the final token alone: **102 of 102 change the
final state**, smallest change **1.87e-3**. The sensitivity is not a knife-edge; every ordering is
distinguished.

### 2. Where the information comes from — exact arithmetic

Give attention uninformative scores (all equal, which is what a query that has learned nothing
produces) and the weight on each visible key is **exactly 1/(i+1)** — worst deviation across the
sentence **0.0e+0**. So the mixed vector at position `i` is the prefix mean, and `1/(i+1)` — the count
of everything up to here — is in it by construction: 1.0000 at `i = 0`, 0.1250 at `i = 7`, 0.0625 at
`i = 15`. No training and no embedding required for the *information* to be present. What training
would supply is a use for it.

### 3. What this app cannot show, measured rather than asserted

A linear probe from hidden states to position index scores **R² = 1.000 in-sample and negative under
leave-one-out** (−2.20 after block 1, −0.15 after block 2). The in-sample figure is meaningless — 33
free parameters against 16 tokens fits anything — and the honest reading of the held-out figure is
that **position is not linearly decodable from this model's states in any generalising way.** That
does not contradict the paper: its claim is that a NoPE transformer *can* reconstruct position, an
existence claim about trained weights. These weights are random. **The card reports the negative
result and the reason, and does not print the in-sample number as if it meant something** — the first
version of this note did, which is why the figure is here at all.

### 4. Defined is not proven — the lesson's point, in one line of arithmetic

The app's own `rope()` returns a rotation at any position, 256K included: the angle for pair `i` is
`pos · freq_i` and nothing in it fails at a large `pos`. That is the whole of what plain RoPE
establishes at 32× its training length, and it is exactly the lesson's sentence: *"the positional rule
is defined at 256K"* is not *"the model can use a 256K context reliably."* The paper's Table 1 puts a
number on the gap the course could only name: **0.0%** at 2×.

---

## What the live view must let the reader do

1. **See both records side by side** — the course's 8K → 256K → 32×, "before annealing", mechanism
   unestablished; and the paper's dated, authored, quotable procedure — with the identity question
   stated rather than resolved.
2. **Swap two tokens** in the sentence, with the causal mask on and off, and watch the final state
   move by 4.46 and then by nothing. This is the card's demonstration and it is exact.
3. **See the 1/(i+1) arithmetic** that puts position into a NoPE model's mixed vector.
4. **Read what removing RoPE bought**, quoted with its conditions, against the zero an unmodified
   RoPE model scores at twice its training length.
5. **Be told what this page cannot show**: the recalibration, the probe that does not generalise, and
   the fact that the deck's baseline is already the paper's endpoint so there is no "before and after"
   to animate.

Not to be drawn: a simulated V4 DroPE (the course material forbids it and the identity is
unestablished), a trained NoPE model, a recalibration curve, or a claim that this app's untrained
NoPE model demonstrates *quality* at any length.

## What it leaves behind

Nothing in this deck — it is the last entry. What it leaves for the work after it is the question its
own evidence opens: the demonstrated zero-shot extension is **2×** on models up to **7B**, the long
figures come from a different family of task, stability needed **QKNorm**, and the deck's oldest
mechanism — the causal mask from concept 1 — turns out to be carrying the positional load. Twenty-six
cards after "attention is all you need", the position scheme that works best at extension is **no
position scheme**, and the honest form of that sentence is that it works best *after* one has been
used and removed.
