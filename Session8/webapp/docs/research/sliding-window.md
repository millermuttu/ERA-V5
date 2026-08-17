# Concept 8 — Sliding window attention with global tokens

**Card id:** `sliding-window` · **Date:** 2020-04-10 (arXiv v1) · **Pressure:** how many comparisons

## What was read

- [arXiv:2004.05150](https://arxiv.org/abs/2004.05150), Beltagy, Peters, Cohan (Allen Institute for
  AI) — *Longformer: The Long-Document Transformer*. Abstract page for the version history; the full
  text via [ar5iv](https://ar5iv.labs.arxiv.org/html/2004.05150) for §1–§5 and Appendices A–B; and
  **both PDFs pulled directly** (`arxiv.org/pdf/2004.05150v1`, `.../v2`, text-extracted) so the two
  versions could be diffed line by line rather than trusted.
- **Version history — there are exactly two, and they differ in ways the card must know:**
  - **v1 — Fri, 10 Apr 2020** (228 KB). The mechanism paper.
  - **v2 — Wed, 2 Dec 2020** (1,357 KB). arXiv's own comments field says: *"Version 2 introduces the
    Longformer-Encoder-Decoder (LED) model"*.
- **What changed between them** (verified by diff, not by the comments field):
  - **The mechanism did not change.** `½w` per side, `O(n×w)`, `ℓ×w`, `ℓ×d×w`, the symmetric global
    attention, and the `Q_s,K_s,V_s` / `Q_g,K_g,V_g` split are word-for-word the same in v1 and v2.
    The character-LM results are also unchanged: 1.10 text8, 1.00 enwik8, 0.99 enwik8-large.
  - **A speed claim was withdrawn.** v1's Figure 1 caption asserted: *"Longformer's GPU-kernel is
    nearly as fast as the highly optimized full self-attention operation, and nearly 6X faster than
    naive Pytorch."* **The "6X" number is gone in v2.** v2 replaces the two-way comparison
    (TVM kernel vs naive loop) with a three-way one — `loop` / `chunks` / `cuda` — the words
    `Longformer-chunks` and `Longformer-loop` appear **zero times in v1** — and adds a footnote
    conceding the kernel is *not* actually beating `n²`:
    > "It is worth noting that theoretically, a perfectly optimized Longformer-cuda should be faster
    > than the $n^2$ computation. However, achieving this level of performance requires special
    > knowledge of low-level GPU programming, similar to implementing a highly optimized matrix
    > multiplication. Our current implementation is sufficiently fast and practical to use."
    v1 had claimed flatly that "our implementation is both fast and memory efficient". v2's own
    Figure 1 caption now says the *vectorized PyTorch chunking*, not the custom kernel, is the
    fastest. **Do not repeat the 6× figure; it is a retracted number.**
  - v2 adds the contemporaneous-work paragraph (ETC, GMAT, BigBird) and one sentence to §3.1 calling
    global attention "a easy way to add inductive bias" [sic].
- Everything quoted below is the authors' wording, taken from the v2 text unless marked v1.

## The mechanism, precisely

The framing is the same pressure Concept 6 measures, stated in one line:

> "The original Transformer model has a self-attention component with $O(n^2)$ time and memory
> complexity where $n$ is the input sequence length. To address this challenge, we sparsify the full
> self-attention matrix according to an 'attention pattern' specifying pairs of input locations
> attending to one another."

Three patterns are stacked on top of each other. They are cumulative, not alternatives.

### 1. Sliding window

> "Given the importance of local context (Kovaleva et al., 2019), our attention pattern employs a
> fixed-size window attention surrounding each token. Using multiple stacked layers of such windowed
> attention results in a large receptive field, where top layers have access to all input locations
> and have the capacity to build representations that incorporate information across the entire
> input, similar to CNNs (Wu et al., 2019). **Given a fixed window size $w$, each token attends to
> $\frac{1}{2}w$ tokens on each side** (Fig. 2(b))."

So `w` is the **total** width, split evenly: `w/2` left, `w/2` right. The cost and the depth formula
are one sentence:

> "**The computation complexity of this pattern is $O(n \times w)$, which scales linearly with input
> sequence length $n$. In a transformer with $\ell$ layers, the receptive field size at the top layer
> is $\ell \times w$** (assuming $w$ is fixed for all layers)."

That is the whole comparison to full attention: `O(n²)` becomes `O(n × w)` **per layer**, and the
price is that reaching distance `D` now costs `⌈D / (w/2)⌉` layers instead of one hop. The depth is
doing the work the width used to do — which is exactly the CNN analogy they draw.

And the licence for the staged design that follows:

> "Depending on the application, it might be helpful to use different values of $w$ for each layer to
> balance between efficiency and model representation capacity (§4.1)."

### 2. Dilated sliding window

> "To further increase the receptive field **without increasing computation**, the sliding window can
> be 'dilated'. This is analogous to dilated CNNs (van den Oord et al., 2016) where **the window has
> gaps of size dilation $d$** (Fig. 2(c)). Assuming a fixed $d$ and $w$ for all layers, **the
> receptive field is $\ell \times d \times w$**, which can reach tens of thousands of tokens even for
> small values of $d$."

Note precisely what is and is not bought: the number of cells read per query is **unchanged** (still
`w`), so the cost is unchanged; only the *span* those `w` cells cover is multiplied by `d`. This is
the cleanest "same cost, more distance" statement in the whole timeline, and it is the thing the card
should make the reader feel.

**Not all heads.** The paper is explicit that uniform dilation is not what they want:

> "In multi-headed attention, each attention head computes a different attention score. **We found
> settings with different dilation configurations per head improves performance by allowing some
> heads without dilation to focus on local context, while others with dilation focus on longer
> context.**"

And in the LM setting, the exact recipe:

> "**We do not use dilated sliding windows for lower layers** to maximize their capacity to learn and
> utilize the immediate local context. For the higher layers, **we use a small amount of increasing
> dilation only on 2 heads.** This gives the model the ability to directly attend to distant tokens
> without sacrificing local context."

Appendix B pins it down: **2 heads only** (out of 8), and for the 12-layer small model the dilation
schedule is `0 (layers 0–5), 1 (layers 6–7), 2 (layers 8–9), 3 (layers 10–11)`; for the 30-layer
large model `0 (layers 0–14), 1 (layers 15–19), 2 (layers 20–24), 3 (layers 25–29)`.

So: **six of eight heads never dilate, and half the layers never dilate.** Dilation is a garnish,
not the dish. What it buys is measured in §4.2.2, and it is small — see Numbers.

### 3. Global attention — and the exact wording that matters

The motivation is that a window is a *content-blind* rule and some tasks have content-specific needs:

> "In our case, the windowed and dilated attention **are not flexible enough to learn task-specific
> representations.** Accordingly, we add 'global attention' on **few pre-selected input locations.**
> Importantly, **we make this attention operation symmetric: that is, a token with a global attention
> attends to all tokens across the sequence, and all tokens in the sequence attend to it.**"

Two halves, and the word "symmetric" is load-bearing: a global token is both a *reader of everything*
and a *thing everyone reads*. The second half is what turns it into a two-hop shortcut for the whole
sequence; the first half alone would only make one token well-informed.

**Which tokens? A human picks them, by task convention:**

> "Fig. 2(d) shows an example of a sliding window attention with global attention at a few tokens at
> **custom locations**. For example **for classification, global attention is used for the `[CLS]`
> token while in QA global attention is provided on all question tokens.**"

> "Since the number of such tokens is small relative to and independent of $n$ the complexity of the
> combined local and global attention is still $O(n)$."

> "**While specifying global attention is task specific**, it is a easy way to add inductive bias to
> the model's attention, and it is much simpler than existing task specific approaches that use
> complex architecture to combine information across smaller input chunks."

There is **no procedure** in the paper for choosing global tokens — no learning, no scoring, no
search. The choice is an architectural constant supplied by the practitioner per task, in the same
way Sparse Transformer's stride was. That is the hinge of this card's forward edge.

### Separate projections — and a correction to the brief

> "We use two sets of projections, $Q_s$, $K_s$, $V_s$ to compute attention scores of **sliding
> window** attention, and $Q_g$, $K_g$, $V_g$ to compute attention scores for the **global**
> attention. The additional projections provide flexibility to model the different types of
> attention, **which we show is critical for best performance on downstream tasks.** $Q_g$, $K_g$,
> $V_g$ are all initialized with values that match $Q_s$, $K_s$, $V_s$."

**The subscripts are the other way round from the task brief.** `s` = sliding window, `g` = global.
The global path gets its *own* `Q_g, K_g, V_g`, initialised as copies of the window ones and then
allowed to diverge. Get this right in the card copy; it is the kind of detail a reader will check.

### Implementation — why a kernel was needed

> "For Longformer, the dilated sliding window attention computes only a fixed number of the diagonals
> of $QK^T$. As shown in Fig. 1, this results in a linear increase in memory usage compared to
> quadratic increase for full self-attention. **However, implementing it requires a form of banded
> matrix multiplication that is not supported in existing deep learning libraries like
> PyTorch/Tensorflow.**"

Three implementations, and the paper is blunt about what the naive one costs:

> "*loop* is a memory efficient PyTorch implementation that supports dilation but is **unusably slow**
> and only used for testing; *chunks* only supports the non-dilated case and is used for the
> pretraining/finetuning setting; and *cuda* is our fully functioning highly optimized **custom CUDA
> kernel implemented using TVM** (Chen et al., 2018) and used for the language modeling experiments."

Appendix A, in more detail:

> "*Longformer-loop* is a naive implementation that computes each diagonal separately in a loop. It is
> memory efficient because it only computes the non-zero values, but **it is unusably slow. We only
> use it for testing** … *Longformer-chunks* only supports the non-dilated case. It chunks $Q$ and $K$
> into overlapping blocks of size $w$ and overlap of size $\frac{1}{2}w$, multiplies the blocks, then
> mask out the diagonals. This is very compute efficient because it uses a single matrix
> multiplication operation from PyTorch, but **it consumes 2x the amount of memory a perfectly
> optimized implementation should consume** because it computes some of the zero values …
> *Longformer-cuda* … is the most memory efficient, and it is **as fast as** the highly optimized full
> self-attention."

> "We build our custom CUDA kernel using TVM (Chen et al., 2018), a deep learning compiler stack …
> Using TVM, we describe our banded matrix multiplication in high-level python constructs, then TVM
> generates the corresponding CUDA code and compiles it for GPUs."

Read that ladder carefully, because it is the honest version of "linear attention": the pure-PyTorch
route that *actually* saves the memory is **unusable**; the route that is fast **cannot do dilation**
and wastes 2× memory; and only a compiler-generated custom kernel gets both — and even then it merely
*ties* full attention on speed. **The pattern is a paper result; the saving is a kernel result.**

## Numbers that matter

**Shape of the mechanism**
| Quantity | Value |
|---|---|
| Tokens read per query, sliding window | `w` — `½w` each side |
| Complexity, per layer | `O(n × w)` vs `O(n²)` full |
| Receptive field at layer `ℓ`, window only | `ℓ × w` |
| Receptive field at layer `ℓ`, dilated | `ℓ × d × w` |
| Extra cost of global attention | `O(n)` — "small relative to and independent of `n`" |

**Autoregressive character LM (§4, the sliding-window-plus-dilation experiment)**
- Small model: **12 layers, 8 heads, 512 hidden** (as Transformer-XL). Large: **30 layers, 8 heads,
  512 hidden** (as Sparse Transformer).
- Staged training, **5 phases**: "in the first phase we start with a short sequence length and window
  size, then on each subsequent phase, **we double the window size and the sequence length, and halve
  the learning rate**." Reason given: "the model needs a large number of gradient updates to learn the
  local context first, before learning to utilize longer context."
- Sequence length **2,048 (phase 1) → 23,040 (phase 5, "gpu memory limit")**. Steps per phase (small):
  430K, 50k, 50k, 35k, 5k. Batch per phase: 32, 32, 16, 16, 16.
- **Evaluation at length 32,256**: "We evaluate with sequences of length 32,256 … we split the dataset
  into overlapping sequences of size 32,256 with a step of size 512, and report the performance on the
  **last 512 tokens**."
- Hardware: **48GB RTX8000** GPUs; small model 4 GPUs × 16 days, large model 8 GPUs × 13 days.
- Results: **text8 1.10**, **enwik8 1.00** test BPC at 41M params (new SOTA at the time); large model
  **0.99 on enwik8 at 102M params**, which "matches the performance of the comparable Sparse
  Transformer (Child et al., 2019)" at ≈100M.

**The ablation that justifies the staged windows and the 2-head dilation (Table 4, 150K steps, text8 dev BPC)**
| Configuration | Dev BPC |
|---|---|
| Decreasing `w` (512 → 32, big windows at the bottom) | 1.24 |
| Fixed `w` (= 230, the average) | 1.23 |
| **Increasing `w` (32 → 512, big windows at the top)** | **1.21** |
| No dilation | 1.21 |
| **Dilation on 2 heads** | **1.20** |

The staged-window effect is **0.03 BPC** between best and worst ordering **at identical cost**, and
the whole dilation effect is **0.01 BPC**. The authors flag their own caveat: "the ordering of end
performance will not agree with that at step 150K."

**Pretrain/finetune (§5 — a different experiment; do not merge its numbers with the LM ones)**
- **Sliding window `w = 512`, sequence length 4,096** ("8 times longer than BERT"), continuing from
  the RoBERTa checkpoint. "We use sliding window attention with window size of 512, **therefore using
  the same amount of computation as RoBERTa**." Footnote: "Sequences up to 16K are possible on
  current GPUs."
- **Dilation was dropped here**: "Adding dilation on a few heads as in §4.1 **hurt performance**,
  likely because it is not compatible with the pretrained RoBERTa weights."
- WikiHop dev ablations (Table 10) — the only direct evidence for global attention and the separate
  projections:

| Model | Accuracy / Δ |
|---|---|
| Longformer (seqlen 4,096) | **73.8** |
| Longformer (no linear proj.) | 72.2 / **−1.6** |
| Longformer (no linear proj., **no global atten.**) | 65.5 / **−8.3** |
| Longformer (seqlen 512, `n²` attention) | 71.7 / −2.1 |

**Global attention is worth ~6.7 points on its own; the separate projections ~1.6.** That is the
strongest number in the paper for the half of the mechanism this card is named after.

- LED (v2 only): encoder uses "local attention with window size **1,024** tokens and global attention
  on the **first `<s>` token**"; decoder uses full attention.

**A defect in the paper worth knowing about.** Appendix B Table 12 is mis-typeset **identically in v1
and v2** (confirmed in both PDFs and in the ar5iv HTML, so it is in the LaTeX source, not a rendering
artifact):

```
Phase 1 window sizes    32 (bottom layer) - 8,192 (top layer)
Phase 5 window sizes    512 (bottom layer) - (top layer)          <- number missing
Phase 5 LR              000015625                                 <- mangled 0.000015625
```

Phase 1's sequence length is **2,048**, so a phase-1 top-layer window of 8,192 is impossible. The
cells are split wrong. The doubling rule ("we double the window size … each phase", 5 phases = 4
doublings) forces the intended reading: **phase 1 = 32 → 512, phase 5 = 512 → 8,192.** Use that, and
say it is reconstructed, not quoted.

## What the live view must let the reader do

The app's seam is `readable(i, j)` in `softmaxMixer` (`app/model/mixers.js`), and all three of
Longformer's patterns are one predicate:

```
readable(i,j)  =  j === g  ||  i === g                        // global, both directions
               || ( (i - j) % d === 0 && (i - j) <= d*w/2 )   // dilated window, causal half
```

(The app is causal, so only the left half of the paper's `½w`-each-side window survives — `w/2`
tokens back, `d·w/2` once dilated. Every number below assumes that; see the honesty note in §2.)

`toyCost` in `app/model/cost.js` already counts `reads` per head and exposes `readsPerQuery` against
`fullReads = (T+1)/2`, so the cost side needs no new machinery.

**One prerequisite.** `forward()` currently hands the *same* mixer to every block and head — it calls
`mixer(Q, K, V, DH)` exactly `BLOCKS × HEADS = 8` times, block-major. Per-layer `w` and per-head `d`
(items 3 and 5 below) need the block and head index at the seam. **The lazy fix is one argument**:
pass `{ block: b, head }` as a fifth parameter in `app/model/transformer.js` and let `softmaxMixer`
forward it to `readable`. Every existing card ignores extra arguments, so nothing else changes. (A
call-counting closure inside the card would avoid touching the model, but it silently depends on
`forward`'s loop order — don't.)

Concept 6 already shows a fixed-pattern grid with a two-hop reachability probe, so **none of the five
below is a grid-plus-reachability repeat.** The distinctive assets here are depth, constant-cost
distance, and a reader-chosen hub.

### 1. The window slider, against the bill — the baseline interaction
**Control:** `w` from 2 to 16 (even), on the reader's own ~16-token sentence.
**Displayed:** the band on the grid, plus a readout — at `T = 16, w = 4`: `reads/query: 2.81 · full
causal: 8.5 · saved: 67%` — taken from `toyCost(res).readsPerQuery` vs `fullReads`.
**The number that proves it:** hold `w` and lengthen the sentence. `readsPerQuery` stays pinned near
`w/2 + 1` while `fullReads` climbs with `(T+1)/2`. That *is* `O(n × w)` versus `O(n²)`, measured on
the page rather than asserted. Keep it small; it is table-setting for the three below.

### 2. Depth — proved on activations, not on the mask *(the distinctive one)*
This is where the app's 2 blocks earn their keep, and the trap to avoid is drawing a second-hop
shading and calling it proof — that is Concept 6's probe again, and it only shows the *mask*.
Instead: **prove the receptive field grew by changing a word and watching which block notices.**

**Interaction:** the reader clicks a query token `i`, then clicks a far token `j` that is *outside*
`i`'s window but inside its two-block reach, and swaps `j` for a different word.
**Displayed:** two norms, computed from `res.trace[b].hidden[i]` on the before/after runs:

```
edit "harbour" → "engine" at position 6, watching position 9   (w = 4, so 2 tokens back per block)
  after block 1:  ‖Δh₉‖ = 0.000     position 6 is outside 9's window {7,8,9} — block 1 cannot see it
  after block 2:  ‖Δh₉‖ = 0.734     it arrived via position 7, which did read 6 in block 1
```

**The number that proves the point is the exact zero in the first row and the non-zero in the
second** — and both are real activations from `forward`, not mask arithmetic. Alongside it, the
counted reach: `reachable after 1 block: 3 · after 2 blocks: 5 · ℓ×w/2 + 1 = 5`. Then a third row
labelled **"arithmetic, not computed — this model has 2 blocks"**: `after 12 blocks: 25`. Extrapolate
in the copy, never in the display.

**Be honest about `ℓ × w` here.** The paper's formula is for a *bidirectional* window; the app is
causal, so each block only extends the reach **leftward by `w/2`**, and the measured growth is
`ℓ·w/2 + 1`. Show the measured number as the truth and the paper's `ℓ × w` as the bidirectional case.
A reader who does the arithmetic will otherwise think the app is broken.

### 3. Dilation — the same bill, twice the distance
**Controls:** `d` from 1 to 3 with `w` fixed; plus a per-head selector implementing the paper's
recipe — `heads 0,1: d = 1` (local), `heads 2,3: d = 2,3` (dilated).
**Displayed:** a two-column readout that must be *identical* on the left and *different* on the right:

```
              reads/query      furthest token back reached (1 block / 2 blocks)
  d = 1          2.81                    2  /  4
  d = 2          2.81                    4  /  8
  d = 3          2.81                    6  / 12
```

**The number that proves it is the unchanging left column.** "Increase the receptive field without
increasing computation" is a claim about two quantities, and the reader should watch one stand
perfectly still while the other doubles. Under the per-head split, add a per-head table of *how many
tokens each head puts more than 1% of its softmax weight on* — real weights from `res.trace[b].heads
[h].weights` — so the reader sees the two local heads concentrating while the two dilated heads
spread. Caption it with the paper's own reason ("some heads without dilation to focus on local
context, while others with dilation focus on longer context") and its own price: **0.01 BPC**.

### 4. A global token the reader picks — and the "symmetric" toggle
**Interaction:** click any word in the sentence to make it global (`g`). A second control turns the
symmetry **off**, keeping only "the global token attends to everyone" and dropping "everyone attends
to it".
**Displayed, three things, all computed:**
- **Cost:** total `reads` before and after. The arithmetic is unusually clean and worth showing: under
  the causal mask the global row and column contribute `(T − g) + (g + 1) − 1 = T` cells — **exactly
  16, wherever the reader puts `g` and whatever `w` is** (less whatever the window already covered).
  A hub costs `O(n)`, once, independent of both its position and the window width. That is the
  paper's "small relative to and independent of `n`" made countable.
- **Reach:** for a query after `g`, `tokens reachable in 2 blocks: 5 → 12`. One hand-picked token
  pulls the entire prefix-up-to-`g` within two hops. That arrow is the card's headline.
- **Symmetry off:** the reach number **falls back to 5** for every token except `g` itself, while `g`
  keeps its full row. This is the operational meaning of the word "symmetric", and it is a genuine
  computed difference rather than a definition restated.

**The causal caveat, which the card should surface rather than hide.** Longformer's global attention
is bidirectional-encoder machinery, and this app is a causal decoder. A hub at position `g` can only
have read `j ≤ g`, and only queries at `i ≥ g` can read it — so the shortcut covers the prefix up to
`g`, not the whole sentence, and a hub at position 0 is worthless. Let the reader discover this by
dragging `g` and watching the reach number peak in the middle. It is the most instructive thing on
the card, and it is the concrete face of the fact recorded below: **the paper never uses global
attention in its autoregressive experiments at all.**

Also show the model's actual top-3 next-word probabilities shifting when `g` moves — with the
standing caveat that the weights are untrained, so the correct claim is *"the computation changed"*,
never *"the prediction improved"*.

### 5. Staged windows across the two blocks — and what the app honestly cannot show
**Control:** two windows, `w_block0` and `w_block1`, with a swap button (`32 → 512` vs `512 → 32`, at
the app's scale `2 → 8` vs `8 → 2`).
**Displayed:** total `reads` for both orderings — **they are identical**, because it is the same
multiset of windows — and the two-block reach, which is **also identical** (`(w₀ + w₁)/2 + 1`).
**The number that proves the point is that nothing in the app changes.** That is the honest finding,
and the card should say so out loud: the ordering effect the paper measured is **1.21 vs 1.24 BPC**,
a *training* outcome invisible to an untrained model. Show the paper's Table 4 as a static three-row
bar next to the reader's identical-cost readout, captioned: *the cost is the same either way; which
order learns better is a fact about training that this page cannot reproduce.* An interaction that
demonstrates the limits of the toy is worth more than a fourth animated mask.

## What the source does *not* establish

- **Global attention is never used in an autoregressive model in this paper.** §4.1 says only: "For
  autoregressive language modeling we use our dilated sliding window attention." The word *global*
  does not occur anywhere in §4 — verified by grep over both PDFs. Every global-attention result in
  the paper (`[CLS]`, question tokens, the −8.3 WikiHop ablation, LED's first `<s>`) comes from a
  **bidirectional encoder**. So the app's causal card is combining two things the paper deliberately
  kept apart: the char-LM half (dilated windows, no globals) and the pretrain/finetune half (windows
  plus globals, no dilation, no causality). Say so on the card — it is not a flaw in the app, it is
  the honest boundary of the source.
- **It does not tell you which tokens should be global.** "Global attention is task specific" is the
  entire selection procedure. `[CLS]` for classification, question tokens for QA — conventions, not
  results. There is no ablation over *which* tokens, no learned or scored alternative, and no cost
  model for adding more. The card's reader-chosen global token is faithful to the paper precisely
  because it is arbitrary.
- **`ℓ × w` is a connectivity bound, not an information-flow measurement.** Nothing in the paper
  measures how far influence actually propagates; it counts which cells are non-zero. A path existing
  in the mask is not the same as a signal surviving `ℓ` layers of averaging.
- **The window itself is barely tuned.** `w = 512` in the pretrain setting is chosen so the model
  uses "the same amount of computation as RoBERTa" — a compute-matching decision, not an optimum. The
  only window search reported is the three-row ordering ablation at 150K steps.
- **Dilation is a character-LM-only result in this paper, and a tiny one.** 1.21 → **1.20** BPC at
  150K steps with the authors' own warning about 150K-step orderings, and it **actively hurt** in the
  pretrain/finetune setting (footnote 6: "Adding dilation on a few heads as in §4.1 hurt performance").
  The pretrained Longformer everyone actually used has **no dilation at all**.
- **The separate `Q_g,K_g,V_g` result is one dataset.** −1.6 on WikiHop dev. There is no LM ablation
  for it. And **the app cannot show it at all** — `forward` has one `q`/`k`/`v` per block. Report the
  number; do not build a control for it.
- **`O(n × w)` is per layer, and linear only while `w` is held fixed.** The staged schedule grows `w`
  to 8,192 as `n` grows to 23,040 — at which point the top layers are within a small factor of dense.
  "Scales linearly" is a statement about the pattern, not about how they trained.
- **The pattern does not deliver the saving; the kernel does.** Banded matmul "is not supported in
  existing deep learning libraries"; the naive version is "unusably slow"; the fast version "only
  supports the non-dilated case" and burns "2x the amount of memory"; and the TVM kernel is only "as
  fast as" dense. **v1's "nearly 6X faster than naive Pytorch" was removed in v2** — treat any
  speed claim on this card as retracted.
- **The paper says nothing about the KV cache.** It is an encoder-and-char-LM paper; the quantity it
  reduces is *attention* memory within a forward pass, not the per-token state a decoder carries
  between steps. The card must not imply otherwise — see the forward edge.
- **The app proves connectivity, never speed.** A `readable(i,j)` predicate over a dense loop skips
  FLOPs in the toy but demonstrates nothing about wall-clock, and the 32-dim, 2-block, untrained
  model produces seeded noise shaped by a rule. Read the geometry; never the predictions.
- **One number in the paper is simply wrong** (Table 12, both versions) — see Numbers. Anything the
  card says about per-phase window sizes is a reconstruction and should be labelled as one.

## Leaves behind

**Backward — what this answers.** Concept 6 legislated a pattern chosen for *periodic* data: strided
attention with stride `l` ≈ the image row width, a rule that works because pixels really do repeat
every `l` positions. On text it failed outright — 1.13 vs 1.00 BPC on Enwik8 — and the 2019 fix was
for a human to notice and switch to the fixed pattern. Longformer's claim is that text has a
*different* shape and needs a differently shaped rule: **local context everywhere, plus a handful of
positions that matter to everyone.** "Given the importance of local context" is the window's entire
justification; "the windowed and dilated attention are not flexible enough to learn task-specific
representations" is the global token's. Where Sparse Transformer buys two-hop connectivity from a
*periodic* second head, Longformer buys it from a *designated hub* — and the WikiHop ablation puts a
price on the difference: removing global attention costs **8.3 points**, far more than any window
tuning in the paper. Both papers still choose their pattern from the indices alone, before a token is
seen; this one just chooses a pattern with a better prior about language.

**Forward — the two weaknesses, stated precisely.**

1. **Bounding the compute does not bound the cache.** `O(n × w)` says each query reads `w` keys. It
   does not say the *other* `n − w` keys can be thrown away — at layer `ℓ+1` a different query needs
   them, and in a decoder every one of them must still be sitting in memory when the next token
   arrives. The app makes this visible for free: `toyCost` reports `cacheNumbers` and
   `growsWithContext`, and **under a sliding window both are unchanged from full attention.** The
   reads-per-query number falls; the bytes-per-token number does not move. That gap is the whole
   subject of the cards that follow — the ones that shrink what is *stored* per token (sharing K/V
   across heads, compressing it to a latent, or replacing the growing list with a fixed-size state)
   rather than what is *read* per query. Longformer never claimed to fix it; the card should be the
   place a reader learns they are two different bills.

2. **The hub is chosen by hand.** This is the same fixity Concept 6 leaves behind, but sharper,
   because it is now *per task* rather than per modality: the practitioner must know, before training,
   which positions everything else will need to see. `[CLS]` because that is where classifiers read
   from; the question tokens because that is what QA compares against. It works, and it works well
   — and it does not generalise to a task whose important tokens are not identifiable from their
   position in the input format. **Every later method attacks exactly this**: let the model decide
   what to attend to from the *content* — by routing, by hashing, by learned or adaptive spans, by
   retrieval — or give up on choosing altogether and make dense attention cheap enough that no one
   has to. Longformer is the high-water mark of hand-designed sparsity: the best that a rule written
   by a person, looking only at indices and task conventions, ever did. The pressure this card
   measures — *how many comparisons* — it genuinely reduces. The question it hands forward is whether
   a human should be the one deciding **which** comparisons to keep.
