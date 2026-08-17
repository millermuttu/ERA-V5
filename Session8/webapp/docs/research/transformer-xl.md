# Concept 5 — Segment recurrence across contexts

**Card id:** `transformer-xl` · **Date:** 2019-01-09 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:1901.02860](https://arxiv.org/abs/1901.02860), Dai, Yang, Yang, Carbonell, Le, Salakhutdinov
  — *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. Abstract page for the
  version history; [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1901.02860) for §1
  (fragmentation), §3.1 (vanilla model), §3.2 (segment-level recurrence with state reuse), §3.3
  (relative positional encodings), §4.2–4.5 (ablations, RECL, evaluation speed), Tables 6–9.
- Version history: v1 **9 Jan 2019**, v2 18 Jan 2019, v3 2 Jun 2019. The timeline uses **v1**.

Everything quoted below is verbatim from the ar5iv rendering of the paper. Notation is transcribed
from the LaTeX `alttext`, so `SG(·)`, `∘`, `R_{i−j}`, `u`, `v` are the paper's own symbols.

## The mechanism, precisely

### The problem it names: context fragmentation

The paper coins the term in §1, describing Al-Rfou et al. (2018), who trained deep Transformers on
*"separated fixed-length segments of a few hundred characters, without any information flow across
segments."* Two consequences, in the paper's words:

> *"As a consequence of the fixed context length, the model cannot capture any longer-term dependency
> beyond the predefined context length. In addition, the fixed-length segments are created by
> selecting a consecutive chunk of symbols without respecting the sentence or any other semantic
> boundary. Hence, the model lacks necessary contextual information needed to well predict the first
> few symbols, leading to inefficient optimization and inferior performance. **We refer to this
> problem as context fragmentation.**"*

So fragmentation is specifically about the **first few tokens of each segment**: they are asked to
predict with almost no left context, not because the corpus lacks it but because the chunking threw
it away. §3.1 restates it as two limitations of the vanilla model: *"the largest possible dependency
length is upper bounded by the segment length"*, and *"simply chunking a sequence into fixed-length
segments will lead to the context fragmentation problem."* Under that paradigm, *"information never
flows across segments in either the forward or backward pass."*

**What happens at segment boundaries during evaluation** is the other half, and it is the expensive
half. The vanilla model does not evaluate segment-by-segment; it slides:

> *"During evaluation, at each step, the vanilla model also consumes a segment of the same length as
> in training, but only makes one prediction at the last position. Then, at the next step, the
> segment is shifted to the right by only one position, and the new segment has to be processed all
> from scratch. … this procedure ensures that each prediction utilizes the longest possible context
> exposed during training, and also relieves context fragmentation issue encountered in training.
> However, this evaluation procedure is extremely expensive."*

That is the trade the whole paper is built on: the vanilla model **can** buy back full context at
evaluation time, but only by recomputing an entire segment per predicted token. Transformer-XL gets
the same context for one forward pass per segment. The 1,874× number below is exactly this ratio.

### The recurrence: cache the previous segment's hidden states

Let two consecutive segments of length `L` be `s_τ = [x_{τ,1}, …, x_{τ,L}]` and
`s_{τ+1} = [x_{τ+1,1}, …, x_{τ+1,L}]`, and let `h_τ^n ∈ ℝ^{L×d}` be the n-th layer hidden state
sequence for segment `s_τ`. Then (paper's §3.2, schematically):

    h̃_{τ+1}^{n−1} = [ SG(h_τ^{n−1}) ∘ h_{τ+1}^{n−1} ]

    q_{τ+1}^n , k_{τ+1}^n , v_{τ+1}^n  =  h_{τ+1}^{n−1} W_q^⊤ ,  h̃_{τ+1}^{n−1} W_k^⊤ ,  h̃_{τ+1}^{n−1} W_v^⊤

    h_{τ+1}^n = Transformer-Layer( q_{τ+1}^n , k_{τ+1}^n , v_{τ+1}^n )

> *"where the function SG(·) stands for stop-gradient, the notation `[h_u ∘ h_v]` indicates the
> concatenation of two hidden sequences along the length dimension, and `W_·` denotes model
> parameters."*

Three things are load-bearing in those three lines, and a card that misses any of them is wrong:

1. **The query comes from the current segment only** — `q` is built from `h_{τ+1}^{n−1}`, not from
   `h̃`. **Keys and values come from the extended context** `h̃`. Cached tokens are attended *to*;
   they never attend. The paper: *"the critical difference lies in that the key `k_{τ+1}^n` and value
   `v_{τ+1}^n` are conditioned on the extended context `h̃_{τ+1}^{n−1}` and hence `h_τ^{n−1}` cached
   from the previous segment."*
2. **`SG` is stop-gradient**, and it is there because *"During training, the hidden state sequence
   computed for the previous segment is **fixed and cached** to be reused as an extended context when
   the model processes the next new segment."* Fixed = it is a constant, not a function of the
   parameters, for backprop's purposes. *"Although the gradient still remains within a segment, this
   additional input allows the network to exploit information in the history, leading to an ability
   of modeling longer-term dependency and avoiding context fragmentation."* The cost of the design is
   named in that clause: **the gradient still remains within a segment.** Backprop never crosses a
   segment boundary. This is what makes the scheme cheap (no unrolled graph, constant memory per
   step) and what makes it *not* BPTT.
3. **The recurrence is diagonal, not horizontal.** *"notice that the recurrent dependency between
   `h_{τ+1}^n` and `h_τ^{n−1}` shifts one layer downwards per-segment, which differs from the
   same-layer recurrence in conventional RNN-LMs. Consequently, the largest possible dependency
   length grows linearly w.r.t. the number of layers as well as the segment length, i.e. `O(N × L)`."*

The paper marks the relationship to the older technique explicitly: *"This is analogous to truncated
BPTT (Mikolov et al. 2010) … However, different from truncated BPTT, our method caches a sequence of
hidden states instead of the last one, and should be applied together with the relative positional
encoding technique described in Section 3.3."*

**Memory length is a free parameter, generalised beyond one segment:** *"we can cache a predefined
length-`M` old hidden states spanning (possibly) multiple segments, and refer to them as the memory
`m_τ^n ∈ ℝ^{M×d}` … In our experiments, we set `M` equal to the segment length during training, and
increase it by multiple times during evaluation."* So yes — **evaluation memory length can and does
differ from training** (see Numbers).

### Why absolute positional encoding breaks under state reuse

This is the part most summaries skip, and it is the reason the second contribution exists at all.
In the standard Transformer the positional encodings are `U ∈ ℝ^{L_max × d}`, where *"the i-th row
`U_i` corresponds to the i-th **absolute position within a segment**."* Naively bolting that onto the
recurrence gives:

    h_{τ+1} = f( h_τ ,  E_{s_{τ+1}} + U_{1:L} )
    h_τ     = f( h_{τ−1} , E_{s_τ}   + U_{1:L} )

> *"Notice that, both `E_{s_τ}` and `E_{s_{τ+1}}` are associated with the same positional encoding
> `U_{1:L}`. As a result, the model has no information to distinguish the positional difference
> between `x_{τ,j}` and `x_{τ+1,j}` for any `j = 1, …, L`, resulting in a sheer performance loss."*

The failure is exact and easy to demonstrate: **position 3 of the memory and position 3 of the
current segment carry the identical encoding row `U_3`**, so a query cannot tell "three tokens into
the segment I am in" from "three tokens into the segment before". The abstract calls the general
condition *"without disrupting temporal coherence"*; §1 calls the fix *"the necessity of using
relative positional encodings rather than absolute ones, in order to enable state reuse without
causing temporal confusion."*

The repair, in the paper's framing: *"the fundamental idea is to only encode the relative positional
information in the hidden states … instead of incorporating bias statically into the initial
embedding, one can inject the same information into the attention score of each layer."* And the
justification that it loses nothing: *"we won't lose any temporal information, as the absolute
position can be recovered recursively from relative distances."*

### The four terms

Standard absolute-encoding attention score between query `i` and key `j` in the same segment,
expanded (`E` = word embedding, `U` = absolute position encoding):

    A^abs_{i,j} = E_{x_i}^⊤ W_q^⊤ W_k E_{x_j}   (a)
                + E_{x_i}^⊤ W_q^⊤ W_k U_j       (b)
                + U_i^⊤    W_q^⊤ W_k E_{x_j}    (c)
                + U_i^⊤    W_q^⊤ W_k U_j        (d)

Re-parameterised:

    A^rel_{i,j} = E_{x_i}^⊤ W_q^⊤ W_{k,E} E_{x_j}   (a)
                + E_{x_i}^⊤ W_q^⊤ W_{k,R} R_{i−j}   (b)
                + u^⊤            W_{k,E} E_{x_j}    (c)
                + v^⊤            W_{k,R} R_{i−j}    (d)

Three edits, each with its own stated reason:

- **`U_j` → `R_{i−j}`** in (b) and (d). *"This essentially reflects the prior that only the relative
  distance matters for where to attend. Note that `R` is a sinusoid encoding matrix (Vaswani et al.
  2017) **without learnable parameters**."*
- **`U_i^⊤ W_q^⊤` → learnable `u ∈ ℝ^d`** in (c), and **→ learnable `v ∈ ℝ^d`** in (d). The reason is
  an argument about query-position invariance: *"since the query vector is the same for all query
  positions, it suggests that the attentive bias towards different words should remain the same
  regardless of the query position."* Both `u` and `v` are trainable vectors, one per head.
- **`W_k` split into `W_{k,E}` and `W_{k,R}`**: *"we deliberately separate the two weight matrices
  `W_{k,E}` and `W_{k,R}` for producing the content-based key vectors and location-based key vectors
  respectively."*

And the reading the card should quote directly, because it is the paper's own gloss:

> *"Under the new parameterization, each term has an intuitive meaning: term (a) represents
> **content-based addressing**, term (b) captures a **content-dependent positional bias**, term (c)
> governs a **global content bias**, and (d) encodes a **global positional bias**."*

Note what these mean operationally. (a) is ordinary QK content matching. (b) is "this particular
query cares about this particular distance". (c) is "this key word is intrinsically interesting to
everyone" — a per-key constant, independent of `i`. (d) is "this distance is intrinsically
interesting to everyone" — a per-distance constant, independent of content. (c) and (d) are the two
terms Shaw et al. do not have.

### Ordering subtlety: this came *after* Shaw et al. 2018, and differs

Shaw, Uszkoreit and Vaswani, *Self-Attention with Relative Position Representations* (NAACL 2018,
arXiv:1803.02155), predates this by roughly ten months. Transformer-XL acknowledges it — *"Previously,
the idea of relative positional encodings has been explored in the context of machine translation
(Shaw et al. 2018) and music generation (Huang et al. 2018). Here, we offer a different derivation,
arriving at a new form of relative positional encodings"* — and then states the two differences
precisely:

> *"In comparison, the formulation in Shaw et al. 2018 only has terms (a) and (b), dropping the two
> bias terms (c) and (d). Moreover, Shaw et al. 2018 merge the multiplication `W_k R` into a single
> trainable matrix `R̂`, which abandons the inductive bias built into the original sinusoid
> positional encoding (Vaswani et al. 2017). In contrast, our relative positional embedding `R`
> adapts the sinusoid formulation. As a benefit of the inductive bias, a model trained on a memory of
> some certain length can automatically **generalize to a memory several times longer during
> evaluation**."*

So the ordering is: relative position was Shaw's idea; **length extrapolation** is Transformer-XL's,
and it falls out of keeping `R` sinusoidal instead of learning a lookup table. This is measurable in
Table 6, not just asserted — see Numbers.

### Efficiency footnote

*"a naive way to compute `A` requires computing `W_{k,R}^n R_{i−j}` for all pairs `(i,j)`, whose cost
is quadratic w.r.t. the sequence length. However, noticing that the value of `i−j` only ranges from
zero to the sequence length, we show a simple computation procedure in Appendix B, which reduces the
cost to be linear w.r.t. the sequence length."* (This is the "left-shift trick" every later
implementation inherits.)

## Numbers that matter

**Dependency length (Table 8, RECL — Relative Effective Context Length, in words):**

| model group | model | r = 0.1 | r = 0.5 | r = 1.0 |
|---|---|---|---|---|
| group 1 | Transformer-XL 151M | **900** | 800 | 700 |
| group 1 | QRNN | 500 | 400 | 300 |
| group 1 | LSTM | 400 | 300 | 200 |
| group 2 | Transformer-XL 128M | **700** | 600 | 500 |
| group 2 | — use Shaw et al. 2018 encoding | 400 | 400 | 300 |
| group 2 | — remove recurrence | 300 | 300 | 300 |
| group 2 | Transformer | 128 | 128 | 128 |

> *"Transformer-XL manages to model dependency of 900 words long on average with r = 0.1. The RECL of
> Transformer-XL is **80% and 450% longer than recurrent networks and Transformer respectively**.
> Both the recurrence mechanism and our positional encodings contribute to a longer RECL."*

Read the table before repeating the headline. **80%** is 900 vs QRNN's 500, inside group 1.
**450%** is 700 vs the vanilla Transformer's 128, inside group 2 — a *different, smaller*
Transformer-XL. The paper is explicit that *"RECL is computed on a model group rather than a single
model"* and *"each group has the same parameter budget"*, so the two percentages are two separate
comparisons the abstract puts in one sentence. Also note the vanilla Transformer's RECL is exactly
**128** at every `r` — identical to its attention length. Its context is capped by its segment, which
is the fragmentation claim shown as a number.

**Evaluation speed (Table 9, enwik8, per-token time on one GPU, vs Al-Rfou et al. 2018):**

| attention length | how much Al-Rfou et al. 2018 is slower |
|---|---|
| 3,800 | **1,874×** |
| 2,800 | 1,409× |
| 1,800 | 773× |
| 800 | 363× |

> *"due to the state reuse scheme, Transformer-XL achieves an up to **1,874 times speedup** during
> evaluation."*

The abstract rounds this down to *"up to 1,800+ times faster than vanilla Transformers during
evaluation."* It is a **ratio of per-token evaluation time**, against the sliding-window-by-one
evaluation procedure described in §3.1 — not a training speedup, and not a generic inference number.
It scales with attention length precisely because the vanilla model's cost per predicted token scales
with the window it must recompute.

**Training length vs evaluation length (does memory length differ? yes):**

| setting | train attention length | eval attention length |
|---|---|---|
| WikiText-103 | 384 | 1,600 |
| enwik8 (18/24-layer) | 784 | 3,800 |
| §4.2 ablation | backprop length 128 | attention length up to 640 |

> *"Although the backpropagation length during training is only 128, with the two techniques the
> attention length can be increased to 640 at test time. In the standard setting with 151M
> parameters, the perplexity decreases as the attention length increases."*

**The ablation that proves it is the encoding, not just the cache (Table 6, WikiText-103):**

| recurrence | encoding | loss | PPL init | PPL best | Attn Len |
|---|---|---|---|---|---|
| ✓ | Ours | Full | 27.02 | **26.77** | 500 |
| ✓ | Shaw et al. 2018 | Full | 27.94 | 27.94 | 256 |
| ✗ | Ours | Full | 29.59 | 29.02 | 260 |
| ✗ | Vaswani et al. 2017 | Half | 30.97 | 30.97 | 120 |
| ✗ | Al-Rfou et al. 2018 (vanilla) | Half | 31.16 | 31.16 | 120 |

> *"Increasing the attention length during evaluation improves performance only when our positional
> encoding is used."*

The `PPL init == PPL best` rows are the tell: with Shaw's learned relative embeddings, or with any
absolute encoding, evaluating with a longer context buys **nothing** (27.94 → 27.94). Only the
sinusoidal-`R` formulation improves (27.02 → 26.77).

**Fragmentation isolated from long context (Table 7, One Billion Word — sentences shuffled, so there
is no long-term dependency to capture):**

| method | PPL |
|---|---|
| Ours | 25.2 |
| With Shaw et al. 2018 encodings | 25.7 |
| Without recurrence | 27.1 |

> *"we deliberately choose a dataset that does not require long-term dependency, so that any
> improvement from establishing the recurrence can be attributed to solving the context
> fragmentation. … using segment-level recurrence substantially improves performance even when
> long-term dependency is not needed."*

**1.9 perplexity** on a dataset with no long-range structure at all. That is the cleanest evidence in
the paper that fragmentation is a real and separate cost, and it is the number the card should use
when a reader says "but my sentence is short."

**Headline LM results (abstract):** bpc/perplexity **0.99** enwik8, **1.08** text8, **18.3**
WikiText-103, **21.8** One Billion Word, **54.5** Penn Treebank (without finetuning).

## What the live view must let the reader do

The app runs a real toy transformer: `d_model = 32`, 4 heads (`d_k = 8`), 2 blocks, seeded untrained
weights, causal masking, on the reader's own ~16-token sentence. The seam in `app/model/mixers.js`
exposes `readable(i, j)`, `bias(i, j)`, `rotate(v, pos)` and kv head sharing. **Segment recurrence is
a `readable(i, j)` mechanism**; the relative encoding half is a `bias(i, j)` mechanism. Both must be
independently switchable, because the paper's own ablation separates them.

One structural requirement before the interactions: the card must compute the segments
**sequentially**, caching each block's hidden states, rather than running one T-wide pass with a
fancy mask. Only then is the layer-shift real, and only then is the reach counter honest.

1. **Cut the sentence, with memory at zero — meet fragmentation.**
   A segment-length slider `L ∈ {2, 4, 8}` over the reader's ~16 tokens, with `M = 0`. Display the
   `T × T` readable matrix with segment boundaries drawn as hard walls, plus a per-token strip
   labelled with the reader's actual words. **The number that proves it:** for the *first* token of
   segment 2 — name it, e.g. "`brown` can see 1 word: itself" — show `readable keys = 1` against
   `causally legal keys = 9`. Sweep the whole sentence and report the aggregate: *"with L = 4, 4 of
   your 16 tokens can see fewer than 2 words. Those are the fragmented positions."* That count is the
   paper's "lacks necessary contextual information needed to well predict the first few symbols",
   rendered in the reader's own sentence.

2. **Slide the memory length `M` from 0 upward and watch a named word become reachable.**
   `M ∈ {0, L, 2L, 3L}`. `readable(i, j)` becomes: same segment, **or** within the `M` cached
   positions immediately preceding this segment's start. Display the same matrix with memory columns
   tinted a distinct colour, and a one-line reachability probe the reader controls: pick a query word
   and a target word. **The numbers that prove it:** (i) a binary *"is `<target>` reachable from
   `<query>`? no → yes"* at the `M` where it flips; (ii) the actual softmax weight that word receives
   once reachable (e.g. 0.000 → 0.071) — it must be a computed attention weight, not a mask entry;
   (iii) `readsPerQuery` from `toyCost`, which rises with `M` and is the price. Also show the
   **layer-shift reach counter**: max reachable index at block 1 is `i − M`, at block 2 is `i − 2M` —
   `O(N × L)` with `N = 2`, live, next to the paper's claim.

3. **Toggle the stop-gradient shading.**
   `SG(·)` cannot be shown as a gradient in an untrained, forward-only model, so show it as
   *provenance*: paint every key column that came from cache in the "frozen" colour and every key
   column from the current segment in the "live" colour, and report the split. **The number:** *"of
   the 12 keys `<query word>` reads, 8 are frozen memory and 4 carry gradient"* — and, as `M` grows,
   the frozen fraction grows towards 1. The caption is the paper's own sentence: *the gradient still
   remains within a segment.* This is the single most-misread line in the paper and the card should
   make the reader see that most of what a late token reads is a constant.

4. **Switch the positional treatment: absolute (broken) vs relative (the fix).**
   With absolute encodings under reuse, display the encoding row assigned to memory position `j` and
   to current-segment position `j` side by side. **The number:** their L2 distance is exactly
   `0.000` — *"the model has no information to distinguish the positional difference between `x_{τ,j}`
   and `x_{τ+1,j}`"*, proven rather than asserted, and the resulting score matrix shows two columns
   at genuinely different distances receiving an identical positional contribution. Flip to relative:
   `bias(i, j)` computed from `R_{i−j}`, with the **four terms broken out as four separately
   toggleable numbers** for one selected `(i, j)` cell — (a) content, (b) content-dependent
   positional, (c) global content (constant down a column, since it has no `i`), (d) global
   positional (constant along a diagonal, since it depends only on `i − j`). Turning (c) and (d) off
   *is* the Shaw et al. 2018 formulation; label that toggle with his name, and show the same two
   columns now separated by a nonzero bias delta.

5. **Count what the vanilla model would have had to recompute.**
   A cost readout comparing three evaluation regimes on the reader's sentence at the current `L`:
   segment-at-a-time with no memory, segment-at-a-time with memory `M`, and the vanilla
   slide-by-one-and-recompute procedure. **The number:** token-forward-passes to score all `T`
   tokens — `T` for Transformer-XL versus `T × L` for the sliding vanilla model, printed as a ratio
   (with `T = 16, L = 8` that is 16 vs 128, an **8×**). State plainly that the paper's 1,874× is this
   same ratio at `L = 3,800`, so the reader can see that the speedup is not a kernel trick but the
   arithmetic of not recomputing a window per token.

## What the source does *not* establish

- **It does not give unlimited context.** The bound is stated: `O(N × L)`, growing linearly in depth
  and segment length, because the recurrence *"shifts one layer downwards per-segment"*. A card that
  says "Transformer-XL removes the context limit" contradicts §3.2.
- **It is not BPTT and it is not an RNN.** `SG(·)` means no gradient ever crosses a segment boundary.
  The memory is a read-only cache of activations; nothing in the model is trained to *decide* what to
  keep. Any "the model learns to compress its past" reading is unsupported here.
- **Memory tokens never attend.** Queries come only from the current segment. Cached states are not
  updated, refined or re-contextualised by later segments — they were computed under whatever context
  existed when their segment ran, and are frozen at that.
- **`R` is not learned.** It is Vaswani's sinusoid. Only `u`, `v`, `W_{k,E}`, `W_{k,R}` are trainable.
  The extrapolation-to-longer-memory property is credited specifically to keeping the sinusoid.
- **The two headline percentages are not one comparison.** 80% and 450% come from two different model
  groups and two differently sized Transformer-XLs (151M and 128M). RECL is a metric the paper
  *introduces* in the same work, defined on a model group, with a tunable parameter `r`; it is not an
  independently established measure.
- **1,874× is the top of a range** (363× at attention length 800) and is per-token *evaluation* time
  against one specific baseline procedure, on enwik8, on one GPU.
- **Recurrence costs memory.** The paper concedes *"the recurrence mechanism costs additional memory"*
  and has to run a separate same-GPU-memory comparison (Table 10, Appendix A) to be fair. Caching
  `M × d` states per layer is not free.
- **No claim about bidirectional or encoder models.** Every result is causal language modeling.
- **The app's own model establishes nothing about quality** — 32 dimensions, untrained seeded weights.
  Reachability, read counts and bias decompositions are honest; predictions are not.

## Leaves behind

**Backward.** This card only makes sense after the fixed-length-segment card: concept 1 gave an `O(1)`
path between any two tokens *inside a window*, and concept 2's absolute positional encoding is the
thing that silently assumed the window always starts at position 0. Transformer-XL is the first entry
on the timeline where the *boundary* of the context — not the cost inside it — is the object of
attack. It also retires an assumption: before it, "longer context" meant "bigger `L`, quadratically
more compute". Here it means "carry state".

**Forward.** This is the ancestor of every cross-chunk state-carrying scheme:

- **Compressive Transformer** (2019) takes the obvious next step — instead of discarding the oldest
  memory when `M` is exceeded, compress it — attacking exactly the `M × d × N` storage this paper
  concedes.
- **Recurrent Memory Transformer**, **Memorizing Transformers**, **Infini-attention** all keep the
  shape (queries from now, keys/values from a carried store) and change what the store is.
- **KV caching at inference** in every modern serving stack is this idea with `M` unbounded and the
  segment length 1 — and **StreamingLLM**'s sliding window with attention sinks is the same
  reachability question this card's slider asks, answered differently.
- **The stop-gradient is the limitation later work still lives with.** Because the cache is never
  differentiated through, nothing learns *what is worth caching*. Compressive Transformer bolts on a
  separate local compression loss precisely to get a training signal back; state-space models (Mamba)
  answer instead by making the carried state differentiable and same-layer, which is the RNN property
  §3.2 explicitly gave up. The card should end on that: the recurrence here is a plumbing change, not
  a learning change.
- **On the positional side**, the four-term decomposition is the template. T5's learned relative
  buckets keep terms in this shape; **ALiBi** is essentially term (d) alone, made a fixed linear
  function of distance and nothing else; **RoPE** rejects the additive-bias framing entirely and
  rotates `q` and `k` instead — which is why the app's seam needs both `bias(i, j)` and
  `rotate(v, pos)`.

The unresolved thing this card hands on: **the memory is a fixed-size FIFO of raw activations,
uncompressed, unselected, and frozen.** Everything downstream is a different answer to "what should
we keep, and who decides?"
