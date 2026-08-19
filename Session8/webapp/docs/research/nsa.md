# Concept 24 — Native Sparse Attention
**Card id:** `nsa` · **Date:** 2025-02-16 (arXiv v1) · **Pressure:** how many comparisons

## What was read

- [arXiv:2502.11089](https://arxiv.org/abs/2502.11089), Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu
  Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang,
  Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng — *Native Sparse Attention: Hardware-Aligned
  and Natively Trainable Sparse Attention*. Categories `cs.CL`, `cs.AI`, `cs.LG`.
- **Version history, read off the abstract page — two versions:**
  - **v1 — Sun, 16 Feb 2025, 11:53:44 UTC**, 915 KB.
  - v2 — Thu, 27 Feb 2025, 09:01:21 UTC, 916 KB (1 KB apart; no diff was performed).

  **The app's record of `2025-02-16` is verified correct.** The abstract page carries **no comments
  field** — no conference line, no award line. Whatever this paper has since collected, the record
  this deck cites is an arXiv posting, and the card says nothing about prizes.
- Full text read from the **v2 HTML render** (`arxiv.org/html/2502.11089v2`). Sections used: §1,
  §2 (rethinking sparse attention), §3.1–3.2 (overall framework and the gated three-branch form),
  §3.3.1 (token compression), §3.3.2 (token selection, including the blockwise rationale and Eq. 9),
  §3.3.3 (sliding window), §3.4 (kernel design), §4.1 (pretraining setup), §4.2–4.3 (benchmarks,
  LongBench, chain-of-thought), §5 (efficiency, including the decoding memory-access table), §6
  (discussion of alternative strategies).
- **What could not be reached.** The per-row detail of Table 1 (only the averages and two deltas came
  back), the full LongBench per-subset columns (only the averages), and the exact figure axes for the
  training-loss curve and the kernel-speed plots. The card quotes only what came back with a number,
  and the note says where a figure is a figure.

---

## The one-sentence version

Every sparse-attention method before this one decided at inference time which keys to skip, on a
model trained to read all of them; NSA makes the sparsity part of the architecture — three
branches, a learned gate, and a block-selection rule whose scores are a by-product of a branch the
model was computing anyway — so the model is trained sparse from the first token.

## The mechanism, exactly

### The frame, §3.2

Replace the full key/value set for query `t` with a smaller remapped set per branch, attend within
each branch, and mix the branches with a learned gate (Eq. 5):

> **o\*_t = Σ_{c∈C} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)**, with **C = {cmp, slc, win}**

and the gate scores **g^c_t ∈ [0,1]** *"derived from input features via an MLP and sigmoid
activation"*. Three branches, three separate softmaxes, one weighted sum. **Note what this is not:**
it is not a mask over one softmax. Every earlier sparse method in this deck — concept 6's strided
pattern, concept 8's window, concept 19's sinks — hides keys inside a single attention. NSA runs
attention three times over three different key sets and adds the results. That is why it can be
trained: each branch is differentiable and the gate decides how much each matters.

### Branch 1 — compression, §3.3.1

> **K̃^cmp_t = { φ(k_{id+1 : id+l}) | 0 ≤ i ≤ ⌊(t−l)/d⌋ }**

Blocks of **l = 32** consecutive keys, stride **d = 16** (so blocks overlap by half), each mapped to
one vector by **φ, "a learnable MLP with intra-block position encoding"**. The branch's key count is
about `s/d` — it grows with the context, but sixteen times more slowly.

### Branch 2 — selection, §3.3.2, and this is the paper's real idea

The selection branch needs to know which blocks matter. Computing that would normally mean scoring
every key, which is the cost being avoided. NSA reuses the compression branch's own softmax:

> **p^cmp_t = Softmax(q_tᵀ K̃^cmp_t)**  (Eq. 8)

and derives block importance for selection blocks of size **l' = 64** by summing the compressed
scores that overlap each one (Eq. 9), valid under the stated conditions **l ≤ l', d | l and d | l'**:

> **p^slc_t[j] = Σ_{m=0}^{l'/d−1} Σ_{n=0}^{l/d−1} p^cmp_t[ (l'/d)·j − m − n ]**

Then heads inside a GQA group vote together (Eq. 10), because they must share the keys that get
loaded:

> **p^slc′ = Σ_{h=1}^{H} p^{slc,(h)}**

and the top **n = 16** blocks are read at full resolution (Eq. 11–12). §4.1 adds that the count
includes *"fixed activating the 1 initial block and 2 local blocks"* — so thirteen blocks are
actually chosen and three are always on.

**Why blocks rather than tokens**, in the paper's own words — two reasons, one hardware and one
statistical:

> "Modern GPU architectures exhibit significantly higher throughput for continuous block accesses
> compared to random index-based reads."

> "Attention scores often exhibit spatial continuity, suggesting that neighboring keys tend to share
> similar importance levels."

### Branch 3 — sliding window, §3.3.3

**w = 512** recent tokens, in their own branch, for a reason that is about training rather than
about quality:

> "Local patterns typically adapt faster and can dominate the learning process, preventing the model
> from effectively learning from compression and selection tokens."

Isolating the window stops the gradient from taking the easy road: if local keys were mixed into the
same softmax, the model would learn to lean on them and the other two branches would never receive a
useful signal. This is a **gradient-flow argument, not an accuracy argument**, and the card should
present it as such — it is the clearest example in the deck of an architectural choice made because
of how training behaves rather than because of what inference needs.

### §3.4 — why a custom kernel, and what it does

FlashAttention loads a contiguous block of queries and streams keys past it. That is wrong here:
*"queries within a block may require disjoint KV blocks"*, because each query selects its own
sixteen. NSA inverts the loop — **group-centric loading**: take one position `t`, load **all heads of
its GQA group at once**, then walk that position's selected key blocks. The GQA group is the unit
because the whole group shares one selection (Eq. 10), so one KV block load serves every head in it.
The stated goal is arithmetic intensity: *"arithmetic intensity above [the critical threshold]
becomes compute-bound, while below it becomes memory-bound"*.

---

## The evidence, quoted with its conditions

**The model**: 27B total parameters, **3B active** (MoE — 72 routed experts, 2 shared, top-6
routing), 30 layers, hidden 2560, 64 heads in **4 GQA groups**, `d_q = d_k = 192`, `d_v = 128`.
**Pretrained on 270B tokens of 8k-length text**, then continued at 32k with YaRN (concept 18).
NSA hyperparameters: **l = 32, d = 16, l' = 64, n = 16, w = 512**.

- **General benchmarks (Table 1)**: Full Attention average **0.443**, NSA **0.456**. Largest named
  gains: GSM8K **+0.034**, DROP **+0.042**.
- **LongBench (Table 2), averages**: H2O **0.303** · InfLLM **0.383** · Quest **0.392** ·
  **Exact-Top 0.423** · Full Attention **0.437** · **NSA 0.469**.
  The row that matters is Exact-Top: an oracle that scores every key and keeps the best ones, i.e.
  the thing every heuristic is trying to approximate. **NSA beats it by 0.046** — which cannot be an
  approximation-quality result, because you cannot approximate an oracle and win. The only
  explanation the paper offers is the one in its title: the model was *trained* this way.
- **Needle in a haystack**: *"NSA achieves perfect retrieval accuracy across all positions in 64k-
  context needle-in-a-haystack test"*.
- **Chain of thought (Table 3, AIME)**: NSA-R **0.121** vs Full Attention-R **0.046** at 8k;
  **0.146** vs **0.092** at 16k.
- **Speed, §5**: **9.0× forward**, **6.0× backward**, **11.6× decoding**, all at **64k**.
- **Training (Figure 4)**: *"Both models exhibit stable convergence, with NSA consistently
  outperforming the Full Attention model"* — a figure, no table, so no number is quotable.

### Decoding memory access, Table 4 — the one the app can recompute

| context | full attention | NSA | speedup |
|---|---|---|---|
| 8,192 | 8,192 | 2,048 | 4.0× |
| 16,384 | 16,384 | 2,560 | 6.4× |
| 32,768 | 32,768 | 3,584 | 9.1× |
| 65,536 | 65,536 | 5,632 | 11.6× |

Per step, NSA loads `⌊s/d⌋` compressed + `n·l'` selected + `w` window tokens.

### What the source does not establish

- **No limitations section.** §6 discusses what *other* approaches fail at — key-clustering's
  *"non-trivial computational overhead"* and *"load imbalances, especially in Mixture-of-Experts
  systems"*, and heuristic blockwise methods' non-differentiable selection causing *"low recall
  rates"* — but states no limitation of its own.
- **No ablation isolating the three branches** came back. The gate is learned and the paper does not
  report what it learns, so nothing on the card may claim which branch carries the model.
- **Nothing about storage.** Every number in the paper is about *access* or *time*. The KV cache
  itself is untouched — see the measurement below.
- **No comparison at short context.** Every efficiency figure is 8k and above, which the measurement
  below shows is not an accident.

---

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

NSA written from §3.2–3.3 and run over this app's own queries, keys and values, on the default
sentence (16 tokens). At this scale the paper's block sizes do not fit, so the card runs
**l = 4, d = 2, l' = 4, n = 2, w = 4** — which satisfies the paper's own conditions (`l ≤ l'`,
`d | l`, `d | l'`) and is stated on the card as a scaled-down setting with the paper's values printed
beside it. **φ is mean pooling here**, a declared stand-in: the paper's φ is a learned MLP and this
model is untrained, so there is nothing to learn from.

### 1. The corners are exact

| claim | measured |
|---|---|
| every block selected, all gate mass on the selection branch | **0.00e+0** difference from plain causal attention |
| window at `w = T`, all gate mass on the window branch | **0.00e+0** difference from plain causal attention |

Both branches are ordinary attention over a subset, so at the degenerate setting they must *be*
ordinary attention. They are, bit for bit. The compression branch has no such corner — its keys are
derived objects and no setting makes them the originals, which is itself worth saying.

### 2. The scores really are free

Over all 8 head-passes and every query with at least one full block behind it (96 queries):

- **extra dot products spent on selection scoring: 0.** The compression branch computes
  `q · k̃` for each compressed block to produce its own output; Eq. 9 is then a sum over numbers
  already in hand. This is the card's central claim and it is a count, not an argument.
- **the top-n blocks contain the key full attention itself weights highest: 81.3%** of the time
  (a rough control — two blocks drawn at random from four — sits near 50%).
- **share of full attention's weight mass inside everything NSA reads: 85.1%.**

Both percentages carry the standing caveat: **untrained weights**, so the attention distribution
they are measured against is itself meaningless as language. What they establish is narrower and
still worth having — *the derived score is not noise*. It agrees with the full computation far more
often than chance on the same numbers.

### 3. The trade is a crossover, and this app is on the wrong side of it

Reads over the whole sentence at the app's settings: **NSA 201, full attention 136.** At the last
query, **19 against 16**. NSA is *more* expensive here, and that is not a bug — it is the mechanism
seen at the wrong scale. Full attention reads `s`; NSA reads `⌊s/d⌋ + n·l' + w`, so it saves only
when

> **s > (n·l' + w) · d / (d − 1)**

- at the **paper's** hyperparameters: **break-even at 1,638 tokens**. Below that, NSA reads more —
  at s = 1,024 it reads 1,600 against 1,024. Every efficiency number in the paper is at 8k or above,
  and now that is visibly not a presentational choice.
- at the **app's** hyperparameters: break-even at **24 tokens**, and the sentence is 16.

| s | full | NSA | ratio |
|---|---|---|---|
| 1,024 | 1,024 | 1,600 | 0.64× |
| 1,638 | 1,638 | 1,638 | 1.00× |
| 2,048 | 2,048 | 1,664 | 1.23× |
| 8,192 | 8,192 | 2,048 | **4.00×** |
| 65,536 | 65,536 | 5,632 | **11.64×** |

**The paper's Table 4 reproduces exactly** from that formula at its own hyperparameters — 2,048 /
2,560 / 3,584 / 5,632 and 4.0 / 6.4 / 9.1 / 11.6×. The card computes it rather than quoting it.

### 4. It reduces reading, not keeping

Nothing in the mechanism removes a key from the cache: the selection branch reads **original** key
and value blocks, so all of them must still be there, and the compressed keys are an **addition**.

| s | full attention keeps | NSA keeps | change |
|---|---|---|---|
| 8,192 | 8,192 | 8,192 + 512 | **+6.25%** |
| 65,536 | 65,536 | 65,536 + 4,096 | **+6.25%** |

This is the sharpest thing the card can say against the reading a headline invites. Concepts 7, 15
and 21 all attacked the size of the cache. **This one makes it slightly bigger and reads a small
part of it.** Both are ways to survive a long context and they are not the same trade, and a reader
who does not see the difference will pick the wrong one.

---

## What the live view must let the reader do

1. **See three separate softmaxes.** Toggle each branch, move its gate, and watch the output move
   against full attention's. The reader has spent twenty-three cards inside one attention; the thing
   to notice here is that there are three.
2. **Watch the free score being derived.** For a chosen query: the compressed blocks and their
   `p^cmp` bars, then Eq. 9's sum turning those into `p^slc` per selection block, then the top-n mark.
   With the running count of extra dot products at zero beside it.
3. **See what each branch reads**, on the grid, in the three branches' own colours, against the row
   full attention would have used — and the share of its weight mass that lands inside.
4. **Reach the corners**: all blocks selected, or `w = T`, and watch the difference from plain
   attention go to zero.
5. **Meet the crossover.** A context-length control that recomputes reads at both parameter sets and
   shows the app's own sentence sitting on the losing side of it, then the paper's Table 4 appearing
   from the same formula at 8k and above.

Not to be drawn: a learned φ (mean pooling here, said plainly), a learned gate (sliders, said
plainly), the Triton kernel or any wall-clock figure, and the MoE architecture the results were
measured on.

## What it leaves behind

Forward, to **concept 25 (`dsa`)**: the selection here is a hierarchy the architect fixed —
compression blocks of 32, stride 16, selection blocks of 64, sixteen of them, window 512 — seven
numbers chosen before any data arrived, plus three blocks that are always on regardless of what the
query wants. DeepSeek's own next sparse attention keeps the idea of choosing at query time and drops
most of the scaffolding.

Backwards, the record names **concept 23 (`gated-deltanet`)** as what it answers, and the
limitation is that card's own closing line: a fixed-size memory with better manners is still a
fixed-size memory. Concepts 19–23 all answered "the past is too big" by compressing it into
something bounded, and every one of them paid in exact recall. NSA answers the same pressure from
the other side — keep every key, read a chosen few — which is why its needle result can be perfect
where a fixed state's cannot be. Concept 21's record points here too, for a different reason it
states itself: a cache that has been compressed has to have every cache-editing technique
re-derived for it, and this is the first cache-editing technique after it.
