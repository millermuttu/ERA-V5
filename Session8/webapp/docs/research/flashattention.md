# Concept 14 — FlashAttention

**Card id:** `flashattention` · **Date:** 2022-05-27 (arXiv v1) · **Pressure:** how it meets the hardware

## What was read

- [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri
  Rudra, Christopher Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness*. Abstract page for title/authors/version history; the full text via
  [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.14135), read end to end including Appendices A–E,
  because the backward pass, the block sizes and the benchmarking protocol are all appendix-only and
  the main body's claims are not checkable without them.
- **Version history — two versions:**
  - **v1 — Fri, 27 May 2022**, 1,325 KB.
  - **v2 — Thu, 23 Jun 2022**, 1,653 KB. The card's date is the v1 date, per the repo convention.
- Everything in quotation marks below is the authors' wording. Every arithmetic result presented as
  a computation (traffic counts, the float64 difference, the crossover table) was run, not recalled;
  where a constant is mine rather than the paper's, it says so.

**Where this sits in the deck.** `data/mechanisms.js` already routes `performer` → `flashattention`
("an estimate whose error falls only as the square root of the features, so accuracy and speed trade
against each other directly"), so `answers: "performer"` is fixed by the existing record. Concepts
11–13 (`delta-rule`, `rope`, `alibi`) are still `pending` — this card is being researched out of
chronological order and must not assume a `rope` or `alibi` card exists to link to.

---

## The mechanism, precisely

### The claim: exact, and the accusation against everyone before it

The abstract states the target and the accusation in two sentences:

> "Approximate attention methods have attempted to address this problem by trading off model quality
> to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a
> missing principle is making attention algorithms IO-aware -- accounting for reads and writes
> between levels of GPU memory."

The introduction sharpens it into the sentence this whole card exists to carry:

> "Although these methods reduce the compute requirements to linear or near-linear in sequence
> length, many of them do not display wall-clock speedup against standard attention and have not
> gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate
> with wall-clock speed) and tend to ignore overheads from memory access (IO)."

That is a direct indictment of Concepts 6, 8, 9 and 10 in this very deck — Sparse Transformer,
Longformer, linear attention, Performer. And the deck should note that the paper's own Table 3 makes
the accusation concrete: the approximate methods it benchmarks *do* get speedups on LRA (Linformer
2.5×, Linear Attention 2.3×, Performer 1.8×, Local 1.7×, Reformer 1.3×), and **exact FlashAttention
beats all of them at 2.4× while scoring higher than every one of them** (59.8 avg vs 54.9–59.6).
The pitch is not "approximation is slow"; it is "approximation was not necessary to be fast".

The mechanism itself changes **no arithmetic**:

> "We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer
> memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from
> HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not
> storing the large intermediate attention matrix for the backward pass."

Theorem 1 states the guarantee:

> "Algorithm 1 returns $O = \mathrm{softmax}(QK^\top)V$ with $O(N^2d)$ FLOPs and requires $O(N)$
> additional memory beyond inputs and output."

**Read that carefully: it is an algebraic identity, not a bitwise one.** The paper nowhere claims bit
identity, and it cannot — the rescaling reorders the summation, and floating-point addition is not
associative. See "What the source does *not* establish".

### The hardware argument

> "The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of memory of different sizes and
> speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth
> memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming
> multiprocessors with bandwidth estimated around 19TB/s. The on-chip SRAM is an order of magnitude
> faster than HBM but many orders of magnitude smaller in size."

Those five figures are the entire load-bearing structure of the paper: **40–80 GB HBM at 1.5–2.0
TB/s; 192 KB SRAM per SM × 108 SMs at ~19 TB/s.** SRAM is roughly 10× the bandwidth and roughly
10⁵–10⁶× smaller. The taxonomy that follows is what makes attention the target:

> "Memory-bound: the time taken by the operation is determined by the number of memory accesses,
> while time spent in computation is much smaller. Examples include most other operations:
> elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer
> norm)."

**Softmax is a reduction, therefore softmax is memory-bound.** That is the pivot. And the reason
ordinary kernel fusion does not rescue it:

> "However, in the context of model training, the intermediate values still need to be written to HBM
> to save for the backward pass, reducing the effectiveness of naive kernel fusion."

Standard attention, Algorithm 0, verbatim in structure: load `Q,K`, compute `S=QKᵀ`, **write `S` to
HBM**; read `S`, compute `P=softmax(S)`, **write `P` to HBM**; load `P` and `V`, compute `O=PV`,
write `O`. Four round trips through HBM of an `N×N` object.

### Tiling and online softmax — the technical heart

The paper's own words for why this is hard, and the escape:

> "We compute attention by blocks. Softmax couples columns of $K$, so we decompose the large softmax
> with scaling."

The stable single-block softmax, defined so the decomposition can be stated:

    m(x) := max_i x_i
    f(x) := [ e^{x_1 − m(x)}  …  e^{x_B − m(x)} ]
    ℓ(x) := Σ_i f(x)_i
    softmax(x) := f(x) / ℓ(x)

And the decomposition of the softmax of a **concatenation** `x = [x⁽¹⁾ x⁽²⁾]`, which is the whole
mechanism in three lines:

    m(x) = max( m(x⁽¹⁾), m(x⁽²⁾) )

    f(x) = [ e^{m(x⁽¹⁾) − m(x)} f(x⁽¹⁾)   e^{m(x⁽²⁾) − m(x)} f(x⁽²⁾) ]

    ℓ(x) = e^{m(x⁽¹⁾) − m(x)} ℓ(x⁽¹⁾) + e^{m(x⁽²⁾) − m(x)} ℓ(x⁽²⁾)

> "Therefore if we keep track of some extra statistics ($m(x), \ell(x)$), we can compute softmax one
> block at a time."

The footnote names the pattern: *"This style of aggregation is called algebraic aggregation."*

**Algorithm 1, the update as executed.** Block sizes first, and note the second one is odd:

    B_c = ⌈M / 4d⌉ ,    B_r = min( ⌈M / 4d⌉ , d )

`B_r` is capped at the **head dimension** `d`. That is a proof convenience (it keeps `B_r × B_c ≤
M/4` while bounding the `B_r × d` tiles of `Q`, `O`), not what a tuned CUDA kernel does. The app
should not inherit this cap; it should expose `B_c` directly and say the cap exists.

Initialisation: `O = 0` (`N×d`), `ℓ = 0` (`N`), `m = −∞` (`N`), all in HBM. Outer loop over `T_c =
⌈N/B_c⌉` blocks of `K,V`; inner loop over `T_r = ⌈N/B_r⌉` blocks of `Q`. Per `(i,j)` tile:

    line  9:  S_ij = Q_i K_jᵀ                        ∈ R^{B_r × B_c}
    line 10:  m̃_ij = rowmax(S_ij)
              P̃_ij = exp(S_ij − m̃_ij)                 (pointwise)
              ℓ̃_ij = rowsum(P̃_ij)
    line 11:  m_i^new = max(m_i, m̃_ij)
              ℓ_i^new = e^{m_i − m_i^new} · ℓ_i  +  e^{m̃_ij − m_i^new} · ℓ̃_ij
    line 12:  O_i ← diag(ℓ_i^new)^{-1} ( diag(ℓ_i) e^{m_i − m_i^new} O_i
                                          + e^{m̃_ij − m_i^new} P̃_ij V_j )
    line 13:  ℓ_i ← ℓ_i^new ,  m_i ← m_i^new

**That is the card.** Line 11 and line 12 are the two rescalings, and they are different in kind and
must be explained differently:

- **Line 11 rescales two scalars.** The running denominator `ℓ_i` was accumulated relative to the old
  maximum `m_i`. If the new tile contains a larger score, every previously-summed exponential is now
  measured against the wrong reference, and the factor `e^{m_i − m_i^new}` (which is `≤ 1`, shrinking
  the old sum) corrects for it. Symmetrically `e^{m̃_ij − m_i^new}` corrects the new tile's own sum.
- **Line 12 rescales a `d`-vector, and undoes a division.** `O_i` currently holds a value already
  divided by the *old* `ℓ_i`. Multiplying by `diag(ℓ_i)` **un-normalises** it back to a raw weighted
  sum, applies the same max correction as line 11, adds the new tile's contribution `P̃_ij V_j`, and
  re-divides by the *new* `ℓ_i^new`. The output is kept normalised at every step, which is what makes
  the extra memory `O(N)` rather than `O(N·T_c)`.

Appendix B.5 makes clear that the second of those is the delta against the prior art:

> "Rabe and Staats 2021 summarizes each block with its temporary output along with the softmax
> normalization statistics. At the end of the forward pass, the temporary outputs of all the blocks
> are combined using the statistics to produce the final output. FlashAttention instead incrementally
> updates the output (Algorithm 1 line 12) after processing each block, so only one copy of the
> output is needed (instead of $K$ copies for $K$ blocks)."

and, crucially for the framing of this whole card:

> "The first major difference is that Rabe and Staats 2021 focuses on the reducing the total memory
> footprint (maximum amount of GPU memory required) while FlashAttention focuses on reducing memory
> accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory
> access is the primary determining factor of runtime. … As a result, FlashAttention is faster than
> standard attention (2-4×) while Rabe and Staats 2021 is around the same speed or slightly slower
> than standard attention."

**Tiling for memory had already been done. Tiling for memory *traffic* is the paper.**

### Recomputation in the backward pass

> "One of our goals is to not store $O(N^2)$ intermediate values for the backward pass. The backward
> pass typically requires the matrices $S, P \in \mathbb{R}^{N\times N}$ to compute the gradients
> with respect to $Q,K,V$. However, by storing the output $O$ and the softmax normalization
> statistics $(m,\ell)$, we can recompute the attention matrix $S$ and $P$ easily in the backward
> pass from blocks of $Q,K,V$ in SRAM. This can be seen as a form of selective gradient
> checkpointing."

**Stored:** `Q, K, V` (inputs, already there), `O` (`N×d`), `ℓ` and `m` (`N` each), plus the RNG state
`R`. **Not stored:** `S`, `P`, `dP`, `dS`, and the dropout mask — every `N×N` object.

Why more FLOPs is faster, in the authors' words:

> "While gradient checkpointing has been suggested to reduce the maximum amount of memory required,
> all implementations (that we know off) have to trade speed for memory. In contrast, even with more
> FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses (Fig. 2)."

The measured evidence for that sentence is the Figure 2 table: **66.6 GFLOPs → 75.2 GFLOPs (+13%),
while HBM traffic falls 40.3 GB → 4.4 GB (−89%), and runtime falls 41.7 ms → 7.3 ms (−82.5%).**
Recomputation is bought with FLOPs and paid for in bytes, and bytes are the currency.

Two implementation observations from Appendix B.4 that a card claiming to explain the backward pass
must not skip:

> "We do not need to store the dropout mask of size $O(N^2)$ from the forward pass. Instead, we can
> save the pseudo-random number generator states from the forward pass and re-generate the dropout
> mask in the backward pass."

> "When computing the softmax gradient, we use Eq. 4 to compute $D_i = P_{i:}^\top dP_{i:}$ without
> reducing over $P_{i:}$ and $dP_{i:}$ of size $N$ (they might not fit into SRAM). Instead we can
> rewrite $D_i = do_i^\top o_i$ and compute the dot product between vectors of size $d$."

That second one is a genuinely pretty identity and is worth stating on the card, because it is the
reason `O` has to be stored at all:

    D_i = P_{i:}ᵀ dP_{i:} = Σ_j (e^{q_iᵀk_j}/L_i) · do_iᵀ v_j = do_iᵀ Σ_j (e^{q_iᵀk_j}/L_i) v_j = do_iᵀ o_i

An `N`-length reduction collapses into a `d`-length dot product **because that `N`-length sum is
already the output**. `O` is not merely saved to avoid recompute; it is the compressed form of the
softmax-gradient reduction. With `d=8` in this app, that is a 16-term reduction replaced by an
8-term one; at `d=64, N=4096` it is 4096 → 64.

Note also that the backward pass **recomputes with the stored `ℓ` and `m` directly** — Algorithm 4
line 13 is `P_ij = diag(ℓ_i)^{-1} exp(S_ij^masked − m_i)`, using the *final* row statistics, so no
online rescaling is needed on the way back. The backward pass is a plain tiled loop; only the forward
pass has a running maximum.

Theorem 5 gives the backward pass the same complexity as the forward: `Θ(Nd + N²)` for standard,
`Θ(N²d²M⁻¹)` for FlashAttention.

### Block-sparse FlashAttention

Given a mask `M̃ ∈ {0,1}^{N×N}` constrained to block form, `P = softmax(S ⊙ 1_{M̃})`, and:

> "Given a predefined block sparsity mask $M \in \{0,1\}^{N/B_r \times N/B_c}$ we can easily adapt
> Algorithm 1 to only compute the nonzero blocks of the attention matrix. The algorithm is identical
> to Algorithm 1, except we skip zero blocks."

Proposition 4:

> "Block-sparse FlashAttention (Algorithm 5) requires $\Theta(Nd + N^2d^2M^{-1}s)$ HBM accesses where
> $s$ is the fraction of nonzero blocks in the block-sparsity mask."

The sparsity multiplies only the large term; `Nd` survives as a floor. With the usual choices:

> "For large sequence lengths $N$, $s$ is often set to $N^{-1/2}$ or $N^{-1}\log N$, resulting in
> $\Theta(N\sqrt N)$ or $\Theta(N\log N)$ IO complexity. For downstream experiments, we use the fixed
> butterfly sparsity pattern."

**And it is approximate.** §3.3's opening sentence is unambiguous: *"We extend FlashAttention to
approximate attention"*. The abstract calls it *"an approximate attention algorithm that is faster
than any existing approximate attention method"*. So the paper contains both an exact algorithm and
an approximate one, and the card must not blur them — the exactness claim covers Algorithm 1 only.

---

## Numbers that matter

**Memory hierarchy (A100), the figures the whole argument rests on**

| | size | bandwidth |
|---|---|---|
| HBM | 40–80 GB | 1.5–2.0 TB/s |
| SRAM | 192 KB per SM × 108 SMs | ~19 TB/s (estimated) |

RTX 3090 for contrast, from §E.5: *"the memory bandwidth on an RTX 3090 is lower than on an A100
(roughly 900 GB/s vs. 1.5 TB/s)"* — and the speedup there is **higher** (2.5–4.5×), which is the
theory working: slower memory, bigger win. On a T4, *"T4 SRAM is smaller than A100, so we need to
make the block sizes smaller … As a result, we observe less speedup on T4, which matches the IO
complexity analysis in Section 3.2."* Smaller `M` → smaller win. That is the `M` in the theorem
showing up in a measurement.

**IO complexity (Theorem 2)**

> "Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \le M
> \le Nd$. Standard attention (Algorithm 0) requires $\Theta(Nd+N^2)$ HBM accesses, while
> FlashAttention (Algorithm 1) requires $\Theta(N^2d^2M^{-1})$ HBM accesses."

    standard        Θ(N d + N²)
    FlashAttention  Θ(N² d² M⁻¹)
    ratio (N ≫ d)   ≈ M / d²

> "For typical values of $d$ (64-128) and $M$ (around 100KB), $d^2$ is many times smaller than $M$,
> and thus FlashAttention requires many times fewer HBM accesses than standard implementation."

The proof sketch, which is the sentence to put on the card because it explains *where the `M` enters*:

> "The main idea of the proof is that given the SRAM size of $M$, we can load blocks of $K,V$ of size
> $\Theta(M)$ each. For each block of $K$ and $V$, we iterate over all blocks of $Q$ to compute the
> intermediate values, resulting in $\Theta(NdM^{-1})$ passes over $Q$. Each pass loads $\Theta(Nd)$
> elements, which amounts to $\Theta(N^2d^2M^{-1})$ HBM accesses."

`Q` is read `T_c` times. That is the entire cost. **FlashAttention does not read less data than
standard attention because it is cleverer about `Q` — it reads `Q` many times over, and still wins,
because it never writes `S` or `P` at all.**

Proposition 3 closes it off: *"There does not exist an algorithm to compute exact attention with
$o(N^2d^2M^{-1})$ HBM accesses for all $M$ in the range $[d, Nd]$."*

**Concrete traffic counts.** The paper gives asymptotics with no constants, so the card needs its own
element-by-element instantiation of Algorithms 0 and 1. **These constants are mine, derived by
counting the loads and stores each algorithm's lines actually name; the paper does not state them.**

    standard        4N² + 4Nd            elements
                    (Q,K in: 2Nd; S out, S in, P out, P in: 4N²; V in: Nd; O out: Nd)

    FlashAttention  2Nd + T_c(3Nd + 4N)  elements,  T_c = ⌈N / B_c⌉
                    (K_j,V_j in once: 2Nd; per outer pass: Q_i in, O_i in/out, ℓ_i,m_i in/out)

The dominant term is `3N²d / B_c`, and with `B_c = M/4d` that is `12N²d²/M` — `Θ(N²d²M⁻¹)`, as
required. The asymptotic ratio under this counting is `M/(3d²)`, a factor of 3 below the paper's
informal `M/d²`; both are the same order and neither should be quoted as *the* ratio.

Computed, `d = 64`, `M = 50,000` elements (≈ 100 KB of fp16), giving `B_c = 196`:

| N | standard (elements) | FlashAttention | ratio |
|---|---|---|---|
| 1024 | 4,456,448 | 1,335,296 | 3.34× |
| 2048 | 17,301,504 | 4,677,632 | 3.70× |
| 4096 | 68,157,440 | 17,383,424 | 3.92× |

The paper's own measured ratio, from the Figure 2 table, is **40.3 GB / 4.4 GB = 9.16×**, and the
introduction claims *"up to 9× fewer, as shown in Fig. 2"*. The gap between 3.3× and 9.2× is honest
slack in the constants (fp16, real tile shapes, a fused kernel that also avoids the dropout mask and
the mask matrix). **The card must present its counter as an instantiation of the paper's formula, not
as a prediction of the paper's measurement.**

**Speedups, with what each is relative to — and they are not the same measurement**

| claim | number | relative to | where |
|---|---|---|---|
| BERT-large, seq 512, end-to-end | **15% faster** (17.4 ± 1.4 min vs 20.0 ± 1.5 min, 8×A100, 10 runs, to 72.0% MLM accuracy) | Nvidia MLPerf 1.1 training-speed record | Table 1 |
| GPT-2 small, seq 1K, end-to-end | **3.5×** (2.7 days vs 9.5) / **1.7×** (vs Megatron's 4.7 days) | HuggingFace / Megatron-LM | Table 2 |
| GPT-2 medium, seq 1K, end-to-end | **3.0×** (6.9 days vs 21.0) / **1.7×** (vs Megatron's 11.5) | HuggingFace / Megatron-LM | Table 2 |
| Long-range arena, seq 1K–4K | **2.4×** (block-sparse: **2.8×**) | standard attention, same model | Table 3 |
| attention op, fwd+bwd | **up to 3×** | PyTorch attention | §4.3 |
| attention computation only | **7.6×** | PyTorch attention, GPT-2 | Fig. 1 caption |
| attention, A100 sweep | **2–4×**, *"more speedup when using dropout and masking due to kernel fusion"* | PyTorch attention | §E.5 |
| attention, head dim 128 | **up to 3×** with a causal mask, less otherwise | PyTorch attention | §E.5 |
| vs Apex FMHA (short seq) | **4% slower** at 128, **8% faster** at 256, **5% faster** at 512 | FMHA, the MLPerf kernel | Table 7 |

That last row is the most honest number in the paper and belongs on the card: against the *actual*
best-in-class fused kernel at short sequence length, FlashAttention is roughly a wash — *"Generally
FlashAttention is slightly faster than FMHA in the forward pass and slightly slower than FMHA in the
backward pass. This is because we do not store the attention matrix in the forward pass and recompute
it in the backward pass."* **The recomputation is a real cost and the paper measures it losing,
sometimes.**

**Memory**

> "FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows
> linearly with sequence length. FlashAttention is up to 20× more memory efficient than exact
> attention baselines, and is more memory-efficient than the approximate attention baselines. All
> other algorithms except for Linformer run out of memory on an A100 GPU before 64K, and
> FlashAttention is still 2× more efficient than Linformer."

**Quality, and the caveats the abstract hides**

- GPT-2 at 4K context is *"still 30% faster than GPT-2 from Megatron with context length 1K, while
  achieving 0.7 better perplexity"* — 17.5 vs 18.2.
- The "6.4 points of lift on long-document classification" is an **average of two very different
  numbers**: *"sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and … length 8K
  outperforms length 512 by 8.5 points on ECtHR."* And the underlying table is **not monotone**:
  MIMIC-III goes 52.8 → 50.7 (worse) → 51.7 → 54.6 → 56.4 → 57.1 across 512→16K; ECtHR peaks at 80.7
  at 8192 and **falls to 79.2 at 16384**. Longer is not uniformly better even in the paper's own
  headline experiment.
- Path-X 61.4 (FlashAttention), Path-256 63.1 (block-sparse), both first-ever better-than-chance.
  **Block-sparse scores 56.0 on Path-X — worse than dense FlashAttention's 61.4.** Sparsity buys
  reach, not accuracy.

**Block sizes and the point of diminishing returns**

> "As block size increases, the number of HBM accesses decreases (as we make fewer passes over the
> input), and runtime decreases. For large enough block size (beyond 256), the runtime is then
> bottlenecked by other factors (e.g., arithmetic operations). Moreover, larger block size will not
> fit into the small SRAM size."

That 256 is worth flagging next to the *different* 256 in the Session 8 lesson (the V4 sparse budget
cap) — see "Leaves behind". They are not the same 256 and the card should say so rather than let a
reader connect them.

---

## What the live view must let the reader do

**The framing problem, stated first, because everything else follows from it.** FlashAttention changes
no arithmetic. Run it on the reader's sentence and the attention grid is *the same grid*, the
dataflow picture is *the same picture*, the next-token distribution is *the same distribution*. The
app cannot demonstrate speed — there is no HBM in a browser, no SRAM, no kernel, and JavaScript
timings on a 16-token toy would measure the JIT, not the algorithm. **Any stopwatch on this card
would be a lie.** What the app *can* do, exactly and reproducibly, is two things:

1. Run the **real online-softmax recurrence** — Algorithm 1 lines 10–13, unmodified — over the
   reader's own sentence, one tile at a time, showing `m_i` and `ℓ_i` update.
2. Count **bytes moved** under both algorithms using the paper's own structure, at settable `N`, `d`
   and `B_c`.

The card's thesis sentence should be: *every earlier mechanism in this deck changed the answer to
make it cheaper; this one changes nothing about the answer and moves the cost anyway.*

**What is already available, so no seam change is needed.** `forward()` stores `Q`, `K`, `V` and
`out` on every head (`trace[b].heads[h]`), so a tiled recomputation can run straight off
`head.Q, head.K, head.V` and be compared against `head.out`. **This card must not be a mixer.** A
mixer producing identical output would be invisible in every existing panel and would add a
configuration branch that does nothing — the right shape is a standalone `tiledAttention(Q,K,V,dh,Bc)`
in the card file returning `{ out, tiles }`, where `tiles` is the per-step log the view animates.

**Three implementation facts that must not be papered over.**

- **The causal mask makes rows that are entirely `−∞` inside a tile.** With `B_r = T` there is one
  row-block, so the skip is **per row, not per tile**: for query row `i` and key tile `[j₀, j₁)` with
  `j₀ > i`, every score in that row of the tile is `−∞`, so `m̃_ij = −∞`, and line 10's
  `exp(S_ij − m̃_ij)` evaluates `exp(−∞ − (−∞)) = exp(NaN) = NaN`. Line 11 then poisons `ℓ_i` for
  good. **Verified: this fires on the very first configuration you would try.** The fix is the one
  the real kernel uses — when `m̃_ij = −∞` for a row, contribute `ℓ̃ = 0` and leave `m_i` and `O_i`
  alone. Guard the `m_i = −∞` initial state the same way (`e^{−∞ − m_new}` must be forced to `0`,
  not left to produce `NaN` when `m_new` is also `−∞`). At `T = 16, B_c = 4` this fires for **24 of
  the 64 (row, tile) pairs** — 4 rows in the `[4,8)` tile, 8 in `[8,12)`, 12 in `[12,16)`. Show it
  rather than hide it: a "skipped — all scores −∞" label is the honest visual bridge to block-sparse
  FlashAttention, which skips work for a different reason using the same machinery.
- **`B_r = min(⌈M/4d⌉, d)` should not be inherited.** With `d = d_k = 8`, the paper's own formula
  caps query tiles at 8 rows regardless of SRAM. Fix `B_r = T = 16` (all rows at once, which is what
  the per-row display wants) and expose only `B_c`. Say on screen that `B_r` is being overridden and
  why.
- **`M` is stated in bytes in §2.1 (`192KB`, `around 100KB`) and used as a count of elements in
  Theorem 2** (`d ≤ M ≤ Nd` only makes sense in elements). The counter must pick one, label it, and
  not silently switch. Recommend: **elements**, with a note that 50,000 elements ≈ 100 KB at fp16.

---

**1. The tile walk — the headline, and the only place the real mechanism is visible.**

A tile-size control `B_c ∈ {1, 2, 4, 8, 16}` and a step/play control that advances one `(i,j)` tile at
a time over the reader's sentence, for one selected head. The display, per step:

- The `16 × 16` causal score grid with the current key tile `[j₀, j₁)` highlighted as a column band,
  and everything to the right of it greyed as "not yet seen".
- A **live table of the 16 running rows**: `m_i`, `ℓ_i`, and `‖O_i‖`, with the two correction factors
  `e^{m_i − m_i^new}` and `e^{m̃_ij − m_i^new}` printed for the rows that actually changed this step.
- The partial output row for a reader-selected query token, shown as 8 numbers, next to the final
  correct row.

- **The number that proves it:** for the selected row, print `m_i` before and after each tile, and
  the correction factor. **When a later tile contains a larger score, the factor is `< 1` and every
  number in `ℓ_i` and `O_i` shrinks at that instant.** That visible shrink *is* the online softmax;
  nothing else on this card is the mechanism. Label it "the rescale".
- **The second number:** the count of rows for which `m_i` changed on this step. It starts at 16 and
  falls toward 0 as the walk proceeds — the running maximum stabilises. A reader who sees that
  understands why the trick works at all.
- Fully-masked tiles must render as **"skipped — all scores −∞"**, with a counter of how many were
  skipped. At `T=16, B_c=4, B_r=16` with a causal mask that is **24 of the 64 (row, tile) pairs**.

**2. The identity assertion — computed, not claimed, and honestly reported.**

A permanent readout comparing the tiled result against the one-shot result the app already computed:

    maxAbs = max over all 16 tokens and all 8 dims of | O_tiled[i][d] − head.out[i][d] |

- **The number that proves it, and the honest reading of it — measured, not predicted:**

  | `B_c` | tiles per row | maxAbs |
  |---|---|---|
  | 1 | 16 | 2.22e−16 |
  | 2 | 8 | 2.22e−16 |
  | 4 | 4 | 2.22e−16 |
  | 8 | 2 | 1.67e−16 |
  | 16 | 1 | 1.11e−16 |

  (float64 epsilon is `2⁻⁵² = 2.22e−16`; output magnitudes are `O(1)`.)

- **The card must say, in words: this is not zero, and it will never be zero.** It is one unit in the
  last place. Rescaling changes the *order* of the additions, and floating-point addition is not
  associative, so a different order gives a different last bit. "Exact" in the paper's title is an
  **algebraic** claim — Theorem 1 is an identity over the reals — and the app is in the rare position
  of being able to display exactly how big the gap between algebra and arithmetic is. Print the ratio
  `maxAbs / (2⁻⁵² · max|O|)` so the reader sees it is ~1 ulp and not "small-ish".
- Contrast it deliberately with the Performer card's readout, which reports `maxAbs` still visibly
  nonzero at every `m` and *falling only as `1/√m`* — that note's measured figures are `0.007` at
  small query/key norms and `0.99` at `m = 2048` at this untrained model's actual norms. Same metric,
  same sentence, same head: **approximation error between `10⁻²` and `10⁰`; exact-with-tiling error
  `~10⁻¹⁶`, and it does not depend on the tile size.** Those two numbers, side by side, are the
  entire argument of this paper.

**3. The traffic counter — where the cost actually moved.**

Three inputs: sequence length `N` (log slider, 16 → 65536), head dimension `d` (16/32/64/128, the
paper's supported set), SRAM size `M` in elements (slider, with the fp16 byte equivalent shown). It
derives `B_c = ⌈M/4d⌉` and `T_c = ⌈N/B_c⌉`, then displays both counts, the ratio, and the two
asymptotic expressions with the current values substituted.

- **The number that proves it:** at `N = 4096, d = 64, M = 50,000`: **68,157,440 vs 17,383,424
  elements, 3.92×**. At `N = 1024` the same settings give **3.34×** — the ratio *rises with `N`*,
  which is the `N²` term dominating the `Nd` term, and is the one asymptotic fact a reader can watch
  happen.
- **The number that proves the `M` dependence, which is the paper's real claim:** halve `M` from
  50,000 to 25,000 at `N=1024, d=64` and the traffic goes 1,335,296 → 2,338,816 (ratio 3.34× →
  1.91×). Put the T4/A100/3090 sentences from §E.5 next to that slider — *"T4 SRAM is smaller than
  A100, so we need to make the block sizes smaller … we observe less speedup on T4, which matches the
  IO complexity analysis"*. **Slider position ↔ quoted measurement.**
- Show the paper's measured pair as a fixed anchor row that the counter does *not* try to reproduce:
  `GPT-2 medium, N=1024, d=64, 16 heads, batch 64, A100: 40.3 GB → 4.4 GB (9.16×), 41.7 ms → 7.3 ms`.
  Label it "measured, fp16, fused kernel" against the counter's "counted from Algorithms 0 and 1".
- **A permanent one-line disclaimer, not a footnote:** *this counts elements moved, not time. The app
  has no HBM. The paper's claim is that the first causes the second; the app can only show the first.*

**4. The honesty panel — at this sentence's scale, tiling loses.**

Fix `N = 16, d = 8` (the reader's actual model) and sweep `B_c`:

| `B_c` | `T_c` | FlashAttention | standard | verdict |
|---|---|---|---|---|
| 1 | 16 | 7,424 | 1,536 | **4.8× worse** |
| 2 | 8 | 3,840 | 1,536 | **2.5× worse** |
| 4 | 4 | 2,048 | 1,536 | **1.3× worse** |
| 8 | 2 | 1,152 | 1,536 | 1.33× better |
| 16 | 1 | 704 | 1,536 | 2.18× better — but `T_c = 1` is *no tiling* |

- **The number that proves it:** at every tile size that is genuinely a tiling of this sentence, the
  algorithm moves **more** data, because `Q` is re-read `T_c` times and `Q` here is only `16 × 8 =
  128` numbers — smaller than the `N² = 256`-element score matrix it is avoiding, but not by enough
  to pay for 16 passes. The break-even is where `4N² ≈ 3N²d/B_c`, i.e. `B_c ≈ 0.75 d`; with `d = 8`
  that is `B_c ≈ 6`. **Print that inequality live.**
- And the punchline: with a real `M` and the paper's own `B_c = ⌈M/4d⌉`, `d = 8` gives `B_c ≥ 3000`
  for any plausible SRAM, so **`T_c = 1` and the paper's algorithm would not tile this sentence at
  all.** The reader is being shown a mechanism that their own model does not need. Say so. That is
  the same shape of honesty as the Performer card's crossover panel, and it is what makes the
  hardware pressure legible: FlashAttention is not a better algorithm, it is the *same* algorithm
  arranged for a machine with a specific `M`, and if `N²d` fits in SRAM there is nothing to arrange.

**5. Block sparsity, reusing the tile walk — the bridge to what comes next.**

A toggle that overlays a block-sparsity mask on the tile grid from interaction 1: `dense` / `causal
only` / `local (butterfly-ish band)`, at the current `B_c`. Skipped blocks are drawn hollow.

- **The number that proves it:** `s` = fraction of non-zero blocks, displayed, and the traffic counter
  switching from `2Nd + T_c(3Nd+4N)` to Proposition 4's `Θ(Nd + N²d²M⁻¹s)` — showing the `Nd` floor
  that sparsity **cannot** remove. At high sparsity the counter stops falling, and the label should
  read "floor: `Nd`, the inputs must be read once".
- **The number that proves the cost:** carry `maxAbs` from interaction 2 into this view. Under dense
  tiling it is `~1e−16`. Under a sparsity mask it jumps by fourteen or more orders of magnitude
  (not measured here — the exact figure depends on which blocks the pattern drops, and the card
  should print whatever it actually computes rather than a number from this note) — because
  **block-sparse
  FlashAttention is not exact.** Put §3.3's own first sentence on screen: *"We extend FlashAttention
  to approximate attention"*. One metric, one toggle, and the reader watches the paper cross its own
  line. This is the single most valuable thing this card can teach, and it costs almost nothing to
  build on top of interaction 1.

---

## What the source does *not* establish

- **It does not claim bit-identical output, and the app must not either.** Theorem 1 is an identity
  over the reals. The paper's own §4.1 sentence *"FlashAttention achieves the same perplexity as the
  other two implementations, as we do not change the model definition"* is contradicted by the table
  immediately below it: **GPT-2 medium is 14.2 for HuggingFace and 14.3 for FlashAttention.** (It is
  also 14.3 for Megatron, so the difference is not evidence against FlashAttention — it is evidence
  that "the same" means "the same to within run-to-run noise", not "the same numbers".) The card's
  measured `1.1e−16 … 2.2e−16` is the honest version of this claim.
- **The IO-complexity ratio is stated loosely and cannot be pinned down from the paper.** `M` is
  "around 100KB" (bytes) in §2.1 but must be an element count in Theorem 2 for `d ≤ M ≤ Nd` to
  typecheck. Reading it as 50,000 fp16 elements gives `M/d² ≈ 12`; as 25,000, ≈ 6; the measurement is
  9.16×; my line-by-line count gives 3.3–3.9×. **There is no single "the ratio", and a card that
  prints one as if the paper had is inventing it.**
- **Headline speedups are inconsistent between the abstract and the tables.** The abstract says "3×
  speedup on GPT-2"; Table 2 says 3.5× (small) and 3.0× (medium). The introduction says "1.8× over
  Megatron"; §4.1 says "1.7×"; Table 2's medium row is 11.5/6.9 = 1.67×. Quote the tables.
- **"Speedup" means at least four different measurements.** End-to-end training (15%, 3×, 2.4×), the
  attention op forward+backward (up to 3×), the attention computation alone (7.6×), and the
  cross-hardware sweep (2–4×). Any card that reports "FlashAttention is 7.6× faster" without saying
  *what* is 7.6× faster is repeating the largest number in the paper out of context.
- **It does not show FlashAttention beating the best existing fused kernel at short sequences.**
  Table 7: 4% *slower* at 128. The advantage begins where memory pressure begins.
- **It does not show that longer context monotonically helps.** Table 5's MIMIC-III row *falls* from
  512 to 1024, and ECtHR falls from 8192 to 16384. The "6.4 points" in the abstract is an average of
  4.3 and 8.5 across two datasets, and the paper itself offers only a speculation for the gap
  (*"may be due to subtle distribution shifts"*).
- **Block-sparse FlashAttention is approximate and is not covered by the exactness claim, the
  optimality claim, or Theorem 1.** It also scores *worse* than dense FlashAttention on Path-X (56.0
  vs 61.4). Its butterfly pattern is fixed before training — the same objection this deck already
  raised against Sparse Transformer and Longformer, restated here by the paper that claimed to make
  approximation unnecessary.
- **Proposition 3's lower bound is weaker than it sounds.** It says no algorithm achieves
  `o(N²d²M⁻¹)` **for all `M` in `[d, Nd]`**, and the proof works by pinning `M = Θ(Nd)`. The paper
  says as much: *"This type of lower bound over a subrange of $M$ is common in the streaming
  algorithms literature. We leave proving parameterized complexity lower bounds in terms of $M$ as
  exciting future work."* It does not establish optimality at any particular `M` — including the `M`
  of an actual A100.
- **The whole result is an artifact of one hardware ratio and the paper knows it.** §5: *"Our current
  approach to building IO-aware implementations of attention requires writing a new CUDA kernel for
  each new attention implementation. … Implementations may also not be transferrable across GPU
  architectures."* Change the SRAM-to-HBM ratio and every number moves; §E.5 measures exactly that
  across A100/3090/T4.
- **Nothing here is about inference.** Every headline number is a training measurement (forward +
  backward). The T4 forward-only plot is the only nod to it. The KV-cache pressure that Concepts 7
  and 15 are about is untouched by this paper.

---

## Leaves behind

**Backward — every mechanism before this one changed the mathematics to dodge the cost. This one asks
whether the mathematics was ever the problem.**

Trace the deck to here. Sparse Transformer (6) legislated which cells to compute. Longformer (8)
chose a window, a dilation and a set of global tokens. Linear attention (9) removed the exponential
so the sum could be regrouped. Performer (10) put the exponential back as an unbiased estimate with
`m` random features and an error falling as `1/√m`. **Four different mechanisms, four different
mathematical objects, four different answers — every one of them a change to what attention
computes.** Each was justified by a FLOP count.

FlashAttention computes the *original* object, bit-for-bit-in-the-limit, and is faster than three of
those four on the paper's own LRA table. The introduction's sentence — *"they focus on FLOP reduction
(which may not correlate with wall-clock speed) and tend to ignore overheads from memory access
(IO)"* — is a claim that the entire preceding branch of the deck was **optimising the wrong
quantity**. The quadratic that mattered was never `N²` multiply-accumulates; it was `N²` elements
written to HBM and read back four times. Softmax is a reduction, reductions are memory-bound, and
`P = softmax(S)` was being round-tripped through 1.5 TB/s memory for no reason at all.

The specific inheritance from Concept 10 is sharp enough to state as a swap. Performer's card ends on
a trade: error `1/√m` against cost `2·T·m·d_k`, with no setting of `m` that makes the error zero.
FlashAttention's card ends on: error `1 ulp`, cost strictly lower, and a slider (`B_c`) that trades
**nothing** — every position gives the same answer. The only thing `B_c` buys is traffic. Put the two
`maxAbs` readouts side by side and the argument needs no prose.

There is a second backward thread, to the very first card. Concept 1 left "two bills — a score matrix
that grows with the square of the context, and a key/value cache that grows with every token
generated". This card pays part of the *first* bill without approximating anything, and does not
touch the second at all. Attention is still `O(N²)` in FLOPs after FlashAttention. It is `O(N)` in
memory and roughly `O(N²d²/M)` in traffic. **The quadratic did not go away; it stopped being paid in
the expensive currency.**

**Forward — a fast exact kernel changes what sparsity is for, and then constrains what sparsity can
be.**

Two consequences, and they are both live in Session 8's own material:

1. **It moved the goalposts for every approximate method after 2022.** The paper's own benchmarking
   sets the new bar: *"Up to sequence length of 512, FlashAttention is both faster and more
   memory-efficient than any existing attention method, whereas for sequence length beyond 1K, some
   approximate attention methods (e.g., Linformer) start to become faster."* After this, an
   approximation is not competing with naive PyTorch attention; it is competing with an exact
   algorithm that is 2–4× faster than the thing it was benchmarked against. A method that trades
   quality for a 2× FLOP reduction now buys nothing. This is why the deck's centre of gravity shifts
   after 2022 from *approximating attention* to *shrinking what generation must remember* (GQA, MLA,
   attention sinks) and to *sparsity that is learned rather than legislated* (NSA, DSA).
2. **The kernel becomes the constraint on what sparsity is affordable.** Block-sparse FlashAttention
   already announces the shape of it: sparsity has to be **block** sparsity, aligned to `B_r × B_c`,
   because that is what a tiled kernel can skip. An arbitrary per-token mask cannot be skipped; only
   whole tiles can. Every later sparse-attention design inherits that: the pattern must be
   block-structured, and the *backward* kernel is the tighter constraint, because the backward pass
   is where the recomputation lives and where `dK`/`dV` accumulate across the inner loop.

   The Session 8 lesson records exactly this bill coming due: *"LightningLM V4 used sparse-attention
   G-layers alongside DeltaNet layers. Its maximum budget was reduced from 1024 to 256 because of
   backward-kernel contention in that particular hardware and software stack."* The lesson is careful
   to add that this *"makes 256 an implementation constraint from that run, not a universal law of
   sparse attention"* — and its own open-questions table lists the fix as *"Re-measurement on the
   current kernels and current hardware rather than inheritance"*. That is this card's pressure
   arriving three years later and one layer up: **a top-k budget chosen not by what the model needs
   but by what the backward kernel will tolerate.**

   Two warnings for whoever builds the NSA/DSA cards. First, do not conflate that 256 with §3.2's
   *"For large enough block size (beyond 256), the runtime is then bottlenecked by other factors"* —
   one is a sparse *budget* in tokens on a 2025 stack, the other is a *tile width* in a 2022 forward
   kernel, and their agreement is a coincidence. Second, FlashAttention is the reason the constraint
   is a *kernel* constraint at all: before it, attention was a sequence of library calls and any mask
   was as cheap as any other; after it, attention is one fused kernel and the mask has to be
   expressible in tiles.

The pressure this card names is **how it meets the hardware**, and it is the first card in the deck
whose mechanism is invisible in the model's output. Every card before it changed what the model
computes and could be shown in the attention grid. This one changes only where the numbers live on
the way to the same answer — which is why its live view is a traffic counter and a `1e−16`, and why
the card's honesty about what it cannot show is the card.
