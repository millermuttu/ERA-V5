# Concept 21 — Multi-head latent attention

**Card id:** `mla` · **Date:** 2024-05-07 (arXiv v1) · **Pressure:** what generation must remember

## What was read

- [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), DeepSeek-AI (157 listed contributors) —
  *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. Abstract page
  for the version history; the full v5 HTML (`arxiv.org/html/2405.04434v5`) pulled raw and read end
  to end for §2.1, §3.1.4, §3.2.3, §5 and **Appendices B, C and D**, which is where the mechanism's
  full formulas and its only ablations live.
- **Version history, verified.** v1 **Tue, 7 May 2024, 15:56:43 UTC** (431 KB); v2 8 May 2024; v3 16
  May 2024; v4 24 May 2024; v5 **19 Jun 2024** (432 KB). The app's record of `2024-05-07` is
  **correct**. Version sizes move by 1 KB across the whole series, so unlike concept 18 this paper
  did not change between versions in any way this card depends on.
- **What is and is not being dated here.** MLA is one of two architectural contributions in a paper
  about a 236B mixture-of-experts model. Every headline number in the abstract — 42.5% training
  cost, 93.3% KV cache, 5.76× throughput — is **DeepSeek-V2 against DeepSeek 67B**, which differ in
  the attention *and* in the feed-forward *and* in the deployment precision. The only MLA-versus-
  something comparisons in the paper are Tables 8 and 9, in an appendix, and they are the numbers
  this card should lead with.

## The mechanism, precisely

### Step 0 — the bill this pays, in the paper's own framing

§2.1's opening establishes the problem exactly where concept 15 left it:

> "Conventional Transformer models usually adopts Multi-Head Attention (MHA) (Vaswani et al., 2017),
> but during generation, its heavy Key-Value (KV) cache will become the bottleneck that limit the
> inference efficiency. In order to reduce the KV cache, Multi-Query Attention (MQA) (Shazeer, 2019)
> and Grouped-Query Attention (GQA) (Ainslie et al., 2023) are proposed. They require a smaller
> magnitude of KV cache, but their performance does not match MHA (we provide the ablation of MHA,
> GQA and MQA in Appendix D.1)."

and states the claim:

> "Equipped with low-rank key-value joint compression, MLA achieves better performance than MHA, but
> requires a significantly smaller amount of KV cache."

**Both halves of that sentence are load-bearing and only one of them is obvious.** Cheaper than MHA is
arithmetic. *Better* than MHA is an empirical claim resting on Table 9, and Table 9 has one cell that
goes the other way (below).

### Step 1 — the compression itself

Standard MHA, §2.1.1, with `d` the embedding dimension, `n_h` the heads, `d_h` the per-head dimension
and `h_t ∈ ℝ^d` the layer's input for token `t`:

    q_t = W^Q h_t,   k_t = W^K h_t,   v_t = W^V h_t          W^Q, W^K, W^V ∈ ℝ^{d_h n_h × d}
    o_{t,i} = Σ_{j≤t} Softmax_j( q_{t,i}^T k_{j,i} / √d_h ) v_{j,i}
    u_t = W^O [o_{t,1}; …; o_{t,n_h}]

MLA replaces the key and value projections with a **joint** low-rank pair (§2.1.2):

    c_t^{KV} = W^{DKV} h_t          c_t^{KV} ∈ ℝ^{d_c},  d_c ≪ d_h n_h
    k_t^C    = W^{UK}  c_t^{KV}
    v_t^C    = W^{UV}  c_t^{KV}

One vector `c_t^{KV}` is cached. Both the keys and the values for **every head** are reconstructed
from it. The queries get their own compression, for training memory rather than for the cache:

    c_t^Q = W^{DQ} h_t              c_t^Q ∈ ℝ^{d'_c}
    q_t^C = W^{UQ} c_t^Q

> "In order to reduce the activation memory during training, we also perform low-rank compression for
> the queries, even if it cannot reduce the KV cache."

### Step 2 — the trick that makes it cheap, and it is an identity, not an approximation

Appendix C, after the full formulas:

> "During inference, the naive formula needs to recover **k**_t^C and **v**_t^C from **c**_t^{KV} for
> attention. Fortunately, due to the associative law of matrix multiplication, we can absorb W^{UK}
> into W^{UQ}, and W^{UV} into W^O. Therefore, we do not need to compute keys and values out for each
> query."

(§2.1.2 says "absorbed into W^Q"; Appendix C's `W^{UQ}` is the precise version. Use the appendix.)

The identity is one line and it is checkable:

    q^T k = q^T (W^{UK} c) = (W^{UK⊤} q)^T c

The query is `n_h · d_h`-sized and is computed once per step; the cached object is `d_c`-sized and
there is one per token in the context. Moving the up-projection onto the query side means the
per-token work never touches a `d_h`-sized key at all. **The cache is not a compressed key that gets
decompressed. It is an object the query has been reshaped to talk to directly.** That distinction is
what separates this from every "just store fewer numbers" scheme, and it is why the compression costs
no extra arithmetic at generation time.

### Step 3 — why RoPE breaks it, in the paper's own words

> "If we apply RoPE for the keys **k**_t^C, W^{UK} in Equation 10 will be coupled with a
> position-sensitive RoPE matrix. In this way, W^{UK} cannot be absorbed into W^Q any more during
> inference, since a RoPE matrix related to the currently generating token will lie between W^Q and
> W^{UK} and matrix multiplication does not obey a commutative law."

That is the whole obstruction. Concept 12's rotation is applied to the key *after* it is built, so the
product the absorption needs to reassociate has a position-dependent matrix wedged in the middle of
it, and it is a different matrix for every cached token.

**The fix is to give position its own small channel** (§2.1.3):

    q_t^R = RoPE( W^{QR} c_t^Q )        per head, dimension d_h^R
    k_t^R = RoPE( W^{KR} h_t )          ONE vector, shared by all heads
    q_{t,i} = [ q_{t,i}^C ; q_{t,i}^R ]
    k_{t,i} = [ k_{t,i}^C ; k_t^R ]
    o_{t,i} = Σ_{j≤t} Softmax_j( q_{t,i}^T k_{j,i} / √(d_h + d_h^R) ) v_{j,i}^C

> "During inference, the decoupled key should also be cached. Therefore, DeepSeek-V2 requires a total
> KV cache containing `(d_c + d_h^R) l` elements."

So the head's score is a sum of two dot products: a **content** term computed against the latent, in
which the absorption works because nothing positional is in the way, and a **position** term computed
against a small shared rotated vector, which is not compressed at all and is simply cached as-is. The
decoupled key is `d_h^R` numbers per token per layer for the *whole layer*, not per head — which is
why it costs so little. Every head reads the same one.

### Step 4 — what the cache costs, Table 1

| Attention Mechanism | KV Cache per Token (# Element) | Capability |
|---|---|---|
| Multi-Head Attention (MHA) | `2 n_h d_h l` | Strong |
| Grouped-Query Attention (GQA) | `2 n_g d_h l` | Moderate |
| Multi-Query Attention (MQA) | `2 d_h l` | Weak |
| **MLA (Ours)** | `(d_c + d_h^R) l ≈ (9/2) d_h l` | **Stronger** |

> "The amount of KV cache is measured by the number of elements, regardless of the storage precision.
> For DeepSeek-V2, `d_c` is set to `4 d_h` and `d_h^R` is set to `d_h/2`. So, its KV cache is equal to
> GQA with only 2.25 groups, but its performance is stronger than MHA."

**"2.25 groups" is the sentence to hang the card on**, because it makes the comparison concrete and
because the paper never actually runs it — there is no GQA-at-2.25-groups row anywhere. The app can
run the matched-budget comparison the sentence implies. See `[measured here]`.

Note the "Capability" column has no units, no metric and no citation. It is an assertion in a table,
and *Stronger* for MLA rests on Table 9's four numbers.

### Step 5 — the configuration, and a second data point

DeepSeek-V2, §2 and the architecture table: **60 layers**, `d = 5120`, `n_h = 128`, `d_h = 128`,
`d_c = 512`, `d'_c = 1536`, `d_h^R = 64`, context 128K, 236B total / 21B activated per token.

DeepSeek-V2-Lite, Appendix B.1: **27 layers**, `d = 2048`, 16 heads, `d_h = 128`, `d_c = 512`,
`d_h^R = 64`, context 32K, 15.7B total / 2.4B activated. And a design note worth carrying:

> "Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress
> the queries."

So the query compression is optional and was dropped at the smaller size — confirming it is a
training-memory optimisation and not part of the cache mechanism.

## Numbers that matter

### Table 8 — the premise, and it is measured

7B dense models on 1.33T tokens, layer count adjusted so the parameter counts match, everything but
the attention identical:

| Benchmark (Metric) | # Shots | w/ MQA (7.1B) | w/ GQA, 8 groups (6.9B) | w/ MHA (6.9B) |
|---|---|---|---|---|
| BBH (EM) | 3-shot | 33.2 | 35.6 | **37.0** |
| MMLU (Acc.) | 5-shot | 37.9 | 41.2 | **45.2** |
| C-Eval (Acc.) | 5-shot | 30.0 | 37.7 | **42.9** |
| CMMLU (Acc.) | 5-shot | 34.6 | 38.4 | **43.5** |

> "MHA demonstrates significant advantages over GQA and MQA on these benchmarks."

This is the strongest published statement of concept 15's cost that this deck has: GQA at 8 groups is
**4.0 points of MMLU** and **5.2 of C-Eval** below MHA at matched parameters and matched data. The GQA
paper's own ablation measured a much smaller gap; this one is on hard knowledge benchmarks rather
than summarisation, and it is the reason DeepSeek did not simply use GQA.

### Table 9 — the claim, and the cell nobody quotes

MoE models, two scales, everything but the attention identical. Small: ~16B total, 1.33T tokens.
Large: ~250B total, 420B tokens.

| Benchmark (Metric) | # Shots | Small w/ MHA | Small w/ MLA | Large w/ MHA | Large w/ MLA |
|---|---|---|---|---|---|
| # Activated Params | — | 2.5B | 2.4B | 25.0B | 21.5B |
| # Total Params | — | 15.8B | 15.7B | 250.8B | 247.4B |
| **KV Cache per Token (# Element)** | — | **110.6K** | **15.6K** | **860.2K** | **34.6K** |
| BBH (EM) | 3-shot | 37.9 | **39.0** | 46.6 | **50.7** |
| MMLU (Acc.) | 5-shot | 48.7 | **50.0** | 57.5 | **59.0** |
| C-Eval (Acc.) | 5-shot | **51.6** | 50.9 | 57.9 | **59.2** |
| CMMLU (Acc.) | 5-shot | 52.3 | **53.4** | 60.7 | **62.5** |

> "MLA shows better performance than MHA. More importantly, MLA requires a significantly smaller
> amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA."

**Seven of eight cells go MLA's way and one does not** — small-scale C-Eval, 50.9 against 51.6. The
paper's summary sentence does not mention it. It is a 0.7-point loss on one benchmark at one scale
and it does not overturn the result, but a card that repeats "better than MHA" without it is
repeating a summary rather than a measurement.

**The cache ratios are the real headline and they are cleaner than the abstract's.** 110.6K → 15.6K is
**7.1×** at 16B; 860.2K → 34.6K is **24.9×** at 250B. Both are MLA against MHA *with everything else
held fixed*, which is exactly what the abstract's 93.3% is not.

### The abstract's three numbers, and what each actually compares

> "Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile
> saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation
> throughput to 5.76 times."

- **42.5% training cost** — §3.2.3: 300.6K GPU-hours per trillion tokens for DeepSeek 67B against
  172.8K for DeepSeek-V2. That is a **mixture-of-experts** result (21B activated of 236B), not an
  attention result. MLA contributes nothing to it.
- **93.3% KV cache** — a dense 67B model's MHA cache against a 236B MoE model's MLA cache. Different
  layer counts, different head counts, different everything. Table 9's 4% is the controlled version.
- **5.76× throughput** — §3.2.3, and the conditions matter: parameters converted to **FP8**, KV cache
  **quantised to 6 bits per element on average**, on **a single node with 8 H800 GPUs**, using "the
  prompt and generation length distribution from the actually deployed DeepSeek 67B service". The
  mechanism of the win is stated plainly — a smaller cache "can serve a much larger batch size" — so
  like concept 20's throughput number this is a batch-size win, and it is stacked on two
  quantisation decisions that have nothing to do with MLA.

None of the three is wrong. All three are DeepSeek-V2-versus-DeepSeek-67B, and the card should print
Table 9's controlled ratios beside them.

### The link to concept 18, which is not decoration

§3.1.4, on extending 4K to 128K:

> "we employ YaRN (Peng et al., 2023) to extend the default context window length from 4K to 128K.
> YaRN was specifically applied to the decoupled shared key **k**_t^R as it is responsible for
> carrying RoPE."

> "For YaRN, we set the scale `s` to 40, `α` to 1, `β` to 32, and the target maximum context length to
> 160K."

> "Slightly diverging from original YaRN, due to our distinct attention mechanism, we adjust the
> length scaling factor to modulate the attention entropy. The factor `√t` is computed as
> `√t = 0.0707 ln s + 1`, aiming at minimizing the perplexity."

Three things fall out. `α = 1, β = 32` are **exactly** concept 18's recommended constants, reused
without change. The **only** place YaRN is applied is the decoupled key — the compressed content
channel has no position in it at all, so there is nothing there to interpolate. And the temperature
coefficient is **re-fitted from 0.1 to 0.0707** because the attention geometry changed. That last one
is the sharpest possible evidence for concept 18's own caveat that its `0.1` is a curve fit with no
theory behind it: the first architecture to change the score's shape had to fit a new one.

Also: 1000 steps at sequence length 32K, batch 576, and the model then evaluated at 128K.

## `[measured here]` — this app, 32 dims, 4 heads, `d_h = 8`, 2 blocks, seed 20260817

Driven by node against `app/model/transformer.js` with a `latent` hook added to `forward`. The
compression used is the **best possible** rank-`d_c` joint factorisation of this block's own
`[W^K | W^V]` map — obtained by eigendecomposing `M Mᵀ` and keeping the top `d_c` directions. A
trained `W^{DKV}`/`W^{UK}` pair cannot beat it, so every error below is a **lower bound** on what a
real MLA layer of that width would lose.

**1. The absorption identity holds exactly.** Head 0, last query, `d_c = 16`: computing the score the
naive way (build `k = W^{UK⊤}c`, then `q·k`) gives **6.236172754003**; computing it the absorbed way
(`(W^{UK}q)·c`, never materialising a key) gives **6.236172754003**. Difference **8.88e-16**. This is
the one part of the mechanism a toy settles completely, and it is the part that matters most, because
it is why the compression is free at generation time rather than merely small.

**2. RoPE breaks it, and not by a little.** Same head, key at position 3, query at position 15. With
the rotation applied to the key, the true score is **3.630803**. Absorbing anyway — reassociating as
if the rotation were not there — gives **0.908332**, an error of **2.72**. The relative size of the term
that will not move past the rotation is **87.5%** across 200 random vectors, and **63.5%** measured on
this model's own key vectors — the card uses the second, since it is a statement about this model. It is not a small
perturbation to be waved through; the absorbed form computes a different function.

**And the decoupled design does not repair that number — it avoids needing to.** The decoupled score
is not an approximation of "rotate the key and absorb anyway"; it is a different function, computed by
a model whose content channel never had a rotation in it. On the app's own numbers the two are
3.63 and 8.32 and there is no reason for them to agree. The card must say so, because a readout
showing two scores side by side invites exactly the wrong reading.

**3. The comparison the paper asserts and never runs: matched cache budget.** Both MLA and GQA are
linear compressions of the same object — the map from the layer's input to its keys and values. So
they can be put on one axis, as the fraction of that map each destroys, against what each costs to
cache. Relative Frobenius error of the reconstructed `[W^K | W^V]`:

| cache elements / token / layer | GQA or MQA | error | MLA | error |
|---|---|---|---|---|
| 16 | MQA (1 group) | **86.73%** | `d_c = 12` | **57.87%** |
| 32 | GQA (2 groups) | **69.67%** | `d_c = 28` | **13.41%** |
| 64 | MHA (4 groups) | 0.00% | `d_c = 32` (+4) = 36 | 0.00% |

**At every matched budget the latent keeps more of the same map, and the gap is enormous.** The reason
is structural rather than empirical: GQA is forced to average a *fixed grouping* of heads, while the
latent is free to choose whichever subspace carries the most, and the best `d_c`-dimensional subspace
is by construction at least as good as any grouping of the same size. This is the paper's "equal to
GQA with only 2.25 groups, but stronger" claim, reproduced as linear algebra on an untrained toy.

Two caveats to print with it. The app's weights are random, so the map's spectrum is nearly flat — the
top 16 of 32 directions hold only **78.52%** of the energy, where a trained projection is far more
low-rank and MLA would do proportionally better. And at this width `d_c = 32 = d`, so the truncation
is vacuous and the compression is *exactly free*, which is an artefact of a 32-dimensional model, not
a property of the method.

**4. The cache arithmetic, at both widths.** Per token per layer:

| | this app (`n_h=4, d_h=8`) | DeepSeek-V2 (`n_h=128, d_h=128, l=60`) |
|---|---|---|
| MHA | 64 | 1,966,080 |
| GQA, 2 groups | 32 | — |
| MQA | 16 | — |
| MLA (`d_c=16`, `d_h^R=4`) | 20 | (`d_c=512`, `d_h^R=64`) → 34,560 |
| MLA against MHA | **3.2×** | **56.9×** |
| equivalent GQA groups | 1.25 | **2.25** ✓ reproduces the paper's caption |

**MLA's saving is proportional to the head count**, and a four-head toy can barely show it — 3.2×
here against 56.9× at 128 heads. This must be stated on the card; it is the single largest gap
between what the app can demonstrate and what the mechanism is for.

**5. On the reader's own sentence.** Running the model with the rank-`d_c` latent in place, against
plain MHA, the next-token distribution's KL divergence falls monotonically once the rank is past the
noise floor: `d_c = 12 → 2.41`, `16 → 0.79`, `24 → 0.20`, `32 → 0.0000`. Below 12 the values are
noise (2.6 to 4.4, unordered) — an untrained model's output distribution is arbitrary, so the small-
rank end of that curve should be read as "destroyed" rather than as a ranking. GQA at matched cache:
1 group → 2.73, 2 groups → 0.77.

## What the source does *not* establish

- **No ablation of the mechanism's own parts.** There is no sweep of `d_c`, none of `d_h^R`, no
  comparison of joint against separate key/value compression, and no experiment removing the query
  compression. `d_c = 4 d_h` and `d_h^R = d_h/2` appear as settings, never as findings. Appendix B
  shows the query compression was simply dropped in the Lite model with no measurement either way.
- **No experiment on the decoupled position channel.** The RoPE incompatibility is argued
  algebraically and the fix is presented — but nothing measures what the fix costs. There is no row
  for "MLA with position folded in and the absorption abandoned", which would be the natural control,
  and no measurement of how much a single shared rotated key of 64 dimensions can carry compared with
  a per-head one.
- **"Stronger" in Table 1 is a word, not a measurement**, and the measurement behind it (Table 9) is
  four benchmarks at two scales, single runs, with one cell going the other way.
- **The three headline numbers are not MLA's.** 42.5% is the mixture of experts; 93.3% and 5.76× are
  a 236B MoE against a dense 67B, with FP8 weights and 6-bit cache quantisation stacked on top of the
  throughput figure. The paper is not misleading about this — §3.2.3 states the conditions — but the
  abstract's clause invites the transfer.
- **Nothing isolates the attention from the mixture of experts.** Every model in Table 9 is a MoE.
  There is no dense MLA model anywhere in the paper, so the interaction between a compressed cache and
  sparse feed-forward routing is unexamined.
- **The Limitations section says nothing about MLA.** §5's limitations are the generic ones —
  knowledge cutoff, hallucination, and limited proficiency outside Chinese and English. Not one
  sentence in the paper names a cost, a failure mode or an open question for the mechanism it
  introduces.
- **No long-context ablation of the compression.** The 128K result is a needle-in-a-haystack figure
  after YaRN fine-tuning; nothing tests whether a `d_c = 512` latent loses more at 128K than at 4K,
  which is the obvious question about a fixed-width compression as the context grows.
- **The app establishes nothing empirical.** Untrained weights, four heads, sixteen tokens. What it
  settles exactly is the algebra: the absorption identity, the non-commutativity that breaks it, the
  matched-budget comparison against GQA as linear maps, and the cache arithmetic. Everything about
  quality is quoted.

## Leaves behind

**Backward — this is the answer to the bill concept 15 wrote, and it pays it in a currency that card
did not have.** GQA's own record ends on "an interpolation, which cannot leave the segment — the cache
is still linear in context and in batch, with a smaller constant — and a group count read off a speed
curve rather than derived". Every word of that is still true of GQA here, and Table 8 puts a number on
what the interpolation costs: 4.0 points of MMLU at 8 groups. MLA's move is to stop interpolating
between MHA and MQA along the head axis altogether. The head axis was never the right axis — it is a
*grouping*, chosen before the data arrives, exactly the same kind of hand-set fixity concept 8's
windows and concept 19's index-based cache policy have. A latent is a *learned subspace*, and at
matched budget it keeps four to five times more of the same map `[measured here]`. **The lesson the
deck should carry is that GQA and MLA are both compressions of one object, and the difference is
whether the compression is allowed to choose its own basis.**

It also lands on concept 12, and awkwardly. RoPE's contribution was that position falls out of the
dot product for free; MLA's contribution needs the dot product to be reassociable, and a rotation
sitting inside it is precisely what stops that. The resolution — give position its own small
uncompressed channel, shared across heads — is the first time on this timeline that a mechanism has
had to **route around** rotary position rather than build on it. And concept 18 shows up in the
implementation with its constants intact (`α = 1`, `β = 32`) and its temperature coefficient re-fitted
from 0.1 to 0.0707, which is the strongest available evidence that that coefficient was always a fit
rather than a law.

**Forward — two things.** First, the compression is *static*: `d_c` numbers per token, the same for
every token, whatever is in it. Concept 20 had just argued that what to keep should be decided by the
content; this card decides how much to keep before the data arrives, and gets its win from choosing a
better basis rather than a better subset. Those are different axes and the timeline's remaining cards
(`nsa`, `dsa`) pick up the other one — keep everything, and be selective about what you *read* rather
than about what you store. The two are compatible, and DeepSeek's own later work stacks them, which is
the cleanest possible demonstration that this card's answer was not the whole answer.

Second, and quieter: the absorption trick makes the cached object *architecturally private*. An MLA
cache is not a set of keys and values — it is a set of latents that only that model's absorbed query
matrices can read. Every technique on this timeline that manipulates the cache from outside — evicting
by index (concept 19), renumbering positions, sharing a prefix between requests, or reading a
neighbour's keys — has to be re-derived for an object that no longer has keys in it. The mechanism buys
its cache reduction partly by making the cache less of a general-purpose data structure, and no paper
on this timeline has yet said so.
