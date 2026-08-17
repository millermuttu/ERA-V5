# Concept 20 — Mamba, selective state spaces
**Card id:** `mamba` · **Date:** 2023-12-01 (arXiv v1) · **Pressure:** compressing the past

## What was read

- [arXiv:2312.00752](https://arxiv.org/abs/2312.00752), Albert Gu (Carnegie Mellon,
  `agu@cs.cmu.edu`) and Tri Dao (`tri@tridao.me`) — *Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces*. Authors listed "Alphabetical by first name"; the paper carries no
  conference line on the abstract page. Categories `cs.LG`, `cs.AI`; licence CC BY 4.0.
- **Version history, from the abstract page — two versions:**
  - **v1 — Fri, 1 Dec 2023, 18:01:34 UTC**, 1,264 KB.
  - v2 — Fri, 31 May 2024, 17:55:27 UTC, 1,017 KB (the COLM-era revision).

  **The app's record of `2023-12-01` is verified correct.** The timeline uses v1, per the repo
  convention.
- Full text read from the **ar5iv render** (`ar5iv.labs.arxiv.org/html/2312.00752`), fetched and
  de-marked-up locally so the equations could be read in their LaTeX source form. **ar5iv renders
  the latest source, i.e. v2**, not the v1 the timeline dates — the same situation as the
  `delta-rule` note, and worth stating rather than hiding. Sections used: §1 (Introduction), §2
  (State Space Models, including Discretization / Computation / LTI / Structure and Dimensions /
  SSM Architectures), §3.1 (Motivation: Selection as a Means of Compression), §3.2 (Improving SSMs
  with Selection, Algorithms 1 and 2), §3.3.1–3.3.2 (Efficient Implementation), §3.4 (A Simplified
  SSM Architecture), §3.5.1 (Connection to Gating Mechanisms, Theorem 1), §3.5.2 (Interpretation of
  Selection Mechanisms), §3.6 (Additional Model Details), §4.1–4.6 (all experiments and ablations),
  §5 (Discussion, including the Scaling limitation), §6 (Conclusion), and the appendices:
  **A** (Discussion: Selection Mechanism — the gating / hypernetwork / data-dependence
  disambiguation), **B.1–B.5** (extended related work, including B.4 on linear attention),
  **C** (proof of Theorem 1), **D** (Hardware-aware Algorithm — the fused kernel and recomputation
  accounting), and **E** (E.1 synthetic task protocols and Table 3, E.2 language-modelling
  recipes and Table 4, E.2.2 extra scaling ablations, E.2.3 downstream details, E.3 DNA including
  Table 5, E.4 audio including Table 6 and the S4→S6 ablation, E.5 efficiency benchmark and
  Table 7).
- Every quotation below is the authors' own wording from that source.
- Code and checkpoints released by the authors: <https://github.com/state-spaces/mamba>.
- **A limit on what can be quoted with a number.** Several of this paper's headline results are
  *figures with no accompanying table*: the language-modelling **scaling laws are Figure 6**, the
  throughput and scan-speed results are **Figure 12**, the audio length-scaling is **Figure 9** and
  the S4→S6 audio ablation is **Figure 18**. This note therefore reports those as the paper's own
  prose claims ("5×", "20-40×", "matches Transformer++") and reserves numeric tables for the places
  a table actually exists (Tables 1, 2, 3, 5, 7 and the ablation figures 13–16, which *are*
  tabulated). Anyone who quotes a scaling-law perplexity for Mamba-1.3B from this paper is reading a
  figure, not a number.
- **No app-side measurement was run for this note.** The session budget ran out before the node
  driver could be written, so **nothing below is labelled `[measured here]`**. The app-facing figures
  in "Numbers that matter" are labelled `[derived from shapes]` — arithmetic on this app's declared
  config (`D = 32`, 4 heads, `d_k = 8`, 2 blocks, `T = 16`) and on the paper's stated dimensions, not
  observations of a run. The interactions in "What the live view must let the reader do" specify the
  measurement to perform; the builder must run it and label the result `[measured here]` before
  putting any of those numbers on screen. **Do not copy a number from this note onto a panel.**

---

### Scope, before anything else

This card is the hardest in the deck to keep honest, and the reason is structural: **Mamba is not an
attention variant.** It is a different architecture. There is no `Q`, no `K`, no `V`, no score, no
softmax and no head in it. The Mamba block replaces the whole attention-plus-MLP sandwich, not the
mixing step inside it. Section 3.4 is explicit that the block "combines the H3 block … with the
ubiquitous MLP block of modern neural networks", and the abstract's phrase for the architecture is
"without attention or even MLP blocks".

The timeline includes it as the point where **the state returns** — where the fixed-size recurrent
memory that concepts 9 and 11 built out of a linearised attention is arrived at from the opposite
direction, from classical state-space theory, and made to work at a scale attention had held alone.
That framing is defensible and it is the paper's own framing (§3.1 is literally titled "Selection as
a Means of Compression" and opens "a fundamental problem of sequence modeling is compressing context
into a smaller state"). But it licenses exactly one thing on this card: **the recurrence and the
selection rule**. It does not license drawing Mamba as a mixer inside this app's transformer, and
every panel this card mounts must be labelled as *the selection idea, transplanted*, not *Mamba,
implemented*. The app's seam can host a per-token decay faithfully. It cannot host a Mamba block, and
pretending otherwise would be the folklore this deck exists to resist.

---

## The mechanism, precisely

### Step 0 — the continuous system, and the discrete recurrence it becomes

§2 defines a structured SSM (S4) by four parameters `(Δ, A, B, C)` and a two-stage transformation.
The continuous system, Eq. 1, maps a **one-dimensional** function to a one-dimensional function
through an `N`-dimensional latent state:

    h'(t) = A h(t) + B x(t)                                                      (1a)
    y(t)  = C h(t)                                                               (1b)

> "They are inspired by a particular continuous system (1) that maps a 1-dimensional function or
> sequence `x(t) ∈ ℝ ↦ y(t) ∈ ℝ` through an implicit latent state `h(t) ∈ ℝ^N`."

The discrete recurrence, Eq. 2 — this is the object the card draws:

    h_t = Ā h_{t−1} + B̄ x_t                                                     (2a)
    y_t = C h_t                                                                  (2b)

and the equivalent global convolution, Eq. 3:

    K̄ = (C B̄, C Ā B̄, …, C Ā^k B̄, …)                                            (3a)
    y  = x ∗ K̄                                                                   (3b)

**Read Eq. 2 against `mixers.js` line 127 and the shapes rhyme but the objects do not.** Eq. 2a is
`state ← decay · state + write`, which is the shape of the `gated` branch. The difference is that
`h_t` is a vector per *channel* and the write `B̄ x_t` is a scalar input times a vector — a rank-one
write into a vector, not into a matrix. There is no outer product of a key with a value anywhere in
Eq. 2. That distinction matters when this card names what the app's grid actually is (see
"Leaves behind").

### Step 1 — discretisation, and what Δ is

The first stage converts the continuous parameters to discrete ones:

> "The first stage transforms the 'continuous parameters' `(Δ, A, B)` to 'discrete parameters'
> `(Ā, B̄)` through fixed formulas `Ā = f_A(Δ, A)` and `B̄ = f_B(Δ, A, B)`, where the pair
> `(f_A, f_B)` is called a discretization rule."

The rule they use is **zero-order hold (ZOH)**, Eq. 4:

    Ā = exp(ΔA)          B̄ = (ΔA)^{-1}(exp(ΔA) − I) · ΔB                        (4)

`Δ` is the **step size** of the discretisation — how much continuous time one token is taken to
occupy. Everything the card needs about it is in §3.5.2's "Interpretation of Δ", and it is the
single most quotable paragraph in the paper for a visual app:

> "In general, `Δ` controls the balance between how much to focus or ignore the current input `x_t`.
> It generalizes RNN gates (e.g. `g_t` in Theorem 1): **mechanically, a large `Δ` resets the state
> `h` and focuses on the current input `x`, while a small `Δ` persists the state and ignores the
> current input.** SSMs (1)-(2) can be interpreted as a continuous system discretized by a timestep
> `Δ`, and in this context the intuition is that large `Δ → ∞` represents the system focusing on the
> current input for longer (thus 'selecting' it and forgetting its current state) while a small
> `Δ → 0` represents a transient input that is ignored."

Trace it through Eq. 4 with `A` negative (which the initialisation guarantees — see Step 5):
`Δ → ∞` makes `exp(ΔA) → 0`, so `Ā → 0` and the state is **wiped**; `Δ → 0` makes `exp(ΔA) → 1`, so
`Ā → 1` and `B̄ → 0`, and the token is **skipped entirely**. One scalar per token spans "forget
everything and take this" to "pretend this token did not happen". That is the whole selection
mechanism in one number, and it is the number the live view should put on a slider.

The paper also volunteers that the continuous framing is optional machinery:

> "However, from a mechanical point of view discretization can simply be viewed as the first step of
> the computation graph in the forward pass of an SSM."

and notes that "Alternate flavors of SSMs can bypass the discretization step and parameterize
`(Ā, B̄)` directly instead (Zhang et al. 2023), which may be easier to reason about." **The card
should say this out loud.** The continuous-time story is how the parameterisation was arrived at; it
is not load-bearing for what the model computes. A reader who cannot follow `exp(ΔA)` has not missed
the mechanism.

### Step 2 — LTI, and why every earlier structured SSM was stuck with it

    "An important property of equations (1) to (3) is that the model's dynamics are constant through
    time. In other words `(Δ, A, B, C)`, and consequently `(Ā, B̄)` as well, are fixed for all
    time-steps. This property is called linear time invariance (LTI)."

Constant dynamics is exactly what makes Eq. 2 and Eq. 3 the same function. If `Ā` and `B̄` never
change, expanding the recurrence gives a fixed convolution kernel `K̄` (Eq. 3a) that can be applied
with an FFT — and the paper's accounting is that this was not a stylistic choice but a survival
requirement:

> "Thus far, all structured SSMs have been LTI (e.g. computed as convolutions) because of fundamental
> efficiency constraints, discussed in Section 3.3. However, a core insight of this work is that LTI
> models have fundamental limitations in modeling certain types of data, and our technical
> contributions involve removing the LTI constraint while overcoming the efficiency bottlenecks."

and, in Appendix B.1, the survey sentence that establishes the novelty claim precisely:

> "Notably, all of these methods, and all other structured SSMs that we are aware of, have been
> non-selective and usually strictly LTI (linear time invariant)."

**The shape of the state, which the card needs for its counters.** §2, "Structure and Dimensions":
`A ∈ ℝ^{N×N}` is diagonal, so `A, B, C` are each `N` numbers; the SSM is applied **independently to
each of `D` channels**; and therefore

> "Note that in this case, the total hidden state has dimension `DN` per input, and computing it over
> the sequence length requires `O(BLDN)` time and memory; this is the root of the fundamental
> efficiency bottleneck addressed in Section 3.3."

`DN` per token, with `N ≈ 10–100` — that is the "expanded state" every efficiency argument in the
paper is about.

### Step 3 — selectivity. This is the contribution, and it is three lines of Algorithm 2

The mechanism is stated with almost aggressive plainness:

> "Algorithms 1 and 2 illustrates the main selection mechanism that we use. **The main difference is
> simply making several parameters `Δ, B, C` functions of the input**, along with the associated
> changes to tensor shapes throughout. In particular, we highlight that these parameters now have a
> length dimension `L`, meaning that the model has changed from time-invariant to time-varying."

The two algorithms side by side, with the shapes, because the shapes *are* the argument:

| line | Algorithm 1 (S4) | Algorithm 2 (S6 — selective) |
|---|---|---|
| `A` | `(D, N)` ← Parameter | `(D, N)` ← Parameter — **unchanged, not input-dependent** |
| `B` | `(D, N)` ← Parameter | **`(B, L, N)` ← `s_B(x)`** |
| `C` | `(D, N)` ← Parameter | **`(B, L, N)` ← `s_C(x)`** |
| `Δ` | `(D)` ← `τ_Δ(Parameter)` | **`(B, L, D)` ← `τ_Δ(Parameter + s_Δ(x))`** |
| `Ā, B̄` | `(D, N)` ← discretize | **`(B, L, D, N)`** ← discretize |
| SSM | "Time-invariant: recurrence **or convolution**" | "Time-varying: **recurrence (scan) only**" |

The `L` appearing in four rows is the entire technical story of the rest of the paper. A parameter
that has a length dimension is a parameter that is different at every token, and a kernel that is
different at every token is not a convolution.

**Exactly what the input-dependence is** (§3.2, and this is quotable verbatim):

> "We specifically choose `s_B(x) = Linear_N(x)`, `s_C(x) = Linear_N(x)`,
> `s_Δ(x) = Broadcast_D(Linear_1(x))`, and `τ_Δ = softplus`, where `Linear_d` is a parameterized
> projection to dimension `d`. The choice of `s_Δ` and `τ_Δ` is due to a connection to RNN gating
> mechanisms explained in Section 3.5."

So: `B` and `C` are plain linear maps of the token to `N` dimensions. `Δ` is a linear map of the
token to **one scalar**, softplus'd to stay positive, then **broadcast to all `D` channels** — and
§3.5.1 gives the reason for the broadcast, which is a real design argument and not an implementation
detail:

> "In particular, note that if a given input `x_t` should be completely ignored (as necessary in the
> synthetic tasks), **all `D` channels should ignore it**, and so we project the input down to `1`
> dimension before repeating/broadcasting with `Δ`."

**What deliberately does not become input-dependent, and why.** `A` stays a fixed learned parameter,
and §3.5.2 gives the argument in full:

> "We remark that while the `A` parameter could also be selective, it ultimately affects the model
> only through its interaction with `Δ` via `Ā = exp(ΔA)` (the discretization (4)). Thus **selectivity
> in `Δ` is enough to ensure selectivity in `(Ā, B̄)`, and is the main source of improvement.** We
> hypothesize that making `A` selective in addition to (or instead of) `Δ` would have similar
> performance, and leave it out for simplicity."

This is the sentence the card must not sand off. The claim is *not* "`A` must be fixed"; it is
"`A` selective would be redundant, because `Δ` already multiplies it". The effective transition
`Ā_t = exp(Δ_t A)` is time-varying **even though `A` is constant**, because `Δ_t` is not. A card that
says "Mamba makes A input-dependent" is wrong; a card that says "Mamba's transition matrix is fixed"
is also wrong. The precise statement is: **one fixed matrix, one per-token scalar, and their product
in an exponential is the per-token decay.**

And the roles of `B` and `C`, §3.5.2:

> "In an SSM, modifying `B` and `C` to be selective allows finer-grained control over whether to let
> an input `x_t` into the state `h_t`, or the state into the output `y_t`. These can be interpreted
> as allowing the model to modulate the recurrent dynamics based on content (input) and context
> (hidden states) respectively."

Three input-dependent quantities, three jobs: **`Δ` decides how much of the past to keep, `B` decides
what of this token goes in, `C` decides what of the state comes out.**

### Step 4 — the argument the paper leads with: two synthetic tasks

§3.1 does not open with equations. It opens with a claim about what compression is for:

> "We argue that a fundamental problem of sequence modeling is compressing context into a smaller
> state. In fact, we can view the tradeoffs of popular sequence models from this point of view. For
> example, **attention is both effective and inefficient because it explicitly does not compress
> context at all.** This can be seen from the fact that autoregressive inference requires explicitly
> storing the entire context (i.e. the KV cache), which directly causes the slow linear-time
> inference and quadratic-time training of Transformers. On the other hand, recurrent models are
> efficient because they have a finite state, implying constant-time inference and linear-time
> training. However, their effectiveness is limited by how well this state has compressed the
> context."

Then the two tasks:

> "The **Selective Copying** task modifies the popular Copying task (Arjovsky et al. 2016) by varying
> the position of the tokens to memorize. It requires content-aware reasoning to be able to memorize
> the relevant tokens (colored) and filter out the irrelevant ones (white)."

> "The **Induction Heads** task is a well-known mechanism hypothesized to explain the majority of
> in-context learning abilities of LLMs (Olsson et al. 2022). It requires context-aware reasoning to
> know when to produce the correct output in the appropriate context (black)."

And the diagnosis of why time-invariant models fail them, which is the sentence the whole card turns
on:

> "These tasks reveal the failure mode of LTI models. From the recurrent view, **their constant
> dynamics (e.g. the `(Ā, B̄)` transitions in (2)) cannot let them select the correct information from
> their context, or affect the hidden state passed along the sequence in an input-dependent way.**
> From the convolutional view, it is known that global convolutions can solve the vanilla Copying
> task (Romero et al. 2021) because it **only requires time-awareness**, but that they have
> difficulty with the Selective Copying task because of **lack of content-awareness** (Figure 2).
> More concretely, the spacing between inputs-to-outputs is varying and cannot be modeled by static
> convolution kernels."

The vanilla Copying task is solvable by a *schedule*: build a kernel of exactly the right length and
the answer falls out, with no reference to what the tokens are. Randomise the spacing and the
schedule is worthless, because the position of the thing to remember is now a function of the
content. **That is the whole distinction: forgetting on a schedule versus forgetting on command.**

The paper also pre-empts the obvious objection — that architectural gating already provides
data-dependence:

> "Note that many previous works argue that adding architecture gating (multiplicative interactions)
> can endow models with 'data-dependence' and solve related tasks (Dao et al. 2023; Poli et al.
> 2023). However, we find this explanation insufficient intuitively because **such gating does not
> interact along the sequence axis, and cannot affect the spacing between tokens.** In particular
> architecture gating is not an instance of a selection mechanism (Appendix A)."

Appendix A elaborates with the GLU example: a diagonal `D = σ(Wx)` gives `y = σ(Wx) ∘ x`, which
"technically satisfies the common meanings of gating …, hypernetworks …, and data-dependent …
However, this in fact simply defines a GLU function, which is so simple that it is often considered
just an activation function … instead of a meaningful layer." Their narrowing:

> "More narrowly, we use selection to refer to the mechanistic action of a model to **select or
> ignore inputs and facilitate data interaction along the sequence length** (Section 3.1)."

**Along the sequence length.** A multiplication that acts within a token is not selection. The gate
has to be on the recurrence.

### Step 5 — why selectivity is expensive, and the hardware-aware scan (the concept-14 argument, again)

The cost is stated as an inheritance:

> "This simple change poses a technical challenge for the computation of the model; in fact, **all
> prior SSMs models must be time- and input-invariant in order to be computationally efficient.**"

§3.3.1 lays out the trap in three bullets. The recurrent form is more general than the convolutional
one, "since the latter (3) is derived from expanding the former (2)" — but

> "this would require computing and materializing the latent state `h` with shape `(B, L, D, N)`,
> which is much larger (by a factor of `N`, the SSM state dimension) than the input `x` and output
> `y` of shape `(B, L, D)`. Thus the more efficient convolution mode was introduced which could
> bypass the state computation and materializes a convolution kernel (3a) of size only `(B, L, D)`."

So the convolution was never primarily about FLOPs. **It was about never writing the expanded state
down.** Prior LTI SSMs "leverage the dual recurrent-convolutional forms to increase the effective
state dimension by a factor of `N` (`≈ 10–100`), much larger than traditional RNNs, without
efficiency penalties." Selectivity removes the convolutional escape hatch, so the `N`-times-larger
state has to be paid for somehow.

The FLOP accounting is the first surprise, and it undercuts a folk belief about why convolutions won:

> "The naive recurrent computation uses `O(BLDN)` FLOPs while the convolutional computation uses
> `O(BLD log(L))` FLOPs, and the former has a lower constant factor. **Thus for long sequences and
> not-too-large state dimension `N`, the recurrent mode can actually use fewer FLOPs.**"

**The recurrence was never the arithmetically expensive option.** It was the memory-traffic
expensive option — which is precisely, exactly, and by the same author the FlashAttention argument
(concept 14). The paper's statement of it:

> "The main idea is to leverage properties of modern accelerators (GPUs) to **materialize the state
> `h` only in more efficient levels of the memory hierarchy.** In particular, most operations (except
> matrix multiplication) are bounded by memory bandwidth (Williams et al. 2009; Ivanov et al. 2021;
> **Dao et al. 2022**). This includes our scan operation, and we use kernel fusion to reduce the
> amount of memory IOs, leading to a significant speedup compared to a standard implementation."

> "Concretely, instead of preparing the scan input `(Ā, B̄)` of size `(B, L, D, N)` in GPU HBM
> (high-bandwidth memory), we **load the SSM parameters `(Δ, A, B, C)` directly from slow HBM to fast
> SRAM, perform the discretization and recurrence in SRAM, and then write the final outputs of size
> `(B, L, D)` back to HBM.**"

Appendix D itemises the fused kernel — read `O(BLD + DN)` bytes into SRAM; discretise to `(B,L,D,N)`
**in SRAM**; scan **in SRAM**; multiply by `C` and write `(B,L,D)` out — and states the payoff:

> "This way, we **reduce IOs by a factor of `O(N)`** (the state dimension), which in practice speeds
> up the operation by 20-40 times (Section 4.5)."

Three classical techniques, named as such: "kernel fusion, parallel scan, and recomputation". The
sequential-dependency problem is handled by the scan —

> "To avoid the sequential recurrence, we observe that despite not being linear it can still be
> parallelized with a **work-efficient parallel scan algorithm** (Blelloch 1990; Martin & Cundy 2018;
> Smith et al. 2023)."

— and the backward pass by recomputation:

> "the intermediate states are not stored but **recomputed in the backward pass** when the inputs are
> loaded from HBM to SRAM. As a result, the fused selective scan layer has **the same memory
> requirements as an optimized transformer implementation with FlashAttention.**"

with Appendix D's per-token accounting: FlashAttention stores "around 12 bytes of activations per
token" plus 20 for the MLP, total 32; "Each selective SSM stores around 16 bytes of activations per
token. Hence two layers of selective SSMs have around the same activation memory as an attention
layer and an MLP layer."

**The card must draw the concept-14 connection explicitly, because it is the same argument by the
same person, one paper later, applied to a different object.** FlashAttention's move was: never write
the `N×N` score matrix to HBM; tile it, keep the tile in SRAM, recompute in the backward pass.
Mamba's move is: never write the `(B,L,D,N)` state to HBM; keep it in SRAM, recompute in the backward
pass. Concept 14's note records the hardware constants that make both work — A100, 40–80 GB HBM at
1.5–2.0 TB/s versus 192 KB SRAM per SM × 108 SMs at ~19 TB/s, about 10× the bandwidth and about six
orders of magnitude less space. Mamba's `O(N)` IO reduction with `N = 16` is a 16× traffic cut that
shows up as a measured 20–40×; FlashAttention's Θ(N²d²M⁻¹) is the same shape of claim. **The
generalisable lesson the deck should carry from 14 to 20: an algorithm is not fast because it does
less arithmetic, it is fast because it moves less memory — and that lesson, not the SSM, is what
made selectivity affordable.** Without concept 14's insight, Algorithm 2 is a good idea that trains
too slowly to test.

### Step 6 — the gate, Theorem 1, and the link to the delta-rule family

§3.5.1 is where this card's forward link lives, and it opens with the paper's own emphasis:

> "**We highlight the most important connection: the classical gating mechanism of RNNs is an
> instance of our selection mechanism for SSMs.** We note that the connection between RNN gating and
> the discretization of continuous-time systems is well established (Funahashi & Nakamura 1993;
> Tallec & Ollivier 2018). … More broadly, **`Δ` in SSMs can be seen to play a generalized role of
> the RNN gating mechanism.** In line with prior work, we adopt the view that **discretization of
> SSMs is the principled foundation of heuristic gating mechanisms.**"

**Theorem 1**, stated in full:

> "When `N = 1, A = −1, B = 1, s_Δ = Linear(x)`, and `τ_Δ = softplus`, then the selective SSM
> recurrence (Algorithm 2) takes the form
>
>     g_t = σ(Linear(x_t))                                                       (5)
>     h_t = (1 − g_t) h_{t−1} + g_t x_t."

Appendix C proves it in four lines, and the lines are worth reproducing because they are the exact
bridge between "step size" and "gate": the continuous system `h(t) = −h(t) + x(t)` is "also called a
leaky integrator"; `Δ_t = softplus(Parameter + Linear(x_t)) = softplus(Linear(x_t))` since "the
parameter can be viewed as a learnable bias and folded into the linear projection"; then ZOH gives

    Ā_t = exp(ΔA) = 1/(1 + exp(Linear(x_t))) = σ(−Linear(x_t)) = 1 − σ(Linear(x_t))
    B̄_t = (ΔA)^{-1}(exp(ΔA) − I)·ΔB = −(exp(ΔA) − I) = 1 − Ā

**`Ā + B̄ = 1` exactly, and both are the same sigmoid.** The decay and the write strength are not two
knobs; they are one knob and its complement. That is a fact the live view can display and the app's
current seam cannot express, because `decay` and `beta` are independent parameters there — see
"What the live view must let the reader do", interaction 3.

**The link to the delta-rule family** is Appendix A's opening sentence, and it names the concept-11
paper directly:

> "Our selection mechanism is inspired by and related to concepts such as gating, hypernetworks, and
> data-dependence. **It can also be viewed as related to 'fast weights' (Schmidhuber 1992; Ba et al.
> 2016), which connects classical RNNs with the mechanism of linear attention (Schlag et al. 2021).
> However, we believe that it is a distinct concept that is worth clarifying.**"

Note the shape of that: an acknowledgement and a demurral in one sentence. The paper positions
selectivity *near* the fast-weight/delta-rule lineage and then declines the identification, and its
stated reason (the rest of Appendix A) is that "gating", "hypernetwork" and "data-dependence" are so
broad that "essentially anything with a multiplication, including standard attention mechanisms …
as well" qualifies, and "we find it uninformative to think of them as such". Its own preferred
positioning:

> "Instead, we view it as most closely related to **the gating mechanism of traditional RNNs**, which
> is a special case (Theorem 1) and also has a deeper history of connections to SSMs through variable
> (input-dependent) discretization of `Δ`. **We also eschew the term 'gating' in favor of selection
> to clarify the overloaded use of former.**"

This is the honest form of the timeline's forward link. Mamba does **not** claim to be a delta rule
with a gate. It claims that input-dependent `Δ` *generalises* the classical RNN gate, and it cites
Schlag et al. in the same breath as declining to be filed under it. What comes next in the deck —
gated DeltaNet (`gated-deltanet`) — is the synthesis that takes the release valve seriously *and*
keeps the correction, i.e. it goes where this paper explicitly did not. Say that as a relationship,
not as a lineage. See "Leaves behind".

Two more properties from §3.5.2 the card should quote, because they are what "the state returns"
actually buys:

> "**Filtering Context.** It has been empirically observed that many sequence models do not improve
> with longer context (Shi et al. 2023), despite the principle that more context should lead to
> strictly better performance. An explanation is that many sequence models cannot effectively ignore
> irrelevant context when necessary … On the other hand, **selective models can simply reset their
> state at any time to remove extraneous history**, and thus their performance in principle improves
> monotonicly with context length."

> "**Boundary Resetting.** In settings where multiple independent sequences are stitched together,
> Transformers can keep them separate by instantiating a particular attention mask, while LTI models
> will bleed information between the sequences. Selective SSMs can also reset their state at
> boundaries (e.g. `Δ_t → ∞`, or Theorem 1 when `g_t → 1`)."

A causal mask is a *legislated* reset — the same kind of thing the Sparse Transformer legislated
about which keys to read. A selective `Δ` is a *learned* one. The card can put those two side by side
in one sentence and the whole deck lights up.

### Step 7 — the architecture, briefly, and why this card should mostly leave it alone

§3.4: expand the model dimension by factor `E` (always `E = 2` in their experiments), fold the H3
block and the MLP block into one homogenous block, SiLU/Swish activation, optional LayerNorm
"motivated by RetNet's usage of a normalization layer in a similar location". Most parameters
(`3ED²`, of which `2ED²` in, `ED²` out) are in the linear projections; "The number of SSM parameters
(projections for `Δ, B, C`, and the matrix `A`) are much smaller in comparison." Two Mamba blocks are
used to match the `12D²` of a Transformer's MHA + MLP pair.

Table 2 (below) shows the architecture is nearly irrelevant to the result and the **layer** is nearly
everything: swap S4 for S6 inside either block and perplexity moves by ~1.6; swap the block and it
moves by ~0.26. This card is about the layer. Draw the block only if there is room left over.

---

## Numbers that matter

### The two synthetics — where the argument is won

**Selective Copying (§4.1.1, protocol in E.1).** Sequences of **length 4096**, vocab **16** tokens
including the noise token, **16 data tokens** to memorise, **2-layer** models with **`D = 64`**,
**400K steps** at a constant learning rate of **1e-4**, batch **64**. Accuracy:

| Model | Arch. | Layer | Acc. |
|---|---|---|---|
| S4 | No gate | S4 | **18.3** |
| — | No gate | **S6** | **97.0** |
| H3 | H3 | S4 | 57.0 |
| Hyena | H3 | Hyena | 30.1 |
| — | H3 | **S6** | **99.7** |
| — | Mamba | S4 | 56.4 |
| — | Mamba | Hyena | 28.4 |
| Mamba | Mamba | **S6** | **99.8** |

Read the table by column, not by row, because that is where the paper's claim lives. Holding the
*architecture* fixed and swapping the *layer* S4 → S6: `18.3 → 97.0` with no gate, `57.0 → 99.7` in
H3, `56.4 → 99.8` in Mamba. Holding the layer fixed and swapping the architecture: `18.3 → 57.0 →
56.4`. **Gating the architecture buys ~38 points; selecting in the recurrence buys ~43 more and
finishes the task.** The paper's summary: "gated architectures such as H3 and Mamba only partially
improve performance, while the selection mechanism (modifying S4 to S6) easily solves this task."

**Induction Heads (§4.1.2, Table 3).** 2-layer models, trained at sequence length **2⁸ = 256**, vocab
**16**, `D = 64` for Mamba and `128` for the others; tested from **2⁶ = 64 to 2²⁰ = 1,048,576**.
Adam, no weight decay, LR chosen from {2e-4, 1e-3} per model. `✓` is perfect accuracy, `✗` is out of
memory:

| Model | Params | 2⁸ (train) | 2⁹ | 2¹⁰ | 2¹¹ | 2¹² | 2¹⁴ | 2¹⁶ | 2²⁰ |
|---|---|---|---|---|---|---|---|---|---|
| MHA-Abs | 137K | 100.0 | 58.6 | 26.6 | 18.8 | 9.8 | 7.8 | ✗ | ✗ |
| MHA-RoPE | 137K | 100.0 | 83.6 | 31.3 | 18.4 | 8.6 | 5.5 | ✗ | ✗ |
| MHA-xPos | 137K | 100.0 | 99.6 | 67.6 | 25.4 | 7.0 | 7.8 | ✗ | ✗ |
| H3 | 153K | 100.0 | 80.9 | 39.5 | 23.8 | 14.8 | 5.9 | 8.2 | 7.4 |
| Hyena | 69M\* | 100.0 | ✓ | 44.1 | 12.5 | 6.6 | 7.0 | 6.6 | 9.8 |
| **Mamba** | **74K** | **100.0** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

\* "Most of the parameters are in learnable positional encodings."

> "It generalizes perfectly to million-length sequences, or **4000× longer than it saw during
> training, while no other method goes beyond 2×.**"

Three things the card should print alongside this table so it is not oversold. (1) **The attention
baselines were never run past 2¹⁴** — "all attention models were only tested up to sequence length
2¹⁴ = 16384 due to memory limitations" — so the last three columns are not a comparison, they are
Mamba alone. (2) **Vocabulary 16, two layers, 74K parameters.** This is a toy, deliberately, and its
job is to isolate a mechanism. (3) The row that actually carries weight is **2⁹ = 512**, one doubling
past training, where every attention variant has already fallen to 58–99% and H3/Hyena to 81–100%.
The collapse starts immediately; the million is showmanship on top of a real result at 512.

### Language modelling

**Scaling laws (§4.2.1, Figure 6 — a figure, no table).** Models `≈125M` to `≈1.3B` on the Pile,
Chinchilla token counts, GPT-2 tokenizer. Sizes from Table 4: 125M (12 layers, `d_model` 768, 4800
steps, 2.5B tokens), 350M (24 / 1024, 13500, 7B), 760M (24 / 1536, 29000, 15B), 1.3B (24 / 2048,
50000, 26B); batch 0.5M tokens throughout. The claim:

> "Mamba is the **first attention-free model to match the performance of a very strong Transformer
> recipe (Transformer++)** that has now become standard, particularly as the sequence length grows."

Transformer++ is defined in E.2.1 as rotary embeddings, SwiGLU MLP, RMSNorm, no linear bias, and the
improved recipe (peak LR **5× the GPT-3 value**, cosine decay to 1e-5, AdamW β = (.9, .95)). The
paper is careful that this recipe matters more than the architecture in places: E.2.2 reports that
"A large improvement is achieved by the improved training recipe" and that "The choice of the inner
LTI SSM does not matter (e.g. Hyena vs. S4)". Also flagged in §4.2.1: "full results on context length
8k are missing for the RWKV and RetNet baselines … because of a lack of efficient implementations".

**The ablation table, which is where the real perplexities are (Table 2).** `≈350M` models at
Chinchilla token counts — the same setting as Figure 6, so these numbers are comparable to each
other:

| Arch. | SSM layer | Perplexity |
|---|---|---|
| H3 | Hyena | 10.24 |
| H3 | S4 (complex) | 10.30 |
| H3 | S4 (real) | 10.34 |
| **H3** | **S6** | **8.95** |
| Mamba | Hyena | 10.75 |
| Mamba | S4 (complex) | 10.54 |
| Mamba | S4 (real) | 10.56 |
| **Mamba** | **S6** | **8.69** |

**Every LTI layer lands between 10.24 and 10.75 regardless of architecture; every selective layer
lands at 8.69–8.95.** A 1.6-perplexity gap that is entirely attributable to the layer. This is the
single most persuasive table in the paper and it is an ablation, not a headline.

**Which parameters have to be selective (Figure 13, same setting):**

| Selective Δ | Selective B | Selective C | Perplexity |
|---|---|---|---|
| ✗ | ✗ | ✗ | 10.93 |
| ✗ | ✓ | ✗ | 10.15 |
| ✗ | ✗ | ✓ | 9.98 |
| **✓** | ✗ | ✗ | **9.81** |
| **✓** | **✓** | **✓** | **8.71** |

`Δ` alone is the best single choice (9.81), consistent with Theorem 1 — but note that all three
together (8.71) beat the best single one by more than the best single one beats none. The paper's
caption: "`Δ` is the most important parameter (Theorem 1), but **using multiple selective parameters
together synergizes.**"

**State dimension `N`, and the fact that it only pays when B and C are selective (Figure 16,
Δ-projection fixed at 64):**

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| constant `B`, `C` | 9.88 | 9.86 | 9.82 | 9.82 | **9.81** |
| **selective `B`, `C`** | 9.73 | 9.40 | 9.09 | 8.84 | **8.71** |

> "Of particular note is the dramatic improvement of the selective SSM when the state size `N` is
> increased, with **over a 1.0 perplexity improvement for a cost of only 1% additional parameters.**"

Parameter counts across that whole sweep: 367.1M → 371.5M. **A bigger state is nearly free in
parameters and worthless without selectivity** — 0.07 perplexity across the top row, 1.02 across the
bottom. That contrast is the card's best available argument that this is about the state and not
about parameter count.

Two smaller ablations: the `Δ` projection dimension (Figure 15) runs `none → 9.12`, `1 → 8.97`,
`4 → 8.91`, `8 → 8.83`, `64 → 8.71` at 358.9M → 371.5M params — so **projecting to a single scalar
already captures most of it** (9.12 → 8.97). And `A`'s initialisation (Figure 14): complex
`−½ + ni` → 9.16, real `−½` → 8.85, real `−(n+1)` (S4D-Real) → 8.71, random `exp(𝒩(0,1))` → **8.71**.
A random initialisation ties the theory-derived one, which the paper reports without defensiveness.

**Zero-shot downstream (Table 1).** 300B tokens, GPT-NeoX tokenizer, context 2048 (RWKV 1024).
Pile validation perplexity, LAMBADA perplexity, and accuracies:

| Model | Pile ppl ↓ | LAMBADA ppl ↓ | LAMBADA | HellaSwag | PIQA | Arc-E | Arc-C | WinoGrande | **Average** |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-160M | 29.64 | 38.10 | 33.0 | 30.2 | 61.4 | 43.2 | 24.1 | 51.9 | 40.6 |
| **Mamba-130M** | **10.56** | **16.07** | 44.3 | 35.3 | 64.5 | 48.0 | 24.3 | 51.9 | **44.7** |
| Pythia-410M | 9.95 | 10.84 | 51.4 | 40.6 | 66.9 | 52.1 | 24.6 | 53.8 | 48.2 |
| **Mamba-370M** | **8.28** | **8.14** | 55.6 | 46.5 | 69.5 | 55.1 | 28.0 | 55.3 | **50.0** |
| **Mamba-790M** | 7.33 | 6.02 | 62.7 | 55.1 | 72.1 | 61.2 | 29.5 | 56.1 | **57.1** |
| Pythia-1.4B | 7.51 | 6.08 | 61.7 | 52.1 | 71.0 | 60.5 | 28.5 | 57.2 | 55.2 |
| RWKV-1.5B | 7.70 | 7.04 | 56.4 | 52.5 | 72.4 | 60.5 | 29.4 | 54.6 | 54.3 |
| **Mamba-1.4B** | **6.80** | **5.04** | 64.9 | 59.1 | 74.2 | 65.5 | 32.8 | 61.5 | **59.7** |
| Pythia-2.8B | 6.73 | 5.04 | 64.7 | 59.3 | 74.0 | 64.1 | 32.9 | 59.7 | 59.1 |
| RWKV-3B | 7.00 | 5.24 | 63.9 | 59.6 | 73.7 | 67.8 | 33.1 | 59.6 | 59.6 |
| **Mamba-2.8B** | **6.22** | **4.23** | 69.2 | 66.1 | 75.2 | 69.7 | 36.3 | 63.5 | **63.3** |
| GPT-J-6B | — | **4.10** | 68.3 | 66.3 | 75.4 | 67.0 | 36.6 | 64.1 | 63.0 |
| OPT-6.7B | — | 4.25 | 67.7 | 67.2 | 76.3 | 65.6 | 34.9 | 65.5 | 62.9 |
| Pythia-6.9B | 6.51 | 4.45 | 67.1 | 64.0 | 75.2 | 67.3 | 35.5 | 61.3 | 61.7 |
| RWKV-7.4B | 6.31 | 4.38 | 67.2 | 65.5 | 76.1 | 67.8 | 37.5 | 61.0 | 62.5 |

The intro's claim, and it checks out arithmetically: "**4 points higher avg. on common sense
reasoning compared to Pythia-3B and even exceeding Pythia-7B**" — 63.3 vs 59.1 is +4.2, and 63.3 vs
Pythia-6.9B's 61.7 is +1.6.

But read the last four rows honestly and the "twice its size" claim narrows. Against **GPT-J-6B**
Mamba-2.8B wins the average by **0.3 points** (63.3 vs 63.0) and **loses** on LAMBADA perplexity
(4.23 vs 4.10), HellaSwag (66.1 vs 66.3), PIQA (75.2 vs 75.4), Arc-C (36.3 vs 36.6) and WinoGrande
(63.5 vs 64.1) — it wins the average almost entirely on LAMBADA accuracy and Arc-E. And **no
Transformer++ appears anywhere in this table.** Every Transformer baseline here (Pythia, OPT,
GPT-Neo, GPT-J) uses a GPT-3-era recipe; the paper's own Figure 6 and E.2.2 say that recipe is worth
a large amount of perplexity. So the "matches Transformers twice its size" headline is measured
against precisely the class of Transformer the paper elsewhere argues is undertrained.

### Speed and memory — the number that matters most is the one that goes the wrong way

**Scan speed (§4.5, Figure 12; protocol in E.5).** A100 80GB PCIe, `D = 1024`, `N = 16`, **batch 1**,
BF16, sequence lengths 2⁹ = 512 up to 2¹⁹ ≈ 500K. The baseline scan is "a standard parallel scan in
PyTorch with no kernel fusion", and attention is "the fastest implementation that we are aware of
(FlashAttention-2 …), with causal mask" — with the honest note that "FlashAttention-2 with causal
mask is about 1.7× faster than without causal mask". The claims:

- §4.5: "Our efficient SSM scan is **faster than the best attention implementation that we know of
  (FlashAttention-2) beyond sequence length 2K**, and up to **20-40× faster than a standard scan
  implementation in PyTorch**."
- Appendix D: "**up to 7× times faster than attention at sequence length 32K**".
- §1: "The resulting implementation is faster than previous methods both in theory … and on modern
  hardware (**up to 3× faster on A100 GPUs**)."

**Three different multipliers against three different baselines, exactly as in concept 9.** 20–40×
is against an unfused PyTorch scan nobody would ship. 7× is against FlashAttention-2 at 32K. 3× is
against prior SSM implementations. And the crossover against FlashAttention-2 is at **2K tokens** —
below that, attention is faster. Print all four numbers together or none of them.

**Inference throughput (§4.5, Figure 12 right; E.5).** Prompt length **2048**, generation length
**128**, batch swept 1 → 128, throughput = `batch × 128 / time`, averaged over 3 runs, A100 80GB
PCIe. Baseline: "the standard Transformer implementation in the **Huggingface transformers
library**", Transformer 1.3B and 6.7B (GPT-3 architecture) against Mamba-1.4B and an **untrained**
Mamba-6.9B.

> "Mamba achieves **4-5× higher inference throughput** than a Transformer of similar size, **since
> without the KV cache it can use much higher batch sizes.** For example, a Mamba-6.9B (untrained)
> would have higher inference throughput than a 5× smaller Transformer-1.3B."

The mechanism of the win is stated by the authors themselves and the card should repeat it: **it is a
batch-size win, not a per-token win.** No KV cache means more sequences fit, and throughput is
sequences × tokens. The abstract's "5×" is the top of a 4–5× range measured against an unoptimised
HuggingFace baseline.

**Training memory (Table 7).** 125M models, one A100 80GB, sequences of length 2048, against "the
most memory-efficient Transformer implementation we are aware of (with kernel fusion from
`torch.compile` and with FlashAttention-2)":

| Batch size | Transformer (w/ FlashAttention-2) | Mamba |
|---|---|---|
| 1 | 4.6 GB | **4.8 GB** |
| 2 | 5.2 GB | **5.8 GB** |
| 4 | 6.9 GB | **7.3 GB** |
| 8 | 11.5 GB | **12.3 GB** |
| 16 | 20.7 GB | **23.1 GB** |
| 32 | 34.5 GB | **38.2 GB** |

**Mamba uses more training memory than the optimised Transformer in every single row** — 4% to 12%
more. The paper's own gloss is accurate and modest: "Mamba's memory requirement is **comparable** to
a similar-sized Transformer with an extremely optimized implementation, and we expect further
improvement in Mamba's memory footprint in the future." Anyone who has absorbed "Mamba is the
memory-efficient one" is thinking of *inference* (no KV cache, genuinely constant) and has silently
transferred it to *training*, where this table says the opposite. This is the paper's own number
undercutting its own reputation, and the card should show the table.

### The other modalities

**DNA (§4.3, Table 5 — Great Apes classification, five species sharing 99% of their DNA, random
guessing 20%).** Accuracy after fine-tuning at matched pretraining context length:

| Model | Params | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ | 2¹⁸ | 2²⁰ |
|---|---|---|---|---|---|---|---|
| HyenaDNA | 1.4M | 28.04 | 28.43 | 41.17 | 42.22 | 31.10 | 54.87 |
| Mamba | 1.4M | 31.47 | 27.50 | 27.66 | 40.72 | 42.41 | **71.67** |
| Mamba | 7M | 30.00 | 29.01 | 31.48 | 43.73 | 56.60 | **81.31** |

Note what this table also shows: **at 2¹⁴ and 2¹² the 1.4M Mamba is worse than HyenaDNA** (27.66 vs
41.17; 27.50 vs 28.43). The win is entirely a long-context win and it arrives late. Model-size
scaling (Figure 7 left, no table) is reported as Mamba matching "the Transformer++ and HyenaDNA
models with roughly **3× to 4× fewer parameters**" at ≈40M.

**Audio (§4.4).** SC09 unconditional speech generation, Figure 10 (tabulated):

| Model | Params | NLL ↓ | FID ↓ | IS ↑ | mIS ↑ | AM ↓ |
|---|---|---|---|---|---|---|
| SaShiMi | 5.8M | 1.873 | 1.99 | 5.13 | 42.57 | 0.74 |
| DiffWave + SaShiMi | 23.0M | — | 1.42 | 5.94 | 69.17 | 0.59 |
| **Mamba** | **6.1M** | 1.852 | **0.94** | 6.26 | 88.54 | 0.52 |
| **Mamba** | **24.3M** | 1.860 | **0.67** | 7.33 | 144.9 | 0.36 |
| *Train set* | — | — | 0.00 | 8.56 | 292.5 | 0.16 |

"reducing FID on a challenging speech generation dataset by more than half" — 1.42 → 0.67, so
**52.8%**, which is exactly "more than half" and not more.

**And the counter-result the paper volunteers itself (E.4.1, Figure 18):**

> "Figure 18 shows that **the change from S4 → S6 (i.e. the selection mechanism) is not always
> beneficial. On long-form audio waveforms, it in fact significantly hampers performance**, which may
> be intuitive from the point of view that audio is uniformly sampled and very smooth, and therefore
> benefits from continuous linear time-invariant (LTI) methods."

§5 names this as a general limitation: "**No Free Lunch: Continuous-Discrete Spectrum.** … the
selection mechanism overcomes their weaknesses on discrete modalities such as text and DNA; but this
conversely can impede their performance on data that LTI SSMs excel on." Also from §4.4.1: the audio
pretraining result "is the only experiment in this paper in which we switched from the real
parameterization to complex".

### Translated to this app `[derived from shapes — not measured, not run]`

The app is `D = 32`, 4 heads, `d_k = 8`, 2 blocks, `T = 16`. The paper's default state dimension is
`N = 16` and its block expansion is `E = 2`.

| object | formula | this app |
|---|---|---|
| this app's state per head (`S` + `Z`, from concept 9) | `d_k² + d_k` | **72 numbers** |
| this app's state per block (4 heads) | `4(d_k² + d_k)` | **288** |
| softmax KV cache per head at `T` | `2·T·d_k` | **256** at `T = 16` |
| a faithful Mamba state at `D = 32`, `E = 2`, `N = 16` | `(E·D)·N = 64 × 16` | **1,024 numbers per block** |

Two things fall out and both belong on screen. **First, a faithful Mamba state for a model this width
is about 3.6× larger than the app's whole per-block attention state, and 4× larger than the softmax
KV cache at `T = 16`.** Mamba's state is not small — the paper's entire §3.3 exists because `DN` is
large, `N ≈ 10–100`, and the point of the hardware-aware scan is to afford a state that is *bigger*
than an RNN's, not smaller than a cache. The deck's story ("the state returns") must not slide into
"the state gets smaller"; what returns is a *fixed* state, and the paper's contribution is partly
making a *large* fixed state affordable.

**Second, the objects have different shapes.** The app's state is one `d_k × d_k` matrix per head —
key-dimension by value-dimension, an associative memory. Mamba's state is `N` numbers per channel,
`D` channels, with no key and no value; the write is `B̄_t x_t`, a vector times a scalar. There is no
outer product. Any panel that shows the app's 8×8 grid while saying "Mamba's state" is lying by
picture. Label it: **the recurrence Mamba added a per-token gate to, drawn on the state this app
already has.**

The IO argument, for scale: the fused kernel cuts traffic "by a factor of `O(N)`" with `N = 16`,
which the paper measures as 20–40×. Concept 14's note records the A100 numbers that make that
possible — HBM 40–80 GB at 1.5–2.0 TB/s, SRAM 192 KB per SM × 108 SMs at ~19 TB/s. **Total SRAM on an
A100 is about 20 MB.** A `(B, L, D, N)` state at `B = 1`, `L = 2048`, `D = 1024`, `N = 16` in BF16 is
**64 MB** — which is why Appendix D has the chunking paragraph ("For sequence length `L` too long
where we cannot fit the sequence in SRAM …, we split the sequences into chunks"). That arithmetic is
worth putting on the card, because it shows the SRAM budget is genuinely tight and the fusion is not
a free lunch.

---

## What the live view must let the reader do

Four interactions. Every number below is a **specification for a measurement the builder must run**,
not a result — the driver was not written for this note. Drive `app/model/mixers.js` and
`app/model/transformer.js` with node (ESM, absolute paths), and label everything that reaches the
screen `[measured here]`.

### The seam, as it stands, and the minimal faithful extension

`stateMixer({ write, decay, beta, phi, features, sumNorm, attnNorm })` already runs the recurrence
this card needs. Reading `mixers.js` line by line:

- **line 117** — `const g = write === "gated" ? decay : 1;` — the decay is computed **once, outside
  the per-token work, from a constant**. This single line is the LTI assumption, sitting in this
  codebase. `g` is `Ā`, and it is the same at every token.
- **line 127** — `S[a][b] = S[a][b] * g + target * k[b]` — this is Eq. 2a's shape, `h_t = Ā h_{t−1} +
  B̄ x_t`, with the write being an outer product rather than a scalar-times-vector.
- **line 121** — `cur[a] = dot(S[a], k, 0, m) * g` — the delta rule's read-before-write, which Mamba
  does not have.
- **line 142** — `snapshots.push(...)` — the per-token state is already retained for drawing.
  **The state film is free.**
- **line 129** — the `norm` accumulator decays with the same `g`.

**The minimal faithful extension is one line: let `decay` be a function of the token, not a number.**

    const g = write === "gated" ? (typeof decay === "function" ? decay(i, at, K[i], V[i]) : decay) : 1;

That is the entire change, and it is exactly the change Algorithm 2 makes to Algorithm 1: a parameter
that had no length dimension acquires one. Nothing else in the mixer needs to move — `snapshots`,
`denominators`, `reads` and the views all keep working.

**Two honesty constraints on how the card uses it.**

1. **The mixer never sees `x_t`.** It receives `Q, K, V, dh, at`, so a per-token gate computed inside
   it must be a function of this token's `k` or `v` vector, not of the block's hidden state. Mamba's
   `s_Δ(x) = Broadcast_D(Linear_1(x))` is a projection of the token's *residual-stream* vector. The
   faithful options are (a) pass the block's normed input through `at` in `transformer.js`, or
   (b) use `dot(K[i], w)` with a fixed seeded `w` as a declared stand-in for `Linear_1(x_t)`. Pick
   one and say which in the panel caption. **Do not silently use `k` and call it `x`.**
2. **Tie the write strength to the decay, per Theorem 1.** In the app, `decay` and `beta` are
   independent. In the selective SSM they are not: Appendix C's proof gives `B̄ = 1 − Ā` exactly. A
   faithful selective mode should therefore compute one scalar `Δ_t = softplus(w·x_t + b)`, set
   `g_t = exp(−Δ_t)` (i.e. `A = −1`), and set the write strength to `1 − g_t`. **One knob, two
   coupled effects** — that is the thing the picture can show that a paragraph cannot.

### Interaction 1 — the dataflow picture, with a decay that changes per token

Reuse `flowPanel` and `flow.update({ …, opts: { stateMode: { matrix: snap }, qkvBadge: … } })`
exactly as `linear-attention.js` and `delta-rule.js` do. The picture is concept 9's — no score matrix,
one grid in the middle — and what this card changes is *the arrow*, not the shape.

Above the grid, one strip the earlier cards do not have: **one bar per token showing that token's own
`Δ_t` (or equivalently its `g_t`)**, coloured so that "keeps the past" and "wipes the past" are
visibly different. Set `qkvBadge` to something like `"Δ, B and C are read off each token; A is not"`.

The token slider steps `snapshots[i]`, as in the sibling cards. What the reader should see, and the
caption should name, is that under a **constant** decay the grid fades at a fixed rate regardless of
what the word was, and under a **selective** decay one particular token flattens the grid while
another leaves it untouched. Same sentence, same weights, one line of difference.

### Interaction 2 — the selective-copying lab. This is the card.

A standalone panel in the style of concept 11's capacity and overwrite labs — **not** the sentence
pipeline, because the claim is about a task, not about these 16 words. Build the paper's task at the
app's scale.

**Procedure, exactly.** A stream of `L` positions over a small vocabulary. `m` of them are *data*
tokens; the rest are a *noise* token. The noise positions are **randomised per trial** — that
randomisation is the entire difference between Copying and Selective Copying and the panel must let
the reader turn it off. At the end, read the data tokens back out of the state in order. Report mean
recovery error, or exact-match rate, over ~30 seeds.

**Three rules on the same axes, as three lines:**

| rule | how the decay is set | what it can do |
|---|---|---|
| **no decay** (`write: "add"`, concept 9) | `g = 1` | accumulates everything, noise included |
| **fixed decay** (`write: "gated"`, constant) | `g = c`, a slider | forgets *on a schedule* — the LTI model |
| **selective decay** (`write: "gated"`, per-token) | `g_t = 0` on noise, `g_t = 1` on data | forgets *on command* |

**The toggle that makes it honest: `regular spacing` / `random spacing`.** With regular spacing, the
fixed decay should be able to do well — the paper says exactly this, that global convolutions "can
solve the vanilla Copying task … because it only requires time-awareness". Flip to random spacing and
the fixed decay should collapse while the selective one does not, because now the right amount to
forget depends on what arrived, not on when. **The reader must be able to break the fixed-decay
model by moving one switch that changes nothing about the model.** That is Figure 2 of the paper,
made interactive, and it is the single most important thing on this card.

The headline readout is one line under the chart with four measured numbers:

> **regular spacing — fixed decay X, selective Y · random spacing — fixed decay X′, selective Y′**

with the expectation (to be confirmed, not asserted) that `X ≈ Y` and `X′ ≪ Y′`. If the measurement
does not show that, **report what it does show**; a hand-set gate on a toy is not guaranteed to
reproduce a trained 4096-length result, and the caption should say the gate here is *given*, not
learned.

Alongside, print the paper's own numbers for the same task as a separate, clearly-labelled block:
S4 layer **18.3**, S6 layer **97.0** in the same gateless architecture; 4096-length sequences, 16
data tokens, 2 layers, `D = 64`. The reader should see the toy and the real result side by side and
be told which is which.

### Interaction 3 — one dial, both effects: Δ made visible

A small panel that is pure arithmetic and needs no model run — Appendix C evaluated on screen.

A slider over `Δ` on a log scale, roughly `0.001` to `100` (the paper initialises the `Δ` bias to
`τ_Δ^{-1}(Uniform([0.001, 0.1]))`, so mark that band). Display, live:

    Δ           0.001            1.0            100
    Ā = e^(−Δ)  0.999            0.368          ~0
    B̄ = 1 − Ā   0.001            0.632          ~1
    behaviour   token ignored    blended        state reset, token taken

with a tiny state-trace beneath: a scalar `h` driven by a fixed input sequence at that `Δ`, so the
reader watches the same signal go from "flat line, nothing gets in" to "spiky, only the last token
matters". The number that proves it is **`Ā + B̄ = 1.000` at every `Δ`** — printed and never moving.
That identity is Theorem 1, it is one line of arithmetic, and it is the clearest possible statement
of "the forget gate and the input gate are the same decision".

Caption must carry the paper's own sentence: *"a large `Δ` resets the state `h` and focuses on the
current input `x`, while a small `Δ` persists the state and ignores the current input."*

### Interaction 4 — fixed against selective, on the reader's own sentence

Run three configurations of `stateMixer` on the **same seeded weights and the same sentence** — the
seam is built for exactly this — and report per-token:

- **max |S|** and the state's effective rank or spectral spread, under `add`, fixed-`gated`, and
  selective-`gated`. The fixed decay should show a geometric envelope; the selective one should show
  a *ragged* one, with visible drops at particular words.
- **Half-life in tokens**, computed from the decay actually applied: `ln(0.5)/ln(g)` for the fixed
  case, and the **per-token distribution** of that quantity for the selective case. One scalar versus
  a histogram is the whole concept in one readout.
- **How far the outputs differ** (L2 and cosine, against the `add` baseline), so the reader sees that
  this is a different function and not a speed trick — the same honesty move `linear-attention.js`
  makes at its "how far the outputs differ" readout.

**Hard caveat to print in the panel, in the same words the sibling cards use:** the weights are
**seeded and untrained**, so *which* words the gate opens on is noise. What is not noise is the
**shape** — that a fixed decay produces one envelope and a per-token decay produces many. Read the
envelope, not the words.

### What this card must not build

- **A Mamba block.** No `E = 2` expansion, no SiLU branch, no short convolution. The card explains a
  recurrence and a gate; the block is a paragraph and a picture, not a panel.
- **A speed demonstration.** Nothing in a browser at `T = 16` can show a memory-hierarchy argument.
  Draw the HBM/SRAM picture as a static diagram with concept 14's constants and say plainly that the
  app cannot measure it — the same discipline concept 9's card uses for its 4,000×.
- **A claim of parallel training.** The app's mixer is a `for` loop over tokens; that loop *is* the
  sequential recurrence. The scan is what makes it affordable on a GPU and the app has no GPU. State
  the scan as a fact about the implementation and do not animate it as if the panel were doing it.

---

## What the source does *not* establish

- **The scale ceiling is the paper's own stated limitation, and it is 2.8B.** §5, verbatim: "Our
  empirical evaluation is limited to small model sizes, below the threshold of most strong open
  source LLMs (e.g. Llama) as well as other recurrent models such as RWKV and RetNet, which have been
  evaluated at the 7B parameter scale and beyond. **It remains to assess whether Mamba still compares
  favorably at these larger sizes.** We also note that scaling SSMs may involve further engineering
  challenges and adjustments to the model that are not discussed in this paper." Scaling laws stop at
  1.3B; the largest trained model is 2.8B; the 6.9B in the throughput benchmark is explicitly
  **untrained**. "Transformer-quality" is a claim about ≤2.8B models.
- **"Matches Transformers twice its size" is against Pythia, not against Transformer++.** Table 1
  contains no Transformer++ row. The paper's own Figure 6 and §E.2.2 report that the improved recipe
  is worth a large amount, and that RetNet, H3++, Transformer++ and Mamba all received it while the
  Table 1 baselines did not. The two claims — "matches Transformer++ in scaling laws (≤1.3B)" and
  "beats Pythia at 2× size (2.8B)" — come from different experiments against different baselines and
  are routinely merged into one sentence they do not support.
- **The paper does not test recall-intensive tasks.** Its retrieval evidence is **induction heads at
  vocabulary 16 with 2 layers and 74K parameters**, plus selective copying with 16 data tokens. There
  is no associative-recall benchmark with a realistic vocabulary, no needle-in-a-haystack, no
  phone-book or key-value lookup at scale, and no long-context QA. The million-length result is a
  generalisation result on a toy, not a recall result on language.

  Later work quantified the gap that this paper leaves untested — most directly **Jelassi et al.,
  *Repeat After Me: Transformers are Better than State Space Models at Copying*
  ([arXiv:2402.01032](https://arxiv.org/abs/2402.01032), Feb 2024)**, which proves that a two-layer
  Transformer can copy strings of length exponential in its size while a fixed-state model is bounded
  by its state, and reports that pretrained Transformers substantially outperform pretrained SSMs at
  copying and retrieving from context. **That finding is not in this paper and this note has not
  verified its numbers** — it is flagged here only so the card does not present the induction-heads
  row as a settled result about recall. The structural argument, though, is one the deck already owns
  from concept 11: a fixed-size state has a capacity, and §4.1 of Schlag et al. bounds it. Mamba
  raises the state (`DN` instead of `d_k²`) and learns what to put in it; it does not repeal the
  bound.
- **Selectivity is not universally an improvement, and the paper says so twice.** E.4.1: on long-form
  audio "the change from S4 → S6 … in fact **significantly hampers performance**". §5: "this
  conversely can **impede** their performance on data that LTI SSMs excel on." The one modality where
  the paper switched to complex parameterisation is also the one where its central mechanism hurts.
- **The training-memory claim is "comparable", and Table 7 has Mamba higher in every row** (4.8 vs
  4.6 GB at batch 1, 38.2 vs 34.5 GB at batch 32). "Constant memory" is an *inference* property. Any
  card that says Mamba is the memory-efficient option at training time is contradicted by the paper's
  own appendix.
- **The 5× throughput is a batch-size effect against a HuggingFace baseline.** The authors state the
  mechanism — "since without the KV cache it can use much higher batch sizes" — and E.5 names the
  baseline as "the standard Transformer implementation in the Huggingface transformers library". §4.5
  reports it as a **4–5×** range. This is the same species of comparison as concept 9's 4,462× versus
  its own stateful-softmax 56×: real, correctly described in the body, and quoted out of context
  everywhere else.
- **The scan is not uniformly faster than attention.** It crosses FlashAttention-2 at **2K tokens**;
  below that, attention wins. The 20–40× is against an unfused PyTorch scan. The 7× is at 32K
  specifically. The 3× in the introduction is against prior SSM implementations. Four numbers, four
  baselines, one figure.
- **`A` is not made selective, and the paper's reason is a hypothesis, not a result.** "We
  hypothesize that making `A` selective in addition to (or instead of) `Δ` would have similar
  performance, and leave it out for simplicity." No ablation tests it. The claim that `Δ`-selectivity
  suffices rests on the algebraic observation that `A` acts only through `exp(ΔA)`, which is sound,
  plus an untested guess about performance.
- **The architecture contributes little, and the paper's own table says so.** Table 2: swapping the
  block (H3 ↔ Mamba) moves perplexity by 0.26 (8.95 → 8.69) while swapping the layer (S4 ↔ S6) moves
  it by ~1.6. E.2.2 adds that "neither change matters too much" when interleaving Mamba with MLP or
  MHA blocks, and that "A large improvement is achieved by the improved training recipe". The
  eponymous architecture is the least load-bearing part of the paper.
- **Two of the headline results are figures, not tables.** The scaling laws (Figure 6) and the
  efficiency benchmarks (Figure 12) have no tabulated numbers anywhere in the paper or its
  appendices. Any Mamba scaling-law perplexity circulating as a number was read off a plot.
- **This is not a claim to have replaced attention, and the discussion is candid about what is
  untested.** §5, "Downstream Affordances": Transformers have "a rich ecosystem of properties and
  modes of interaction … such as fine-tuning, adaptation, prompting, in-context learning, instruction
  tuning, RLHF, quantization" and "We are particularly interested in whether Transformer alternatives
  such as SSMs have similar properties and affordances." None of that is evaluated here.
- **The app proves none of the efficiency and cannot.** The hardware-aware scan is the paper's second
  contribution and it is unobservable in a browser. What the app can honestly show is **selectivity**:
  that a per-token decay solves a task a per-schedule decay cannot, that `Ā + B̄ = 1`, and that the
  state's envelope changes shape when the gate becomes a function of the token. It cannot show 5×, or
  20–40×, or 1M tokens.
- **The model is untrained.** Seeded weights shaped by a rule. In this card the gate is *set by the
  reader or by a hand-written rule*, not learned — which is a bigger departure than the earlier
  state cards, because in the paper the entire claim is that the gate is *learned* from data. Say so
  in every caption that shows a gate.

---

## Leaves behind

### Backward — three cards converge here, and they arrived from different directions

**Concept 9 (`linear-attention`)** built a fixed-size state by removing the softmax and regrouping a
sum. **Concept 11 (`delta-rule`)** noticed the state was a weight matrix and gave it a correcting
write. Both arrived at the state *from attention*, by subtraction. This card arrives at the same
object *from control theory*, by discretisation — and the paper's own §2 does the reconciliation
itself, filing linear attention as a member of its family:

> "**Linear attention** (Katharopoulos et al. 2020) is an approximation of self-attention involving a
> recurrence which can be viewed as **a degenerate linear SSM.**"

"Degenerate" is doing real work in that sentence. Appendix B.2 says RetNet "reduces the inner S4 layer
to a special case where the state dimension is `N = 1`", and B.3 says the older gated RNNs "do not
use state expansion (`N = 1`)". **The SSM view's contribution to the deck is a parameter the
attention view never had: `N`, the size of the state per channel.** Concept 9's state is `d_k × d_k`
because that is what an outer product of a key and a value happens to be; nobody chose it. Mamba
chooses `N = 16` and measures what happens at 1, 2, 4, 8, 16 (Figure 16). That is the difference
between a state you inherited and a state you designed.

**Concept 14 (`flashattention`)** is the load-bearing dependency, and the card should say so
explicitly: without IO-awareness, Algorithm 2 is untrainable at scale. The same author, the same
argument, the same three techniques (fuse, tile into SRAM, recompute in the backward pass), applied
to the SSM state instead of the score matrix. §3.3.2 and Appendix D cite Dao et al. 2022 by name and
end at "the same memory requirements as an optimized transformer implementation with
FlashAttention". **Concept 14 taught the deck that the bottleneck is memory traffic, not arithmetic;
concept 20 is the first card where that lesson unlocks a mechanism rather than accelerating one.**

**Concept 5 (`transformer-xl`)** already routes forward to this card in `mechanisms.js` — "a
read-only cache chosen by **recency** rather than learned". That is the sharpest backward link
available and it should be used. Transformer-XL keeps the previous segment and drops what falls off
the end: forgetting by position, decided before the data arrives. A selective `Δ` is forgetting by
content, decided by the token. Between them sits the fixed decay of a plain gated state, which
forgets on a schedule — geometrically rather than by a cliff, but still without looking. **Three ways
to forget, in chronological order: by cliff (2019), by schedule (linear/gated states), by content
(2023).** The middle one is what the app's `stateMixer` currently implements, and the whole of this
card is the move from the middle to the right.

**And a correction to carry back to concept 9.** That card's "Leaves behind" says an add-only state
"cannot take anything back", and names the delta rule as the fix for *correction* and gating as the
fix for *release*. This paper's Theorem 1 sharpens what release means: `h_t = (1 − g_t)h_{t−1} +
g_t x_t` is not just decay-with-a-knob, it is decay whose knob is **read off the token**, with the
write strength forced to be its complement. Concept 9's card should not be edited, but this card
should note that the "release valve" it anticipated arrived in a specific form — coupled to the
write, and content-dependent — and that the coupling is a theorem, not a design choice.

### Forward — what this paper leaves on the table, and who collects

**One: the write is still additive, and there is still no correction.** Look at Eq. 2a again:
`h_t = Ā_t h_{t−1} + B̄_t x_t`. The state decays and the token is added. **Nothing in the recurrence
reads the state before writing to it.** That is concept 11's entire complaint about `S += vkᵀ`,
reappearing in a model that solved a different problem. Mamba can *release* — that is the whole
contribution — but it cannot *correct*: it has no term of the form `(v − S k)`, and the only way it
can revise an association is to decay everything, including the associations it had no opinion
about. Which is precisely the collateral damage Appendix B of the delta-rule paper argues against for
Peng et al.'s gated rule.

So the deck's three operations line up cleanly, and this card completes the set:

| operation | card | rule |
|---|---|---|
| **accumulate** | concept 9, `linear-attention` | `S ← S + v φ(k)ᵀ` |
| **correct** | concept 11, `delta-rule` | `S ← S + β(v − S φ(k)) φ(k)ᵀ` |
| **release** | **concept 20, this card** | `h ← Ā_t h + B̄_t x`, with `Ā_t` read off the token |

and the synthesis is **concept 23, `gated-deltanet`**, which is *both* — decay the past **and**
correct the present. `mixers.js`'s `"gated"` branch already implements that synthesis
(`S = gS + β(v − gSk)kᵀ`), which is why `delta-rule.md` is careful to say that branch is not the rule
Schlag et al. argued against. **This card should be equally careful in the other direction: the
`gated` branch is not Mamba either.** Mamba has no `(v − gSk)` term. What this card borrows from that
branch is the `g`, and what it adds is that `g` becomes a function of the token. The gated-DeltaNet
card is where the two halves finally meet, and it should be able to open by pointing at exactly two
predecessors: the correction came from concept 11, the input-dependent gate came from here.

**Two: the parallel-training problem, restated in a new key.** Concept 11's card ends on the delta
rule destroying parallel training, and concept 22 (`parallel-deltanet`) paying that debt with a
chunked WY representation. Mamba hits the same wall from the other side — "This simple change poses a
technical challenge … all prior SSMs models must be time- and input-invariant in order to be
computationally efficient" — and pays it differently, with a **parallel scan plus a fused kernel**
rather than with a reformulation of the recurrence. Two papers, one year apart, two different
answers to "the recurrence is sequential now": *change the algebra* (2024, WY) or *change the
memory movement* (2023, scan in SRAM). The deck can put those side by side and the comparison is
genuinely instructive, because one of them is a mathematical result and the other is an engineering
one, and they solve the same problem.

**Three: the honest open question this paper hands to the rest of the timeline.** §3.1's framing is
that the efficiency-versus-effectiveness trade "is characterized by how well they compress their
state: **efficient models must have a small state, while effective models must have a state that
contains all necessary information from the context.**" Mamba's answer is *learn what to throw away*.
It is a good answer and Table 2 shows it is worth 1.6 perplexity. But it is an answer to the
*compression* question, and the deck's other live thread — concepts 21 and 24–25, `mla`, `nsa`,
`dsa` — is pursuing the opposite answer: **keep everything, and get cleverer about what you read.**
Mamba is the strongest statement in this deck that the compressed-state branch is viable at scale;
it is not a statement that the branch won. Its own §5 says "It remains to assess whether Mamba still
compares favorably at these larger sizes", and the frontier models that followed did not, in the
main, stop keeping a cache.

The pressure this card names is **compressing the past** — the same pressure as concepts 5, 9 and 11.
What it adds to that pressure is a verb the earlier cards did not have. They asked *how much do you
keep*; this one asks **who decides**. Every mechanism before it answered "the architecture, before
the data arrives". This one answers "the token, as it goes past" — and the price is that a
convolution stops existing, a scan has to be written by hand, and the whole thing only became
affordable because concept 14 had already worked out where the memory is.
