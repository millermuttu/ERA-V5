# KronMath — arithmetic that lives in the embedding, not in the weights

**Session 7 assignment, Problem 1.** *"What if embeddings can store mathematical structure as
well. Say 9 — somehow it has stored the meaning of 9 (in absolute math terms), such that when
we actually do 9 + 9, the mathematical meaning part of the embeddings is itself 18! When we do
9\*9 it becomes 81! How much can we push? ... for that we can use the 32 existing spaces, and
add this new concept into new ones (that are appended)."*

This is a fixed, parameter-free block appended to the Kronecker byte grid. The byte grid says
what a token *spells*. The appended block says what it *is worth*, in a basis where addition
and multiplication are both a single rotation.

```
9 + 9 -> 18     9 * 9 -> 81     (9+9)*3 -> 54     12*12+6 -> 150
```

read straight out of the vectors, with no training and no model involved.

---

## 1. The construction

For each prime `p` we carry two characters of the two cyclic groups that live on `Z/p`:

| channel | encoding | the operation it linearises |
|---|---|---|
| **A** (additive) | `[cos(2πn/p), sin(2πn/p)]` | `n → n+m` rotates the phase |
| **M** (multiplicative) | `[cos(2π·ind(n)/(p−1)), sin(...)]` | `n → n·m` rotates the phase |

`ind(·)` is the discrete logarithm to a primitive root `g mod p`. `(Z/p, +)` is cyclic of
order `p`, and `(Z/p)^×` is cyclic of order `p−1`, so **both** groups admit a faithful phase
representation — and the discrete log is precisely the isomorphism between them.

The consequence is that addition and multiplication become *the same operation on the
embedding*: add the phases, i.e. multiply the `(cos, sin)` pair as a complex number. They just
act on different channel families.

```python
add:  A-channels  ×               sub:  A-channels  × conj
mul:  M-channels  ×               div:  M-channels  × conj
pow:  M-channel angle × k
```

Integers beyond a single prime come from the **Chinese Remainder Theorem**: carry primes
`3·5·7·11·13` and the representation is exact on `Z/15015` in **22 dimensions**.

### Zero handles itself

`n ≡ 0 (mod p)` has no discrete logarithm. It encodes as the **zero vector**, which is
absorbing under complex multiplication — exactly what `0 · x = 0` requires. The degenerate
case falls out of the construction instead of needing a special case.

---

## 2. What is new here

[FoNE](https://arxiv.org/html/2502.09741v1) and
[Prime Fourier Embeddings](https://arxiv.org/abs/2606.23044) both encode integers as
period-indexed `(cos, sin)` pairs. PFE's block is exactly

```
PFE_{p,d}(a) = [cos(2πa / p^(d+1)), sin(2πa / p^(d+1))]
```

and its equivariance theorem is about the action `a → a + b`. Both carry **additive characters
only**. In that basis multiplication is not a rotation and there is nothing for it to act on —
the model has to learn the multiplication table from data.

**The multiplicative character channel is the delta.** It is what makes `9*9` a rotation
rather than an association to be memorised. The `families="A"` variant in `codec.py` is the
published baseline, and it *raises `ValueError` when asked to multiply* — there is no channel
for the operation to live in. That is the gap this work fills, made mechanical.

---

## 3. Result 1 — exact, parameter-free algebra

`python codec.py`

```
primes=(3, 5, 7, 11, 13)  modulus=15015  math dims=22
9 + 9 -> 18    9 * 9 -> 81
add/sub/mul exact on 43,200 cases over Z/15015
pow exact
0 * x = 0 by construction
chained: (9+9)*3 -> 54   12*12+6 -> 150
sub/negatives exact; division exact on 4,179 unit divisors, refuses non-units
appended cost: 22 dims on top of 256*32=8192  (0.27%)
additive-only variant: add works, mul has no channel  <- the delta
single-prime p=97 codec: 4 dims, add and mul exact
```

Every operator is a fixed function of the operand *vectors*. No operator consults the
integers, nothing is learned, and the check is exhaustive over the operand grid rather than
sampled. Chained expressions stay in embedding space the whole way: `canonicalize()` moves a
result between the two character families (a fixed change of basis — the discrete log), so an
expression tree is evaluated without touching an integer until the final decode.

**Cost: 22 dimensions on top of the 8192-dim byte grid — 0.27%.**

---

## 4. The honest limit, stated up front

A vector space has one addition. There is no finite-dimensional embedding that is a faithful
**ring** homomorphism for unbounded integers — you cannot have `+` and `×` both exact and both
unbounded in fixed dimension. Fixing a modulus is the standard price, and PFE pays it too, for
addition alone.

So the claim is precisely scoped: **exact on `Z/M`, where `M` is the product of the carried
primes, at 4 dimensions per prime.** `M = 15015` covers every four-digit result at 22 dims;
adding primes `17·19·23` pushes it to 111,546,435 at 34 dims. The range grows multiplicatively
in the primes while the dimension grows additively — which is the reason to use CRT rather than
one large modulus.

### What is exact, and what refuses

| operation | status |
|---|---|
| `+`, `−`, `×` | exact on all of `Z/M`, verified exhaustively over the operand grid |
| negatives | free — `−k` is the residue `M−k`, and `sub` is a conjugate rotation |
| `a**k` | exact, including `0**0 = 1` |
| `÷` | exact **iff the divisor is a unit mod M** (coprime to every carried prime) |

### Precision

The codec table can be stored at the model's own precision. Over 7,200 operations at
`M = 15015`, decode failures were **0 in fp32, 0 in bf16, 0 in fp16**. With primes at most 13,
adjacent phases sit `2π/13 ≈ 0.48` rad apart, which is far above bf16's resolution — one more
reason to reach the range through CRT rather than through a single large modulus, where the
spacing would be `2π/M`.

Division by a non-unit is the one genuinely dangerous case: `7` has a zero multiplicative
channel under `M = 3·5·7·11·13`, and conjugate-multiplying by it returns a confidently wrong
answer — `14/7` decodes to `8582`. (`49/7` happens to come out right, which is exactly how
this sort of bug survives review.) It is detectable from the vector alone, so `div()` raises
rather than lying. That mirrors the silent-collision failure the Kronecker paper documents at
`pos_dim = 32`: the dangerous errors are the ones nothing warns about.

---

## 5. Result 2 — does a transformer use it?

Setup: `a OP b (mod M)` for `OP ∈ {+, ×}`. A set of operand **values** is held out entirely —
those integers never appear as an operand during training, only in results. Session 7 §2 is why
this split is the right one: an embedding row only moves when its token is gathered, so a dense
row for a held-out operand sits at its initialisation forever. A byte codec at least gives that
token a spelling. Only the math channels give it a value.

### The finding: the M-channel matters in one regime and is redundant in the other

The additive channels alone *can* support multiplication — if the model can memorise the
per-channel multiplication table. That table has `p²` entries, and whether training covers it
is decided by one measurable quantity: **do the held-out operands still expose their residues?**

| task | per-channel residue coverage of held-out operands |
|---|---|
| CRT `7·11·13` (mod 1001) | **1.00, 1.00, 1.00** |
| single `p = 97` | **0.00** |

Under CRT with small primes, an operand held out at the integer level (say 613 of 0..1000)
still has residues `613 mod 7`, `mod 11`, `mod 13` appearing in hundreds of other training
operands. Each channel's table is 49, 121 or 169 entries and is fully covered, so the additive
baseline learns multiplication as three small lookups and **the M-channel is redundant**.

Under a single large prime, held-out operand *is* held-out residue. The table row is never
seen and `p² = 9409` is far too large to interpolate. The additive baseline has nothing to fall
back on, while the M-channel turns the same operation into one continuous rotation.

The general statement, and the one worth keeping:

> Addition is a continuous rotation in the A-basis, so it generalises to operands never seen.
> Multiplication is a continuous rotation **only in the M-basis**; in the A-basis it is a
> discrete table. Small-prime CRT hides this by making the table small enough to memorise.

### 5a. The representational claim, isolated from optimisation luck

Before asking whether a transformer *happens* to find the algorithm, ask the sharper question
underneath: **is the operation a bilinear function of the two operand embeddings?** Phase
addition is complex multiplication, which is bilinear in the two `(cos, sin)` pairs, so if an
operation is a rotation in some carried channel then the smallest possible readout —
a linear map on `emb(a) ⊗ emb(b)`, no depth, no attention, no nonlinearity — can express it,
and it extrapolates to unseen operands because it is continuous in the phase rather than a
table lookup.

`python probe.py --prime 97` — fitted readout, trained on seen operands only, evaluated on
pairs containing a held-out operand:

| channels | dims | op | test | vs chance |
|---|---|---|---|---|
| A | 2 | `+` | 0.536 | **52.0×** |
| A | 2 | `×` | 0.008 | 0.8× — chance |
| M | 2 | `×` | 0.557 | **54.1×** |
| M | 2 | `+` | 0.016 | 1.5× — chance |

Perfectly complementary: **each family carries exactly one operation and is at chance for the
other.** `test ≈ train` throughout, so this is extrapolation, not memorisation.

(`probe.py` also prints the combined `AM` rows, which score *lower* than either family alone.
That is a fitting artifact, not a representational one: with 4 dims the outer product has 16
features instead of 4, twelve of which are cross-channel noise for any single operation, so the
same step budget gets further from the optimum. The analytic construction below shows the exact
solution is present in each family regardless.)

The fitted numbers understate it, because bounded logits give cross-entropy weak gradients.
The exact readout can be *written down* instead of fitted — with
`W[:,c] = cos(t_c)·(1,0,0,−1) + sin(t_c)·(0,1,1,0)` the logit is `cos(t_a + t_b − t_c)`,
maximised exactly when `t_c` is the phase of the result. Evaluated exhaustively over all
`97² = 9409` operand pairs:

```
A/add: all 9409 pairs -> 1.0000 | neither operand 0 -> 1.0000
M/mul: all 9409 pairs -> 0.9795 | neither operand 0 -> 1.0000 | all failures have a 0 operand
```

and it is not specific to `p = 97` — exhaustively over every pair, for every prime tested:

| p | A / `+`, all pairs | M / `×`, all pairs | M / `×`, neither operand 0 |
|---|---|---|---|
| 13 | 1.0000 | 0.8521 | **1.0000** |
| 31 | 1.0000 | 0.9365 | **1.0000** |
| 61 | 1.0000 | 0.9675 | **1.0000** |
| 97 | 1.0000 | 0.9795 | **1.0000** |
| 127 | 1.0000 | 0.9843 | **1.0000** |
| 251 | 1.0000 | 0.9920 | **1.0000** |

The middle column is exactly `1 − (2p−1)/p²`, the fraction of pairs containing a zero operand
(at `p = 13`: `25/169 = 0.1479`, and `1 − 0.1479 = 0.8521`). The match is exact at every prime,
which confirms the residual is the missing bias term and nothing else.

**100%, with zero training.** The 2.05% residual is a readout artifact, not an algebra
failure: a pure bilinear form has no bias term, so an operand that is `0 mod p` produces the
zero vector and every logit ties. The codec's own `decode` handles that case correctly.

To be precise about scope: bilinearity is a **per-channel** property. Each channel's phase
addition is bilinear in that channel's two `(cos, sin)` pairs, which is what the table above
measures. Recombining several channels into an integer is CRT — a fixed lookup, not a bilinear
form. So the honest statement is that every carried operation is bilinear *in the basis the
codec provides*, one prime at a time, and the multi-prime range comes from composing those.

### 5b. End to end, in a transformer

`python train.py --primes 97 --arms math+A math+AM` — 4-layer, `d_model` 128, untied head,
operand values held out entirely. Chance is `1/97 = 0.0103`.

| arm | input params | `+` | `×` | `×` vs chance |
|---|---|---|---|---|
| `kron` (byte grid only) | 262.1 K | 0.912 | 0.017 | 1.6× — chance |
| `math+A` (PFE / FoNE baseline) | 0.3 K | 0.569 | 0.016 | 1.5× — chance |
| `math+AM` (math channels alone) | 0.5 K | 0.075 | **0.100** | **9.7×** |
| **`kron+AM`** (the proposed layer) | 262.7 K | **0.935** | **0.090** | **8.7×** |

Two things to read here, and the second is the more useful one.

**The M-channel produces a real multiplication gain, and it composes with the byte grid.**
`kron+AM` is the configuration the assignment actually describes — byte grid untouched, math
channels appended — and it is the best arm on both operations at once: addition **0.935**
(above the byte grid's own 0.912) and multiplication **5.3× the byte grid's**, for 600 extra
projection parameters on top of 262 K. The additive-only baseline is pinned at chance on
multiplication, as is the byte grid alone. The byte grid generalises *addition* beautifully
without any help, because it is a digit decomposition and digit-wise addition is learnable from
spelling. Multiplication is where representations actually separate.

**But the transformer extracts only a fraction of what is there.** The probe shows the same
4 numbers support 100% multiplication accuracy under a *linear* readout on the outer product.
The model gets 0.090–0.100. And `math+AM` without the byte grid is markedly *worse at addition*
than the additive-only arm (0.075 vs 0.569) — the same pattern the probe's `AM` rows show, for
the same reason: extra channels are extra features to select among, and at a small step budget
the routing cost exceeds the gain. (The byte grid rescues this in `kron+AM`, which reaches
0.935 on addition, because it supplies a second, easier route to the digits.)

Caveat that applies to every number in this table: none of these runs had converged. `math+A`
was still climbing when it was stopped (`+`: 0.425 at 5 K steps, 0.569 at 10 K). These are
fixed-budget comparisons on a 4 GB card, so they show *direction*, and the multiplication
column is only trustworthy because chance is 0.0103 and two of the three arms never leave it.

The bottleneck is **extraction, not representation.** That points somewhere concrete: the
operation is known at the input (there is an operator token sitting right there), so the
channel should be selected architecturally — operator-conditioned gating, or an explicit
bilinear path — rather than left for the optimiser to discover. That is the obvious next
experiment and it is not run here.

### 5c. The regime where the M-channel is redundant (honest negative)

Under CRT with small primes the additive baseline reaches **474× chance** on multiplication
(`math+A`, mod 1001, `*: 0.474`). It is not doing algebra — it is memorising three tiny tables
of 49, 121 and 169 entries, which the training data fully covers because held-out *integers*
still have seen *residues*. Anyone reporting a win for the M-channel without checking residue
coverage would be reporting an artifact of their choice of primes.

---

## 6. Files

| file | what |
|---|---|
| `codec.py` | the codec, the operators, the decoder, and the exhaustive self-check |
| `probe.py` | the bilinear readout probe — fitted, and the exact constructed solution |
| `train.py` | the controlled transformer comparison |
| `sweep.py` | prime-size sweep (see the note on small primes below) |

```bash
python codec.py                                          # exact algebra, no training
python probe.py --prime 97                               # representational claim
python train.py --primes 97 --arms math+A math+AM        # end to end, discriminating regime
python train.py --primes 7 11 13                         # the regime where M is redundant
```

## 7. What I would not claim

- **Small primes are not a fair test.** At `p = 13` the 0.2 holdout leaves 2 held-out operands
  and 121 training pairs — every arm sits at chance, for lack of data rather than lack of
  structure. The sweep is kept for the record, but the discriminating regime is a single large
  prime, where there are enough operands to learn a rotation and too many table entries to
  memorise one.
- **The transformer numbers are point estimates.** The probe results are exhaustive and
  analytic; the end-to-end runs are 1–2 seeds on a 4 GB card and should be read as direction,
  not as a measured effect size.
- **This does not "describe whole mathematics".** It describes the ring `Z/M` — `+`, `−`, `×`,
  powers, and division by units. Anything requiring unbounded magnitude, ordering across the
  modulus, or non-commutative structure is outside it. The magnitude channels (`linear`, `log`)
  are carried for ordering but are approximate, not exact, and no claim is made for them.
- **Coexistence with text is by construction, not by measurement.** The math block is appended
  to the byte grid and the byte grid is untouched, so nothing about spelling changes; but I did
  not run a language-modelling control to confirm the extra 0.27% of input dimensions is free
  in practice.
