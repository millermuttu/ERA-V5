# KronFourier — each character a wave, added to make a word

**Session 7 assignment, Problem 4.** *"What is a REAL Fourier alternative of Kronecker? Why
can't I represent each character like a Fourier wave, and just add them to make a word!"*

Taken literally, a sum of per-character waves is order-blind — `abc` and `cba` give the same
vector. Position has to enter through **phase** rather than through a second factor, and once
it does the construction has a name: **Fourier Holographic Reduced Representations**.

```
phi[v]     unit-modulus phasor per byte value, fixed random, N complex dims
psi^p      position p as a phase rotation exp(i*p*theta), theta fixed random
encode(b)  (1/sqrt(L)) * sum_p  phi[b_p] (*) psi^p        elementwise complex multiply
```

Stored as `[real, imag]`, so `D = 2N`, and `N = 4096` matches the released codec's `D = 8192`
**exactly**. Nothing is learned; `phi` and `theta` come from a seed. Every comparison below is
at that matched `D`.

`python fhrr.py`:

```
fhrr: ok  (N=4096, D=8192)
  shift equivariant, unbind exact on short tokens, length-free
  the lesson's colliding pair now at cosine 0.9346
```

---

## 1. Why phase, and what it buys

Three properties follow from binding by phase, and none of them is available to an outer
product of one-hots.

**`D` does not depend on length.** There is no `pos_dim`, so there is nothing to crop. This
does not patch P3's 32-byte wall; it removes the wall.

**There is an inverse.** `z ⊙ conj(psi^p)` rotates position `p` back out, leaving
`phi[b_p]` plus the other characters as noise. Match against the 256 phasors and the byte
comes back.

**Prepending `k` bytes multiplies the whole code by `psi^k`.** Because
`psi^{p+k} = psi^p ⊙ psi^k`, shifting every character is *one elementwise multiply on the
code* — no access to the token text. `fhrr.py` asserts this to fp32 tolerance, and it is the
property the rest of the submission rests on.

---

## 2. Capacity — how much can one vector hold?

The Fourier answer trades the byte codec's exact sparse recovery for approximate dense recall.
That trade is the entire risk, so it is measured. `python capacity.py` — per-byte recovery
after unbinding, random tokens:

| N (D=2N) | L=8 | L=32 | L=64 | L=128 | L=256 | L=512 | L=1024 |
|---|---|---|---|---|---|---|---|
| 64 (128) | 0.902 | 0.236 | 0.093 | 0.046 | 0.024 | 0.014 | 0.010 |
| 256 (512) | 1.000 | 0.877 | 0.521 | 0.225 | 0.089 | 0.042 | 0.023 |
| 1024 (2048) | 1.000 | 1.000 | 0.995 | 0.864 | 0.506 | 0.222 | 0.093 |
| **4096 (8192)** | 1.000 | 1.000 | 1.000 | 1.000 | **0.996** | 0.865 | 0.504 |

The curve is a clean function of `L/N` and reads off the diagonal:

```
L <= N/16   recovery 1.000        L = N/4   ~0.86
L = N/8     ~0.996                L = N/2   ~0.50
```

**At the released `D = 8192`, that is exact recovery to 256 bytes — eight times the shipped
32-byte window**, and graceful decay past it rather than a cliff. The byte codec at the same
`D` is 1.000 up to 32 bytes and 0.000 after, because the tail was never stored.

On real tokens, `N = 4096`, whole-token round-trip:

| population | n | FHRR | byte codec |
|---|---|---|---|
| `L ≤ 32` (fits the byte codec) | 300 | **100.0%** | 100.0% |
| `L > 32` (byte codec crops) | 300 | **100.0%** | 0.0% |

### Precision costs nothing, which I did not expect

I predicted a dense superposition would be far more fragile than the byte codec's sparse
scatter, and that P1's zero fp16 decode failures would not be matched. Casting the *code* to
bf16 and fp16 (tables kept fp32, since they are constants and the code is the activation that
travels through a model):

| dtype | L=8 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|
| fp32 | 1.000 | 0.877 | 0.521 | 0.225 | 0.089 |
| bf16 | 1.000 | 0.877 | 0.521 | 0.225 | 0.089 |
| fp16 | 1.000 | 0.877 | 0.521 | 0.225 | 0.089 |

Identical to three decimals, measured at `N = 256` where the code is *not* saturated — a
comfortable code would have hidden any precision loss. The recovery margin is set by
superposition crosstalk, which is orders of magnitude larger than fp16 rounding, so precision
never becomes the binding constraint. The prediction was wrong and the measurement says so.

---

## 3. Head to head at matched `D`

`python structure.py`, on this repo's Session 2 BPE vocabulary and a 20k sample of XLM-R.

**Identification** — top-1 retrieval of each token against the whole set, after relative
Gaussian noise on the code:

| codec | 0% | 1% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| byte | 0.9969 | 0.9969 | 0.9965 | **0.9753** | **0.5357** |
| FHRR | **1.0000** | **1.0000** | **0.9973** | 0.9726 | 0.5269 |

FHRR's clean-case win is the collision ceiling P3 measured, and no SNR ever recovers it. The
arithmetic is worth stating precisely: P3 found **57** colliding tokens at `pos_dim = 32` on
this vocabulary (0.57%), but identification loses only **31** (0.31%), because `argmax`
tie-breaking credits one member of each collided group. The 26-token gap is the number of
groups, not a discrepancy. **Under noise the two are a wash, and the byte codec is marginally
ahead from 10% onward.** FHRR removes collisions; it does not buy robustness. Any claim that a
holographic code is more noise-tolerant here would be false.

**Locality** — Spearman between codec cosine and negative byte edit distance:

| codec | Session 2, L≤32 | Session 2, L>32 | XLM-R, L≤32 | XLM-R, L>32 |
|---|---|---|---|---|
| byte (as released, z-normed) | **0.208** | 0.487 | **0.361** | 0.455 |
| byte, z-norm off | 0.023 | 0.496 | 0.146 | **0.470** |
| FHRR | 0.066 | **0.578** | 0.133 | 0.364 |

Two things here, and the first is a finding about the *baseline*, not about FHRR.

**The byte codec's short-token locality is mostly an artifact of z-normalisation.** Turning it
off drops Session 2 from 0.208 to 0.023 and XLM-R from 0.361 to 0.146. z-norm subtracts a
per-token mean, leaving every code with a large shared component whose size tracks token
length — and length correlates with edit distance. Against the raw outer product FHRR is
*better* on Session 2 (0.066 vs 0.023) and slightly worse on XLM-R (0.133 vs 0.146).

**On long tokens the result is vocabulary-dependent and I will not generalise it.** FHRR wins
on Session 2 (0.578 vs 0.487) where tokens run to 73 bytes and cropping discards a lot; it
loses on XLM-R (0.364 vs 0.455) where the longest token is 48 bytes, so the byte codec still
sees most of it. One vocabulary each way is not a trend.

---

## 4. The suffix limitation, and what actually fixes it

The paper lists this as structural:

> "Position-aware encoding weakens on suffix-only families… This is structural to
> position-aware encoding and cannot be mitigated without sacrificing the case-sensitivity and
> prefix structure that make the encoding work."

| pair | byte codec | FHRR | FHRR shift-aligned |
|---|---|---|---|
| **shared prefix** | | | |
| compute / commute | 0.8570 | 0.8599 | 0.8599 (k=0) |
| train / training | 0.7904 | 0.7975 | 0.7975 (k=0) |
| train / trainer | 0.8451 | 0.8495 | 0.8495 (k=0) |
| nation / national | 0.8659 | 0.8664 | 0.8664 (k=0) |
| **shared suffix** | | | |
| nation / creation | −0.0008 | 0.0227 | **0.7257** (k=−2) |
| separate / operate | −0.0009 | 0.0132 | **0.6765** (k=+1) |
| training / running | −0.0009 | 0.0040 | **0.6785** (k=+1) |
| nation / information | −0.0010 | 0.0164 | **0.6211** (k=−5) |

**Plain cosine does not rescue suffix families under FHRR either** — 0.004 to 0.023, which is
orthogonal for practical purposes. Being Fourier is not, by itself, the fix, and a submission
claiming otherwise would be reading its own table wrong. The shared suffix sits at different
absolute positions, so its contribution is rotated by `psi^k` and is near-orthogonal to itself.

What changes is that **the alignment is computable from the code alone**. Searching
`max_k cos(z_a, z_b ⊙ psi^k)` costs one elementwise multiply per candidate shift, recovers
0.62–0.73 on every suffix pair, and **finds the right shift every time**: `k=−2` for
`nation`/`creation`, where `ation` sits at index 1 and 3; `k=−5` for `nation`/`information`,
where it sits at index 1 and 6. `structure.py` asserts this on every run: the recovered `k`
must equal the true offset between the two words' longest shared substring, for all eight
pairs. The byte codec can only be "shifted" by re-encoding from the bytes, which
is not an operation available to something holding a hidden vector.

And the paper's condition is met: **nothing is sacrificed to get it.** The prefix rows are
unchanged — 0.86 against 0.86, with the search selecting `k=0` on its own. Case sensitivity
and prefix structure survive because the aligned similarity is an *additional* operator over
the same fixed code, not a different encoding.

So the honest statement of the result: *the limitation is not fixed by being Fourier, it is
fixed by shift equivariance, which the Fourier form is what makes available.*

---

## 5. The P5 thread — measured, not claimed

Unbinding is the operator P5 (issue #6, drop the output head) needs: §2 shows byte recovery is
exact for `L ≤ N/16`, which at `D = 8192` covers every token any tokenizer produces. That is
the whole of "invert the codec".

It is also not what P5 is actually about. The research note already established that the
*released* codec is exactly invertible too — reshape to `(256, pos_dim)`, argmax per column,
8/8 round-trip including Devanagari and emoji. P5's real problem is inverting `W_proj`, an
`8192 → 8096` projection that is nearly square but not injective. FHRR contributes nothing to
that, and the capacity curve here should not be mistaken for progress on it.

---

## 6. What I would not claim

- **No robustness win.** §3 — under noise the two codecs are a wash and the byte codec is
  marginally ahead past 10%. The win is the collision ceiling, and that is a small number
  (0.31% of this vocabulary).
- **No locality win in general.** §3 — one vocabulary each way on long tokens. What is solid
  is the negative finding about the baseline's z-normalisation, which is a claim about the
  released codec rather than about FHRR.
- **Plain cosine does not solve suffix families.** §4. The aligned similarity does, and it is
  an extra operator someone has to actually call.
- **The shift search is O(span) per pair.** Cheap against one candidate, not free inside an
  attention layer over a batch. Nothing here measures that cost in a model.
- **No downstream model.** Scoped out deliberately: P3's LM on the same 31k-token corpus was
  the weakest evidence in that submission, and repeating it would add none here. Every number
  above is a property of the codec, measured directly.
- **`phi` and `theta` are one seed.** Capacity numbers are averaged over tokens, not over
  draws of the tables; the concentration argument for random phasors says the variance is
  small, but it is not measured here.
- **Real vocabularies, English families.** The suffix pairs are hand-picked English words
  chosen to match the paper's own example. No morphological corpus is used.

---

## 7. Files

| file | what |
|---|---|
| `fhrr.py` | the codec — bind, unbind, shift, decode, aligned similarity, self-check |
| `capacity.py` | recovery vs `N` and `L`, real-token round-trip, precision sweep |
| `structure.py` | head to head with the byte codec: identification, locality, families |
| `results_*.log` | the evidence behind every number above |

Reused rather than rewritten, from [`../kronbudget/`](../kronbudget/): `vocab.py` for the four
real vocabularies as byte strings, `roles.py` for the released byte codec baseline, and
`fidelity.py` for `edit_distance` and the locality protocol.

```bash
python fhrr.py                                          # self-check
python capacity.py --trials 64                          # ~4 min, GPU
python structure.py                                     # Session 2 vocabulary
python structure.py --vocab xlm-roberta-base --sample 20000
```
