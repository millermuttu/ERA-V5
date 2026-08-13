# KronInv — dropping the output head, and what it actually costs

**Session 7 assignment, Problem 5.** *"Kronecker is forward deterministic (same word always
gives the same embedding). How do I make a reverse of this (same embedding gives the same
Kronecker)? If we can do this, then we can get rid of the final head as well! Then we can have
a vocab of 1M as well without any issues."*

Short version of the result: the reverse exists, it is cheaper than the problem statement
implies, the parameter saving is real and large — and *"without any issues"* is wrong. The
issues are measured below.

---

## 1. The codec was never the obstacle

Reshape `D → (256, pos_dim)`, argmax down the byte axis per column, and the bytes come back
exactly. z-normalisation is a per-token affine map and does not move an argmax. That is the
whole of "invert the Kronecker codec", and `invert.py` asserts it.

The obstacle is **`W_proj`**, the one trainable input-side tensor, mapping `D = 8192` down to
`d_model`. It is not injective, so *"invert `W_proj`"* is, read literally, impossible.

That reading is the wrong one, and this is the load-bearing observation of the submission:

> **`kappa` is sparse.** Before z-normalisation the grid holds at most `pos_dim` non-zeros out
> of 8192 — one per byte position. Recovering a sparse vector from few linear measurements is
> not inversion, it is **compressed sensing**. It does not need `W` to be invertible, or
> square, or even well-conditioned. It needs `d_model` to be large enough relative to the
> **sparsity**, which is the token's byte length — not relative to `D`, and not relative to
> the vocabulary.

`python invert.py`:

```
kroninv: ok  (codec exact; d=512 recovers every word through a 16x reduction)
  d=512, length known : 5/5
  d=512, length blind : 5/5
  d=192, length known : 3/5  (43x reduction)
```

`W` is built with orthonormal rows, so `W @ W.T = I` and the pseudo-inverse is exactly `W.T` —
the decoder is one matmul, no factorisation, no iterative solver.

---

## 2. The recovery law

`python recover.py` — exact token recovery through `W_proj`, by token byte length. Two numbers
per cell: **length known / blind**, the second also having to find the token's length from the
code.

Session 2 BPE (10k vocabulary):

| d_model | L 1–4 | L 5–8 | L 9–16 | L 17–24 | L 25–32 |
|---|---|---|---|---|---|
| 64 | 80.7/53.4 | 5.9/2.2 | 0.0/0.0 | 0.0/0.0 | 0.0/0.0 |
| 128 | 99.4/89.4 | 68.1/34.9 | 4.0/1.1 | 0.0/0.0 | 0.0/0.0 |
| 192 | **100.0**/97.7 | 95.1/68.8 | 27.2/11.5 | 0.0/0.0 | 0.0/0.0 |
| 384 | 100.0/100.0 | **100.0**/97.8 | 92.9/63.5 | 37.0/7.9 | 0.0/0.0 |
| 512 | 100.0/100.0 | 100.0/99.5 | 98.1/84.9 | 74.1/30.4 | 19.5/5.3 |
| 1024 | 100.0/100.0 | 100.0/100.0 | **100.0**/99.9 | **100.0**/92.3 | 98.5/69.1 |
| 2048 | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 | **100.0**/95.4 |

Qwen2.5 (151k vocabulary) reproduces it — 100.0/100.0 at `d=192` for L 1–4, `d=512` for L 9–16,
`d=1024` everywhere. **The law tracks sparsity, not vocabulary size**, which is exactly what
the compressed-sensing framing predicts and is the reason the 1M-vocabulary claim has any
foundation at all.

Read off the diagonal: roughly `d ≈ 64·L` for exact recovery with the length known. At the
released `D = 8192` the full 32-byte budget needs `d_model ≈ 2048` — a **4× reduction** from
`D`, and far below the vocabulary sizes the head would otherwise scale with.

**Robustness is essentially free.** Recovery against relative noise on the hidden vector, all
tokens, length known:

| d_model | 0% | 1% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| 512 | 85.8 | 85.8 | 85.8 | 85.6 | 84.9 |
| 1024 | 99.7 | 99.7 | 99.7 | 99.7 | 99.5 |
| 2048 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

Flat to 20%. An argmax over 256 candidates has an enormous margin, so the binding constraint
is `d_model`, never precision or noise.

---

## 3. What the output side costs

The head-free output map is `d_model × D` with `D = 8192` fixed. The tied head is
`d_model × V`. So **the ratio is just `V / D`**, independent of `d_model`:

| vocabulary | tied head (d=512) | head-free | ratio |
|---|---|---|---|
| Session 2, 10k | 5.1M | 4.2M | 1.2× |
| GPT-2, 50k | 25.7M | 4.2M | 6.1× |
| V5, 131k | 67.1M | 4.2M | 16.0× |
| XLM-R, 250k | 128.0M | 4.2M | 30.5× |
| **1M** | **512.0M** | **4.2M** | **122×** |

Head-free wins whenever `V > 8192`, and the assignment's 1M vocabulary is where it stops being
an optimisation and becomes an architectural difference.

---

## 4. Does it work in a model?

`python train.py` — wikitext-103 under the Qwen2.5 tokenizer (1.36M tokens, **V = 151,651**),
`d_model = 256`, 4 layers, 3000 steps, 2 seeds. Everything except the output side is identical.

| arm | output params | top-1 next token | not a token | wrong length |
|---|---|---|---|---|
| `tied-head` — `Linear(d, V)` | 38.82M | **0.187** ±0.000 | 0.0% | 0.0% |
| `byte-head` — `Linear(d, 257·32)` | 2.11M | 0.079 ±0.001 | 30.2% | 75.1% |
| `reuse-W` — `h @ W_proj`, **no head at all** | **0.00M** | 0.069 ±0.001 | 63.3% | 89.1% |

**The head does come off.** `reuse-W` reads the input projection backwards and adds *zero*
output parameters; `byte-head` costs 18× less than the tied head and would cost 122× less at
V=1M. Both learn something real, from a codec that was never trained.

**And it costs most of the accuracy.** 0.187 → 0.079 is a 2.4× drop, and 0.187 → 0.069 for the
zero-parameter arm is 2.7×. Against the problem statement's "without any issues", that is the
finding: a head-free Kronecker decoder is not free, it is *bought*, and at this scale the price
is more than half the top-1 accuracy.

### Where the failure actually is

The `wrong length` column is the useful diagnostic. For `byte-head`, 7.9% of predictions are
correct and 75.1% are length errors — so **more than four fifths of all errors get the token's
length wrong**, not its bytes. The code carries no length marker: with an exact `kappa`,
unoccupied columns sit at the z-norm floor and the decoder walks until the first gap, but a
*predicted* `kappa` has energy everywhere and the gap has to be guessed. `byte-head` predicts
length explicitly through a 257th "no byte here" class and still gets it wrong most of the
time; `reuse-W` cannot even represent that class — its 8192 outputs are exactly `256 × 32` —
and has to infer length from the largest drop in per-column peak height, which is why it is
worse on every column.

**Length prediction, not byte prediction, is what a head-free Kronecker decoder has to solve.**
That is not visible from the assignment's framing and it is the most useful thing here.

### The other catch: parameters go away, normalisation does not

Both factorised arms predict a distribution over **byte strings** that factorises across
positions. It puts mass on byte strings that are not tokens — 30.2% and 63.3% of predictions
land outside the vocabulary. For argmax decoding that is just an error. For *sampling* it is
worse: getting a normalised distribution over the vocabulary means scoring `V` candidates
again, which is `O(V)` compute even though the parameters are gone. The alternatives are
sampling per byte position and accepting invalid tokens, or a constrained decode over a trie
of the vocabulary — neither of which this submission builds.

---

## 5. What I would not claim

- **Not a free win.** §4. Every headline number about parameters is real; the accuracy cost is
  large and is the reason to be suspicious of "vocab of 1M without any issues".
- **This is a small model on a small corpus.** 1.36M tokens, `d_model=256`, 4 layers, 3000
  steps, 2 seeds, `SEQ=64` — a 4 GB card is what set those. The gap between arms is large
  enough to survive the scale, but whether it *closes* with scale is untested and is the
  obvious next experiment.
- **The two objectives differ.** `tied-head` optimises token cross-entropy; the factorised arms
  optimise per-byte cross-entropy. That is the natural loss for each, not a rigged comparison,
  but they are not the same objective and the accuracies are not likelihood-comparable.
- **Collisions set a ceiling I never reached.** P3 measured 205 colliding tokens for this Qwen
  vocabulary at `pos_dim=32` (0.135%), so any byte-decoding head is capped at 99.87%. At 7.9%
  accuracy that cap is nowhere near binding here — but it is the thing that would bite a
  working version, and it links P5 directly to P3 and P4.
- **`reuse-W` conflates two jobs.** The same matrix is the input projection and the output map,
  so training pulls it in two directions. A separate `d × D` decoder (`byte-head`) is the
  cleaner design and does measurably better.
- **No generation quality measured.** Top-1 next-token accuracy only. Sampling, calibration and
  actual text quality are untouched, and §4 explains why sampling is the harder half.
- **`d_model = 2048` for full 32-byte recovery** (§2) is a real constraint on the design: the
  head-free decoder needs a wide enough model, which is a different resource than the head's
  parameters and is not accounted for in the ratio table.

---

## 6. Files

| file | what |
|---|---|
| `invert.py` | the inverse — projection, pseudo-inverse, length inference, self-check |
| `recover.py` | the recovery law: exact recovery vs `d_model` and sparsity, noise, parameters |
| `train.py` | three output sides on a real corpus at V=151k |
| `results_*.log` | the evidence behind every number above |

Reused from [`../kronbudget/`](../kronbudget/): `roles.py` for the released codec and its
decoder, `vocab.py` for the vocabularies as byte strings.

```bash
python invert.py                                       # self-check
python recover.py                                      # Session 2 vocabulary
python recover.py --vocab Qwen/Qwen2.5-0.5B-Instruct   # same law, 151k vocabulary
python train.py --steps 3000 --seeds 2                 # ~35 min on a 4 GB card
```
