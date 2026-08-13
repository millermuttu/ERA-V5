# KronMM — one Kronecker codec for text, images and audio

**Session 7 assignment, Problem 2.** *"What is the natural extension of Kronecker, such that it
can represent images and audio as well!! Yes we'll need to do some preprocessing of image and
audio patches as well, but how we do use this concept to represent all 3!"*

The released codec is a **Tensor Product Representation**: a token is a bundle of
`filler ⊗ role` pairs — one-hot byte *value* bound to one-hot byte *position*. Nothing is
learned, and the whole token is the sum. Extending it to three modalities turns out to need no
new machinery, only two design decisions, and **both of them can be measured rather than
argued**.

---

## 1. The unification

**The filler alphabet is already universal.** All three modalities are natively 8-bit:

| modality | filler | range |
|---|---|---|
| text | UTF-8 byte | 0–255 |
| image | 8-bit pixel intensity | 0–255 |
| audio | **μ-law** companded sample / log-magnitude bin | 0–255 |

μ-law is the load-bearing one — not a convenience cast but the standard perceptual companding
for audio (WaveNet's quantization), so its 256 levels are perceptually spaced.

**Only the role space differs**, and where it needs to be 2-D it is itself a Kronecker product:

| modality | role | slots |
|---|---|---|
| text | byte position in the token | 64 |
| image | `row ⊗ col` in an 8×8 patch | 8×8 = 64 |
| audio | frequency bin of the frame | 64 |

Fixing the role count at **R = 64** makes `D = 256 × 64 = 16,384` identical across modalities,
so **one shared `Linear(16387, d_model)` consumes all three** (+3 dims for a modality tag).
There is no per-modality embedding table anywhere in the input path.

`python mmcodec.py`:

```
text   round-trip exact (ascii, devanagari, emoji)
image  round-trip exact on 50 random 8x8 patches
audio  round-trip exact at 110/440/1760 Hz, max mu-law reconstruction error 0.0164
one shared Linear(16387, 512): torch.Size([3, 16387]) -> torch.Size([3, 512])
```

### Why the prior is earned here, and where it would not be

The codec's entire content is *"two things are similar iff they share values at the same
roles."* For **raw file bytes** that is a bad prior — byte 400 of one JPEG has no stable
relationship to byte 400 of another — which is the real objection to feeding compressed files
to a byte model. Once the input is **preprocessed into patches** the prior becomes correct:
two 8×8 patches sharing pixel values at the same offsets really are visually similar. The
assignment's "we'll need to do some preprocessing" is precisely the step that earns it.

---

## 2. Rule 1 — the role must be the axis the modality is stable along

The first audio front end I built used the obvious role: sample index in time. It does not
work, and the failure is total.

| audio front end | Spearman vs spectral distance, σ=0 → 64 |
|---|---|
| role = time index (μ-law sample) | **0.014 → 0.053** |
| role = frequency bin (log-magnitude) | **0.475 → 0.783** |

Two frames of the same tone at different **phase** are the same sound and have completely
different samples at every index. A time-index role is drowning in phase. Moving the role to
frequency discards phase deliberately — which is exactly what audio similarity is invariant to.
The mechanism, measured directly: applying a 1.1 rad phase shift moves the code by

```
0.2/255 mean  (spectral roles)     vs     80.2/255 mean  (time-domain roles)
```

a ~400× reduction in phase sensitivity. Note *reduction*, not elimination — a 128-sample frame
holds under two periods at 110 Hz, so windowed spectral leakage leaves real phase dependence at
low frequencies (max drift across bins reaches 41/255 there). Magnitude spectra are
phase-invariant for whole periods, not for short frames.

**No filler choice rescues a wrong role space:** the time-index row stays near zero across the
entire σ sweep. Get the role wrong and nothing downstream recovers it.

---

## 3. Rule 2 — the filler must match the *type* of the alphabet

Text bytes are **categorical**: byte 101 and 102 have no reason to be similar, so a one-hot is
right. Pixel intensities and log-magnitudes are **ordinal**: 100 really is almost 101. Under a
one-hot they are orthogonal, and the consequence is brutal:

```
brightness shift on one patch     sigma=0 (one-hot)   sigma=4 (ordinal)
  +1                                       -0.0039             +0.9836
  +2                                       -0.0039             +0.9359
  +4                                       -0.0039             +0.7658
  +8                                       -0.0039             +0.3308
  +16                                      -0.0039             -0.0392
```

A **+1 brightness shift** — visually identical — is as distant as +32. The codec has zero
intensity locality.

The fix changes one thing and keeps everything else: replace the one-hot filler with an
L2-normalised **Gaussian bump of width σ** over neighbouring levels. Still fixed, still
parameter-free, still `filler ⊗ role`, same `D`. `σ = 0` recovers the released codec exactly.

`python structure.py` — Spearman correlation between codec cosine and in-modality similarity
(negative edit distance for text, negative pixel L2 for images, negative spectral distance for
audio), 160 items, 12,720 pairs:

| modality | σ=0 | σ=2 | σ=4 | σ=8 | σ=16 | σ=32 | σ=64 |
|---|---|---|---|---|---|---|---|
| text | **0.636** | 0.195 | 0.265 | 0.406 | 0.538 | 0.568 | 0.583 |
| image (MNIST) | 0.920 | 0.934 | 0.939 | 0.945 | 0.956 | 0.969 | **0.987** |
| photo (continuous-tone) | 0.265 | 0.780 | 0.843 | 0.900 | 0.955 | **0.989** | 0.983 |
| audio (freq roles) | 0.475 | 0.607 | 0.626 | 0.643 | 0.659 | 0.693 | **0.783** |
| audio (time roles) | 0.014 | 0.025 | 0.031 | 0.046 | **0.053** | 0.041 | 0.018 |

- **Text is best at σ=0 and smearing actively hurts it** (0.636 → 0.195). Categorical.
- **Continuous-tone images go 0.265 → 0.989**, a 3.7× improvement. Ordinal.
- **MNIST barely moves** (0.920 at σ=0) because MNIST is nearly binary — the ordinal problem
  hides on near-binary data, which is why the continuous-tone row exists. Testing this idea on
  MNIST alone would have produced a false negative.

---

## 4. σ is a bounded trade, not a free parameter

Wide kernels give better similarity structure. They also destroy the codec's invertibility
under noise. Round-trip accuracy of an 8×8 patch, with relative Gaussian noise on the code:

| σ | 0.1% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|
| 0 | 100 | 100 | 100 | 100 | 100 |
| 2 | 100 | 100 | 100 | **100** | **100** |
| 4 | 100 | 100 | 100 | **100** | 71 |
| 8 | 100 | 100 | 43 | 0 | 0 |
| 16 | 100 | 0 | 0 | 0 | 0 |

Every σ decodes exactly in the clean case; robustness is what separates them. **σ = 2–4** is
the usable band: it keeps perfect decoding at 2% noise while capturing most of the structure
gain (photo 0.265 → 0.843 at σ=4). The σ=32 that maximises rank correlation is not deployable.

---

## 5. End to end — what does unification cost?

`python train.py` — 28 classes (10 MNIST digits, 8 pitch classes, 10 words), sequences of 4
items from one source, **mixed batches with the modality never given to the model**.

| steps | `shared` (one Kronecker path) | `per-mod` (3 linear adapters) | `per-mod-mlp` (3 MLP adapters) |
|---|---|---|---|
| 30 | **0.995** | 0.735 | 0.779 |
| 60 | **1.000** | 0.768 | 0.819 |
| 120 | 1.000 | 0.877 | 0.914 |
| 250 | 1.000 | 0.963 | 0.959 |
| 500 | 1.000 | 0.991 | 0.991 |

One fixed codec and one projection reaches 0.995 in **30 steps**; three learned adapters need
~500 to reach 0.991. **Roughly 16× fewer steps, and an equal-or-better endpoint** — the fixed
expansion means there is almost nothing to learn on the input side. At convergence the gap
nearly closes, so the honest claim is *unification costs nothing and converges much faster*,
not *unification is more accurate*.

---

## 6. What I would not claim

- **The parameter win does not transfer.** The released codec saves parameters by replacing a
  `V × d_model` vocabulary table. Images and audio have no vocabulary to replace: a raw 8×8
  patch is 64 numbers, so a plain linear on it costs 0.02 M against the shared path's 2.10 M.
  For non-text modalities this buys **unification, vocabulary-freeness and sample efficiency —
  not memory.** Any claim of compression here would be false.
- **The `shared` arm has ~100× the input parameters** of `per-mod`, so its win is not a
  like-for-like capacity comparison. That is why `per-mod-mlp` is included, and why the
  convergence-rate table matters more than the endpoint.
- **Audio is synthesised**, not real recordings — mixed harmonic tones with random phase and
  amplitude. The phase-invariance finding is exactly the sort of thing synthetic data shows
  cleanly, but a speech or music corpus would be the real test.
- **`photo` is synthetic too** (smooth gradients plus texture). It exists because MNIST is
  near-binary and hides the ordinal problem. CIFAR or any photographic set would be the
  honest replacement.
- **One shared σ per modality** is assumed. Nothing tests whether σ should vary by role, or
  adapt during training.
- **Only three modalities, one role layout each.** Video (`row ⊗ col ⊗ frame`) is the obvious
  next Kronecker factor and is not built.

---

## 7. Files

| file | what |
|---|---|
| `mmcodec.py` | the unified codec, μ-law + spectral front ends, inverse, self-check |
| `structure.py` | does the prior match each modality? σ sweep, Spearman correlations |
| `train.py` | tri-modal classification: one shared path vs per-modality adapters |

```bash
python mmcodec.py                 # round-trip all three modalities, show the ordinal failure
python structure.py --n 160       # the two design rules, measured
python train.py --steps 400       # end to end
```
