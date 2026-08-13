# Session 7 — Embeddings and Model Internals

Course material: [`reference/session7-lesson.md`](reference/session7-lesson.md) and the 13
interactive widgets under [`reference/widgets/`](reference/widgets/).

The session covers the Kronecker byte-codec embedding — a fixed `256 × pos_dim` grid of
(byte value, byte position) feeding one shared `Linear(8192, d_model)`, which removes the
vocabulary from the input-side parameter count entirely — its silent-collision failure at
`pos_dim = 32` on Indic scripts, and the wall that absolute position tables hit.
Reference paper: [Kronecker Embeddings, arXiv:2605.29459](https://arxiv.org/abs/2605.29459).

## Assignment

Pick one of five open "Kronecker V2" problems, solve it, and prove it with code.

- [`assignment-research.md`](assignment-research.md) — literature review of all five problems,
  what is already published against each, and where the genuine openings are.
- [`kronmath/`](kronmath/) — **Problem 1**, mathematical structure in embeddings.
- [`kronmm/`](kronmm/) — **Problem 2**, one Kronecker codec for text, images and audio.
- [`kronbudget/`](kronbudget/) — **Problem 3**, the position budget: the collision count the
  lesson asks for, and a role map that removes it.
- [`kronfourier/`](kronfourier/) — **Problem 4**, each character as a wave: a Fourier codec
  with an inverse, and what it does to the suffix limitation.
- [`kroninv/`](kroninv/) — **Problem 5**, dropping the output head: the reverse map, and the
  accuracy it costs.

## Problem 1 — arithmetic that lives in the embedding

A fixed, parameter-free block appended to the byte grid that carries a number's *value* in a
basis where addition and multiplication are both a single phase rotation, via the additive and
multiplicative characters of `Z/p`. 22 dimensions on top of 8192 (0.27%) buys exact `+ − × ÷`
and powers on `Z/15015`.

The delta against the published work (FoNE, Prime Fourier Embeddings) is the **multiplicative
character channel**: both carry additive characters only, in which multiplication is not a
rotation and has to be learned as a table. A constructed bilinear readout hits 100% on operands
it never saw; a transformer extracts only part of that, so the bottleneck is extraction rather
than representation.

See [`kronmath/README.md`](kronmath/README.md).

## Problem 2 — one codec for three modalities

The codec is a Tensor Product Representation, `filler ⊗ role`. All three modalities are natively
8-bit (UTF-8 byte / pixel intensity / μ-law sample), so the **filler alphabet is already
universal**; only the role space differs, and fixing it at 64 slots makes `D = 16,384` shared,
so one `Linear` consumes all three with no per-modality table.

Two design rules, each measured rather than argued: the **role** must be the axis the modality
is stable along (time-index roles for audio score 0.014, frequency-bin roles 0.475–0.783), and
the **filler** must match the alphabet type — categorical gets a one-hot, ordinal gets a
smeared bump, without which a +1 brightness shift is as distant as +32.

See [`kronmm/README.md`](kronmm/README.md).

## Problem 3 — the position budget

Section 8 of the lesson asks for one thing outright: encode the vocabulary and count the
collisions per script. Over four real vocabularies, the released `pos_dim = 32` leaves
0.01–0.57% of tokens permanently ambiguous, every one of them Indic, Georgian, Sinhala or
Thai — Latin never collides at any budget — and `pos_dim = 48` removes all of them for 33 M
parameters.

The premise about waste is wrong, though, and the submission says so: the codec is sparse, so
`"a"` costs nothing in the vector. `R` buys `D`, and therefore projection width, which makes
this a global parameter decision rather than a per-token one. Holding `R` fixed and changing
only the role map, wrapping overflow bytes back to slot `i mod R` collides *less at R=16 than
the released codec does at R=32* — half the projection — while leaving every token that
already fits bit-identical.

See [`kronbudget/README.md`](kronbudget/README.md).

## Problem 4 — each character a wave

A sum of per-character waves is order-blind, so position has to enter through phase, and once
it does the construction is Fourier Holographic Reduced Representations: a fixed unit-modulus
phasor per byte value, position as a phase rotation, a word as the sum. `D = 2N` with
`N = 4096` matches the released `D = 8192` exactly.

`D` no longer depends on length, so there is no budget to crop — recovery is exact while
`L ≤ N/16`, which at the released width is 256 bytes against the shipped 32, decaying
gracefully rather than falling off a cliff. Two expectations died in the measurement: fp16
costs nothing (crosstalk dominates rounding by orders of magnitude), and the byte codec's
short-token locality turns out to be mostly an artifact of its z-normalisation.

On the suffix limitation the paper calls structural, the honest result is narrower than the
obvious one: plain cosine fails for the Fourier codec too. What fixes it is shift
equivariance — prepending `k` bytes multiplies the whole code by `ψ^k`, so aligning two words
is one elementwise multiply on the code, recovering 0.62–0.73 on suffix families while leaving
prefix families untouched.

See [`kronfourier/README.md`](kronfourier/README.md).

## Problem 5 — dropping the output head

The codec was never the obstacle: reshape, argmax per column, and the bytes come back exactly.
The obstacle is `W_proj`, which maps 8192 down to `d_model` and is not injective — so
"invert it" is, read literally, impossible. The reframing that makes it possible is that
`kappa` is **sparse**: at most `pos_dim` non-zeros out of 8192. Recovering it is compressed
sensing, and the requirement scales with the token's byte length, not with `D` and not with the
vocabulary — which is why the 1M-vocabulary claim has a foundation.

Measured, the law is roughly `d ≈ 64·L`, flat against 20% noise, and reproduced on two
vocabularies. The output-side parameter ratio is exactly `V / D`: 122× at a 1M vocabulary.

The head does come off — one arm reads the input projection backwards and adds *zero* output
parameters — and it costs more than half the top-1 accuracy (0.187 → 0.069 at V=151k). So
"without any issues" is wrong, and the useful finding is *where* it breaks: over four fifths of
the errors get the token's **length** wrong rather than its bytes.

See [`kroninv/README.md`](kroninv/README.md).
