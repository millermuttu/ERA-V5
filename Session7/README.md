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
