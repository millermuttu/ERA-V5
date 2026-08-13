# Session 7 assignment — research note on the five Kronecker-V2 problems

Working notes, 2026-08-09. Input: the [Kronecker Embeddings paper](https://arxiv.org/abs/2605.29459)
(Shravan, arXiv:2605.29459v1, 28 May 2026), its [reference implementation](https://github.com/theschoolofai/kronecker-embeddings),
and a literature sweep per problem. Goal: pick one problem and know what it is
actually worth before spending a session on it.

---

## 1. The frame that reorganizes all five problems

The Kronecker codec is a **Tensor Product Representation** — Smolensky, 1990. One-hot
*role* (byte position) ⊗ one-hot *filler* (byte value), bundled by summation. The paper
never names it, but that is exactly what `c_{b_p} ⊗ p_p` is, and it means there are ~35
years of results on this specific object under a different flag: **Vector Symbolic
Architectures / Hyperdimensional Computing**.

That matters because the VSA literature's founding complaint about TPRs is the one this
paper inherits:

> "Because the dimension of the tensor product increases with each binding operation, the
> size of the representation grows exponentially… The solution is to collapse the N×N
> role/filler matrix back into a length-N vector."

`D = 256 × 32` growing with `pos_dim` **is** that problem. The standard fix — **circular
convolution binding** (HRR/FHRR), computed as elementwise multiply in the Fourier domain —
collapses the outer product back to fixed `D`, preserves similarity structure, and comes
with an **approximate unbinding operator**.

Against the assignment list:

| VSA property | Assignment problem |
|---|---|
| fixed `D` regardless of sequence length | **P3** dynamic budget |
| Fourier-domain binding | **P4** Fourier alternative |
| unbinding operator | **P5** invertibility |

Three of the five problems are one substitution in the VSA literature. This is the single
most useful finding in this note.

---

## 2. Two claims verified in code

### 2.1 The codec is already exactly invertible

Ran the repo's real `codec.py`. Reshape `D → (256, pos_dim)`, argmax down the byte axis per
column. Each position column has at most one nonzero, so recovery is exact and O(D).
z-normalisation is a per-token affine map and does not disturb the argmax.

```python
"""Is the Kronecker codec exactly invertible? Reshape D->(256,pos_dim), argmax per column."""
import torch
from codec import encode_single          # from theschoolofai/kronecker-embeddings

POS = 32
def decode(kappa, pos_dim=POS, char_dim=256):
    grid = kappa.view(char_dim, pos_dim)          # index = byte*pos_dim + pos
    vals, idx = grid.max(dim=0)                   # best byte value per position
    active = vals > grid.mean()                   # occupied columns sit above the z-norm floor
    L = int(active.sum())
    return bytes(idx[:L].tolist()), L

for w in ["a", "at", "training", "अंतर्राष्ट्रीयकरण", "తెలుగుభాష", "क्ष", "  ", "🙂"]:
    b = w.encode()[:POS]
    got, L = decode(encode_single(b, pos_dim=POS))
    print("PASS" if got == b else "FAIL", repr(w), len(b), "B ->", L, "B")
```

Result: **8/8 exactly recovered** — multi-byte scripts, conjuncts, emoji, whitespace.

**This reframes P5.** The question is not "how do I invert the Kronecker codec" — that is a
solved, trivial argmax. The question is **"how do I invert `W_proj`"**, an `8192 → 8096`
matrix that is *nearly square*. Far more tractable than the assignment's phrasing suggests.

### 2.2 The lesson's collision claim is real and total

```
pos_dim=32: 51B vs 48B, codec cosine = 1.0000, identical = True
pos_dim=48: 51B vs 48B, codec cosine = 0.9582, identical = False
```

`अंतर्राष्ट्रीयकरण` and `अंतर्राष्ट्रीयता` are **bit-identical** at the shipped `pos_dim=32`.
Widening to 48 separates them but only to cosine 0.958 — it demotes them from *identical* to
*very confusable*. Worth knowing before building a solution that buys 0.04 of cosine.

Note: `encode_single` does not do UTF-8-safe truncation; the production path does (backs off
to the previous codepoint boundary). Truncating a 51-byte Devanagari word at 32 in the naive
path leaves an invalid trailing partial codepoint.

---

## 3. P1 — Mathematical structure in embeddings

**Ask:** append dimensions holding true numeric value so `emb(9)+emb(9)` lands on `emb(18)`,
`9*9` on `81`.

**Prior art.** No prior sketch *in the paper*, but by far the **densest external literature**
of the five:

- [FoNE: Precise Single-Token Number Embeddings via Fourier Features](https://arxiv.org/html/2502.09741v1)
  (ICLR) — numbers into embedding space via Fourier features, each digit as cos/sin at
  different periods, recovers `x mod 10^i`. **99% accuracy with 64× less data.** This is the
  problem, already solved and published.
- [Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic](https://arxiv.org/abs/2606.23044)
  (Jun 2026) — integers as prime-indexed (cos, sin) pairs. Proves via **Schur's lemma** that
  any equivariant linear map on PFE is block-diagonal, one block per prime; CRT predicts which
  channels are task-relevant. Specialization ratios >500×.
- [Numbers Already Carry Their Own Embeddings](https://arxiv.org/html/2606.14108),
  [Pre-trained LMs Learn Remarkably Accurate Representations of Numbers](https://arxiv.org/html/2506.08966),
  [Abacus Embeddings](https://arxiv.org/pdf/2405.17399) (120-digit addition, 6× length
  generalization), xVal.

**What is genuinely open:** *composition*. FoNE gives an `emb(9)` that decodes to 9. Nobody
has made the space **closed under the operations** — a fixed operator ⊕ with
`emb(9) ⊕ emb(9) = emb(18)` — *and* had that space coexist with the 8,192-dim byte code for
words.

**The structural limit, which must be stated up front.** The elegant version is a
homomorphism `φ: (ℕ, +, ×) → (V, ⊕, ⊗)`, and `(ℕ, +, ×)` is a **ring**. A group homomorphism
for `+` is easy (Fourier phases add). Both operations exact and simultaneous is a faithful
ring representation in fixed dimension, which fails for unbounded integers — you are forced
into modular arithmetic (hence PFE's primes + CRT) or a bounded range. Declared up front this
reads as principled; discovered at the end it reads as failure.

**Proof protocol.** Small transformer on `a+b` / `a×b` with three arms: BPE digits,
FoNE-style, your codec-appended version. Metric is accuracy vs. **unseen operand magnitude** —
length generalization separates real structure from memorization.

**Verdict: crowded.** Novel to *this* paper ≠ novel. Viable only if aimed squarely at
ring-closure plus coexistence with byte channels.

---

## 4. P2 — Multimodal Kronecker

**Prior art.** [bGPT](https://arxiv.org/abs/2402.19155) (Microsoft) already models text, audio
and images as raw bytes with next-byte prediction, matching specialized models per modality,
plus >99.99% accuracy simulating CPU behaviour. Also [MEGABYTE](https://arxiv.org/abs/2305.07185),
[Multiscale Byte LMs](https://arxiv.org/html/2502.14553) (ICML 2025, 5M-byte contexts on one
GPU), [Byte Latent Transformer](https://arxiv.org/abs/2412.09871).

**The problem nobody says out loud:** *byte-position locality is the wrong inductive bias for
images and audio.* The codec's entire value proposition — two strings are similar iff they
share bytes at the same positions — is a **language** claim. Byte 400 of a JPEG has no stable
relationship to byte 400 of another JPEG. Feed image bytes to this codec and the structure
recovered is an artifact of the file format, not the image.

The honest extension is not "bytes for everything" but **the TPR generalizes to the right
roles per modality**:

| Modality | Filler | Role(s) |
|---|---|---|
| Text | byte value | byte position |
| Image | quantized patch value / VQ code | `pos_x ⊗ pos_y` |
| Audio | codec token (EnCodec/SoundStream) | time index ⊗ frequency band |

`D = 256 × H × W` explodes immediately, and the escape is again circular-convolution binding,
which keeps `D` fixed under any number of role bindings. **P2 done properly routes through P4.**

**Verdict:** most work, least certain payoff, needs image *and* audio pipelines before
anything can be shown. Only worth it with those pipelines already in hand.

---

## 5. P3 — Dynamic byte budget

**Ask:** stop spending 32 position slots on `"a"`; stop cropping words longer than 32.

**Reality check.** The paper measured this: `d_p=32` covers **≥99.82%** of tokens across six
modern tokenizers, residual dominated by whitespace runs and Indic word pieces. Also `"a"`
wastes nothing per-token — the codec is sparse and unused positions contribute zero to the
sum. The waste is in `D` (hence projection width), not in the vector.

So the assignment's framing is slightly off and the submission should say so. The real costs:
(a) `D` scales linearly with `pos_dim`, so multilingual coverage is bought in projection
parameters; (b) at `d_p=16` coverage collapses to **95.98% on Gemma-3's multilingual
SentencePiece**. (b) is the actual motivation and it is script-specific.

**Prior art.** [CANINE](https://aclanthology.org/2022.tacl-1.5.pdf) — hashes character n-grams
into fixed buckets with multiple hash functions, no vocabulary, collisions diluted across
functions. [HashFormers](https://arxiv.org/pdf/2210.07904), Bloom embeddings,
[MYTE](https://aclanthology.org/2024.acl-long.804.pdf) (morpheme-based byte encoding —
**shorter encodings for all 99 languages tested**, biggest wins on non-Latin scripts; the
single most relevant paper for the Indic problem), [fastText](https://arxiv.org/abs/1607.04606)
char n-grams.

**Candidates, cheapest first:**

1. **Relative / log-bucketed positions** — bucket byte position as `{0,1,2,3,4-5,6-7,8-11,…}`.
   Unbounded length, ~12 slots. Loses exact position, gains no-crop.
2. **Hash (byte, position) → fixed buckets**, CANINE-style. Length-free by construction;
   collisions become tunable and measurable rather than silent.
3. **Two-ended window** — first 16 + last 16 bytes. One-line change, kills the exact failure
   mode (Indic words differing only in the suffix, i.e. the `…करण`/`…ता` pair), costs nothing.
   Ugly, effective, and the laziest thing that beats the baseline.
4. **Circular convolution** (→ P4). Fixed `D`, any length, principled.

**Proof protocol.** Encode the full V5 vocabulary; report **collisions per script** at
`pos_dim ∈ {16, 32, 48, 64}`, baseline vs. your scheme — the number the paper never published.
Then a small LM run showing perplexity improves on the affected script. The measurement half
is a couple of hours.

**Verdict: safest, smallest.** Guaranteed publishable measurement, modest ceiling.

---

## 6. P4 — A real Fourier alternative

"Represent each character as a Fourier wave and add them to make a word" is not a loose
analogy. It is **Fourier Holographic Reduced Representations**, a named, mature construction.

Mechanics:

- Each byte value gets a random **unit-modulus complex** vector (phasor).
- Byte is bound to position by **circular convolution** = elementwise complex multiply in the
  Fourier domain, equivalently phase rotation by position.
- Bundling across positions is **addition**. Literally "add them to make a word."

Properties, all targeting this paper's stated weaknesses:

| Property | Consequence |
|---|---|
| `D` fixed, independent of length | **solves P3** — no crop, no waste, no `d_p` |
| approximate **unbinding** via conjugate | **partial P5** — query "what byte at position 3?" |
| position via **fractional power encoding** | continuous/relative positions, not 32 discrete slots |
| similarity-preserving by construction | [Recursive Binding for Similarity-Preserving Hypervector Sequences](https://arxiv.org/pdf/2201.11691) |
| shift structure lives in phase | attacks the paper's **named** `nation`/`creation` suffix failure |

The last row is the strongest argument available. The paper lists as a structural limitation:

> "Position-aware encoding weakens on suffix-only families… This is structural to
> position-aware encoding and cannot be mitigated without sacrificing the case-sensitivity and
> prefix structure that make the encoding work."

A phase-based encoding makes a suffix shift a **phase rotation** rather than a move to
unrelated coordinates. That is directly refuting a limitation the author declared unfixable —
the best possible shape for a submission.

**Prior art to stand on.** [Computing on Functions Using Randomized Vector Representations](https://dl.acm.org/doi/pdf/10.1145/3517343.3522597)
(Plate, Frady, Kleyko, Sommer), [Generalized HRR](https://arxiv.org/html/2405.09689v1),
[HDC/VSA survey](https://redwood.berkeley.edu/wp-content/uploads/2022/11/2022_CSUR_survey_HDCVSA_part_1.pdf),
fractional power encoding.

**Risk, stated plainly.** FHRR trades *exact* sparse recovery for *approximate* dense
superposition. The current codec's collisions are catastrophic-but-rare (0.18%); FHRR's are
graceful-but-universal (everything is slightly noisy). Whether that trade helps or hurts is an
empirical question — which is what makes it a good assignment.

**Proof protocol.**

1. Analytic: collision rate + cosine structure vs. the byte codec on the real vocabulary.
   Show `nation`/`creation` rises while `compute`/`commute` does not.
2. Recovery: unbind each position, measure byte-recovery accuracy vs. `D` and vs. word length.
   A clean curve.
3. Train: small transformer, 3 seeds, matched compute, Kronecker vs. FHRR vs. BPE — the
   paper's own protocol at 1/10 scale.

**Verdict: highest expected value.** Deepest theory, cleanest experiments, structurally
subsumes P3 and opens P5.

---

## 7. P5 — Invertible Kronecker, drop the output head

**Ask:** same embedding → same token, so the `d_model → |V|` head disappears and vocab can be 1M.

Starting from §2.1 — the codec is already exactly invertible; the blocker is `W_proj`. Three
routes:

1. **Constrain `W_proj` to be orthogonal / invertible.** It is nearly square already. Then
   `κ = e W⁻¹`, argmax, done — exactly invertible end to end, no head at all, any UTF-8 string
   ≤ `d_p` bytes decodable. Cost: an orthogonality constraint on the input projection, a real
   but bounded capacity restriction.
2. **Learn a separate decoder** `d_model → D`, train against `κ(bytes(t))`. The paper's
   **Hypothesis A** (§8.5).
3. **Distributional**: predict `μ, σ²` over Kronecker space, train by NLL/KL. The paper's
   **Hypothesis B**.

**Prior art the paper does not cite.**
[Headless Language Models: Learning without Predicting with Contrastive Weight Tying](https://arxiv.org/pdf/2309.08351)
— removes the softmax over vocabulary entirely, reconstructs representations contrastively,
reports being *more* compute- and data-efficient, and explicitly notes it "enables the use of
very large token vocabularies at virtually no increased cost." That is P5's headline claim,
already demonstrated, minus the byte structure. Strongest support and closest competitor at
once.

**The failure mode, named by the author.** Codec collisions the transformer body absorbs
harmlessly on the input side become **hard decode errors** on the output side; the 0.18%
truncation rate stops being cosmetic. Route 1 makes this the only remaining error source,
which is a clean story: *the exact decoder works, and its error rate equals the vocabulary's
collision rate — so fix the collisions (P3/P4) and you get a headless model.*

**Verdict: highest payoff**, and route 1 is far more tractable than the lesson implies. But it
executes hypotheses the author already wrote down, against a 2023 paper that got most of the
way there by another road.

---

## 8. Counterweight to the whole session

[Scaling Embeddings Outperforms Scaling Experts in Language Models](https://arxiv.org/html/2601.21204v1)
(Jan 2026) — **LongCat-Flash-Lite**, 68.5B params with **over 30B allocated to embeddings**,
beating parameter-equivalent MoE baselines, strongest in agentic and coding domains. Finding:
embedding scaling is a *better* sparsity axis than expert scaling in some regimes, degrading
only past ~50% of total parameters.

A direct counterweight to this entire session — a frontier result saying spend *more* on the
input side, not 94% less. It does not invalidate the Kronecker result (different regimes, and
the 124M Kronecker win is a *quality* win, not only compression), but a submission that engages
with it will read as considerably more serious than one that ignores it.

---

## 9. Ranking

> **Outcome: P1 was chosen and implemented — see [`kronmath/`](kronmath/).** The ranking below
> is the pre-work assessment and is left unedited. One part of it proved right and one wrong.
> Right: P1 is crowded, and the solution had to be aimed specifically at the gap (ring-closure,
> and coexistence with the byte channels) rather than at number encoding in general. Wrong: the
> gap turned out to be sharper and more attackable than "crowded" suggested — FoNE and PFE both
> carry *additive characters only*, so multiplication is not a rotation in either, and the
> multiplicative-character channel closes that with 2 dimensions per prime.

1. **P4 (Fourier / FHRR)** — the pick. The only problem where "here is a 30-year-old
   mathematical framework that does exactly this, and applying it refutes a limitation the
   paper declared structural" is available. Absorbs P3 for free, gives a running start on P5.
   Cheap experiments; even the negative result is interesting.
2. **P5 (invertible)** — highest ceiling. The invertibility check says the tractable version is
   route 1 (orthogonal projection), not the author's Hypotheses A/B. The one the author most
   wants.
3. **P3 (dynamic budget)** — safest. Per-script collision count is a guaranteed contribution
   nobody has published. Low ceiling.
4. **P1 (math structure)** — the *most* crowded of the five externally. Viable only aimed at
   ring-closure and coexistence with byte channels, with the bounded-range limit stated up front.
5. **P2 (multimodal)** — most work, least certain; byte-position locality is genuinely the
   wrong prior for pixels and audio. Skip without existing pipelines.

---

## Sources

- [Kronecker Embeddings (arXiv:2605.29459)](https://arxiv.org/abs/2605.29459) · [reference implementation](https://github.com/theschoolofai/kronecker-embeddings)
- [FoNE: Fourier Number Embeddings](https://arxiv.org/html/2502.09741v1)
- [Prime Fourier Embeddings](https://arxiv.org/abs/2606.23044)
- [Numbers Already Carry Their Own Embeddings](https://arxiv.org/html/2606.14108)
- [Pre-trained LMs Learn Remarkably Accurate Representations of Numbers](https://arxiv.org/html/2506.08966)
- [Abacus Embeddings / Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/pdf/2405.17399)
- [bGPT: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155)
- [Multiscale Byte Language Models](https://arxiv.org/html/2502.14553)
- [CANINE](https://aclanthology.org/2022.tacl-1.5.pdf) · [HashFormers](https://arxiv.org/pdf/2210.07904)
- [MYTE: Morphology-Driven Byte Encoding](https://aclanthology.org/2024.acl-long.804.pdf)
- [Computing on Functions Using Randomized Vector Representations](https://dl.acm.org/doi/pdf/10.1145/3517343.3522597)
- [Generalized Holographic Reduced Representations](https://arxiv.org/html/2405.09689v1)
- [HDC/VSA survey, part 1](https://redwood.berkeley.edu/wp-content/uploads/2022/11/2022_CSUR_survey_HDCVSA_part_1.pdf)
- [Recursive Binding for Similarity-Preserving Hypervector Sequences](https://arxiv.org/pdf/2201.11691)
- [Headless Language Models](https://arxiv.org/pdf/2309.08351)
- [Scaling Embeddings Outperforms Scaling Experts](https://arxiv.org/html/2601.21204v1)
