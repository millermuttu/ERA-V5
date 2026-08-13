# KronBudget — the position budget, measured and then made dynamic

**Session 7 assignment, Problem 3.** *"Today Kronecker is limited to 32 positions for every
word (even 'apple' or 'a'). That's a waste of space. What can we do? How can it be dynamic and
not force us to crop a word (currently we cannot have a word of len more than 32)?"*

The lesson asks for one thing outright, and it is not a design:

> take the V5 vocabulary, encode every token, and count the collisions per script... If the
> collision count says buy it, buy it. **The assignment asks you to produce that count.**

So that count is produced first, over four real vocabularies including this repo's own. Then
the design, which the count turns out to constrain quite sharply.

---

## 1. The premise needs correcting before anything is built

**"Even 'apple' or 'a' spends 32 positions — that's a waste of space" is not true of the
vector.** The codec is a scatter into a `256 × R` grid. A one-byte token writes one cell; the
other 31 columns are zero and contribute nothing to the sum, the norm, or the projection's
output. There is no per-token waste to reclaim.

What `R` actually buys is `D = 256 × R`, and therefore the width of the one projection that
holds every input-side parameter:

| `R` | `D` | projection (`D × 8096`) |
|---|---|---|
| 16 | 4,096 | 33.2 M |
| 32 | 8,192 | 66.3 M |
| 64 | 16,384 | 132.7 M |

So the budget is a *global* parameter decision, not a per-token one, and the second half of
the question — the crop — is the real defect. That reframing sets the experiment: **hold `R`
fixed, change only the role map, and see who collides.** Every scheme below has identical `D`
and an identically sized projection, so nothing here is bought with parameters.

---

## 2. The count

A collision is not "these codes are close". It is **bit-identical codes** for two distinct
vocabulary entries — the codec is frozen and the projection is shared, so no amount of
training can ever separate them. It is detected exactly, not by a cosine threshold: two
tokens collide iff their `(byte, slot)` count grids agree up to a positive scalar, since
z-normalisation is affine. That makes the test `O(L)` per token instead of materialising
250k × 8192 floats.

`python collide.py` — **tokens whose code is shared with at least one other token**, under
the released `trunc` scheme:

| vocabulary | tokens | R=16 | R=32 | R=48 | R=64 |
|---|---|---|---|---|---|
| **Session 2 BPE** (this repo) | 10,000 | 16.09% | **0.57%** | 0 | 0 |
| XLM-R (SentencePiece) | 249,997 | 3.89% | **0.04%** | 0 | 0 |
| Qwen2.5 (byte-BPE) | 151,651 | 0.49% | **0.14%** | 0.11% | 0.08% |
| mBERT (WordPiece) | 119,542 | 1.29% | **0.01%** | 0 | 0 |

Per script at the shipped `R = 32`, worst first:

| vocabulary | worst scripts |
|---|---|
| Session 2 BPE | Telugu 1.4%, Devanagari 0.3%, Kannada 0.1%, **Latin 0** |
| XLM-R | Georgian 1.4%, Malayalam 0.5%, Sinhala 0.2%, Thai/Devanagari 0.1%, **Latin 0** |
| mBERT | Tamil 0.5%, Georgian 0.6%, Cyrillic 0.02%, **Latin 0** |
| Qwen2.5 | Latin 0.1%, symbol runs 2.8% |

Three things the table settles.

**The lesson's claim holds and is script-shaped.** Latin never collides at any budget in any
vocabulary. Every collision at `R = 32` is Indic, Georgian, Sinhala or Thai. Typical pairs are
a word and its inflection:

```
▁მნიშვნელოვანი  ==  ▁მნიშვნელოვანია  ==  ▁მნიშვნელოვან         (Georgian, XLM-R)
ുണ്ടായിരുന്ന   ==  ുണ്ടായിരുന്നു                              (Malayalam, XLM-R)
 ప్రధానమంత్ర   ==   ప్రధానమంత్రి  ==   ప్రధానమంత్రికి          (Telugu, Session 2)
```

The collided tokens are not junk. `ప్రధానమంత్రి` is "prime minister"; the codec cannot
distinguish it from "to the prime minister".

**`R = 48` already buys the fix outright**, for every natural-script vocabulary. If the
question is only "is 32 the right window", the answer from the count is: 32 leaves 0.01–0.57%
of the vocabulary permanently ambiguous, 48 leaves none, and the price is 33 M parameters.
That is the buy-it-or-not answer the lesson asks for, and it needs no new scheme at all.

**Qwen never reaches zero**, and that is the interesting residue. Its collisions are runs of
repeated characters — `****…`, `####…`, spaces — that differ only in *length*. No wider
window helps, because the first `R` bytes are identical however far you look.

---

## 3. The design, and the rule that constrains it

Only the role map changes; the filler stays the byte value and `D` stays `256 × R`.

| scheme | role of byte `i` in a token of length `L` |
|---|---|
| `trunc` | `i`, and drop everything past `R` — the released codec |
| `fit-ends` | `i` if `L ≤ R`, else first `R/2` from the head and last `R/2` from the tail |
| `fit-wrap` | `i if L ≤ R`, else `i mod R` — byte `i` and byte `i+R` superpose |
| `fit-rel` | `i` if `L ≤ R`, else `⌊i·R/L⌋` — squeeze the whole token proportionally |
| `rel-always` | `⌊i·R/L⌋` for every token — the control, see below |

**The `fit-` prefix is the whole discipline.** Coverage at `R = 32` is 96.90% (Session 2),
99.66% (XLM-R), 99.84% (Qwen), 99.89% (mBERT), so a scheme that reshuffles short tokens is
trading a certain regression on ~99% for a fix on ~1%. Every `fit-*` scheme is therefore the **identity map whenever the
token fits**, which makes its codes bit-identical to the released codec there — asserted in
`fidelity.py`, not merely intended.

`rel-always` is in the table to show what happens without that rule: it drops exact round-trip
decoding on tokens that fit from **100% to 11.7%** at `R=16` (2.6% at `R=32`), because it
squeezes tokens that never needed squeezing. Its collision behaviour is fine. It is still
disqualified.

---

## 4. What the schemes do at fixed `R`

`python collide.py` — colliding tokens, same `D`, same projection size:

| vocabulary | scheme | R=16 | R=32 | R=48 | R=64 |
|---|---|---|---|---|---|
| Session 2 BPE | `trunc` | 1,609 | 57 | 0 | 0 |
| | `fit-ends` | 169 | 2 | 0 | 0 |
| | `fit-wrap` | **0** | **0** | 0 | 0 |
| | `fit-rel` | **0** | **0** | 0 | 0 |
| XLM-R | `trunc` | 9,720 | 103 | 0 | 0 |
| | `fit-ends` | 1,541 | 2 | 0 | 0 |
| | `fit-wrap` | **0** | **0** | 0 | 0 |
| Qwen2.5 | `trunc` | 749 | 205 | 160 | 118 |
| | `fit-ends` | 350 | 202 | 153 | 112 |
| | `fit-wrap` | **52** | **21** | 4 | 2 |
| mBERT | `trunc` | 1,541 | 10 | 0 | 0 |
| | `fit-ends` | 104 | **0** | 0 | 0 |
| | `fit-wrap` | **0** | **0** | 0 | 0 |

`fit-ends` barely helps on Qwen (202 vs 205) and this is diagnostic rather than
disappointing: head-and-tail cannot separate strings that differ only in length, because
their heads and tails are the same characters. `fit-wrap` can, because wrapping turns extra
length into extra *count* in a cell — 205 → 21.

### The headline

**`fit-wrap` at `R = 16` collides less than the released codec at `R = 32`, on every
vocabulary, with half the projection:**

| | Session 2 | XLM-R | Qwen2.5 | mBERT | projection |
|---|---|---|---|---|---|
| `trunc`, R=32 (released) | 57 | 103 | 205 | 10 | **66.3 M** |
| `fit-wrap`, R=16 | **0** | **0** | **52** | **0** | **33.2 M** |

The dynamic budget does not cost width. It refunds it.

---

## 5. What it costs — the two properties that must survive

Removing collisions is easy if you are allowed to wreck everything else.

**Locality.** `python fidelity.py` — Spearman between codec cosine and negative byte edit
distance, split by population, because on tokens that fit every `fit-*` scheme *is* the
baseline:

| scheme | R=16 fits | R=16 overflow | R=32 fits | R=32 overflow |
|---|---|---|---|---|
| `trunc` | 0.301 | 0.163 | 0.261 | 0.571 |
| `fit-ends` | 0.301 | 0.216 | 0.261 | 0.458 |
| `fit-wrap` | 0.301 | **0.250** | 0.261 | **0.659** |
| `fit-rel` | 0.301 | 0.238 | 0.261 | 0.579 |
| `rel-always` | 0.261 | 0.238 | 0.158 | 0.579 |

The `fits` columns are identical by construction, which is the point. On the overflow tail
`fit-wrap` is the best of the four at both budgets: keeping the tail bytes, even superposed,
orders long tokens better than discarding them.

**Invertibility.** Exact round-trip through the released decoder (reshape, argmax per slot):

| | tokens that fit | tokens that overflow |
|---|---|---|
| `trunc`, `fit-ends`, `fit-wrap`, `fit-rel` | **100%** | **0%** |
| `rel-always` | 11.7% (R=16) | 0% |

**No scheme is invertible past the budget, including the baseline.** `trunc` fails because
the tail was thrown away; `fit-wrap` and `fit-rel` fail because superposed slots cannot be
unscrambled by an argmax. The gain here is *discriminability*, not *recoverability*: the
codes differ, so the model can tell the tokens apart, but the bytes cannot be read back. Any
claim that a wrapped code is decodable would be false.

**The lesson's own pair**, cosine between the two codes:

| scheme | R=16 | R=32 |
|---|---|---|
| `trunc` | 1.0000 (identical) | 1.0000 (identical) |
| `fit-ends` | 0.8118 | 0.7490 |
| `fit-wrap` | 0.9361 | 0.9320 |
| `fit-rel` | 0.7689 | 0.4209 |

`अंतर्राष्ट्रीयकरण` and `अंतर्राष्ट्रीयता` stop being the same vector. Note that they stay
*similar* under `fit-wrap` — they share a 44-byte prefix out of 51 and 48, so they should.

---

## 6. Downstream, and why this arm is weak

`python train.py` — a small causal LM over this repo's Session 2 tokenizer and its
Kannada / Telugu / Hindi / English corpora, everything held fixed except the role map. Val
loss is reported per corpus and, separately, **restricted to positions whose preceding token
is one the released codec cannot disambiguate** — all arms scored on that same baseline
position set, since scoring each arm on its own set would not be a comparison.

Ambiguous-context val cross-entropy at `R = 16`, 3 seeds:

| steps | `trunc` | `fit-ends` | `fit-wrap` | `fit-rel` |
|---|---|---|---|---|
| 200 | 7.360 | 7.324 | 7.326 | 7.359 |
| 400 | 7.313 | 7.281 | **7.253** | 7.314 |
| 800 | 7.513 | 7.459 | **7.404** | 7.496 |
| 1600 | 8.209 | 8.141 | **8.058** | 8.228 |

`fit-wrap` is ahead of `trunc` at every budget past 200 steps, by 0.06–0.15 nats against a
seed spread of 0.02–0.06. The whole-corpus columns barely move, which is expected: collided
types are 16% of the *vocabulary* at `R=16` but only 34% / 32% / 4.9% / 0% of *token
occurrences* in Kannada / Telugu / Hindi / English.

**This is the weakest evidence in the submission and should be read as directional.** 31k
training tokens is far too little; the model overfits past ~400 steps (which is why a step
sweep is reported instead of one number), and the Kannada validation split is 210 tokens.

One result cuts against a tidy story and is kept because of that: **`fit-rel` has zero
collisions and no downstream benefit** — it tracks `trunc` at every step count. Removing
collisions is necessary, not sufficient; the replacement roles have to be *usable*, and
proportional squeezing makes a token's role assignment depend on its own length, so the same
byte sequence lands in different slots in different tokens.

---

## 7. Two silent bugs, both caught by a number that refused to move

**Stripping WordPiece `##`.** The first run read mBERT as **25.7% collided at `R=32`**, and
the tell was that the number did not fall as `R` grew — 25.71% at 32, 25.70% at 64. That is
not truncation. Stripping the marker had merged the continuation token `##ka` with the
standalone token `ka`, an artifact of my normalisation rather than a property of the codec.
Keeping the marker gives **0.01%**, a 2,500× correction.

**Rebuilding the Session 2 vocabulary from `base_chars + merges`.** Index 0 of the
tokenizer's table is the UNK slot, so every merge index refers to a table shifted by one. The
resulting tokens are *plausible* — right length, right scripts — and entirely wrong, e.g.
`'ీఃపళౌbuలಂಥಅಥప'`, a token no Telugu BPE would ever learn. It survived until the mixed-script
junk stopped looking like a quirk of the corpus. The vocabulary now comes from the
tokenizer's own `id_to_str`. Every Session 2 number in this README is post-fix.

---

## 8. What I would not claim

- **Nothing here restores invertibility past the budget.** See §5. The problem statement's
  "cannot have a word of len more than 32" is solved in the sense that longer words now get
  *distinct* codes; it is not solved in the sense of reading the bytes back.
- **`R = 48` is a complete answer to the lesson's actual question** and costs only parameters.
  `fit-wrap` at `R = 16` is better on both axes, but the honest comparison is against
  `R = 48`, not only against the shipped `R = 32`.
- **The LM is underpowered** (§6) — small corpus, small model, tiny validation splits. It
  supports a direction, not a magnitude.
- **Locality is measured against byte edit distance**, a crude proxy for "these tokens mean
  similar things", and on BPE fragments rather than words.
- **Only four vocabularies, all public or in-repo.** The real V5 vocabulary was not available;
  Session 2's tokenizer is this repo's closest stand-in and is 10k tokens, which is small
  enough that its 0.57% is 57 tokens.
- **No compute or throughput measurement.** `fit-wrap` at `R=16` halves the projection, and
  whether that shows up as wall-clock is untested.
- **Hashed roles (CANINE-style) are not built.** They are the principled length-free scheme;
  `fit-wrap` is the one-line version that happens to work.

---

## 9. Files

| file | what |
|---|---|
| `roles.py` | the codec with a pluggable role map, the five schemes, decoder, self-check |
| `vocab.py` | four real vocabularies as the byte strings the codec would actually see |
| `collide.py` | the count: exact collisions per script, per budget, per scheme |
| `fidelity.py` | what the schemes cost — round-trip decode and locality |
| `train.py` | downstream LM on the Session 2 corpora |
| `results_*.log`, `collisions.json` | the evidence behind every number above |

```bash
python roles.py                    # self-check
python collide.py                  # the count the lesson asks for (~15 min, 530k tokens)
python collide.py --vocabs session2-bpe    # ~10 s
python fidelity.py                 # decode + locality
python train.py --steps 400 --seeds 3
```
