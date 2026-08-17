# Concept 4 — Learned absolute position tables

**Card id:** `learned-absolute` · **Date:** 2018-10-11 (BERT, arXiv v1) — *a judgement call, see below* ·
**Pressure:** where a token sits

## What was read

- [arXiv:1810.04805](https://arxiv.org/abs/1810.04805), Devlin, Chang, Lee, Toutanova — *BERT: Pre-training
  of Deep Bidirectional Transformers for Language Understanding*. Abstract page for version history, and
  the [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1810.04805) for §3 (input representation),
  §3.1 and Appendix A.2 (pretraining procedure).
  Version history: v1 **11 Oct 2018**, v2 24 May 2019.
- [ar5iv:1706.03762](https://ar5iv.labs.arxiv.org/html/1706.03762) §3.5 and Table 3 — the Transformer
  paper's own learned-vs-sinusoidal experiment.
- [arXiv:1705.03122](https://arxiv.org/abs/1705.03122), Gehring, Auli, Grangier, Yarats, Dauphin —
  *Convolutional Sequence to Sequence Learning*. PDF, §3.1 (Position Embeddings), §5.4 + Table 4
  (the ablation). Version history: v1 **8 May 2017**, v2 12 May 2017, v3 25 Jul 2017.
- [`google-research/bert/modeling.py`](https://github.com/google-research/bert/blob/master/modeling.py),
  `embedding_postprocessor` (lines ~436–516). Read because the paper does **not** actually contain the
  claim everybody attributes to it (see *What the source does not establish*). The code does.

### The dating decision — flag this as judgement, not fact

This technique has no invention paper. Three candidates, all real:

| candidate | date | what it actually did |
|---|---|---|
| ConvS2S (Gehring et al.) | **2017-05-08** | introduced it; §3.1 is the clearest statement of the mechanism anywhere, and §5.4 already names the wall |
| Transformer §3.5 | 2017-06-12 | *tried* it, found it equal to sinusoids, and rejected it |
| BERT | **2018-10-11** | made it the default that the next five years inherited |

**Recommendation: date the card 2018-10-11 (BERT v1).** Two reasons, one principled and one structural.
The principled one: the card's subject is not "somebody once added a lookup table", it is "the lookup
table became the thing everyone shipped, and its length limit became everyone's length limit" — and
that is BERT, which is also where the concrete 512 the reader has heard of comes from. The structural
one: the timeline has a baseline-first rule, and ConvS2S at 2017-05-08 lands **five weeks before**
`transformer` (2017-06-12), which would put a positional-scheme card ahead of the baseline card that
motivates positional schemes at all. Dating from ConvS2S would be defensible on priority grounds and
would break the timeline; dating from BERT is defensible on influence grounds and does not. That is a
curatorial choice, not a fact about the literature, and the card should say so in one line — something
like *"learned tables predate this (ConvS2S, May 2017); BERT is where they became the default."*
Do not let the card imply BERT invented them.

## The mechanism, precisely

A **parameter matrix**, shape `[max_positions, d_model]`. One trainable row per absolute index. Position
*k* is looked up as row *k* and **added** to the token embedding. That is the whole method — no formula,
no decay, no structure imposed. Whatever "position 7" means, the model learns it from scratch, and
nothing ties row 7 to row 8.

ConvS2S §3.1 states it most cleanly, verbatim:

> "First, we embed input elements x = (x₁, …, x_m) in distributional space as w = (w₁, …, w_m), where
> w_j ∈ ℝ^f is a column in an embedding matrix D ∈ ℝ^{V×f}. We also equip our model with a sense of
> order by embedding the absolute position of input elements p = (p₁, …, p_m) where p_j ∈ ℝ^f. Both are
> combined to obtain input element representations e = (w₁ + p₁, …, w_m + p_m)."

Note `p_j ∈ ℝ^f` — same dimension as the word embedding, because the combination is addition. Same
reason §3.5 of the Transformer gives for sinusoids: *"The positional encodings have the same dimension
d_model as the embeddings, so that the two can be summed."*

BERT stacks a third table on top. Figure 2 caption, verbatim:

> "BERT input representation. The input embeddings are the sum of the token embeddings, the
> segmentation embeddings and the position embeddings."

and in §3:

> "For a given token, its input representation is constructed by summing the corresponding token,
> segment, and position embeddings."

So BERT's input is a **three-way sum** — token + segment + position — all three learned lookups, all
three `d_model` wide, added before block 1. The app has no segments, so it is the two-way sum.

**The wall.** This is the part to get exactly right, because it is routinely described wrongly.

Row *k* exists for `k < max_positions`. At `k = max_positions` there is **no row**. Not a stale row, not
a badly-trained row, not a row that generalises poorly — the array index is out of range. The reference
implementation makes this literal (`modeling.py`, `embedding_postprocessor`):

```python
if use_position_embeddings:
  assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
  with tf.control_dependencies([assert_op]):
    full_position_embeddings = tf.get_variable(
        name=position_embedding_name,
        shape=[max_position_embeddings, width],
        initializer=create_initializer(initializer_range))
```

The table is `[max_position_embeddings, width]`, and the first thing the graph does is **assert** the
sequence fits. Past the limit the model does not answer worse — it does not answer. The comment above
the variable is also the paper's missing sentence:

> "Since the position embedding table is a learned variable, we create it using a (long) sequence
> length `max_position_embeddings`."

This is the whole difference from concept 2. Sinusoidal `PE(pos, 2i) = sin(pos/10000^(2i/d_model))` is a
*function*: hand it position 5000 and it returns a vector, because a function is defined everywhere.
A table is *finite storage*. That is why the Transformer authors, having found the two equal, still
picked the formula — §3.5, verbatim:

> "We also experimented with using learned positional embeddings instead, and found that the two
> versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
> because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
> during training."

Note the hedge: *"may allow"*. They did not demonstrate it. They preferred it on this argument.

**Growing the table is not free.** Adding rows means adding *randomly initialised* parameters that have
never been trained, in a model whose every other weight has. There is no interpolation to fall back on,
because no source read here claims the rows have any exploitable structure. Extending a learned-absolute
model to longer context is a training problem, not a config change.

And ConvS2S said this out loud, in 2017, before the Transformer existed — §5.4, verbatim:

> "These embeddings allow our model to identify which portion of the source and target sequence it is
> dealing with but also impose a restriction on the maximum sentence length."

The wall was known to the people who introduced the technique. It was adopted anyway, because it is one
line of code and it trains marginally better in-distribution. That is the honest story, and it is more
interesting than "nobody noticed".

## Numbers that matter

**Learned vs sinusoidal — the Transformer's own ablation** (Table 3, EN→DE dev, newstest2013;
per-wordpiece PPL):

| row | PPL (dev) | BLEU (dev) | params |
|---|---|---|---|
| base (sinusoidal) | 4.92 | 25.8 | 65 × 10⁶ |
| (E) positional embedding instead of sinusoids | 4.92 | 25.7 | — |

Identical perplexity, 0.1 BLEU apart. *"Nearly identical results"* is not a euphemism — it is the table.

**ConvS2S — the position-embedding ablation** (Table 4, valid PPL / BLEU):

| model | PPL | BLEU |
|---|---|---|
| ConvS2S | 6.64 | 21.7 |
| − source position | 6.69 | 21.3 |
| − target position | 6.63 | 21.5 |
| − source & target position | 6.68 | 21.2 |

Removing *both* costs 0.5 BLEU. Their own comment: *"position embeddings are helpful but that our model
still performs well without them"* — because a convolutional stack has a receptive field
(*"up to 27 and 25 words respectively"*) that leaks relative position anyway. This does **not** transfer
to a transformer, which is permutation-equivariant and has no such fallback. Useful on the card only as
"even the inventors found the signal weak *in their architecture*" — do not generalise it.

**BERT's limit and its cost.** Model sizes, verbatim: *"BERT_BASE (L=12, H=768, A=12, Total
Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M)."* Corpus, verbatim:
*"For the pre-training corpus we use the BooksCorpus (800M words) and English Wikipedia (2,500M words)."*
Training: 4 days each, 4 Cloud TPUs for BASE, 16 for LARGE.

The 512 and the reason for the schedule, Appendix A.2, verbatim:

> "Longer sequences are disproportionately expensive because attention is quadratic to the sequence
> length. To speed up pretraing in our experiments, we pre-train the model with sequence length of 128
> for 90% of the steps. Then, we train the rest 10% of the steps of sequence of 512 to learn the
> positional embeddings."

(`pretraing` is their typo; quote it or fix it silently, but do not silently reword the rest.)

| quantity | value | provenance |
|---|---|---|
| BERT max positions | 512 | `max_position_embeddings=512`, code default; paper says *"combined length is ≤ 512 tokens"* |
| position table, BASE | 512 × 768 = **393,216** params | derived from stated L/H/A |
| position table, LARGE | 512 × 1024 = **524,288** params | derived |
| share of BERT_BASE's 110M | ≈ **0.36%** | derived — the table is tiny, and still it is the ceiling |
| pretraining length schedule | 128 for 90% of steps, 512 for the last 10% | verbatim, A.2 |
| this app | 32 dims, 4 heads, 2 blocks, ~16 tokens | table would be `max_positions × 32` |

The 0.36% row is the one to put on the card. The component that decides the model's entire maximum
context is a third of a percent of its parameters — cheap to store, impossible to exceed.

## What the live view must let the reader do

The app is 32-dim, 4 heads, 2 blocks, causal, seeded **untrained** weights, ~16 editable tokens. That
last fact constrains every claim below: this view can honestly demonstrate *structure* (a vector exists,
is identical, is absent, is noise) and must never claim *quality* (better, worse, more accurate). There
is no quality in an untrained model to degrade. Design the copy accordingly.

1. **See the table as a table.** Render the `max_positions × 32` matrix as a heatmap, one row per
   position, row index labelled down the side. Hovering a token in the sentence lights its row; clicking
   a row shows its 32 raw numbers. Beside it, the arithmetic for the hovered token: `token embedding +
   position row = input to block 1`, all three vectors shown. What this proves: position is a *stored
   vector per index*, not a rule — and the reader can see that row 3 and row 4 have no visible
   relationship, which is exactly the difference from the sinusoidal card's smooth banded stripes. Show
   the two side by side if the sinusoidal card's rendering can be reused; the contrast between
   "structured stripes" and "unstructured noise" is the whole point and costs nothing to display.

2. **The wall — centrepiece.** A `max_positions` spinner, default **12**, range 4–24, over a sentence
   the reader can freely extend past it. Three linked panels:

   - *The table*, drawn with exactly `max_positions` rows, and the region below the last row rendered as
     empty hatching labelled **"no rows here"** — not greyed-out rows, visibly *absent* rows. That
     distinction is the concept.
   - *The sentence*, tokens 0…`max_positions−1` normal; tokens at and past the limit marked
     **"no position row"** in red.
   - *The gate*, shown as the literal check BERT's code runs:
     `seq_length (17) ≤ max_positions (12) → FALSE`. And then the model **refuses to run** — the
     attention maps and logits go blank with that message, rather than showing a degraded output. A
     reader who sees a number appear will conclude the model coped. It did not; there is nothing to
     compute.

   Then, and only after the refusal has been seen, offer three repair buttons — all of which the reader
   should come away distrusting:

   - **Truncate** — drop the overflow tokens. Display: the attention map loses those columns entirely,
     and the words the reader just typed provably influence nothing. This is what production systems
     actually do.
   - **Clamp** — reuse row `max_positions−1` for every overflow token. Display: the input vectors for
     those tokens now differ *only* by their token embedding, and their attention rows become visibly
     near-identical. Print the cosine similarity between two clamped positions' rows as **1.000**.
     Position information is gone, mechanically and measurably.
   - **Extend with fresh rows** — append randomly initialised rows. Display: the new rows in the heatmap
     are visibly uncorrelated with the trained block above them, and the boundary is a hard seam. Label
     it *"these rows have never been trained"*.

   **What proves the point:** the reader adds one word inside the limit and nothing dramatic happens;
   they add one word crossing the limit and the model stops. There is no gradient, no curve, no gentle
   falloff — a cliff at exactly index `max_positions`. Make the spinner and the sentence the two things
   the reader is most tempted to fiddle with, and let them find the cliff themselves.

3. **A/B against sinusoidal, past the wall.** Toggle the scheme with everything else held fixed, and
   show two things in sequence. *Inside* the limit: both schemes produce working attention maps that are
   different but neither is obviously better — the reader cannot tell which is which, which is Table 3
   row (E) reproduced at a scale where nobody can fake it. *Past* the limit: the sinusoidal panel keeps
   rendering, because `sin(pos/10000^(2i/d))` is defined at position 19 as surely as at position 3; the
   learned panel shows the failed assert. Caption it with the paper's own sentence about extrapolation,
   including the hedge *"may allow"*.

4. **Same word, two positions.** Type a sentence with a repeated token. Show its two input vectors,
   identical in the token half, different only by the position row, and show that its two attention rows
   differ as a result. Then zero the position table and watch the two rows become identical. This is the
   minimal proof that the table is what carries position at all, and it takes one toggle.

5. **The price of the wall.** A live readout as the spinner moves: table parameters =
   `max_positions × 32`, next to BERT's `512 × 768 = 393,216` and the **0.36%** figure. Alongside it,
   the asymmetry BERT states: doubling `max_positions` doubles the table linearly but costs attention
   **quadratically** — quote *"Longer sequences are disproportionately expensive because attention is
   quadratic to the sequence length"* and let the reader see why 512 was a budget decision, not a
   discovery about language.

## What the source does *not* establish

- **BERT never says its position embeddings are learned.** Searched the full text: the word "learned"
  appears about the *segment* embedding (*"we add a learned embedding to every token indicating whether
  it belongs to sentence A or sentence B"*) and nowhere about position embeddings. The paper also never
  states a flat "the maximum sequence length is 512" — the closest sentences are *"They are sampled such
  that the combined length is ≤ 512 tokens"* (about generating pretraining examples) and the A.2
  schedule. The hard 512 and the learnedness both come from the **released code**
  (`max_position_embeddings=512`, `tf.get_variable(...)`), not the paper. This is worth a line on the
  card: the most-cited fact about BERT's positions is sourced from its repository.
- **Undertrained far positions: NOT established.** No source read here measures, or claims, that
  high-index rows are worse-trained or worse-behaved. The nearest real evidence is BERT's *"we train the
  rest 10% of the steps of sequence of 512 to learn the positional embeddings"* — which shows the authors
  knew rows 128–511 needed dedicated exposure and budgeted for it, and is therefore evidence they
  *addressed* the problem, not evidence the problem survived. The popular claim that late rows are
  undertrained is plausible and widely repeated; it is not in these papers. The card must not assert it.
  If it is wanted, it needs its own citation (later analysis work), and should be marked as such.
- **"Learned is worse than sinusoidal" is not established — the single measurement says they tie.**
  4.92 vs 4.92 PPL, 25.8 vs 25.7 BLEU. The sinusoidal preference in §3.5 rests entirely on an
  *untested hypothesis about extrapolation*, hedged as "may allow". Any card copy implying learned
  tables are the inferior choice on quality grounds contradicts the only number either paper reports.
- **ConvS2S's 0.5 BLEU ablation does not transfer.** A CNN has a receptive field
  (*"up to 27 and 25 words"*) that carries relative position implicitly; a transformer is permutation-
  equivariant and has none. Do not use "positions only cost 0.5 BLEU" as a general claim.
- **Nothing here is about extrapolation working.** No source read demonstrates that *any* scheme
  successfully extrapolates past training length. Sinusoids were *hoped* to; that is all.
- **The app's outputs establish nothing.** Seeded, untrained. Every claim in the live view is about
  vectors existing, matching, or being absent — never about predictions being good.

## Leaves behind

A hard ceiling nailed into the parameter file, and the awkward fact that everyone adopted it knowing
about the ceiling (ConvS2S §5.4 named it in May 2017) because it was simpler and tied on the benchmark.
Three costs go forward:

- **The ceiling itself** — a model's maximum context is a table dimension chosen before pretraining and
  unfixable afterwards without more training. This is what relative-position schemes attack: if the
  model only ever sees *offsets* rather than *absolute indices*, there is no table to run off the end
  of. Picked up next by relative position representations (Shaw et al., 2018) and then by the
  bucketed/biased approaches.
- **The unstructured rows** — nothing relates row *k* to row *k+1*, so there is nothing to interpolate
  or extrapolate along. RoPE and ALiBi later re-impose exactly that structure, from opposite directions
  (rotation vs. a linear distance penalty on the scores), and both recover the extrapolation property
  the Transformer authors wanted from sinusoids but never demonstrated.
- **The quadratic reason 512 was 512** — BERT's own explanation for the limit is attention cost, not
  positions. That is the `transformer` card's O(n²) bill coming due in a second place, and it is the
  thread the efficiency cards pick up.
