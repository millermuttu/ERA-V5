# Concept 19 — Attention sinks / StreamingLLM

**Card id:** `attention-sinks` · **Date:** 2023-09-29 (arXiv v1) · **Pressure:** what generation must
remember

## What was read

- [arXiv:2309.17453](https://arxiv.org/abs/2309.17453), Guangxuan Xiao (MIT; "part of the work done
  during an internship at Meta AI"), Yuandong Tian (Meta AI), Beidi Chen (CMU), Song Han (MIT, and
  NVIDIA from v3 on), Mike Lewis (Meta AI) — *Efficient Streaming Language Models with Attention
  Sinks*. Abstract page for the version history; the full text via
  [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.17453) end to end — §1–§5 and **Appendices A–I** —
  and **all four PDFs pulled and text-extracted** (`arxiv.org/pdf/2309.17453v1` … `v4`) so the
  versions could be diffed sentence by sentence rather than trusted. Code:
  `https://github.com/mit-han-lab/streaming-llm`.
- **The app's record says 2023-09-29. Verified — that is the v1 date.** Four versions:

  | | date | size | what it is |
  |---|---|---|---|
  | **v1** | **Fri, 29 Sep 2023** | 9,548 KB | the paper. **12 pages, no appendices at all.** |
  | v2 | Tue, 21 Nov 2023 | 13,230 KB | adds **Appendices A–I** — including the Limitations paragraph |
  | v3 | Tue, 12 Dec 2023 | 13,230 KB | affiliation (+NVIDIA), acknowledgements, two-word edits |
  | v4 | Sun, 7 Apr 2024 | 13,231 KB | ICLR 2024 camera-ready; adds Reproducibility and Impact statements |

- **What the diff establishes, and it matters for the honesty section.** Every headline number is in
  v1 unchanged: `5158.07`, `5.40`, `5.60`, Table 2, Table 3, `22.2×`, "4 million tokens", and the
  Figure 1 panel labels. **Nothing was retracted.** But the two sentences everyone should quote when
  they cite this paper were **not in v1**:
  - The Limitations paragraph — *"it does not extend the models' context window or enhance their
    long-term memory capabilities"* — arrived with the appendices in **v2**.
  - The Introduction's own disclaimer — *"Finally, we emphasize that StreamingLLM efficiently
    generates coherent text from tokens within the KV cache **without extending the LLMs' context
    length**"* — was added in **v4**, six months after v1.

    v1 said only *"StreamingLLM firstly decouples the LLM's pre-training window size and its actual
    text generation length"* (Conclusion, identical in all four). The folklore that this paper gives
    you infinite context had a six-month window in which the paper did not explicitly contradict it.
    Say so on the card; it explains the misreading rather than just scolding it.
- Everything quoted below is the authors' wording. Section and table references are to the v4/ar5iv
  text; where a passage is v2-or-later only, it is marked.

## The mechanism, precisely

### The observation comes first, and it is violent

The setup is a decoder generating token `T` from a model pre-trained at length `L`, with `T ≫ L`.
Figure 1 lays out **four** ways to run it, with cost and quality on each. Perplexities are *"measured
using the Llama-2-13B model on the first book (65K tokens) in the PG-19 test set"*:

| | complexity | PPL | the paper's own one-line verdict (Figure 1 panel) |
|---|---|---|---|
| (a) Dense attention | `O(T²)` | **5641** | *"Has poor efficiency and performance on long text."* |
| (b) Window attention | `O(TL)` | **5158** | *"Breaks when initial tokens are evicted."* |
| (c) Sliding window w/ re-computation | `O(TL²)` | **5.43** | *"Has to re-compute cache for each incoming token."* |
| (d) **StreamingLLM** | `O(TL)` | **5.40** | *"Can perform efficient and stable language modeling on long texts."* |

Read the middle column and the right column together: **(b) and (d) have the same complexity and the
same cache size, and differ by a factor of 955 in perplexity.** That is the whole card in one row
pair. And (c), the only baseline with acceptable quality, pays `O(TL²)` — it *"rebuilds the KV states
from the L recent tokens for each new token"*, which is why it is the thing the speed result is
measured against.

The caption on (b) is precise about how little it takes:

> "Window Attention caches the most recent $L$ tokens' KV. While efficient in inference, performance
> declines sharply once the starting tokens' keys and values are evicted."

And §1 puts the knife-edge in a subclause:

> "Although it ensures constant memory usage and decoding speed after the cache is initially filled,
> **the model collapses once the sequence length exceeds the cache size, i.e., even just evicting the
> KV of the first token**, as illustrated in Figure 3."

§3.1's heading for this is *"Identifying the Point of Perplexity Surge"*:

> "Figure 3 shows the perplexity of language modeling on a 20K token text. It is evident that
> **perplexity spikes when the text length surpasses the cache size, led by the exclusion of initial
> tokens.** This suggests that the initial tokens, **regardless of their distance from the predicted
> tokens**, are crucial for maintaining the stability of LLMs."

The collapse is not a decay curve. It is a step, at the exact token index where eviction begins.

### The explanation, in the paper's own argument

Two sentences carry it, and the card should quote both because the second is the part people drop:

> "We attribute the reason to the Softmax operation, **which requires attention scores to sum up to
> one for all contextual tokens.** Thus, even when the current query does not have a strong match in
> many previous tokens, **the model still needs to allocate these unneeded attention values somewhere
> so it sums up to one.** The reason behind initial tokens as sink tokens is intuitive: **initial
> tokens are visible to almost all subsequent tokens** because of the autoregressive language
> modeling nature, making them more readily trained to serve as attention sinks."

Restated in §3.1 under the heading *"LLMs attend to Initial Tokens as Attention Sinks"*:

> "The nature of the SoftMax function (Equation 1) **prevents all attended tokens from having zero
> values.** This requires aggregating some information from other tokens across all heads in all
> layers, **even if the current embedding has sufficient self-contained information for its
> prediction.** Consequently, the model tends to **dump unnecessary attention values** to specific
> tokens."

So the mechanism is a *plumbing* argument, not a semantic one: softmax has no "attend to nothing"
output, a head that wants nothing must still emit a distribution summing to 1, and the cheapest place
to put the surplus is a position that is (i) always present and (ii) carries little enough meaning
that dumping onto it costs nothing. Position 0 is the only index in a causal model that satisfies (i)
for every query.

The failure mode follows immediately, and it is arithmetic:

> "We find that, beyond the bottom two layers, the model consistently focuses on the initial tokens
> across all layers and heads. The implication is clear: **removing these initial tokens' KV will
> remove a considerable portion of the denominator in the SoftMax function** (Equation 1) in
> attention computation. This alteration leads to a **significant shift in the distribution of
> attention scores** away from what would be expected in normal inference settings."

Equation 1 is written to make the point typographically — the sink term is pulled out of the sum:

    SoftMax(x)_i = e^{x_i} / ( e^{x_1} + Σ_{j=2..N} e^{x_j} ),   with   x_1 ≫ x_j,  j ∈ 2,…,N

Delete `e^{x_1}` from a denominator it dominated and **every surviving weight is multiplied by the
same factor `Z / Z′ > 1`**. The pattern of the remaining attention is untouched; its *scale* is
inflated by however much the sink was holding. That renormalisation is the entire collapse, and it is
the one part of the mechanism this app can reproduce exactly (see §Numbers, [measured here]).

### Semantic or positional? — the paper tests it, and the answer decides the design

This is the fork the card must not blur:

> "There are two possible explanations for the importance of the initial tokens in language modeling:
> (1) Either their semantics are crucial, or (2) the model learns a bias towards their absolute
> position. To distinguish between these possibilities, we conduct experiments (Table 1), wherein
> **the first four tokens are substituted with the linebreak token "\n".** The observations indicate
> that the model still significantly emphasizes these initial linebreak tokens. Furthermore,
> reintroducing them restores the language modeling perplexity to levels comparable to having the
> original initial tokens. This suggests that **the absolute position of the starting tokens, rather
> than their semantic value, holds greater significance.**"

Table 1 is three rows and settles it: window `5158.07`, four real initial tokens `5.40`, **four
linebreaks `5.60`**. Four newlines recover 99.6% of the gap. Whatever the sink is holding, it is not
the meaning of the sentence's first four words — it is a *slot*.

That is why the fix is legal at all. If the sinks were semantically necessary you would have to keep
the *right* early tokens; because they are positional, any four will do, and you can keep the same
four forever while the rest of the cache rolls.

The paper also says why it is *four* and not one:

> "We believe this pattern emerges because these models didn't include a consistent starting token
> across all input samples during pre-training. Although **Llama-2 does prefix each paragraph with a
> "\<s\>" token, it's applied before text chunking, resulting in a mostly random token occupying the
> zeroth position.** This lack of a uniform starting token leads the model to use several initial
> tokens as attention sinks."

Four is a *contingent* number — an artefact of how training data was chunked, not a property of
attention. §3.3 proves that by removing the contingency (below).

### The method: what is kept, what is evicted, and how positions are numbered

§3.2, in full on the cache shape:

> "Alongside the current sliding window tokens, **we reintroduce a few starting tokens' KV in the
> attention computation.** The KV cache in StreamingLLM can be conceptually divided into two parts,
> as illustrated in Figure 4: (1) **Attention sinks (four initial tokens)** stabilize the attention
> computation; 2) **Rolling KV Cache** retains the most recent tokens, crucial for language modeling.
> StreamingLLM's design is versatile and can be seamlessly incorporated into any autoregressive
> language model that employs **relative positional encoding, such as RoPE and ALiBi.**"

Note what is *not* here: no retraining, no fine-tuning, no kernel, no change to the model at all. The
cache is `x + y` — `x` frozen entries at the front, `y` rolling entries at the back, everything in
between dropped and never recovered.

**And now the detail everyone gets wrong.** The positions are renumbered:

> "**When determining the relative distance and adding positional information to tokens,
> StreamingLLM focuses on positions within the cache rather than those in the original text. This
> distinction is crucial for StreamingLLM's performance.** For instance, if the current cache
> (Figure 4) has tokens [0, 1, 2, 3, 6, 7, 8] and is in the process of decoding the 9th token, **the
> positions assigned are [0, 1, 2, 3, 4, 5, 6, 7], rather than the positions in the original text,
> which would be [0, 1, 2, 3, 6, 7, 8, 9].**"

Read the two lists carefully — the paper's own example has the query token in it, which is why there
are 7 cached tokens and 8 assigned positions. The cache is *contiguous by construction*: the gap
between the sinks and the rolling window is closed, and the sinks sit at distance `|cache|` from the
current token, not at distance `T`. A model asked about position 4,000,000 would be far outside
anything it saw in training; a model asked about position 1,023 is at home. **This is where the
length generalisation actually comes from** — not from the sinks, from the renumbering. The sinks fix
the denominator; the renumbering keeps the position arithmetic inside the training distribution. Both
are needed, and the paper says so in the same breath:

> "**This method of assigning positional embedding within the cache is crucial to StreamingLLM's
> functionality**, ensuring that the model operates efficiently even beyond its pre-training
> attention window size."

The per-scheme implementation detail, which is the seam-level fact:

> "For encoding like RoPE, **we cache the Keys of tokens prior to introducing the rotary
> transformation.** Then, we apply position transformation to the keys in the rolling cache at each
> decoding phase. On the other hand, integrating with ALiBi is more direct. Here, **the contiguous
> linear bias is applied instead of a 'jumping' bias** to the attention scores."

Two words worth putting on the card: *prior to*, and *jumping*. RoPE bakes position into the key, so
the cache must store the key **un-rotated** and re-rotate on every step — a real implementation cost
nobody mentions when they call this method free. ALiBi's bias is computed on the fly from an index,
so you just feed it the cache index; the bias that would result from using text positions is the
"jumping" one, and it is the wrong one.

### The pre-training variant

§3.3 takes the "four sinks is an artefact" hypothesis and removes the artefact:

> "A potential remedy can be the intentional inclusion of a **global trainable attention sink token,
> denoted as a 'Sink Token'**, which would serve as a repository for unnecessary attention scores.
> Alternatively, replacing the conventional SoftMax function with a variant like SoftMax-off-by-One …
> which does not require the attention scores on all contextual tokens to sum up to one, may also be
> effective. **Note that SoftMax₁ is equivalent to prepending a token with an all-zero Key and Value
> features** in the attention computation. We denote this method as **'Zero Sink'** to fit our
> framework."

    SoftMax₁(x)_i = e^{x_i} / ( 1 + Σ_{j=1..N} e^{x_j} )        (Equation 2)

Three 160M models trained from scratch, identical settings. Table 3 is the payoff, and it contains a
result the paper is quiet about (see Numbers). The recommendation:

> "Introducing a sink token is highly effective in stabilizing the attention mechanism. Simply
> pairing this sink token with recent tokens sufficiently anchors the model's performance, and the
> resulting evaluation perplexity is even marginally improved. Given these findings, **we recommend
> training future LLMs with a sink token in all samples** to optimize streaming deployment."

## Numbers that matter

**Table 1 — the collapse and the semantic/positional test** (Llama-2-13B, first book of PG19, 65K
tokens):

| cache config | PPL |
|---|---|
| `0 + 1024` (window attention) | **5158.07** |
| `4 + 1020` | **5.40** |
| `4"\n" + 1020` (first four replaced by linebreaks) | **5.60** |

**Table 2 — how many sinks, and it is not one** (400K tokens, concatenated PG19):

| model | `0+y` | `1+…` | `2+…` | `4+…` | `8+…` |
|---|---|---|---|---|---|
| Falcon-7B (`y=2048`) | 17.90 | 12.12 | 12.12 | 12.12 | 12.12 |
| MPT-7B (`y=2048`) | **460.29** | 14.99 | 15.00 | 14.99 | 14.98 |
| Pythia-12B (`y=2048`) | 21.62 | 11.95 | 12.09 | 12.09 | 12.02 |
| Llama-2-7B (`y=4096`) | **3359.95** | 11.88 | 10.51 | **9.59** | 9.54 |

The paper's reading: *"(2) Introducing one or two initial tokens doesn't fully restore model
perplexity, showing that the model doesn't solely use the first token as the attention sink. (3)
Introducing four initial tokens generally suffices; further additions have diminishing returns."*
§4.4: *"a threshold of four initial tokens appears enough, with subsequent additions contributing
marginal effects. This result justifies our choice of introducing 4 initial tokens."*

**Look at the row spread before repeating "the collapse is 1000×".** It is 1000× on Llama-2 and MPT
and **1.5× on Falcon-7B** (17.90 → 12.12) and **1.8× on Pythia-12B**. The headline `5158 → 5.40` is
the worst case, not the typical one. And Llama-2-7B is the only model where the 1→4 step does real
work (11.88 → 9.59); on Falcon the first token alone recovers everything to the last decimal, and on
MPT and Pythia the 1-token and 8-token columns are within 0.15. **"Four" is a safe default, not a
measured optimum, and on three of four models one would have done.**

**Table 3 — pre-training with a sink** (160M from scratch, first sample of PG19):

| | `0+1024` | `1+1023` | `2+1022` | `4+1020` |
|---|---|---|---|---|
| Vanilla | **27.87** | 18.49 | 18.05 | 18.05 |
| Zero Sink (SoftMax₁) | **29214** | 19.90 | 18.27 | 18.01 |
| Learnable Sink | **1235** | **18.01** | 18.01 | 18.02 |

Two things here, and the second is not in the paper's prose. First, the intended result: with a
learnable sink token, **one** cached entry is enough — `18.01` at `1+1023`, matching its own `4+1020`
— whereas vanilla needs two-to-four. Second, the awkward one: **in the `0+y` column, both remedies are
catastrophically worse than vanilla** (27.87 → 29214 for Zero Sink, → 1235 for Learnable Sink). The
paper never comments on it. It is coherent — a model given a dedicated dump concentrates *everything*
there, so losing it is worse than losing a diffuse sink — but it means these methods make the model
*more* fragile to the one failure they are designed around, and only the discipline of always keeping
the sink saves them. Worth a line on the card; it is the paper's own number.

Note also the vanilla `0+1024` here is **27.87**, not 5158. A 160M model has a much softer sink than a
13B one. The size of the collapse scales with the model.

**Appendix I — more sinks do not help** (v2+): a second sink token gives benchmark scores within noise
(Table 9), and Table 10 shows `+2 Sink Tokens` at `1+1023` is **25.73** against `+1 Sink Token`'s
`18.01` — *"the model appears to rely on both sink tokens"*, so two sinks means you must now cache
two. *"This contrasts with findings in Vision Transformers (ViT), where multiple 'registers' have
been found to be beneficial."*

**Speed and memory (§4.5, Figure 10).** Against sliding-window-with-recomputation, the only baseline
with acceptable quality, Llama-2-7B/13B on **a single NVIDIA A6000**, Huggingface Transformers:

> "As the cache size increases, StreamingLLM's decoding speed has a **linear growth**. The sliding
> window with re-computation baseline has a **quadratic rise** in decoding latency. Thus,
> StreamingLLM achieves an impressive speedup, reaching **up to 22.2× per token**. Despite its
> reduced latency, StreamingLLM **sustains a memory footprint consistent with the re-computation
> baseline.**"

So: **22.2× faster, ~equal memory** — and both halves need saying. The memory is *not* improved
against that baseline; the baseline was already `O(L)` in cache. What is improved is that the `O(L²)`
recomputation per token goes away, so the gap *widens with cache size* (linear vs. quadratic) — 22.2×
is the value at the largest cache tested, not a constant. (Figure 10's per-bar numbers are inside the
graphic; the extracted labels include latencies up to `2355 ms` for the baseline against double-digit
values for StreamingLLM, but the pairing is not recoverable from text — do not quote individual bars.)

**Streaming length and coverage.** *"models including Llama-2-[7, 13, 70]B, MPT-[7, 30]B,
Falcon-[7, 40]B, and Pythia-[2.9,6.9,12]B can reliably model 4 million tokens, and potentially even
more"* (Figure 5, concatenated PG19, 100 books). Cache sizes in §4.1: **2048 for Llama-2, 1024 for
Falcon/Pythia/MPT** — *"This is half the pre-training window size chosen to enhance visualization
clarity."* Coverage spans both major position schemes: RoPE (Llama-2, Falcon, Pythia) and ALiBi
(MPT).

**Streaming QA (Table 5, ARC, cache 1024).** Dense: **OOM**. Window attention: **3.58 / 1.39** (7B),
**0.25 / 0.34** (13B), **0.12 / 0.32** (70B) — i.e. zero. StreamingLLM: **71.34 / 55.03**, **80.89 /
65.61**, **91.37 / 80.20**, against a one-shot sample-by-sample baseline of 71.25 / 53.16, 78.16 /
63.31, 91.29 / 78.50.

**Table 6 — the ablation that undercuts the headline.** More cache does not reliably mean better:

| Cache | `4+252` | `4+508` | `4+1020` | `4+2044` |
|---|---|---|---|---|
| Falcon-7B | 13.61 | 12.84 | **12.34** | 12.84 |
| MPT-7B | **14.12** | 14.25 | 14.33 | 14.99 |
| Pythia-12B | 13.17 | 12.52 | **12.08** | 12.09 |

| Cache | `4+508` | `4+1020` | `4+2044` | `4+4092` |
|---|---|---|---|---|
| Llama-2-7B | 9.73 | 9.32 | **9.08** | 9.59 |

MPT-7B is **monotonically worse** with more context. The paper's own gloss: *"Contrary to intuition,
increasing the cache size doesn't consistently lower the language modeling perplexity. This
inconsistency shows a potential limitation where these models might not maximize the utility of the
entire context they receive."*

**Appendix C (v2+) — the number that kills the "infinite context" reading.** StreamEval, Llama-2-7B-32K-Instruct,
23 tokens per line, accuracy vs. query-to-answer distance:

| line distance | token distance | `4+2044` | `4+4092` | `4+8188` | `4+16380` |
|---|---|---|---|---|---|
| 20 | 460 | 85.80 | 84.60 | 81.15 | 77.65 |
| 80 | 1840 | 75.30 | 77.15 | 76.40 | 73.80 |
| **100** | **2300** | **0.00** | 61.60 | 50.10 | 40.50 |
| 200 | 4600 | 0.00 | **0.00** | 62.75 | 46.90 |
| 400 | 9200 | 0.00 | 0.00 | **0.00** | 45.70 |
| 800 | 18400 | 0.00 | 0.00 | 0.00 | **0.00** |

Each column goes to **exactly zero** the moment the answer falls outside that column's cache. Not
degraded — zero. *"StreamingLLM retains accuracy when the token distance between the query and answer
is within the cache size. However, accuracy diminishes as this distance increases and eventually
drops to zero when it surpasses the cache capacity."* And: *"these results demonstrate that while
StreamingLLM is effective in generating coherent text based on recent context, **it cannot extend the
context length of language models.**"*

**Appendix D (v2+) — LongBench**, Llama-2-7B-chat vs. LongBench's own middle-truncation baseline
(1750 head + 1750 tail):

| | NarrativeQA | Qasper | HotpotQA | 2WikiMQA | GovReport | MultiNews |
|---|---|---|---|---|---|---|
| Truncation 1750+1750 | **18.7** | 19.2 | **25.4** | **32.8** | **27.3** | 25.8 |
| StreamingLLM `4+3496` | 11.6 | 16.9 | 21.6 | 28.2 | 23.9 | 25.5 |
| StreamingLLM `1750+1750` | 18.2 | **19.7** | 24.9 | 32.0 | 26.3 | **25.9** |

The published configuration **loses on five of six tasks**, *"likely due to the loss of crucial
initial input prompt information"*. It only ties when the "sink" is inflated to 1750 tokens — i.e.
when it stops being a sink and becomes the prompt.

**Appendix F (v2+) — how big a sink actually is.** Llama-2-7B, 256 sequences of 4096 tokens, the
4096th token's attention to the first: *"the attention scores for the first token are significantly
high, **often exceeding half of the total attention**, except for the two bottom layers."* Figure 2's
observation that fixes the layer structure: *"(1) The attention maps in the first two layers (layers
0 and 1) exhibit the 'local' pattern, with recent tokens receiving more attention. (2) Beyond the
bottom two layers, the model heavily attends to the initial token across all layers and heads."*
Confirmed at length 128 (App. E) and on Llama-2-70B (App. G).

**Where the effect is and is not.** App. H: BERT-base-uncased shows the same phenomenon but the sink
is *"the **[SEP]** token in most layers"* — in a bidirectional encoder the always-visible token is
not the first one, it is the separator. App. B links it to Darcet et al.'s ViT "registers", with the
distinction stated: *"'registers' in Vision Transformers function as global information holders
within intermediate layers, whereas our 'attention sinks' are positioned as initial tokens in
autoregressive models."*

### [measured here] — this app, 32 dims, 4 heads, 2 blocks, seed 20260817, 16-token default sentence

Driven by node against `app/model/transformer.js` and `app/model/mixers.js` directly. Sentence: *"the
lighthouse keeper wrote the code in a notebook and hid it under the third stair"*.

**1. Does this untrained model have an attention sink? No.** Mean attention *received* by each key
position, averaged over heads and over queries `i ≥ 8`:

```
block 0   j=0: 0.069   j=1: 0.059   j=5: 0.243   j=15: 0.245      uniform ≈ 0.083
block 1   j=0: 0.031   j=1: 0.141   j=6: 0.179   j=11: 0.131
```

Position 0 is **below uniform in both blocks**. Averaged over every (block, head, query ≥ 4) its
weight is **0.068** against a uniform expectation of **0.095**, and its mean rank among the keys a
query may read is **5.15**. There is no sink. This is exactly what the paper predicts — a sink is
*"more readily **trained** to serve as"* one — and it is the single most important fact for the card's
design. The model *is* peaky (mean largest weight in a row **0.662**, and block-0 head 1 puts
**1.000** on one key), but it is peaky about arbitrary mid-sentence tokens: block 0's four heads want
`5 (code)`, `5 (code)`, `15 (stair)`, `6 (in)`. Random projections produce confident nonsense, not
sinks.

**2. The denominator arithmetic, which the app reproduces exactly.** Block 0, last query, comparing
the full causal row against a `0+8` window (block 0's Q/K are unaffected by the mechanism, so this is
a true subset of the same score row):

| head | full `Z` | `0+8` keeps | `4+8` keeps |
|---|---|---|---|
| 0 | 1.580 | **29.5%** | 36.4% |
| 1 | 1.000 | **0.0%** | 0.0% |
| 2 | 1.026 | 97.5% | 100.0% |
| 3 | 1.891 | **45.7%** | 47.1% |

And the identity that *is* the collapse, checked numerically: with `Z/Z′ = 3.3872` on head 0, the
surviving weight at `j=15` goes `0.00401 × 3.3872 = 0.01359`, against a measured `0.01359`. **Every
surviving weight is multiplied by exactly the same number.** The shape is preserved; the scale is
not. That is the mechanism, and the app has it to five decimal places.

**3. What eviction moves.** Block 1, head 0, last query — the one head that does put visible mass on
token 0:

```
j        0       1       5       7       8      10
full   0.1473  0.0953  0.1163  0.4227  0.0817  0.0100
0+8    0.0000  0.0000  0.0000  0.0000  0.5050  0.4483
4+8    0.8619  0.0058  0.0000  0.0000  0.0859  0.0280
```

**81% of this head's attention sits on tokens the `0+8` window evicts**, and when four sinks are kept,
token 0 goes from 0.147 to **0.862** — it absorbs the renormalisation. That is the picture of a sink
doing its job, produced by a model that has no sink, purely because the surplus has nowhere else to go.

**4. Positions in the cache vs positions in the text.** Cache `4+8` at `T=16` keeps text positions
`[0,1,2,3,8,9,10,11,12,13,14,15]` and assigns `[0,1,2,3,4,5,6,7,8,9,10,11]`. Under RoPE, the last
query's block-0 head-0 weights over those same twelve keys:

```
text positions   0.944  0.000  0.001  0.000  0.000  0.052  ...
cache positions  0.048  0.000  0.005  0.000  0.000  0.874  ...
```

**Total variation 0.896** — the two are almost disjoint distributions over the *same twelve key
vectors*. The query's distance to the sink is 15 under text numbering and 11 under cache numbering.
This is the sharpest demonstration the app can give, and it needs no training: renumbering is not a
bookkeeping detail, it changes which token wins.

**5. Does the recipe help *this* model? No, and it should not.** KL of the next-word distribution
against the full-attention run, **at equal cache budget `B`** (the paper's own `x+y` design):

| | `0+B` | `1+…` | `2+…` | `4+…` |
|---|---|---|---|---|
| B=8 | 1.332 | 1.371 | 1.089 | 3.963 |
| B=10 | 2.391 | 1.782 | 2.240 | 1.189 |
| B=12 | **0.023** | 0.021 | 2.585 | 2.137 |

No ordering, no trend — noise. On a model with no sink, spending cache entries on position 0 buys
nothing and costs recency. **Print this table on the card as a negative result.** It is the honest
inverse of Table 2, and its disagreement with Table 2 is the evidence that the phenomenon is learned.

**6. Cache arithmetic at `T=16`** (`toyCost`-style counting, causal):

| config | keys read per query | KV entries held |
|---|---|---|
| full causal | 8.50 | 16 (grows with `T`) |
| `0+8` | 6.25 | 8 |
| `4+8` | 7.88 | 12 |
| `4+4` | 6.25 | 8 |

At `T=16` the sentence is barely longer than the cache, so the saving looks trivial — which is itself
worth showing, because the entry that matters is the second column being **constant in `T`**.

## What the live view must let the reader do

The seam needs **no changes at all**. `softmaxMixer({readable})` expresses the whole method in one
predicate, and `at` already carries `{block, head}`:

```js
const readable = (i, j) => j < sinks || j >= i - recent + 1;   // x + y, exactly
```

Positions are the other half, and `rotate` is the hook: RoPE's `rotate(v, pos)` is called with the
*text* index today, so the cache-position variant is `rotate(v, cachePos.get(pos) ?? pos)` built from
the same `readable`. Both halves already exist. What does **not** exist in this model is a sink — and
the card is honest or it is worthless.

**Concept 8 already owns:** the `w` slider against the reads-per-query bill, the depth/receptive-field
probe on activations, dilation, and the hand-picked global hub. **None of the panels below repeats
any of those.** This card's assets are the *denominator*, the *renumbering*, and the *learned-artefact
boundary*.

### 1. Evict the first token — the step, not the slope *(the opening)*
**Control:** one number, `sinks` (0–4), with `recent` fixed; and a "cache is full" marker on the
sentence showing which words are gone.
**Displayed:** for the selected head, the full-attention weight row and the cached row side by side,
plus three computed numbers: **mass on the evicted tokens** (`0.810` at block 1 head 0, `w=8`),
**fraction of the denominator kept** (`29.5%` on block 0 head 0), and **the multiplier every survivor
gets** (`Z/Z′ = 3.39`).
**The number that proves it is the multiplier**, because the reader can check it by hand on any row:
`0.00401 × 3.3872 = 0.01359`. Caption with Equation 1 and the paper's own sentence — *"removing these
initial tokens' KV will remove a considerable portion of the denominator"* — and state plainly that in
a trained model the removed term is *"often exceeding half of the total attention"* (App. F), so the
multiplier there is ~2× on **every** remaining weight in **every** head above layer 1.

### 2. The four-way comparison, as a cost table the reader drives
**Control:** the four settings of Figure 1 as radio buttons.
**Displayed:** a live four-row table — complexity, keys read per query, KV entries held, and whether
the entry count grows with the sentence — recomputed from the reader's own sentence via the app's
counting, next to the paper's static PPL column (5641 / 5158 / 5.43 / 5.40) clearly labelled *"paper,
Llama-2-13B, PG-19"*.
**The row that does the work is (b) against (d):** identical complexity, identical cache size, four
extra entries, 955× the perplexity. Put the two rows adjacent and let the width of that gap be the
card's headline. And mark (c)'s `O(TL²)` as the thing the 22.2× is measured against — not against
dense, not against window.

### 3. Positions are relative to the cache — the interaction nobody else has *(the distinctive one)*
**Control:** a toggle, "number positions by the text / by the cache", with RoPE on.
**Displayed:** the kept positions and the assigned positions as two aligned strips (`[0,1,2,3,8,…,15]`
above `[0,1,2,3,4,…,11]`), the query's distance to the sink under each (`15` vs `11`), and the last
query's attention weights under each numbering with the **total variation between them: 0.896**
[measured here].
**The number that proves it is that TV,** because it says the same twelve key vectors produce almost
disjoint attention depending only on what integer you hand the rotation. Quote the paper's example
verbatim — `[0,1,2,3,6,7,8]` → `[0,1,2,3,4,5,6,7]` — and its judgement, *"This distinction is crucial
for StreamingLLM's performance."* Add the implementation consequence as a note: RoPE caches keys
*"prior to introducing the rotary transformation"* and re-rotates every step; ALiBi gets *"the
contiguous linear bias … instead of a 'jumping' bias"*. Let the reader switch the position scheme
between RoPE and ALiBi (`position.js` has both) and see that the renumbering matters for each in a
different place — inside the key for one, inside the bias for the other.

### 4. The panel that says what this page cannot show *(mandatory, and the most valuable one)*
This card's subject is a **learned artefact**, and the app's model is untrained. Do not fake it, do
not hint at it, do not let a suggestive heatmap do the implying. Build the boundary as a panel with
three parts, all real:

- **Here is the effect on a trained model.** Static, sourced, labelled: Llama-2-13B `5158.07 → 5.40`;
  App. F's *"often exceeding half of the total attention"* on the first token from the 4096th token;
  Figure 2's split — local in layers 0–1, sink-dominated everywhere above.
- **Here is why this page cannot reproduce it.** The measured table from §Numbers item 1: position 0
  receives **0.069** (block 0) and **0.031** (block 1) against a uniform **0.083**, mean rank
  **5.15**. Caption: *a sink is a thing training builds — "initial tokens are more easily trained to
  serve as attention sinks" — and nothing here has been trained.*
- **Here is the negative result that follows.** The equal-budget KL table from item 5: at `B=8`,
  `0+8 → 1.332` and `4+4 → 3.963`; at `B=12`, `0+12 → 0.023` and `4+8 → 2.137`. No ordering. Caption:
  *on a model with no sink, keeping the first four tokens buys nothing and costs four slots of
  recency. That the paper's ablation shows the opposite is the evidence that the phenomenon is
  learned rather than architectural.*

An interaction that measures its own inapplicability is worth more than a fourth animated mask — and
concept 8's card already set this precedent with its staged-windows panel.

### 5. Build a sink by hand and watch the collapse appear
The one way to demonstrate the *causal* claim on this model: let the reader **impose** a sink rather
than hope for one. Add a slider "extra logit on token 0", feeding `bias(i, j) => j === 0 ? boost : 0`
through the seam that already exists. Show, live: the weight token 0 receives, the denominator share
it holds, and — with the boost held — the `0+w` versus `4+w` KL from item 5.
**The number that proves it:** on block 1 head 0's last query the current logit on token 0 is
**3.703**, the row's max is **4.757**, and **+1.756 nats** would give token 0 half the row [measured
here]. Push the slider past that and the model now has a sink; the eviction panel's collapse appears
and the `4+w` column starts beating `0+w`. Nothing was trained — a sink was *installed* — and the
card should say exactly that: this is a demonstration that the mechanism follows from the sink, not
evidence that the sink exists.

### 6. Softmax₁, in one checkbox
`SoftMax₁(x)_i = e^{x_i} / (1 + Σ e^{x_j})` is three characters in `ops.js` terms and the paper hands
you the equivalence: *"SoftMax₁ is equivalent to prepending a token with an all-zero Key and Value
features."* Show both framings on the same row — the extra `1` in the denominator, and a phantom key
at position −1 — and the resulting weights, which now sum to **less than one**. Then the paper's own
verdict, which is not what the folklore says: Zero Sink at `0+1024` is **29214**, *worse* than
vanilla's **27.87**, and at `4+1020` it is 18.01 against vanilla's 18.05 — a wash. *"While the zero
sink alleviates the attention sink problem to some extent, the model still relies on other initial
tokens as attention sinks."* One checkbox, one number, one deflated claim.

## What the source does *not* establish

- **It does not extend the context window, and the paper is explicit — eventually.** Appendix A
  (v2+): *"While StreamingLLM improves the efficiency of LLMs in streaming contexts, **it does not
  extend the models' context window or enhance their long-term memory capabilities.** … the model is
  limited to operating within the confines of its current cache. Consequently, StreamingLLM is **not
  suitable for tasks that demand long-term memory** and extensive data dependency, such as long
  document question-answering (QA) and summarization."* §2 says the same from the other end: *"We do
  not expand the attention window size of LLMs or enhance the model's memory and usage on long
  texts."* The proof is Appendix C's table of exact zeros. **"4 million tokens" means the model keeps
  producing sensible text for 4 million tokens, not that it can refer to any of them.** The sharp
  formulation for the card: *the cache is a window that slides forever; the memory is still the
  window.* And note the version history — the Introduction's own version of this sentence appears
  only in **v4**.
- **The distinction between length extrapolation and context extension is the paper's, and it is
  worth quoting because the timeline has both:** *"progress in one direction doesn't necessarily lead
  to progress in the other. For example, extending the context size of LLMs doesn't improve the
  model's performance beyond the context size, and neither approach ensures effective use of the long
  context."*
- **It does not explain why a sink forms, beyond an intuition.** The softmax-must-sum-to-one argument
  is a plausibility story, stated as *"We attribute the reason to"* and *"is intuitive"*. There is no
  derivation, no measurement of "unneeded" attention, no training-dynamics experiment showing a sink
  forming. The linebreak substitution (Table 1) is a genuine test — but it tests *semantic vs.
  positional*, not *why softmax produces a sink at all*. The 160M pre-training runs show a sink can be
  **relocated**, not why it exists.
- **Four sinks is a default, not a finding.** Table 2 shows `1` suffices for Falcon (12.12 at every
  column) and is within 0.2 for MPT and Pythia; only Llama-2-7B needs the extra three. The paper's own
  explanation for the number is a data-chunking accident (Llama-2's `<s>` applied before chunking).
  Anyone who reads "four" as a property of transformers has over-read it.
- **The pre-training sink is one model, one size, one dataset.** 160M parameters, Pythia codebase,
  deduplicated Pile, 143,000 steps, batch reduced to 256, 8×A6000. No scaling study. The
  recommendation *"we recommend training future LLMs with a sink token in all samples"* rests on that
  single pair of runs, plus a 7-benchmark zero-shot table where the differences (18.6→19.6, 45.2→45.6,
  50.1→50.8) are well inside noise for a 160M model.
- **Both remedies make the `0+y` case catastrophically worse** (Table 3: 27.87 → 29214 Zero Sink, →
  1235 Learnable Sink) and the paper does not mention it.
- **The 22.2× is against one baseline, on one GPU, in one framework.** *"Sliding window with
  re-computation"* on a single A6000 in Huggingface Transformers. It is **not** a speedup over dense
  attention, not over an optimised long-context serving stack, and not over FlashAttention-backed
  recomputation. Memory is explicitly *"consistent with the re-computation baseline"* — no memory win
  is claimed there at all.
- **On long-context benchmarks the published configuration loses** (App. D, five of six tasks) and
  ties only when the sink is inflated to 1750 tokens. That is in the paper and almost never cited.
- **Bigger caches are not reliably better** (Table 6; MPT-7B monotonically worse). The paper flags
  this as a limitation *of the models*, but it also means "increase the cache" is not a lever this
  method reliably offers.
- **Perplexity is not coherence.** Almost every quality number here is a PPL on PG19. The two
  non-perplexity results are ARC accuracy (Table 5) and StreamEval, both of which the paper is careful
  to describe as *"questions typically pertain to recent information"*. There is no measurement of
  whether a 4-million-token conversation stays *consistent*, only that it stays fluent.
- **Claims commonly attributed to this paper that are not in it:**
  - *"attention sinks are how models represent a null attention operation"* — the paper says the model
    *"tends to dump unnecessary attention values"*; it never characterises what the sink's value
    vector contributes to the output, and never ablates it.
  - *"StreamingLLM gives infinite context"* — see above; it gives infinite *generation*.
  - *"you should always keep 4 tokens"* — Table 2 says 1 suffices on 3 of 4 models; the learnable-sink
    result says 1 is right if you control pre-training.
  - *"the sink is always the BOS token"* — the paper says the opposite for Llama-2 (*"a mostly random
    token occupying the zeroth position"*), and in BERT it is `[SEP]`, not the first token (App. H).
  - *"it reduces memory"* — versus its own baseline it explicitly does not; versus *dense* attention
    it does, but so does plain window attention, which is the thing being fixed.
  - *"it's ~free"* — with RoPE you must cache keys un-rotated and re-apply the rotation to the whole
    cache every decoding step (§3.2). Small, but not nothing, and not mentioned in any summary.
- **The app cannot show the phenomenon, only the mechanism.** Untrained 32-dim weights produce no
  sink — measured: **0.069 / 0.031 received at position 0 against 0.083 uniform** — so eviction here
  removes an arbitrary chunk of the denominator rather than a dominant term, and the equal-budget KL
  table shows the recipe buying nothing. What the app *can* show exactly: the renormalisation identity,
  the cache-position renumbering, the cost/entry arithmetic, and — with an imposed bias — the collapse
  appearing once a sink is installed by hand. Read the arithmetic; never the predictions.

## Leaves behind

**Backward — what this answers.** Concept 8 left the timeline with a precise, unpaid debt, recorded in
its own note: *"Bounding the compute does not bound the cache … the reads-per-query number falls; the
bytes-per-token number does not move."* Window attention is the obvious way to pay it — throw the old
KV away and the cache stops growing — and this paper is the discovery that **the obvious way does not
work**, by a factor of 955 on Llama-2-13B, at the exact token where the first entry is evicted. It
also lands squarely on concept 13. ALiBi's whole promise was that a relative bias lets you test longer
than you trained; §2 here reports *"our tests on MPT models highlighted a breakdown when the text
length was vastly greater than the training length"*, and Table 2 gives the number: MPT-7B at `0+2048`
is **460.29**. ALiBi's bias is defined at any distance — concept 13's note is careful that "defined
beyond `L`" and "works beyond `L`" are different properties — and this paper shows the gap is not the
bias at all. It is the denominator, and it is fixed by four cached entries plus renumbering. The
correction to concept 8 is equally sharp: Longformer's global tokens are *chosen by a person for a
task* (`[CLS]`, question tokens) and App. B here dismisses the whole family — *"LongFormer, ETC, and
BigBird all rely on a global attention pattern, **which is unsuitable for autoregressive language
models**"*. StreamingLLM's four sinks look like global tokens and are not: nobody chose them, they
carry no meaning (four linebreaks work: 5.60 vs 5.40), and they were not designed in — they were
**found already there**, put there by training, and merely *not deleted*. That is the difference
between a designed pattern and an observed one, and it is this card's contribution to the timeline's
argument.

**Forward — two things this hands on.**

1. **The sink is a bug the architecture forces, and the fix belongs in the softmax.** The paper says
   so and then does not take its own advice: SoftMax₁ *"may also be effective"*, and the version they
   actually recommend is a learnable token — a workaround that reserves a cache slot forever rather
   than removing the constraint that created it. The constraint is one line: a softmax cannot output
   "nothing", so `Σ w = 1` even when the right answer is `0`. Every later attempt to give attention an
   escape valve — off-by-one softmax, learned per-head bias logits, registers in ViTs (App. B's own
   comparison), gated/selective attention — is chasing the same line. And notice that the pressures
   this timeline tracks meet here: the sink is *also* the quantisation-outlier problem (the paper
   cites Xiao et al. and Bondarenko et al. on exactly this), so an architectural wart shows up as a
   *systems* bill downstream. One mechanism, three pressures.

2. **The cache is now a policy, not a buffer — and this paper picks the dumbest policy that works.**
   Once you accept that the KV cache will be edited, the question becomes *which entries to keep*, and
   StreamingLLM's answer is a rule written from indices alone: the first four, plus the last `y`.
   Nothing is scored, nothing is learned, nothing looks at the content — the same hand-designed
   fixity concept 8's note flags, one level up the stack, now applied to *memory* instead of to
   *comparisons*. It works because the thing it protects is positional (Table 1's linebreaks prove
   it), and it fails exactly where content would have mattered (App. C's zeros, App. D's five losses).
   That gap is the opening for everything that scores entries by what they contain rather than where
   they sit — importance-based eviction, heavy hitters, learned compression of the evicted span — and
   for the other branch, which refuses the question entirely by not keeping a growing list at all.
   The timeline should read this card as the last word on **which** past to keep by rule, and the
   cards around it as the argument about whether "keep" is the right verb.
