# How attention got here

Every attention mechanism from ERA V5 Session 8, on one timeline, **ordered by the date each was
launched** rather than the order it was taught — so the page reads as the field changing its mind
rather than as a taxonomy. It opens on scaled dot-product attention and works forward; everything
after that is presented as a reply to a limitation of what came before.

**26 concepts. A real transformer runs in the page.** 32 dimensions, 4 heads, 2 blocks, seeded
weights, no dependencies, no build step. Every concept is that same model with one thing changed, so
the comparison between two mechanisms is fair: same weights, same sentence, same everything except
the rule being explained. Type your own sentence and every card recomputes.

## Run it

ES modules need to be served over http, so opening `index.html` from the filesystem will not work:

```bash
cd Session8/webapp && python3 -m http.server 8765
# then http://localhost:8765/index.html
```

| | |
|---|---|
| `?selfcheck=1` | runs all **85 assertions** and paints pass/fail at the foot of the page |
| `#<concept-id>` | deep-links a card — `#rope`, `#mla`, `#nsa`, `#drope` |
| `← →` | previous / next concept; the timeline slider is draggable and is itself the chronology |

## The model is untrained, and that is load-bearing

The weights are seeded noise. **Nothing in this app is a claim about language quality**, and the
next-word bars are structurally real and semantically meaningless. That is a deliberate choice, not a
shortcut: it means every number on a card is either

- **measured live** from the model in front of you — an identity, a count, a norm, a difference — or
- **quoted** from the primary source, labelled as quoted, with the conditions that produced it.

An untrained model is the right instrument for an exact claim (a degenerate setting reducing to the
baseline, an algebraic identity, an operation count) and the wrong one for a quality claim. Several
cards therefore report **negative results about their own subject** rather than faking a demonstration:

- `attention-sinks` measures that this model has no attention sink, and says so.
- `gated-deltanet` measures that its gate adds almost nothing on top of the delta rule at this scale,
  and explains why the paper's gain cannot appear here.
- `nsa` measures that it reads **more** than full attention on a 16-token sentence, and computes the
  1,638-token break-even that explains it.
- `dsa` measures that an untrained lightning indexer is indistinguishable from choosing at random —
  which is the honest form of a mechanism whose entire value is learned.
- `drope` reports that a position probe scores R² 1.000 in-sample and **negative** held-out, so the
  page can show position is present and used but not that it is decodable.

## The chronology, and its sources

Dates are where an agent sounds most confident and is most likely to be wrong, so every one of them
was checked against a primary source rather than recalled — and then **all 24 arXiv-backed dates were
re-verified in one query against the arXiv API** at the end of the build. The date used is the **v1
submission date**, not the latest revision, because the question the timeline asks is when a technique
first became public.

| # | date | mechanism | primary source | provenance | research note |
|---|---|---|---|---|---|
| 1 | `2017-06-12` | Scaled dot-product attention, multi-head | [Vaswani et al., Attention Is All You Need — arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | arXiv v1 | [note](docs/research/transformer.md) |
| 2 | `2017-06-12` | Sinusoidal position encoding | [Vaswani et al., Attention Is All You Need §3.5 — arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | arXiv v1 | [note](docs/research/sinusoidal.md) |
| 3 | `2018-03-06` | Relative position representations | [Shaw et al., Self-Attention with Relative Position Representations — arXiv:1803.02155](https://arxiv.org/abs/1803.02155) | arXiv v1 | [note](docs/research/relative-positions.md) |
| 4 | `2018-10-11` | Learned absolute position tables | [Devlin et al., BERT — arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | arXiv v1 | [note](docs/research/learned-absolute.md) |
| 5 | `2019-01-09` | Segment recurrence across contexts | [Dai et al., Transformer-XL — arXiv:1901.02860](https://arxiv.org/abs/1901.02860) | arXiv v1 | [note](docs/research/transformer-xl.md) |
| 6 | `2019-04-23` | Sparse Transformer, strided and fixed patterns | [Child et al., Generating Long Sequences with Sparse Transformers — arXiv:1904.10509](https://arxiv.org/abs/1904.10509) | arXiv v1 | [note](docs/research/sparse-transformer.md) |
| 7 | `2019-11-06` | Multi-query attention | [Shazeer, Fast Transformer Decoding — arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | arXiv v1 | [note](docs/research/mqa.md) |
| 8 | `2020-04-10` | Sliding window attention with global tokens | [Beltagy et al., Longformer — arXiv:2004.05150](https://arxiv.org/abs/2004.05150) | arXiv v1 | [note](docs/research/sliding-window.md) |
| 9 | `2020-06-29` | Linear attention, the kernel regrouping | [Katharopoulos et al., Transformers are RNNs — arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | arXiv v1 | [note](docs/research/linear-attention.md) |
| 10 | `2020-09-30` | Performer, FAVOR+ | [Choromanski et al., Rethinking Attention with Performers — arXiv:2009.14794](https://arxiv.org/abs/2009.14794) | arXiv v1 | [note](docs/research/performer.md) |
| 11 | `2021-02-22` | The delta rule, fast-weight programmers | [Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers — arXiv:2102.11174](https://arxiv.org/abs/2102.11174) | arXiv v1 | [note](docs/research/delta-rule.md) |
| 12 | `2021-04-20` | RoPE, rotary position embedding | [Su et al., RoFormer — arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | arXiv v1 | [note](docs/research/rope.md) |
| 13 | `2021-08-27` | ALiBi, attention with linear biases | [Press et al., Train Short, Test Long — arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | arXiv v1 | [note](docs/research/alibi.md) |
| 14 | `2022-05-27` | FlashAttention | [Dao et al., FlashAttention — arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | arXiv v1 | [note](docs/research/flashattention.md) |
| 15 | `2023-05-22` | Grouped-query attention | [Ainslie et al., GQA — arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | arXiv v1 | [note](docs/research/gqa.md) |
| 16 | `2023-06-27` | Position interpolation | [Chen et al., Extending Context Window via Position Interpolation — arXiv:2306.15595](https://arxiv.org/abs/2306.15595) | arXiv v1 | [note](docs/research/position-interpolation.md) |
| 17 | `2023-06-29` | NTK-aware base scaling | [bloc97, NTK-Aware Scaled RoPE on r/LocalLLaMA — no paper; written up later in YaRN](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | **no paper** | [note](docs/research/ntk-aware.md) |
| 18 | `2023-08-31` | YaRN | [Peng et al., YaRN — arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | arXiv v1 | [note](docs/research/yarn.md) |
| 19 | `2023-09-29` | Attention sinks, StreamingLLM | [Xiao et al., Efficient Streaming Language Models with Attention Sinks — arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | arXiv v1 | [note](docs/research/attention-sinks.md) |
| 20 | `2023-12-01` | Selective state space | [Gu and Dao, Mamba — arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | arXiv v1 | [note](docs/research/mamba.md) |
| 21 | `2024-05-07` | Multi-head latent attention | [DeepSeek-AI, DeepSeek-V2 — arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | arXiv v1 | [note](docs/research/mla.md) |
| 22 | `2024-06-10` | Parallelizable DeltaNet | [Yang et al., Parallelizing Linear Transformers with the Delta Rule over Sequence Length — arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | arXiv v1 | [note](docs/research/parallel-deltanet.md) |
| 23 | `2024-12-09` | Gated DeltaNet | [Yang et al., Gated Delta Networks — arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | arXiv v1 | [note](docs/research/gated-deltanet.md) |
| 24 | `2025-02-16` | Natively trainable sparse attention | [Yuan et al., Native Sparse Attention — arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | arXiv v1 | [note](docs/research/nsa.md) |
| 25 | `2025-09-29` | DeepSeek sparse attention | [DeepSeek-V3.2-Exp release — DeepSeek_V3_2.pdf shipped in the model repo, no arXiv v1; the later arXiv:2512.02556 covers V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) | **no paper** | [note](docs/research/dsa.md) |
| 26 | `2025-12-13` | DroPE | [Gelberg et al., Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings — arXiv:2512.12167](https://arxiv.org/abs/2512.12167) | arXiv v1 | [note](docs/research/drope.md) |

### Four dates that changed because they were checked

| concept | corrected to | how |
|---|---|---|
| `mqa` | **2019-11-06** | arXiv abstract page, during the date-verification pass |
| `parallel-deltanet` | **2024-06-10** | arXiv abstract page, same pass |
| `ntk-aware` | **2023-06-29** | the archived post's own metadata — the 30th is the date of the *follow-up* post |
| `drope` | **2025-12-13** | it had no public source and a placeholder sort key of `2026-01-01`; its research pass found a paper (arXiv:2512.12167) |

(The first three were wrong by a day before that pass. The exact wrong values were not kept, so they
are not reproduced here — what the record preserves is the correction and how it was made.)

### The two entries with no paper behind them

Both carry a badge on the card saying so, and the app marks an entry verified **only** when a paper
backs it.

- **`ntk-aware`** — a community post on r/LocalLLaMA by `bloc97`, never published as a paper. It is in
  the timeline because YaRN's own paper treats it as prior work and names it.
- **`dsa`** — DeepSeek Sparse Attention, introduced in the **DeepSeek-V3.2-Exp release** of
  2025-09-29. The primary source is the six-page `DeepSeek_V3_2.pdf` shipped inside the model
  repository; the date is verified against DeepSeek's own dated release note, which is also where the
  "API prices cut by 50%+" figure comes from. A full paper naming DSA as its first contribution
  appeared later — **arXiv:2512.02556, v1 2025-12-02** — but it is two months after the release and
  about a different model, so the timeline dates the release and cites the paper as further reading.

### One identity the sources do not establish

`drope` has **two records under one name** and the card presents both without merging them. The course
material records LightningLM V4's *"positional recalibration: DroPE, applied before annealing"* with
**8K trained → 256K reported, 32×**, and states plainly that it does not establish the algorithm. The
paper found during the research pass describes **removing RoPE from a pretrained model and briefly
recalibrating**. Same name, compatible description, and neither document mentions the other. The card
simulates neither mechanism — which is also the lesson's own instruction.

### What the ordering shows

Each entry carries the pressure it answers — the mechanism itself, where a token sits, what
generation must remember, how many comparisons, compressing the past, how it meets the hardware — and
read in date order those labels **circle rather than progress**. Position is worked on in 2017–2018,
dropped, returned to in 2021, again in 2023 across four consecutive cards, and once more at the very
end in 2025. Compressing the past appears in 2019, is abandoned for years, and comes back in force
from 2020 through 2024. Sparsity appears in 2019 and returns in 2025. The last card answers a
positional problem by **deleting the position scheme altogether**, which is the clearest single
illustration of the point: standard attention was not bad and then replaced by something better. It
solved a real problem and created new costs, and later work attacked different parts of those costs in
an order driven by what had become expensive at the time.

## The three questions, on every card

Each concept answers the assignment's three questions in two registers — a technical
`what it buys / what it gives up / when to choose it` block, and a plain-language `In plain words`
verdict with no formulas or jargon in it. **The self-check enforces that both exist and that neither
cost list is empty**, because a mechanism with five advantages and no meaningful downside means the
research was not finished.

## Architecture

```
index.html            the shell; no framework, no build, no dependencies
app/main.js           wiring: sentence input, timeline, deck, ?selfcheck
app/runner.js         one shared sentence and view state; the mounted card subscribes
app/deck.js           one concept in view; mount on arrival, unmount on leaving
app/timeline.js       the slider that is also the chronology — entries spaced by real date
app/model/            the transformer: ops, vocab, forward pass, position schemes, cost model
app/model/mixers.js   THE SEAM — every mechanism is a configuration of one of two mixers
app/cards/<id>.js     one file per concept: prose, live panels, trade-offs, verdict
app/data/mechanisms.js  the chronology; the deck, timeline, this README and the self-check read it
app/views/            shared visuals: attention grid, bars, curves, the dataflow panel
docs/research/<id>.md   the research note that had to be written before each card's code
```

**The seam is the point.** `app/model/mixers.js` stayed small on purpose: softmax attention
parameterised by a readability rule, a position bias and a rotation; and fixed-size state
parameterised by a write rule, a decay and a write strength. Adding a mechanism should mean
configuring that seam, not extending it. Where an extension was genuinely required it was one line,
and the research note said in advance what it had to be. Two mechanisms needed their own
implementation in their own card file — `parallel-deltanet`, because the whole point is that two
independent implementations agree, and `nsa`, because three separate attentions added together is not
a mask over one softmax. `drope` also carries two mixers, but they are **controls rather than
mechanisms**: attention with the causal mask removed, and attention with uninformative scores, both
there to make one claim falsifiable.

### How a concept was built

One at a time, never two: **research → build → verify → commit.** Research means fetching the primary
source and reading it, not recalling it, and writing `docs/research/<id>.md` **before any code**. Each
note records what was read and what could not be reached, the mechanism in precise terms with quoted
equations, the numbers that matter, the `[measured here]` figures obtained by driving this app's own
model from node, what the live view must let the reader do, and **what the source does not establish**.

## Verification

`app/lib/selfcheck.js` — **85 assertions, no framework**, visible in the page at `?selfcheck=1`:
model invariants; every mechanism's degenerate setting reducing exactly to the baseline; the cost
formulas reproducing the lesson's 6.44 GB and 51.54 GB; and the integrity of the chronology — unique
ids, parseable dates, nothing answering a limitation that had not happened yet, every card rendering
without throwing and carrying a complete trade-off record.

Beyond the suite, an end-to-end pass mounted **every card against three different sentences and drove
every control it exposes to both ends of its range** — 78 mount combinations, 1,506 control changes —
checking that nothing throws and that no `NaN`, `undefined` or `Infinity` reaches visible text; walked
the deck through all 26 concepts forwards and back, past both ends, and onto a shared fragment link;
confirmed every one of the 26 source URLs resolves; and re-verified every arXiv date against the API.

## What this app cannot show

Stated per card, and worth stating once here. It cannot show **quality**, because the weights are
untrained. It cannot show **wall-clock time**: FlashAttention, the chunkwise delta rule and NSA are
all claims about hardware — tiled SRAM, tensor cores, hand-written Triton kernels — so those cards
count operations, steps and bytes, which are exact, and say plainly that the step from "fewer
operations" to "less time" is the one a browser tab cannot take. It cannot show **training**, which is
the entire mechanism of `dsa` and `drope` and the central claim of `nsa`. And where a paper's evidence
is a figure with no table, no number is printed from it.

## Research notes

One per concept, in build order, each the spec for its card and the citation for this table:
[`docs/research/`](docs/research/).

## Submission

- **Live link** — not deployed yet; the repository's Pages source has to be set to GitHub Actions
  first.
- **Repository** — <https://github.com/millermuttu/ERA-V5/tree/master/Session8/webapp>
- **Sources for the chronology** — the table above, with a research note behind every row.
