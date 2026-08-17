# Concept 15 — Grouped-query attention

**Card id:** `gqa` · **Date:** 2023-05-22 (arXiv v1) · **Pressure:** what generation must remember

## What was read

- [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), Joshua Ainslie, James Lee-Thorp, Michiel de
  Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (Google Research) — *GQA: Training
  Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. Abstract page for the
  version history and abstract text.
- [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2305.13245) for §1 Introduction, §2 Method
  (§2.1 Uptraining, §2.2 Grouped-query attention), §3 Experiments (§3.1 setup, §3.2 main results,
  §3.3 ablations), Table 1, §4 Related Work, §5 Conclusion, Limitations, and **Appendix A Training
  Stability** — which is where the instability claim actually lives and which most summaries skip.
- Version history: **v1 22 May 2023**, v2 24 Oct 2023, v3 23 Dec 2023 (EMNLP 2023 camera-ready). The
  timeline uses v1. **I diffed the v1 and current ar5iv renderings across §1–§3 and the appendix: the
  Method, the Table 1 numbers, the ablation wording and the Training Stability appendix are
  identical.** Nothing quoted here is version-sensitive.
- The paper is short — five pages of body — and it is dense with claims that get flattened in
  retelling. Everything quoted below is from 2305.13245 unless labelled otherwise.

Two things were read outside the paper, both only to settle questions the paper leaves open, and both
labelled as external wherever they are used:

- **Hugging Face configs for the public T5.1.1 checkpoints** the paper says it uptrains from
  (`google/t5-v1_1-large`, `google/t5-v1_1-xxl`). The paper never states head counts, and the whole
  "why 8" question is meaningless without them. XXL: `d_model 4096`, `num_heads 64`, `d_kv 64`, 24
  decoder layers. Large: `d_model 1024`, `num_heads 16`, `d_kv 64`, 24 decoder layers.
- **[Llama 2](https://ar5iv.labs.arxiv.org/html/2307.09288), Appendix A.2.1** — read for exactly one
  reason: the "8 groups because a single node has 8 accelerators" argument is routinely attributed to
  *this* paper and is not in it. See the contradiction note below.

## The mechanism, precisely

### There are two contributions, and they are separable

The paper says so in a sentence that most summaries collapse into one idea. §1:

> *"This work contains two contributions for faster inference with large language models. First, we
> show that language model checkpoints with multi-head attention (MHA) can be uptrained (Komatsuzaki
> et al. 2022) to use MQA with a small fraction of original training compute. This presents a
> cost-effective method to obtain fast multi-query as well as high-quality MHA checkpoints."*

> *"Second, we propose grouped-query attention (GQA), an interpolation between multi-head and
> multi-query attention with single key and value heads per subgroup of query heads. We show that
> uptrained GQA achieves quality close to multi-head attention while being almost as fast as
> multi-query attention."*

The abstract numbers them explicitly: *"We (1) propose a recipe for uptraining existing multi-head
language model checkpoints into models with MQA using 5% of original pre-training compute, and (2)
introduce grouped-query attention (GQA)…"*

**Contribution (1) is about MQA, not GQA.** Read it carefully: the uptraining recipe is stated for
converting MHA → MQA. GQA is the *second*, independent idea, and the paper then applies the same
conversion machinery to it. You could accept uptraining and never use GQA (you would get a fast MQA
model from your existing checkpoint); you could accept GQA and never uptrain (you would train GQA from
scratch — which the paper never does, and lists as a limitation). The card should present them as two
levers, because that is what they are.

The motivation for (1) is a supply-chain fact, not an architecture fact. §1:

> *"while some language models already use multi-query attention, such as PaLM (Chowdhery et al.
> 2022), many do not, including publicly available language models such as T5 (Raffel et al. 2020)
> and LLaMA (Touvron et al. 2023)."*

and

> *"it may not be feasible to train separate models optimized for quality and inference."*

That is the real problem being solved: you already own a multi-head checkpoint that cost millions of
dollars, and you do not want to pay for it twice.

### The uptraining recipe, exactly as stated

§2.1 gives it in two steps and three sentences:

> *"Generating a multi-query model from a multi-head model takes place in two steps: first, converting
> the checkpoint, and second, additional pre-training to allow the model to adapt to its new
> structure."*

**Step 1 — the conversion.** The operation is on the *projection matrices*, not on the cached
activations:

> *"The projection matrices for key and value heads are mean pooled into single projection matrices,
> which we find works better than selecting a single key and value head or randomly initializing new
> key and value heads from scratch."*

Figure 1's caption repeats it: *"Key and value projection matrices from all heads are mean pooled into
a single head."*

For the grouped case, §2.2 states the same operation restricted to a group:

> *"When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value
> head by mean-pooling all the original heads within that group."*

So: for a group `g` containing query heads `{h₀ … h_{m−1}}`, the group's key projection is the
elementwise mean of those `m` heads' key projections, and likewise for value. Nothing is learned at
conversion time; it is an average of existing weights. Because the projection is linear, mean-pooling
the projection matrices and mean-pooling the resulting key vectors are the same thing — which is why
this is implementable in the app at the activation level and still faithful to the paper.

**Step 2 — the uptraining.** §2.1:

> *"The converted checkpoint is then pre-trained for a small proportion α of its original training
> steps on the same pre-training recipe."*

§3.1 pins down what "same recipe" means and what α cost:

> *"Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are
> mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion
> of original pre-training steps with the original pre-training setup and dataset from (Raffel et al.
> 2020). For α = 0.05, training took approximately 600 TPUv3 chip-days."*

**α = 0.05.** Five percent of the original pre-training *steps*, same data, same optimiser (Adafactor,
*"the same hyperparameters and learning rate schedule as T5"*), and the absolute cost is stated: ~600
TPUv3 chip-days for XXL. That last number is the one that makes "5%" concrete — it is not free, it is
merely twenty times cheaper than starting over.

### GQA itself

§2.2, the definitional paragraph, in full:

> *"Grouped-query attention divides query heads into G groups, each of which shares a single key head
> and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and
> therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number
> of heads, is equivalent to MHA."*

That is the whole framing, and it is a *parameterisation*, not a new operation: MHA and MQA are the
two endpoints of one axis, and everything the paper does is choose an interior point on it. Figure 2's
caption states the same in words: *"Multi-head attention has H query, key, and value heads. Multi-query
attention shares single key and value heads across all query heads. Grouped-query attention instead
shares single key and value heads for each group of query heads, interpolating between multi-head and
multi-query attention."*

Note what does **not** change: the number of query heads. Every query head keeps its own `Q`
projection and computes its own attention row. `G` only counts distinct key/value heads.

### Why the middle is not just a compromise — the scaling argument

This is the part worth getting right, because it is an argument about *what happens as models grow*,
not about averaging two options. §2.2, four claims in sequence:

> *"An intermediate number of groups leads to an interpolated model that is higher quality than MQA but
> faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA
> reduces H key and value heads to a single key and value head, reducing the size of the key-value
> cache and therefore amount of data that needs to be loaded by a factor of H. **However, larger models
> generally scale the number of heads, such that multi-query attention represents a more aggressive
> cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in
> bandwidth and capacity as model size increases.**"*

This is the answer to "why does MQA degrade more at larger scale", and it is a *definitional* answer
rather than an empirical one: MQA is not a fixed intervention. `H → 1` is an `H`-fold cut, and `H`
grows with model size, so the same named mechanism is a harsher amputation on a bigger model. GQA-8 is
a fixed 8-way reduction whatever the model size; MQA is a 16× cut on T5-Large and a 64× cut on T5-XXL
(head counts from the public configs — external, see above). **The paper argues this; it does not
measure it.** There is no scale sweep, no MQA-Large-vs-MQA-XXL degradation curve. Two model sizes are
trained and only the XXL one is converted.

Then the second-order term, which cuts the other way:

> *"Moreover, larger models suffer relatively less from memory bandwidth overhead from attention, as
> the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of
> model dimension."*

So at large scale the *penalty* for keeping more key/value heads shrinks (attention's share of the
bandwidth bill falls as `d` vs `d²`) while the *damage* from collapsing to one grows. Both terms point
the same way: the middle gets more attractive as models get bigger. That is the paper's actual thesis
about scale, and it is stated as reasoning from the shape of the cost, not from a measurement.

And then the sentence this card exists to rescue:

> *"Finally, standard sharding for large models replicates the single key and value head by the number
> of model partitions (Pope et al. 2022); GQA removes the waste from such partitioning. Therefore, we
> expect GQA to present a particularly good trade-off for larger models."*

**Read that carefully.** When you serve a large model with tensor parallelism across `P` chips, the
attention heads are split across chips. With MHA you can give each chip its own subset of key/value
heads. With MQA there is exactly *one* key/value head and `P` chips that all need it, so it gets
**replicated on every chip** — and the KV memory you thought you saved comes back `P`-fold. GQA with
`G ≥ P` restores a clean partition: one or more key/value heads per chip, no replication, no waste.
This is the argument that makes 8 a *systems* number rather than a taste number — but see the
contradiction note: **the paper makes this argument for GQA in general and never explicitly ties it to
the choice of 8.**

### Where GQA is not applied

Two scoping statements that matter for anyone reading Table 1:

> *"We note that GQA is not applied to the encoder self-attention layers; encoder representations are
> computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck."*

and §3.1: *"We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder
self-attention."*

These are encoder-decoder T5 models. The mechanism is applied to the two attention types that run one
token at a time. This is the same observation Shazeer made in 2019 (his encoder time barely moved),
now used as a design rule.

### Training instability — what the paper actually did

§1 asserts it in passing: *"multi-query attention (MQA) can lead to quality degradation and training
instability"*. The evidence is in **Appendix A**, and it is worth quoting entire because its epistemic
status is unusual:

> *"We find that multi-query attention can lead to training instability during fine-tuning, in
> particular combined with long input tasks. We trained multiple T5-Large models with multi-query
> attention from scratch. In each case, pre-training suffered from frequent loss spikes and the final
> models diverged immediately when fine-tuning on long-input tasks. Uptrained multi-query attention
> models are more stable but still display high variance, so for multi-query models on unstable tasks
> we report average performance over three fine-tuning runs. Uptrained grouped-query attention models,
> however, appear to be stable, so we did not investigate futher on the root causes of multi-query
> instability."*

(The typo *"futher"* is in the source.) So instability is **observed and acted upon, but not
quantified**: no loss curves, no spike counts, no divergence rates, "multiple" models with no number,
and an explicit statement that they stopped investigating. It also changed the experimental protocol —
MQA numbers on unstable tasks in Table 1 are three-run averages while the others are not, which is a
methodological asymmetry the paper discloses in an appendix and nowhere else. And note the shape of
the finding: **from-scratch MQA was the unstable case; uptrained MQA was better; uptrained GQA was
fine.** That is a third, quiet argument for uptraining as a path.

## Numbers that matter

### Table 1 — the headline comparison, complete

Inference time is *"time per sample per TPUv4 chip, as measured by xprof"*, on *"8 TPUs with the
largest batch size that fits up to 32 per TPU, and parallelization optimized separately for each
model."* Averages are over the seven dev sets. R1 = ROUGE-1, WMT = BLEU, TriviaQA = F1. The two XXL
non-MHA rows are 5%-uptrained.

| Model | T_infer (s) | Average | CNN R1 | arXiv R1 | PubMed R1 | MediaSum R1 | MultiNews R1 | WMT BLEU | TriviaQA F1 |
|---|---|---|---|---|---|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

**The claim, in real figures.** "Quality close to MHA at speed close to MQA" cashes out as:

- **Quality gap to MHA-XXL: 0.1 average points** (47.1 vs 47.2), against MQA's 0.6 (46.6 vs 47.2).
  GQA-8 recovers **0.5 of the 0.6-point MQA→MHA gap — 83% of it.**
- **Speed gap to MQA: 0.04 s per sample** (0.28 vs 0.24), i.e. GQA-8 is **17% slower than MQA** while
  MHA-XXL is **529% slower** (1.51 vs 0.24).
- **GQA-8 vs MHA-XXL: 5.39× faster** (1.51 / 0.28) for 0.1 average points.
- Per task, "close to" hides two outright wins: GQA-8-XXL **beats** MHA-XXL on PubMed (**47.7 vs
  47.5**) and MultiNews (**47.2 vs 46.9**), ties WMT (28.4), and loses the other four by 0.1–0.3
  (CNN 43.5/43.8, arXiv 45.4/45.6, MediaSum 36.3/36.4, TriviaQA 81.6/81.9). The average is not a
  uniform small deficit; it is a mixed bag whose mean lands at −0.1.
- The MQA anomaly from Shazeer's own table repeats here: **MQA-XXL posts the best WMT BLEU of all four
  models (28.5)** while being worst-or-near-worst everywhere else. Same warning as concept 7 — a single
  translation score moving the other way is noise, not evidence.

**The other comparison in Table 1, which is contribution (1) standing alone.** MQA-XXL vs MHA-Large:
**46.6 at 0.24 s versus 46.0 at 0.37 s.** The uptrained multi-query XXL model is *both* better *and*
1.54× faster than a genuinely smaller multi-head model. §3.2:

> *"We see that a larger uptrained MQA model provides a favorable trade-off relative to MHA models,
> with higher quality and faster inference than MHA-Large. Moreover, GQA achieves significant
> additional quality gains, achieving performance close to MHA-XXL with speed close to MQA."*

That MQA-XXL-vs-MHA-Large row is the argument for uptraining *as such*: the alternative to converting
your XXL checkpoint is serving a smaller model, and converting wins on both axes at once.

### The α ablation — how much uptraining was needed

§3.3, in full (Figure 5's underlying claim; the figure's numeric values are not in the text):

> *"Figure 5 shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA.
> First, we note that GQA already achieves reasonable performance after conversion while MQA requires
> uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from
> 10%."*

Three separate facts, all worth keeping:

1. **α = 0 is already usable for GQA.** Mean-pooling a checkpoint into 8 groups and doing *nothing
   else* gives *"reasonable performance"*. This is the strongest possible statement of how mild the
   grouped conversion is.
2. **α = 0 is not usable for MQA** — it *"requires uptraining to be useful"*. The full collapse breaks
   the model until it is repaired.
3. **The knee is at 5%, flat by 10%.** α = 0.05 is where the paper's headline models sit, and doubling
   it buys little. Cost anchor: α = 0.05 ≈ **600 TPUv3 chip-days** for XXL.

**Point 1 and point 2 together are the real relationship between the two contributions.** Uptraining
is what makes MQA viable from a checkpoint; GQA is what makes uptraining nearly unnecessary. The
better the interior point, the less repair the conversion needs.

### The checkpoint-conversion ablation

§3.3, in full:

> *"Figure 4 compares the performance of different methods for checkpoint conversion. Mean pooling
> appears to work best, followed by selecting a single head and then random initialization.
> Intuitively, results are ordered by the degree to which information is preserved from the
> pre-trained model."*

Figure 4's caption states the setting exactly: *"Performance comparison of different checkpoint
conversion methods for T5-Large uptrained to MQA with proportion α = 0.05. 'Mean' mean-pools key and
value heads, 'First' selects the first head and 'Random' initializes heads from scratch."*

What is and is not established here:

- The comparison is **three-way and strictly ordered**: mean > first > random.
- It is on **T5-Large → MQA** (the full collapse, `G = 1`), **not** on GQA and **not** on XXL.
- It is measured **after α = 0.05 uptraining**, on CNN/Daily Mail, MultiNews and TriviaQA (§3.3's
  *"representive subsample"*; the typo is the paper's). So it is not "which conversion is closest to
  the original" — it is "which conversion recovers best after repair". Those are different questions
  and the paper only answers the second.
- **The margins are in a figure and nowhere in the text.** I cannot report how much better mean is
  than first, and neither can any honest summary. The word used is *"appears"*.
- The stated mechanism is a hypothesis, flagged as one by *"Intuitively"*: the ranking tracks *"the
  degree to which information is preserved from the pre-trained model"*.

### The number of groups, and why 8

§3.3, in full:

> *"Figure 6 demonstrates the effect of the number of GQA groups on inference speed. For larger models
> the memory bandwidth overhead from the KV cache is less constraining (Shazeer 2019), while the
> reduction in key-value size is sharper due to the increased number of heads. As a result, increasing
> the number of groups from MQA only results in modest slowdowns initially, with increasing cost as we
> move closer to MHA. **We selected 8 groups as a favorable middle ground.**"*

Figure 6's caption: *"Time per sample for GQA-XXL as a function of the number of GQA groups with input
length 2048 and output length 512. Going from 1 (MQA) to 8 groups adds modest inference overhead, with
increasing cost to adding more groups."*

**The shape of the curve is the argument.** Time as a function of `G` is *flat near MQA and steep near
MHA* — so the cheapest place to buy quality back is immediately above `G = 1`. You get most of the
quality for almost none of the time. Eight is chosen from that curve, at input length 2048 / output
length 512, and the only stated justification is *"a favorable middle ground"*. Again: **the figure's
values are not in the text.** The one anchor that is: Table 1's 0.24 → 0.28 s, the MQA→GQA-8 step,
+17%.

**The partitioning argument, and who actually made it.** The systems reason for 8 that everyone
repeats — "8 groups because you serve on one node of 8 accelerators, so each gets its own KV head" —
**is not stated in this paper.** What this paper states is the general sentence in §2.2: *"standard
sharding for large models replicates the single key and value head by the number of model partitions
(Pope et al. 2022); GQA removes the waste from such partitioning."* That is the same physics, applied
to GQA-vs-MQA generically, never to the number 8. Two circumstantial facts sit next to it: the timing
rig is *"8 TPUs"* (§3.1), and 8 divides T5-XXL's 64 heads evenly. Neither is the paper drawing the
connection.

The explicit version is **Llama 2, Appendix A.2.1** (external, arXiv:2307.09288, July 2023):

> *"To optimize for latency, we host our largest models using 8 A100s in a single node with tensor
> parallelism (Shoeybi et al. 2019)."*

> *"In this setting, sharding for MQA cannot be done across heads anymore, given the number of heads is
> lower than the number of GPUs. Either you duplicate the KV values in all GPUs (making the KV cache
> size equal to GQA), or an alternative is to shard across the batch dimension instead (Pope et al.
> 2022). The latter, however, can complicate an inference service, as it works only when batch sizes
> are larger than the number of shards and the additional communication cost is not worth it in all
> cases."*

> *"Therefore, based on the ablation results and ease of scaling inference, for the 34B and 70B Llama 2
> models we chose to use GQA instead of MQA."*

That is the "single node" argument, made two months later by a different team, and it contains the
punchline: **duplicating the single MQA head across 8 GPUs makes its KV cache exactly the size of
GQA-8's anyway.** MQA's advantage over GQA-8 evaporates at the node boundary — you pay GQA-8's memory
and get MQA's quality. Llama 2 confirms it empirically: *"In these runs we simply duplicated the KV
heads for MQA in all GPUs, so the KV cache size for MQA became equal to the GQA and the two variants
behaved very similar."* **This is the strongest available answer to "why 8" and the card must
attribute it to Llama 2, not to Ainslie et al.**

### Scale numbers the paper implies but never prints

The paper never states head counts, which makes the reduction factors invisible. From the public
T5.1.1 configs it uptrains from (external):

| | heads | d_kv | decoder layers | MQA cut | GQA-8 cut |
|---|---|---|---|---|---|
| T5.1.1-Large | 16 | 64 | 24 | 16× | 2× |
| T5.1.1-XXL | 64 | 64 | 24 | 64× | 8× |

So **GQA-8-XXL keeps 8× more key/value state than MQA-XXL and 8× less than MHA-XXL** — the reduction
is 8-fold, not 64-fold. That number is the one to put next to "close to MQA speed": GQA-8 gives up 7/8
of MQA's cache saving and loses only 17% of its speed advantage, because by that point the bandwidth
bottleneck has largely been paid off. It also makes §2.2's scaling sentence concrete: the *same name*
"MQA" means a 16× cut at Large and a 64× cut at XXL, while "GQA-8" means 2× and 8×.

### What this app already computes, measured

All figures below were computed by running the app's own `forward()` and `cost.js` (32 dims, 4 heads,
`d_k = 8`, 2 blocks, seed 20260817) on the default 16-token sentence *"the lighthouse keeper wrote the
code in a notebook and hid it under the third stair"*. They are properties of the *conversion
geometry*, which is untrained-safe; none of them is a quality claim.

**Fidelity of the converted key/value heads to the originals.** For each of the 4 heads × 2 blocks ×
16 tokens, compare the vector the head actually reads after conversion against the vector it read
under MHA. `cos` is the mean cosine; `rel` is the mean relative L2 error `‖pooled − original‖ /
‖original‖`; `TV` is total-variation distance between the model's next-token distribution and the MHA
baseline's.

| conversion | key cos | key rel | value cos | value rel | TV from MHA | top word |
|---|---|---|---|---|---|---|
| G = 4 (MHA, identity) | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | her |
| G = 2, mean-pool | **0.5286** | 0.8457 | 0.5616 | 0.8430 | **0.5222** | block |
| G = 2, select first | 0.3987 | 1.0203 | 0.3611 | 1.0098 | 0.6831 | block |
| G = 1, mean-pool | **0.3227** | 0.9948 | 0.3338 | 0.9685 | **0.6330** | memory |
| G = 1, select first | 0.1576 | 1.2989 | 0.2193 | 1.2566 | 0.7139 | context |
| G = 1, random init | 0.0046 | 1.5026 | — | — | — | — |

Mean-pooling wins on every metric at every `G`, and the ordering **mean > select > random** reproduces
the paper's ordering exactly. Random init was measured separately by drawing a fresh head at the same
scale as the originals (RMS 2.043): mean cosine **0.0046**, mean |cosine| 0.3103, relative error
**1.5026** ≈ √2, which is what "no information preserved" looks like numerically.

**Honesty about what that table proves.** The mean is, by construction, the minimiser of total squared
distance to the vectors it replaces — so "mean-pooling stays closest to the originals" is a theorem,
not an experiment, and the app cannot pretend otherwise. What the app *does* add is the **magnitude**:
how much is lost, and how the loss grows as `G` falls. The paper's actual finding is the step our app
cannot take — that staying closest also *finishes* best after uptraining.

**How aligned the heads were before pooling** — the quantity that determines how much pooling costs.
Mean cosine between original key vectors of heads inside the same group, block 0 + block 1:

- `G = 2` (2 heads per group): **−0.0363**
- `G = 1` (4 heads per group): **−0.0559**

These heads are essentially orthogonal, which is the *worst* case for averaging: pooling `m` mutually
orthogonal unit-norm vectors gives a mean whose cosine to each is `1/√m` — **0.7071 at m = 2, 0.5000 at
m = 4.** Measured: 0.5286 and 0.3227, below the ideal because the head norms differ. This is the honest
frame for an untrained toy: the app shows the *pessimistic* end of mean-pooling, and the paper's
comparatively mild real-world damage is consistent with trained heads inside a group being more alike
than these are.

**The head-key cosine matrix, block 0, mean over 16 tokens** — the picture that is GQA-specific:

```
G = 4                       G = 2                       G = 1
1.000  0.014  0.008 -0.103  1.000  1.000 -0.038 -0.038  1.000 1.000 1.000 1.000
0.014  1.000  0.009  0.040  1.000  1.000 -0.038 -0.038  1.000 1.000 1.000 1.000
0.008  0.009  1.000  0.074 -0.038 -0.038  1.000  1.000  1.000 1.000 1.000 1.000
-0.103 0.040  0.074  1.000 -0.038 -0.038  1.000  1.000  1.000 1.000 1.000 1.000
```

At `G = 2` the matrix is **block-diagonal**: 1.000 inside a group, ≈ 0 across. That is grouping, made
visible in one image — and it is a different picture from concept 7's, which reports a single scalar
going to 1.0000.

**Cost, unchanged from concept 7 but needed here for the slope argument.** Toy, per forward pass:
cache bytes **4096 → 2048 → 1024** at `G = 4 → 2 → 1`; mixing multiplies **4352 at every setting**;
reads per query 8.500 at every setting. Serving scale (`cacheBytes` at 48 layers, head dim 128, 32768
tokens, bf16, batch 1) as a function of key/value heads:

| kv heads | per user | × batch 64 |
|---|---|---|
| 64 (MHA) | 51.540 GB | 3298.5 GB |
| 32 | 25.770 GB | 1649.3 GB |
| 16 | 12.885 GB | 824.6 GB |
| **8 (GQA-8)** | **6.442 GB** | 412.3 GB |
| 4 | 3.221 GB | 206.2 GB |
| 2 | 1.611 GB | 103.1 GB |
| 1 (MQA) | 0.805 GB | 51.5 GB |

**The lesson's own 6.44 GB configuration is a GQA-8 configuration.** That is worth saying out loud on
the card: the serving number the whole app has been quoting since concept 7 already assumes this
paper's answer. The MHA version of that same model is 51.5 GB.

**Attention-row agreement with MHA, last token, all heads and blocks** (cosine between the attention
weight rows; argmax agreement = fraction of heads still attending most to the same token):

| | row cos vs MHA | argmax agreement |
|---|---|---|
| G = 2, mean | 0.6744 | 50.0% |
| G = 2, select | 0.6180 | 50.0% |
| G = 1, mean | 0.6231 | 37.5% |
| G = 1, select | 0.3008 | 25.0% |

Mean-pooling preserves *where the heads look* better than selecting does, and the gap widens at `G = 1`
(0.6231 vs 0.3008). This is the closest the app gets to a behavioural, rather than geometric,
demonstration of the conversion ablation.

## What the live view must let the reader do

The seam is `kvGroups` plus the already-implemented `kvPool` option (`"mean"` default, `"select"`), in
`app/model/transformer.js` lines 53–72 — which means **the paper's checkpoint-conversion ablation is
already wired into this app and nothing has demonstrated it yet.** Everything below moves a real
computed quantity. Nothing below reports a quality result.

**What this card must not do.** Concept 7 already owns: the `kvGroups` slider as a collapse story, the
head-key cosine reaching 1.0000, arithmetic staying constant while cache falls, Shazeer's memory-to-
arithmetic ratio, and the serving cache bar chart. Every one of those is off-limits as the *point* of
an interaction here, though the cache numbers may be reused as background. This card's three
distinctive subjects are: **the middle of the axis**, **uptraining by mean-pooling**, and **why 8**.

1. **The conversion bench: mean vs select vs original, as a measured distance.** The one interaction
   this card exists for. Controls: `G ∈ {4, 2, 1}` and a three-way conversion toggle
   (**mean-pool / select first head / random**). Display: for each of the 4 heads, the original key
   vector and the converted key vector as paired strips, plus one big readout — **mean cosine to the
   original** and **mean relative error**. It reads, at `G = 2`: **0.5286 / 0.8457** for mean-pool and
   **0.3987 / 1.0203** for select. At `G = 1`: **0.3227 / 0.9948** vs **0.1576 / 1.2989**, with random
   at **0.0046 / 1.5026**. *The number that proves the point is the ordering — mean > select > random
   on both metrics at both settings, the same ordering the paper reports for downstream quality.*
   Caption must carry the caveat: the mean is the least-squares centre of the group by definition, so
   this shows what the conversion **costs**, not that mean-pooling **wins**; the paper's Figure 4 is
   what shows the second thing, and it measures quality after 5% uptraining, which this page cannot
   do. `random` is not currently implemented in the seam — it is three lines, and the card is not
   complete without the third bar.

2. **The head-key cosine matrix, as a 4×4 grid that goes block-diagonal.** Same `G` control. Display:
   the 4×4 matrix of mean cross-head key cosines, drawn as a heatmap with the numbers printed. It
   reads: near-identity at `G = 4` (off-diagonals 0.014, 0.008, −0.103, 0.009, 0.040, 0.074), a clean
   **two-by-two block structure at `G = 2` (1.000 inside a group, −0.038 across)**, and all-ones at
   `G = 1`. *The number that proves the point is the −0.038 sitting next to the 1.000 in the same
   matrix:* at the interior point, some heads share and some do not, which is precisely what "groups"
   means and precisely what a single scalar cosine cannot show. This is the visual that separates this
   card from concept 7's.

3. **The predictor: how much pooling costs depends on how aligned the group already was.** Display two
   numbers side by side for the current `G`: **within-group cosine before pooling** (−0.0363 at
   `G = 2`, −0.0559 at `G = 1`) and **cosine retained after pooling** (0.5286, 0.3227), with the
   idealised orthogonal-case prediction `1/√m` printed alongside (**0.7071**, **0.5000**). *The number
   that proves the point is the pair (−0.036, 0.529): these heads are near-orthogonal, so averaging
   two of them keeps only about half of each.* The honest reading, which the card should state plainly:
   this is the worst case, our weights are untrained and therefore unaligned; the paper converts
   trained checkpoints whose heads have had reason to become correlated, and the paper reports that a
   grouped conversion is *"already… reasonable performance after conversion"*. The app shows the floor
   of that argument, not the paper's result.

4. **The slope panel: what the middle buys at serving scale.** The toy has 4 heads, so the interior of
   the axis is a single point (`G = 2`) — the group-count *curve* only exists at real scale, which is
   where 8 lives. Display: `cacheBytes` from the existing cost model at 48 layers / head dim 128 /
   32768 tokens / bf16, with a **kv-heads control stepping 64 → 32 → 16 → 8 → 4 → 2 → 1**, drawn as
   cache-vs-context-length **lines** rather than bars (bars are concept 7's). It reads **51.540 GB
   (MHA-64) → 6.442 GB (GQA-8) → 0.805 GB (MQA)**. *Two numbers prove the point.* First: GQA-8 gives
   up 7/8 of MQA's saving and Table 1 says it costs **0.04 s per sample (0.28 vs 0.24)** — the
   quality-per-second is bought on the flat part of the curve. Second, and this is the forward link
   made arithmetic: **the line still rises.** GQA-8 reaches the same 6.442 GB at 32768 tokens that
   MHA-64 reaches at 4096 and MQA reaches at 262144. Grouping divides the slope by 8; it does not
   change the fact that it is a line through the origin.

5. **The uptraining panel — two levers, not one.** A small non-interactive-first, then-interactive
   panel that states the paper's actual recipe as a pipeline: *existing MHA checkpoint → mean-pool the
   key/value projections → pre-train for α of the original steps → serve*. Display the paper's three
   anchors as a row of figures: **α = 0.05**, **≈ 600 TPUv3 chip-days**, **knee at 5%, flat by 10%**.
   Then one live element: the `α = 0` column of the app's own conversion bench (interaction 1),
   labelled *"this is what the conversion alone does — the paper's α = 0"*, next to the paper's two
   sentences: *"GQA already achieves reasonable performance after conversion while MQA requires
   uptraining to be useful."* *The number that proves the point is the TV distance from the MHA output:
   **0.5222 at `G = 2` versus 0.6330 at `G = 1`, both with mean-pooling*** — converting further from
   MHA moves the model further from where it started, in the same direction the paper's α = 0 finding
   points, and the card may say exactly that much and no more. The panel's job is to plant the idea
   most readers arrive without: **the conversion is a weight-space edit to a model you already own,
   and it is cheap.**

**Banner the card must carry, not bury.** The model on this page is untrained. `kvGroups` and `kvPool`
visibly change its output (TV from the MHA baseline: 0.5222 at `G = 2` mean-pool, up to 0.7139 at
`G = 1` select) — that proves the switches are real and nothing more. The card may say *what the
conversion does to the vectors*; it may never say GQA is better, worse, or "close to" MHA on this page.
Every quality claim on this card is Table 1's, quoted and attributed.

## What the source does *not* establish

- **It does not establish that GQA beats MHA, or that the gap is negligible.** The gap is **0.1 average
  points**, one number, on seven dev sets, on one model at one size, with a decode-time difference of
  1.23 s per sample in GQA's favour. The paper's own Limitations section says the evaluation is shaky:
  *"For summarization we employ Rouge score, which we know is a flawed evaluation that does not tell
  the whole story; for that reason, it is difficult to be certain our trade-offs are correct."*
- **It does not establish that uptraining matches training from scratch.** Stated outright:
  *"Due to limited computation, we also do not compare our XXL GQA model to a comparitive model
  trained from scratch, so we do not know the relative performance of uptraining vs training from
  scratch."* (typo the paper's). Every "GQA is as good as MHA" claim on the internet rests on an
  uptrained model whose from-scratch counterpart was never built.
- **The "why 8" systems argument is not in this paper.** The paper's stated reason is *"We selected 8
  groups as a favorable middle ground"*, read off an inference-speed curve. The partitioning sentence
  in §2.2 is generic to GQA and is never applied to the number 8. The single-node version is **Llama 2,
  Appendix A.2.1, July 2023** — and it is the better argument. Attribute it correctly.
- **The scale claim is argued, not measured.** *"larger models generally scale the number of heads,
  such that multi-query attention represents a more aggressive cut"* is reasoning about `H`, not a
  measured degradation curve across sizes. No MQA-Large-vs-MQA-XXL quality comparison exists in the
  paper. The card may present it as the paper's argument; it may not present it as a finding.
- **Instability is reported, not quantified.** Appendix A gives frequency-free, count-free
  observations (*"multiple T5-Large models"*, *"frequent loss spikes"*, *"diverged immediately"*), and
  ends by declining to look further: *"we did not investigate futher on the root causes of multi-query
  instability."* It is real enough that it changed the protocol — MQA numbers on unstable tasks are
  three-run averages — but there is no measurement to cite. Say "observed", never "showed".
- **Three of the paper's five results are figures whose values are not in the text.** Figure 4 (mean vs
  first vs random), Figure 5 (α sweep), Figure 6 (time vs group count). All that can be quoted from
  them is ordering and shape: *"appears to work best"*, *"diminishing returns from 10%"*, *"modest
  slowdowns initially, with increasing cost"*. **Any specific number attributed to those figures in a
  summary is invented.** The only numeric anchors for the group-count curve are Table 1's 0.24 s and
  0.28 s.
- **It is an encoder-decoder result, and the paper says the decoder-only case should be different —
  better.** *"we evaluate the impact of uptraining and GQA only on encoder-decoder models. Recently,
  decoder-only models are extremely popular, and since these models do not have separate self-attention
  and cross-attention, we expect GQA to have a stronger advantage over MQA."* Note the direction of the
  caveat: the paper expects its own result to *understate* the benefit on the architecture everyone
  actually uses now, including this app's toy — a prediction, not a measurement.
- **The timings are one hardware configuration.** Per sample per TPUv4 chip, 8 TPUs, batch up to 32 per
  TPU, *"parallelization optimized separately for each model"* — that last clause means the four rows
  are not running the same parallelisation, which is fair to each model and makes the ratios
  hardware-and-tuning specific. The 5.39× does not transfer.
- **GQA was not solely theirs.** §4: *"Rabe 2023 independently developed GQA with public
  implementation."* The paper cites a flaxformer source file. Priority here is genuinely shared, and
  the paper says so.
- **The app's own outputs establish nothing about quality.** Seeded noise weights, no training, 32
  dims, 4 heads. `G = 2` is the only interior point that exists on this page, and the paper's headline
  configuration (8 groups out of 64 heads) is not representable in the toy at all.

## Leaves behind

**Backward.** Concept 7 ended on a residue it named precisely: multi-query took the key/value cache
down by a factor of `h` and paid a measured quality cost — +0.015 ln(PPL) on WMT14 dev, +0.3
perplexity on Billion-Word — and never asked what happens between `h` and `1`. **This is the paper
that asks.** It opens by quoting exactly that problem — *"MQA can lead to quality degradation and
training instability"* — and its answer is not a new operation but a parameter on the old one:
*"GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while
GQA-H, with groups equal to number of heads, is equivalent to MHA."* Concept 7's slider was already
this axis; this concept is the discovery that the *interior* of it is where you want to live, and that
the good part is right next to the MQA end — *"increasing the number of groups from MQA only results
in modest slowdowns initially"*.

It also brings something concept 7 did not have and that the app's timeline has not yet shown anywhere:
**an architecture change you can apply to a model that already exists.** Every mechanism on this page
so far has been a decision made before training. Mean-pooling the key/value projections of a finished
checkpoint and repairing it with 5% of the original compute is a different kind of move — retrofit
rather than design — and it is the reason GQA spread as fast as it did. Llama 2 shipped with it two
months after this preprint (external: arXiv:2307.09288, §A.2.1, *"for the 34B and 70B Llama 2 models we
chose to use GQA instead of MQA"*).

**One correction to carry back.** Concept 7's note says of GQA: *"Note the GQA abstract says nothing
about training stability either — if the card mentions instability at all, it belongs to neither of
these two sources."* The abstract does not, but the **body does** — §1 asserts it and **Appendix A**
reports from-scratch MQA models with *"frequent loss spikes"* that *"diverged immediately when
fine-tuning on long-input tasks"*, while *"Uptrained grouped-query attention models, however, appear to
be stable."* Instability belongs to *this* source, attributed to 2023 and to Appendix A, and described
as observed rather than measured. Concept 7's line needs amending.

**Forward.** What GQA leaves behind is a **slope, not a wall**. Grouping divides the cache by `H/G` —
a constant — and the cache is still `2 · layers · G · d_k · tokens · batch`, still linear in context
length and still linear in batch. At the app's own serving configuration that is the difference between
51.540 GB and 6.442 GB per conversation; it is not the difference between a growing quantity and a
fixed one. **GQA-8 buys 8× more context before you hit the same wall, and then you hit it.** The
paper's own framing concedes the shape: it is *"an interpolation"*, and interpolation cannot leave the
segment.

Two later families attack what remains, and each attacks a different factor in that product:

- **The `d_k` factor, by compression rather than sharing.** GQA reduces the *count* of stored key/value
  vectors per token; it never questions that what you store is a key and a value at full head width.
  Multi-head latent attention compresses each token's key/value state into a smaller latent vector that
  is decompressed at read time — attacking the width of what is cached rather than how many copies of
  it exist. The app's cost model already anticipates this: `toyCost` takes a `latentDim` that replaces
  `2 · kvGroups · d_k` with `2 · latentDim`, and the two are alternative reductions of the *same*
  product.
- **The `tokens` factor.** Sequence-compression and eviction methods shrink the number of positions
  cached at all — the direction Shazeer himself pointed at in 2019 (*"reduce the number of positions
  being attended-to"*) and which concept 7 correctly noted he did not take.

There is a third thing this paper hands forward that is not an architecture at all: **the idea that
conversion is a legitimate move.** Uptraining is cited to *"Komatsuzaki et al. 2022, which uptrains
standard T5 checkpoints into sparsely activated Mixture-of-Experts models"* — GQA is the second entry
in a line, not the first — and the recipe generalises past attention. Once you accept that you can edit
a trained model's weights structurally and repair it for 5% of the original cost, the question stops
being "what should we have trained?" and becomes "what can we convert?"

For the card's slider, the honest summary of where this concept sits: `kvGroups = heads` is 2017,
`kvGroups = 1` is 2019, and **every value strictly between them is this paper — including, quietly, the
`kvHeads = 8` that the app's serving panel has been using as its default since concept 7.**
