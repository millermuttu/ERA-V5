# Session 9: Loss Functions & Output Heads

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms9mhq4k7p2v9x3t6bd/lesson
Captured: 2026-09-02

Widgets (13) saved under `widgets/`, inline scripts collapsed into `widgets-scripts-dump.txt`.
Videos embedded on the page: https://youtu.be/GbL2pI1K7-A and https://youtu.be/alAoXtJ4qY8
The page notes the studio recording was corrupted and a cropped GMeet version stands in for it.

## 1. What this session is

At the end of Session 8, we had reached this point:

    tokens → embeddings [B, T, D] → transformer blocks → hidden state h [B, T, D]

Every token now has a vector that has seen the whole context before it. The obvious next question
is: that is still just numbers. Where is the prediction?

A hidden state of 4,096 numbers is not a token and it is not a probability. Something has to turn
it into a prediction. And once it is a prediction, something has to turn the difference between
that prediction and the truth into one number the optimiser can push down. That is this session.

    Session 8 — how information is mixed
    Session 9 — prediction to scalar loss
    Session 10 — loss to gradients and updates

The spine is simple:

    hidden state → output head → logits → probabilities → cross-entropy
                 → perplexity + implementation constraints → SFT, preferences, RL, distillation

Two terms we need to focus on today. **Logits** (the prediction head's output) are sent to softmax,
as cross-entropy and sampling depend on them. **Perplexity** is defined before we use it as a
training sanity check.

Session 8 focused on attention, but the hidden state came from a full transformer block, not
attention alone. Section 2 shows the residual stream, FFN, SwiGLU, RMSNorm and pre-norm, just
enough for the output head to have a proper upstream object. Later, when we talk about fused and
chunked cross-entropy, the word *kernel* appears; Section 9 explains just enough GPU vocabulary for
that implementation discussion. Sessions 10 to 13 go deeper.

The session is broad, but it is not random. Every section answers one of two questions: what number
is the model optimizing, and what does V5 have to pay to compute that number?

## 2. The rest of the block

Attention is one part of a transformer layer. Assembled:

    x → RMSNorm → attention → (+) → RMSNorm → FFN → (+) → out
    └──────────────────────────┘ └──────────────────────┘
       residual added back          residual added back

Two sub-layers. Each one is normalised on the way in, and each one adds its result back to what it
was given.

### The residual stream

That line running straight through the middle is the residual stream. Every sub-layer reads from
it, computes something, and adds the answer back.

    x ← x + attention(norm(x))
    x ← x + ffn(norm(x))

Nothing overwrites. Everything accumulates.

This is why depth works at all. Without the addition, a gradient reaching layer 1 from layer 48 has
been multiplied by 47 matrices, and it has either vanished or exploded long before it arrives. With
the addition there is a path from the loss to every layer that passes through no matrices at all.
The derivative of `x + f(x)` with respect to `x` contains a 1. That 1 is the whole logic.

> Remember: a deep transformer is not just a chain of transformations. It is a running residual
> state that each sub-layer reads from and writes back into.

### The feed-forward network

Attention moves information between tokens. The feed-forward network is where each token thinks
about what it just received, on its own, with no reference to any other token.

    h [D] → up-projection D → d_ff → nonlinearity → down-projection d_ff → D

Wide in the middle, back to D at the ends. Traditionally `d_ff = 4 × D`. Why four? Nobody derived
it as a law. The 2017 paper used it, it worked, and the field kept variants of the pattern. Treat
numbers like this as engineering defaults, not mathematical constants.

### SwiGLU

Modern models replace the single nonlinearity with a gate: two projections instead of one,
multiplied together elementwise. SwiGLU is not like a ReLU — it is a concept, a design pattern.

| FFN | Computation | Learned matrices |
| --- | ----------- | ---------------- |
| Classic design | `down(act(up(h)))` | 2 |
| SwiGLU design | `down(silu(gate(h)) ⊙ up(h))` | 3 |

One branch decides what to pass, the other decides how much. It is the same idea as the gates in
Session 8's Gated DeltaNet.

Three matrices instead of two would cost 50% more, so `d_ff` shrinks to compensate:

| Design | Parameter calculation | Parameters |
| ------ | --------------------- | ---------- |
| Two-matrix, 4× | 2 × 4,096 × 16,384 | 134.2M |
| SwiGLU at 11,008 | 3 × 4,096 × 11,008 | 135.3M |

Same budget, better function. 11,008 rather than 10,922 because hardware likes multiples of 256.

### RMSNorm

LayerNorm centres and scales each token's vector using its own mean and standard deviation.
RMSNorm drops the centring and keeps only the scaling.

| Norm | Computation | Work |
| ---- | ----------- | ---- |
| LayerNorm | `(x − mean(x)) / std(x) ⊙ g + b` | subtract, divide, scale, shift |
| RMSNorm | `x / rms(x) ⊙ g` | divide, scale |

Dropping the mean often costs little in LLM training and saves work. RMSNorm is now the common
default in modern decoder-only LLMs.

### Pre-norm

Where the norm sits matters more than which norm you use.

| Layout | Computation | Where the norm sits |
| ------ | ----------- | ------------------- |
| Post-norm (2017) | `x = norm(x + attention(x))` | on the stream |
| Pre-norm (modern) | `x = x + attention(norm(x))` | on the branch |

Post-norm puts a normalisation directly in the residual path, which breaks the clean gradient road
we just built. Deep post-norm models need careful warmup to train at all. Pre-norm leaves the
residual path cleaner. It is the common modern default for deep decoder-only LLMs.

### What a block costs

At `d_model = 4,096`:

| Component | Parameter calculation | Parameters |
| --------- | --------------------- | ---------- |
| Attention: Q, K, V, O | 4 × 4,096 × 4,096 | 67.1M |
| FFN: SwiGLU | 3 × 4,096 × 11,008 | 135.3M |
| Two norms | 2 × 4,096 | 0.008M |
| **Per layer** | | **202.4M** |

> Remember: attention is the famous part, but the FFN is often the larger parameter block. The
> hidden state entering the output head is the result of both.

## 3. The output head

We have a hidden state. We want a score for every word in the vocabulary. So give every word a
vector of its own, the same width as the hidden state, and compare the two with a dot product. A
word whose vector points the same way as the hidden state scores high.

| Vector | Example | Dot product with h |
| ------ | ------- | ------------------ |
| hidden state h | [0.4, −0.2, 0.9, …] | |
| row for Delhi | [0.3, −0.1, 0.8, …] | 1.06 (high) |
| row for banana | [−0.5, 0.6, −0.2, …] | −0.50 (low) |

Stack one such row per token and you have a matrix. That matrix is the layer, and it has three
names in the literature: the **output head**, the **unembedding**, or the **LM head**. Same thing.

    z = h · W_vocabᵀ

| Tensor | Meaning | Shape |
| ------ | ------- | ----- |
| h | one hidden state | [D] |
| W_vocab | one learned row per token | [V, D] |
| z | one score per token | [V] |

Those scores have a name, and this course has never used it.

> A **logit** is one raw, unnormalised score for one vocabulary token. It can be negative. It can
> be large. It is not a probability and it does not mean anything on its own. The whole vector z is
> what we call the logits.

### Weight tying

The input embedding table also holds one row per token, in the same width. So people often tie the
two: use the same matrix for reading tokens in and scoring tokens out.

| Setup | Matrices | Cost |
| ----- | -------- | ---- |
| Untied | `W_embed [V,D]` and `W_vocab [V,D]` | two matrices |
| Tied | `W_embed [V,D]` used for both | one matrix |

Tying halves the parameters and often helps quality, because a token's input meaning and its output
meaning are related. It also couples them, which is a real constraint: the vector that means "this
token arrived" is forced to be the vector that means "predict this token". Section 8 below returns
to this, where it becomes a problem specific to us.

*Widget: One hidden state meeting a vocabulary, row by row. Hover any word to see its weight row
and the dot product that becomes its logit. Change the context and watch which rows rise. Our real
head, V = 131,072 by d_model = 4,096, is printed beneath.*

## 4. Scores are not probabilities

Suppose the vocabulary has five tokens and the model produces these logits:

| Token | Logit |
| ----- | ----- |
| Delhi | 2.0 |
| Mumbai | 1.0 |
| Chennai | 0.5 |
| banana | −1.0 |
| runs | −1.5 |

We want a probability distribution. Softmax is how we get one:

    p_i = exp(z_i) / Σ_j exp(z_j)

Two things it does, and both matter: `exp` makes every number positive, which a probability has to
be; and dividing by the sum makes them add to one.

By hand, for the five above:

| Token | exp(z_i) | Probability |
| ----- | -------- | ----------- |
| Delhi | 7.389 | 7.389 / 12.347 = 0.598 |
| Mumbai | 2.718 | 2.718 / 12.347 = 0.220 |
| Chennai | 1.649 | 1.649 / 12.347 = 0.134 |
| banana | 0.368 | 0.368 / 12.347 = 0.030 |
| runs | 0.223 | 0.223 / 12.347 = 0.018 |
| **Total** | **12.347** | **1.000** |

Notice that the gap between Delhi and Mumbai was 1.0 in the logits and became a factor of e in the
probabilities. Softmax turns differences into ratios. Only the gaps matter: add 100 to every logit
and the probabilities are unchanged.

## 5. Cross-entropy asks one question

The model gave us five probabilities. The truth is one token. What number do we push down?

### Surprise

If something you believed had probability p happens, how surprised should you be? `−log p`.

Test it before trusting it. A one-in-a-thousand event should surprise you a certain amount. Two of
them, independently, should surprise you twice as much.

| Events | Probability | Surprise |
| ------ | ----------- | -------- |
| One event | 1/1,000 | −log p = 6.9 |
| Two independent events | 1/1,000,000 | −log p = 13.8 |

6.9 + 6.9 = 13.8. Surprises add. That is why the logarithm is there.

### The loss we actually want

Say the true next word was Chennai. The model gave it 0.134.

    L = −log(0.134) = 2.010

That is the whole loss. What probability did you give the right answer?

| Probability assigned to the truth | Loss |
| --------------------------------- | ---- |
| 1.000 | 0.000 |
| 0.500 | 0.693 |
| 0.134 | 2.010 |
| 0.010 | 4.605 |

Nothing about the other four tokens appears. They matter only because softmax made them share the
total.

> Carry this forward: for one training position, cross-entropy asks only how much probability the
> model assigned to the correct next token.

### The general version (used in model distillation)

That single term is a special case of something larger, and the larger thing is what the rest of
this session is built from. Suppose the true distribution is not one token but a spread. Over many
examples, the word after "the capital of India is" actually comes out like this:

| Token | p (the world) | q (the model) |
| ----- | ------------- | ------------- |
| Delhi | 0.60 | 0.598 |
| Mumbai | 0.25 | 0.220 |
| Chennai | 0.10 | 0.134 |
| banana | 0.03 | 0.030 |
| runs | 0.02 | 0.018 |

This must be explicitly understood as an example of the distillation process. Otherwise the actual
numbers would be 1, 0, 0, 0, 0 for Delhi, Mumbai, Chennai, banana and runs from our dataset.

**Entropy** is the average surprise of the world against itself:

| Token | Contribution to H(p) |
| ----- | -------------------- |
| Delhi | 0.60 × −log(0.60) = 0.3065 |
| Mumbai | 0.25 × −log(0.25) = 0.3466 |
| Chennai | 0.10 × −log(0.10) = 0.2303 |
| banana | 0.03 × −log(0.03) = 0.1052 |
| runs | 0.02 × −log(0.02) = 0.0782 |
| **H(p)** | **1.0668** |

**Cross-entropy** is the average surprise of the world when you believed q instead:

| Token | Contribution to H(p,q) |
| ----- | ---------------------- |
| Delhi | 0.60 × −log(0.598) = 0.3085 |
| Mumbai | 0.25 × −log(0.220) = 0.3785 |
| Chennai | 0.10 × −log(0.134) = 0.2010 |
| banana | 0.03 × −log(0.030) = 0.1052 |
| runs | 0.02 × −log(0.018) = 0.0803 |
| **H(p,q)** | **1.0736** |

Being wrong cost you 1.0736 − 1.0668 = 0.0068 nats. That excess has a name. It is the **KL
divergence**, and there is a formula for it that never mentions entropy at all:

    H(p,q) − H(p)      = 1.0736 − 1.0668 = 0.0068
    Σ_i p_i log(p_i/q_i) = 0.0068

Both routes give the same number: `D_KL(p ‖ q) = H(p,q) − H(p)`.

### The collapse

Go back to a real training step. The target is one token, Chennai. As a distribution that is
[0, 0, 1, 0, 0]. Put it through cross-entropy and watch four terms die:

    H(p,q) = −(0·log(0.598) + 0·log(0.220) + 1·log(0.134) + 0·log(0.030) + 0·log(0.018))
           = −log(0.134) = 2.010

Now entropy, on that same one-hot: `H(p) = 0` — a certain outcome carries no surprise. So the
excess is the whole thing: `D_KL(p ‖ q) = 2.010 − 0 = 2.010`, which is the number we started this
section with.

> For a one-hot target, cross-entropy and KL divergence are the same number.

Hold on to that. Distillation, the RLHF penalty, DPO and GRPO are all the same KL with something
other than a one-hot in the first slot. Section 22 is the table.

### The gradient

Differentiate the loss with respect to the logits and almost everything cancels:

    ∂L/∂z = softmax(z) − onehot(y)

What you predicted, minus what was true. Nothing else. On our five numbers, with Chennai correct:

    softmax(z) = [+0.598, +0.220, +0.134, +0.030, +0.018]
    onehot(y)  = [0, 0, 1, 0, 0]
    ∇_z L      = [+0.598, +0.220, −0.866, +0.030, +0.018]

The correct token is pushed up by 1 − 0.134 = 0.866. Every wrong token is pushed down by exactly
its own probability. Confident mistakes are punished hardest.

Now add the gradient up:

    Σ_i ∂L/∂z_i = 0.598 + 0.220 − 0.866 + 0.030 + 0.018 = 0

It is zero, and it is always zero. Softmax sums to one, the one-hot sums to one, so the difference
sums to nothing.

> The gradient can move logit mass around. It can never move the whole vector up or down together.

That unwatched direction is what Section 11 is about.

One more consequence, for Section 8. Look at the gradient again: every one of the five entries is
non-zero. At our real vocabulary that is 131,072 rows receiving a gradient on every single token.

> Carry this forward: the loss is simple, but the output-head gradient is dense over the whole
> vocabulary. That is why the head becomes both a memory problem and a stability problem.

*Widget: Drag a logit and watch the loss move. Raise the correct token and the loss falls. Raise a
wrong one and it climbs. Underneath, softmax − onehot is drawn beside a numerical finite difference
computed live, so you can check the gradient rather than take my word for it.*

## 6. Where do the correct answers come from?

This is the part that gets skipped. It is also where the silent bugs live.

We have a loss that needs a correct token. Nobody labelled this data. So where does the target come
from? From the next token. That is the whole trick of language modelling: the text is its own
supervision.

| Input position | The | capital | of | India |
| -------------- | --- | ------- | -- | ----- |
| Target at that position | capital | of | India | is |

Every position predicts the one after it. One sequence of T tokens gives T−1 training examples for
free. In code that is a shift by one:

```python
logits = model(tokens)                     # [B, T, V]
loss = cross_entropy(
    logits[:, :-1].reshape(-1, V),         # drop the last position: nothing follows it
    tokens[:, 1:].reshape(-1),             # drop the first token: nothing predicts it
)
```

Four things go wrong here, and all four are quiet.

**Padding.** Sequences in a batch have different lengths, so short ones get padded. A padding token
is not a prediction. If you include it, you are training the model to predict padding, and your
loss looks better than it is because padding is trivially predictable. Padding positions must be
excluded from the mean, usually with an ignore index.

**Document boundaries.** Session 6 packed many documents into fixed-length sequences to avoid
wasting compute on padding. That packing creates a trap:

    ... Telugu news article → <eos> ⇢ (mask this target) ⇢ Once upon a time in ...

The last token of one document has no relationship to the first token of the next. Training that
pair teaches the model that unrelated things follow each other. The fix is a loss mask at every
document boundary, and if you are also using a document-aware attention mask from Session 6, the
two must agree.

**The mean.** The loss is an average, and you must divide by the number of positions that actually
counted, not by B × T. Divide by the wrong denominator and your loss is scaled by whatever fraction
of your batch was real, which changes with every batch.

**The off-by-one.** Shift the wrong way, or forget to shift, and you have handed the model the
answer. The loss will drop beautifully. The model will have learned to copy its input.

> Sanity check: a suspiciously good loss early in training is usually a target-alignment bug. Print
> the actual token strings for one example, side by side, and read them. Every time.

*Widget: Inputs on top, targets below, offset by one. Four switches, one at a time. Count the
padding into the loss. Let the join between two packed documents predict across. Introduce an
off-by-one. Divide by the wrong denominator. Each one moves the loss, and the contributing-token
count tells you which lie you just told.*

## 7. Loss and perplexity

The loss is in nats and does not mean much on sight. Perplexity is the same number in a form you
can reason about:

    perplexity = exp(mean loss)

Interpret it as how many equally likely options the model is effectively choosing between at each
token.

| Loss | Perplexity | Interpretation |
| ---- | ---------- | -------------- |
| 0.000 | 1 | certain |
| 2.303 | 10 | as unsure as a fair 10-way choice |
| 4.605 | 100 | |
| 11.784 | 131,072 | as unsure as guessing uniformly from our vocabulary |

That last line is the useful anchor. An untrained model on our vocabulary starts at a perplexity of
131,072 and a loss of ln(131,072) = 11.784. If your run does not start near there, inspect your
target alignment before trusting the run. It is the cheapest sanity check you will ever run.

### The warning

> Perplexity is not comparable across tokenizers.

Perplexity is per token, and a tokenizer decides what a token is. Session 2 measured this: a
tokenizer that splits Telugu into roughly three times as many tokens as English for the same
meaning is being asked an easier question at each step, because each step covers less meaning. Its
perplexity will look better while the model is not better.

Comparing two models' perplexity is only meaningful when they share a tokenizer. Across tokenizers,
use bits per byte or bits per character, which normalise by something the tokenizer cannot move.

> Carry this forward: perplexity is a good within-tokenizer training signal and a bad
> cross-tokenizer scoreboard.

## 8. The vocabulary is a second memory problem

Session 8 had two bills. The output head has its own, and it is not the one people expect.

### The parameters

Our vocabulary is 131,072 and our width is 4,096, so:

    head parameters = V × D = 131,072 × 4,096 = 536,870,912 = 536.9M

That number should look familiar. It is exactly the size of the dense embedding table Session 7
threw away. Session 7 replaced a 536.9M table with a 33.55M projection and won 93.75%.

> The head is 16.0× the entire input side we worked so hard to compress. Keep a dense head, and
> Session 7's 93.75% saving across both ends becomes 46.9%.

And weight tying, the standard escape, is closed to us. There is no input table to tie to. Session
7's input side is a fixed byte codec plus one projection. You cannot tie a [V, D] matrix to a thing
that has no rows.

### The tensor, which is worse

Parameters are a fixed cost. The logits are not:

| Tensor | Shape |
| ------ | ----- |
| Hidden states | [B, T, D] |
| Logits | [B, T, V] |

With D = 4,096 and V = 131,072, the logits tensor is 32 times larger than the hidden states that
produced it. Multiply it out:

| batch | context | logits, bf16 | with the backward |
| ----- | ------- | ------------ | ----------------- |
| 8 | 8,192 | 16 GiB | 32 GiB |
| 4 | 32,768 | 32 GiB | 64 GiB |
| 1 | 262,144 | 64 GiB | 128 GiB |

Read the last row again. At the 256K context Session 8 spent three hours earning, one intermediate
tensor in the loss is 64 GiB, and the backward pass needs its gradient too. That is larger than any
accelerator you can buy, for a tensor whose only purpose is to be collapsed into a single scalar.

> Carry this forward: attention's bill grows with context. The logits bill grows with tokens times
> vocabulary, and it arrives at the last layer.

*Widget: Two blocks drawn to scale. The hidden states on the left, the logits on the right. Slide
the context out and watch one of them stay still.*

## 9. How a GPU actually runs this

This is an implementation interlude. The next section compares four ways to compute the same
cross-entropy objective. They give the same loss. They differ in what has to exist in memory at
once. To see why that matters, we need three facts about the machine.

**A GPU has thousands of small cores.** A CPU core is fast and clever. A GPU core is slow and
stupid, and there are tens of thousands of them. For work that is the same operation repeated over
a great many numbers, which is all of deep learning, the second design wins.

**Memory is a hierarchy, and the far end is slow.**

| Memory | Approximate latency | Capacity |
| ------ | ------------------- | -------- |
| Registers | instant | tiny |
| Shared memory | ~30 cycles | ~228 KB per SM |
| L2 cache | ~200 cycles | ~50 MB |
| HBM (VRAM) | ~500 cycles | 80 to 192 GB, where tensors live |

HBM is the card's main memory. It is enormous compared with the caches and it is slow compared with
the cores. Everything you call "GPU memory" is this.

**Bandwidth is the budget.** An H100 does roughly 1,000 trillion floating-point operations a second
and moves about 3.35 TB/s from HBM. Divide those and you get the number that governs everything:

    compute   ≈ 1,000 TFLOP/s
    bandwidth ≈ 3.35 TB/s ≈ 1.7 trillion bf16 numbers/s
              ⇒ roughly 590 operations while fetching one number

> Useful approximation: many deep-learning kernels are limited less by arithmetic and more by
> moving data through the memory hierarchy.

### What a kernel is

A kernel is one function that runs on the GPU. It reads its inputs from HBM, computes, and writes
its outputs back to HBM. That last part is the expensive part.

    x in HBM → kernel 1 → y in HBM → kernel 2 → z in HBM

`y` was written to slow memory and read straight back. If nothing else needed it, that round trip
was pure waste.

### Fusion

Fusing two kernels means doing both steps in one pass, keeping the intermediate in registers, and
never writing it to HBM at all.

| Execution | HBM traffic |
| --------- | ----------- |
| Unfused | read x, write y, read y, write z; 2 writes, 2 reads |
| Fused | read x, write z; 1 write, 1 read |

The arithmetic is identical. The traffic halves, and the intermediate never needs to exist. That is
the whole idea behind the next section. Our logits tensor is 16 GiB written to HBM and read
straight back, purely so it can be collapsed into one number.

A fused kernel does not compute anything different. It just refuses to write down its working.

## 10. Four implementations of exactly one objective

> None of what follows changes what the model learns. The loss is the same number to the last
> decimal. What changes is what has to exist in memory at once.

The GPU interlude is why this works. Without it, fusion sounds like a new objective. It is not. It
is the same objective with less memory traffic.

**1. Materialise everything.** The naive path. Compute all the logits, keep them, softmax, gather
the right one, mean.

    h [B, T, D] → logits [B, T, V], 16 GiB retained for backward → softmax → loss

Simple, and it hits the wall in the table above.

**2. Fuse the projection into the loss.** A fused kernel does the matrix multiply and the loss in
one pass over the data, writing only the scalar out. The full logits tensor is never written to
memory at all. This is the memory-bandwidth argument from the pre-read: the arithmetic was never
the problem, moving 16 GiB to and from HBM was.

**3. Chunk it.** Process the tokens in blocks. Take 1,024 tokens, compute their logits, get their
loss, throw the logits away, take the next 1,024.

    peak logits memory = chunk × V × bytes = 1,024 × 131,072 × 2 = 256 MiB

16 GiB becomes 256 MiB, a 64× reduction, and the loss is bit-for-bit the same. The backward pass
recomputes each chunk's logits when it needs them, trading a little extra arithmetic for a great
deal of memory. This is the trade the whole technique rests on. Cut Cross-Entropy takes this
furthest: it computes the loss without ever materialising a full row, and reports the classifier
head on one Gemma 2 example dropping from tens of gigabytes to about one.

**4. Shard the vocabulary across GPUs.** Give each GPU a slice of the vocabulary. Each computes the
logits for its own slice, and a single communication step combines them into the global
log-sum-exp. This is vocabulary parallelism, and it is what Megatron-style stacks do at scale. It
costs communication and it is the only one of the four that needs more than one device.

| Approach | Peak logit memory | Loss changes? | Needs |
| -------- | ----------------- | ------------- | ----- |
| Materialise | B·T·V | no | nothing |
| Fused kernel | ~0 | no | a kernel |
| Chunked | chunk·V | no | recompute in backward |
| Vocabulary parallel | B·T·V / N | no | N devices and a collective |

> Carry this forward: implementation can change memory by orders of magnitude without changing the
> objective. V5 should specify both the loss and how it is computed.

*Widget: The same loss, computed two ways. Drop the chunk size and watch peak memory fall from
16 GiB to 256 MiB. The two losses agree to every decimal shown.*

## 11. Stability at the head

The head is also where a training run can quietly go wrong. Look again at the loss written the
numerically sane way:

    L = −z_y + log Σ_j exp(z_j)
                └──── log Z ────┘

Only the difference between logits matters to the probabilities. And Section 5 showed exactly why
nothing stops the whole vector drifting: the cross-entropy gradient sums to zero, so it can never
push the logits up or down as a group. That degree of freedom is unconstrained by the loss.

So log Z goes for a walk. The numbers get large, bf16 starts losing precision, and a run that
looked healthy produces a NaN some thousands of steps later.

> This is not a mysterious instability. It is a direction the cross-entropy loss does not directly
> control, and each fix below adds a different kind of control.

Three fixes, in the order the field found them.

**z-loss.** Add a small penalty on the normaliser itself:

    L_total = L_cross-entropy + λ(log Z)²

It pins log Z near zero. Used in OLMo and Chameleon. Costs one hyperparameter.

**Logit soft-capping.** Squash the logits through a bounded function before the softmax:

    z ← c · tanh(z / c)

| With c = 30 | Becomes |
| ----------- | ------- |
| logit 10 | 9.6 |
| logit 60 | 28.9 |
| logit 600 | 30.0, the ceiling |

Gemma 2 uses c = 30 on the final logits and c = 50 on attention logits. Hard ceiling, guaranteed.
It also distorts the distribution near the cap, and it costs a hyperparameter you have to pick.

**Output embedding centering.** Subtract the mean from the output embedding rows, so the head
cannot express a uniform shift in the first place. No hyperparameter to tune.

Now be careful here, because the widget below caught me writing something loose. Centering pins the
mean logit to zero. It does **not** pin log Z. Once the mean is fixed, log Z is governed by how far
the logits spread, and centering says nothing about spread. In the widget's run, centering drives
the mean logit to −5.07e-16 and leaves log Z at 52.17, which is higher than the plain run's 46.42.

> These three fixes do not do the same job.

z-loss attacks the normaliser directly. Soft-capping bounds the logits, so it bounds log Z with
them. Centering removes the uniform component and leaves the spread alone. The 2026 result has
centering winning on training stability. That is a different measurement from the size of log Z,
and reading it as the latter is the mistake I made. The widget runs all four from one identical
initialisation. Read the differences as mechanisms, not as interchangeable fixes.

*Widget: Four runs from one identical start: plain, z-loss, soft-cap, centering. Scrub the step
slider and watch log Z walk away from zero. Read the final numbers off the panel. z-loss pins it to
0.07. Soft-capping bounds it near c. Centering does not lower it at all, which is the point of
Section 11.*

## 12. Adaptive softmax, and when it still applies

Before fused kernels existed, the standard answer to a large vocabulary was to stop scoring all of
it.

Adaptive softmax exploits the fact that word frequency is wildly skewed. Put the few thousand
common tokens in a cheap direct path. Push the rare tail into clusters, reached by a second,
smaller decision.

    hidden state → token frequency? → frequent: score directly, full width
                                    → rare: score cluster, then token, narrower width

It works, and it is still in PyTorch as `AdaptiveLogSoftmaxWithLoss`. It is worth knowing when it
applies: an extremely large vocabulary, heavy frequency skew, and a willingness to accept a
hierarchical approximation of the true distribution.

> 2026 frontier practice: adaptive softmax is worth knowing, but the mainline frontier path is
> exact full-vocabulary softmax with better implementations.

The field went the other way. Keep the exact full-vocabulary softmax, and make the implementation
cheap with fused kernels, chunking and sharding. Exactness is worth keeping when memory can be
handled through fused kernels, chunking and sharding.

## 13. Why predict only one token?

Everything so far assumed one head predicting the next token. That assumption is now worth
questioning.

    h_t → head → token t+1

    h_t → head 1 → token t+1
        → head 2 → token t+2
        → head 3 → token t+3
        → head 4 → token t+4

The losses simply add. Multi-token prediction has two separate motivations, and they should be kept
separate.

**In training, it densifies the signal.** Every position now receives four gradients instead of
one, and the hidden state is forced to carry information useful beyond the immediate next word. A
representation that can only predict one token ahead has learned something shallower than one that
can predict four.

**At inference, the extra heads become drafts.** Heads 2, 3 and 4 propose tokens the model has not
properly computed yet. The main path then verifies them in a single pass and keeps the longest
correct prefix.

> This is where MTP is usually oversold. The extra heads are guesses and they get rejected. You do
> not get four tokens per step. You get some fraction of them, and the acceptance rate is what
> decides whether the technique pays.

The argument that actually matters operationally:

> MTP is speculative decoding where the draft model *is* the model.

It composes with continuous batching and prefix caching, and needs nothing extra resident in
memory. Speculative decoding needs a second model in VRAM.

By 2026, MTP-style heads are no longer just a curiosity. They appear in serious frontier-style
architecture discussions because they connect training signal density with inference
draft-and-verify speedups. It belongs in V5's design discussion rather than in a list of
interesting papers.

The cost is honest but not small for us: k heads means k times the head parameters, and our head is
536.9M. Four dense heads would be 2.1B parameters. That number argues for a factored head even
harder than Section 8 did.

> Carry this forward: MTP is not "free tokens." It is extra supervision and a draft-and-verify path
> whose value depends on acceptance rate and head cost.

*Widget: One head, then four. The trunk is shared, each head predicts one position further out, and
the four losses add. Watch head 4's loss sit above head 1's: predicting four tokens ahead is
genuinely harder.*

*Widget: The extra heads propose, the main pass decides. Accepted tokens are kept, the rest are
thrown away. Head 2 lands 75% of the time, head 4 only 27%, and the overall acceptance is 50.7%.
That is the number that decides whether MTP pays.*

## 14. The other losses in pre-training

Next-token cross-entropy is the main objective, and it is not the only term in a real pre-training
loss. This section is a catalog, not a new spine. Each item below is an auxiliary term or data
arrangement that modifies the training objective while keeping cross-entropy as the centre.

**Fill-in-the-middle.** Rearrange a document so the model must predict a span given both what came
before and what comes after. Same cross-entropy, different arrangement of the same text. It is why
code models can complete inside a function instead of only at the end of one.

**Label smoothing.** Replace the one-hot target with a slightly softened distribution: 1 − ε on the
truth, ε / (V−1) spread over everything else. The model stops being rewarded for infinite
confidence. Note what this does to our spine: the target is no longer zero-entropy, so
cross-entropy and KL stop being the same number, and the gap is exactly H(p).

**MoE load balancing.** Session 14 covers mixture-of-experts properly. The part that belongs here
is that a router which sends every token to the same expert is useless, so an auxiliary loss pushes
the routing towards even usage. It is added to the main loss with a small coefficient.

**Router z-loss.** Exactly the z-loss from Section 11, applied to the router's logits instead of the
vocabulary's, for exactly the same reason.

> Carry this forward: a real training objective is a weighted sum. Each coefficient is a training
> decision, not decoration.

Every one of these is added to the main loss with a coefficient. Every coefficient is a decision
that should be measured rather than inherited blindly.

## 15. Supervised fine-tuning is the same loss, masked

Pre-training is done. Now we want a model that answers questions rather than continuing text. The
loss does not change at all. This is the first post-training bridge. Nothing new is needed
mathematically. What changes is which tokens are allowed to contribute to the same cross-entropy.

    <user> What is the capital of India?   →   <assistant> New Delhi.
    prompt: masked out                          completion: trained on

Cross-entropy, masked to the completion. The prompt is context, not something the model should
learn to generate. Train on the prompt tokens too and you spend capacity teaching the model to
write user questions.

That is the whole of it, mechanically. Session 17 covers what goes in the dataset, which is where
the difficulty actually lives.

> Carry this forward: SFT is not a new loss. It is next-token cross-entropy with a response-only
> loss mask.

*Widget: The prompt greyed out, the completion trained on. Flip the toggle to train on the prompt
too, and watch the contributing token count change.*

## 16. Reward models, and the Bradley-Terry loss

Now the harder problem. Some things have no correct token. If a user asks for advice, there is no
single right answer to cross-entropy against. But a person can look at two answers and say which is
better. That is a different kind of supervision, and it is what preference data is.

    prompt x → completion y_w (preferred: winner)
             → completion y_l (not preferred: loser)

Suppose each completion has some hidden quality score r. A human comparing them should usually pick
the higher one, and should be close to a coin flip when they are level.

| Reward gap r_w − r_l | Winner preferred |
| -------------------- | ---------------- |
| 2.0 | 88% |
| 0.5 | 62% |
| 0.0 | 50%, a coin flip |

A sigmoid does exactly that. Giving it a name, this is the **Bradley-Terry model**:

    P(winner preferred) = σ(r_w − r_l)

Train a reward model by maximising the likelihood of the preferences you observed, which is the
same as minimising:

    L_reward model = −log σ(r_w − r_l)

Two things worth noticing. **Only the gap matters.** Add a constant to every reward and the loss is
unchanged. The reward model has no absolute scale, exactly like logits under softmax. And **this is
cross-entropy again**, over a two-outcome distribution. The sigmoid is softmax with two classes.

> Carry this forward: preference data does not give a correct next token. It gives a pairwise
> comparison, and Bradley-Terry turns that comparison into a loss.

*Widget: Drag the two rewards. The loss follows the gap between them. Now shift both by the same
amount and watch nothing happen. That is the property DPO exploits.*

Kept here only for reference. Beyond today's scope, but will come back to this in future sessions.

## 17. RLHF, and what PPO is actually doing

We have a reward model. Now make the language model score well under it. This section is not asking
you to master policy gradients yet. It is here because PPO shows why alignment training became
expensive: it needs extra models, fresh samples and a KL anchor.

The naive objective is to maximise expected reward. It fails immediately. The reward model learned
from humans who preferred thorough answers. So it scores long answers slightly higher. The policy
finds this in a few hundred steps:

| Step | Response | Reward |
| ---- | -------- | ------ |
| 0 | "Delhi." | 0.61 |
| 200 | "Delhi. I hope this helps!" | 0.68 |
| 600 | "Delhi. I hope this helps! Let me know if you need anything else at all." | 0.74 |

Nothing here is a better answer. The policy found a direction the reward model rewards and walked
along it. This is **reward hacking**, and it is the default outcome. So we anchor the model to
where it started:

    max E[reward] − β · D_KL(π_policy ‖ π_reference)

| Part | Role |
| ---- | ---- |
| E[reward] | go where the reward is |
| −β D_KL(π_policy ‖ π_reference) | do not wander far from the starting model |
| β | how tight the leash is |

That KL is the same KL from Section 5. The reference distribution is no longer a one-hot token, it
is the frozen starting model. Same object, different first argument.

PPO optimises this with three pieces.

**A clipped surrogate.** Compare what the old policy gave a token with what the new one gives it:

| Quantity | Value |
| -------- | ----- |
| Old policy | 0.20 |
| New policy | 0.35 |
| Ratio | 0.35 / 0.20 = 1.75 |
| Clip range, ε = 0.2 | [0.80, 1.20] |
| Ratio actually used | 1.20 |

The step wanted to be 1.75 and is allowed to be 1.20. One batch cannot wreck the model.

**A value network.** A second model that estimates how good a state is, used as a baseline. Section
19 explains what a baseline is and then deletes this.

**An entropy bonus.** So the policy does not collapse onto one output.

It works. It is also heavy.

| Model | State |
| ----- | ----- |
| Policy | being trained |
| Reference | frozen |
| Reward | frozen |
| Value | being trained |

Four models in memory. And the whole thing is on-policy, so you must generate fresh samples
constantly. The next two sections show how later objectives delete two of those four roles.

*Widget: Four boxes: policy, reference, reward, value. Two are trained, two are frozen. Switch to
DPO and GRPO and watch four become two.*

## 18. DPO: the reward model cancels

Here is the result that changed alignment practice. That constrained objective in Section 17 has a
closed-form optimum. You can write the best policy down without searching for it.

> What I am asking you to take on trust here: the closed form below is a standard result for a
> KL-constrained maximisation and I am not deriving it in class. Everything after it, the inversion
> and the cancellation, is arithmetic you can check line by line.

    π*(y | x) = (1 / Z(x)) · π_ref(y | x) · exp(r(x,y) / β)

Z(x) is the partition function that makes it sum to one. It is a sum over every possible
completion, so it is completely intractable. Hold that thought. Now solve it for the reward
instead:

    r(x,y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x)

The reward is just a log-ratio between the tuned model and the reference, plus that intractable
term. Now substitute this into Bradley-Terry from Section 16, which only ever uses the difference
of two rewards on the same prompt:

    r(x,y_w) − r(x,y_l) = β log( π(y_w)/π_ref(y_w) ) + β log Z(x)
                        − β log( π(y_l)/π_ref(y_l) ) − β log Z(x)

The two log Z(x) terms cancel because both completions share the same prompt x. Z(x) depends only
on the prompt. Both completions share the prompt. It cancels exactly. And what remains has no
reward model in it at all:

    L_DPO = −log σ( β log(π(y_w|x)/π_ref(y_w|x)) − β log(π(y_l|x)/π_ref(y_l|x)) )

> DPO is the RLHF objective with the reward model substituted away.

No reward model to train, no value network, no sampling loop. Two models in memory instead of four,
and an ordinary supervised loss over a fixed dataset.

Read the formula as an instruction and it is almost obvious: raise the probability of the winner
relative to where the reference model had it, and lower the loser's, and the sigmoid stops pushing
once the gap is comfortable.

The honest limitations, because DPO is not free:

- It is **offline**. It learns from a fixed set of pairs, so it cannot discover that a completion
  neither annotator saw is better than both.
- It is **sensitive to the reference**. If π_ref already assigns a pair strange probabilities, the
  log-ratio starts distorted.
- It can push down the winner's absolute probability while still increasing the gap, because only
  the gap is in the loss.

> Carry this forward: DPO is cheaper because it turns a KL-constrained preference problem into a
> fixed-dataset supervised loss. It is not automatically better than online RL; it trades
> exploration for simplicity.

## 19. GRPO: the value network cancels too

DPO removed the reward model but also gave up online learning. For reasoning, where a correct answer
can be checked, we want to stay online. So delete something else.

First, what a baseline is, because PPO's value network exists only to provide one. A completion
scores 0.9. Is that good? It depends entirely on what the other answers to that same prompt scored.

| Other group rewards | Group mean | What reward 0.9 means |
| ------------------- | ---------- | --------------------- |
| 0.1, 0.2, 0.1 | 0.325 | excellent |
| 0.9, 0.9, 0.9 | 0.900 | unremarkable |

Same completion, same reward, opposite lesson. A baseline is what you subtract so the model learns
from the difference and not from the raw score. PPO trains a whole second network to predict it.

**Group Relative Policy Optimisation** gets the baseline for free. Sample a group of G completions
for the same prompt, score them all, and let the group be its own baseline:

    A_i = (r_i − mean(r_1, …, r_G)) / std(r_1, …, r_G)

    prompt → completion 1: reward 0.9 · advantage +1.526
           → completion 2: reward 0.2 · advantage −0.723
           → completion 3: reward 0.5 · advantage +0.241
           → completion 4: reward 0.1 · advantage −1.044

    mean = 0.425,  std = 0.311,  Σ_i A_i = 0

The advantages are zero-mean by construction. That is precisely what the value network was being
trained to achieve, and here it falls out of arithmetic.

> PPO needed four models. DPO removed two. GRPO removes the value network and stays online.

Two cancellations, the same shape both times. Something expensive turned out to be recoverable from
things you already had.

**RLVR**, reinforcement learning from verifiable rewards, is what makes this practical for
reasoning. Where the answer can be checked by running the code or evaluating the maths, you do not
need a learned reward model at all. The reward is 1 if the answer is right and 0 if it is wrong.

> A verifier is a stronger reward source than a learned preference model when the task is genuinely
> checkable. It can still have bugs, loopholes and coverage gaps, so the verifier itself becomes
> part of the training system.

This is why reasoning training scaled so fast in 2025 and 2026. Where answers can be checked, the
most fragile component simply disappeared.

> Carry this forward: GRPO keeps online sampling but removes the separate value network by using
> the group as the baseline.

*Widget: One prompt, two completions, one preference, three routes to the same update. Spend the
time on the DPO tab. log Z is on a slider, both reward values move as you drag it, and the loss does
not change by a single digit. That is the cancellation happening in front of you. The GRPO tab
shows the advantages summing to zero, which is why no value network is needed.*

## 20. The rest of the preference family

DPO opened a design space and a dozen variants followed. They differ in what they assume and what
they need, and they are all the same shape. This section is a map, not a derivation. The purpose is
to recognise the family names when they appear in papers and training plans.

| Method | What it changes | Needs pairs? | Needs a reference model? |
| ------ | --------------- | ------------ | ------------------------ |
| DPO | the baseline | yes | yes |
| IPO | replaces the sigmoid objective to stop overfitting to deterministic preferences | yes | yes |
| KTO | works from single thumbs-up or thumbs-down labels, no pairing needed | no | yes |
| ORPO | folds preference into the SFT loss with an odds-ratio term | yes | no |
| SimPO | uses average log-probability as the implicit reward, length-normalised | yes | no |

Two of them are worth remembering for practical reasons. KTO, because unpaired feedback is far
cheaper to collect than pairs, and a thumbs-down in production is already unpaired data. ORPO,
because it collapses SFT and alignment into one stage, which removes a whole training run from the
pipeline.

> Carry this forward: preference objectives mostly differ in what feedback they can consume and
> whether they need a reference model.

## 21. Distillation

One more, and it completes the pattern. Sometimes the target is neither a token nor a preference. It
is another model. A large teacher produces a full distribution over the vocabulary; a small student
is trained to match it.

    L = D_KL(teacher ‖ student)

Now the first argument to KL is a full distribution rather than a one-hot, and the H(p) term from
Section 5 is no longer zero. It is a constant with respect to the student, so it does not change the
gradient, but the loss no longer bottoms out at zero.

The direction matters, and it is a real design decision. Take a teacher with two good answers and
one bad one, and two candidate students:

| Distribution | Answer 1 | Answer 2 | Answer 3 | Forward KL | Reverse KL |
| ------------ | -------- | -------- | -------- | ---------- | ---------- |
| Teacher, two modes | 0.45 | 0.10 | 0.45 | | |
| Student A, hedging | 0.34 | 0.32 | 0.34 | 0.136 | 0.182 |
| Student B, committed | 0.90 | 0.05 | 0.05 | 0.746 | 0.479 |

Read the columns. Forward KL picks student A. Reverse KL picks student B. Same two students,
opposite verdicts.

- **Forward KL**, `KL(teacher ‖ student)`, is **mass-covering**. Assigning near-zero probability
  where the teacher has mass is punished hard, so the student spreads out.
- **Reverse KL**, `KL(student ‖ teacher)`, is **mode-seeking**. Putting mass where the teacher has
  none is punished, so the student picks one mode and commits.

> Neither direction is correct in general.

A hedging student produces bland text. A committing student produces confident text and may drop
whole behaviours the teacher had. Choose against what the model is for.

**On-policy distillation** takes the student's own generations and asks the teacher to score them,
which fixes the mismatch where a student trained on teacher text never sees its own mistakes.

> Carry this forward: distillation is KL against a teacher distribution, and the KL direction
> changes student behaviour.

## 22. The loss map

Every loss in this course, in one place. Read the middle column.

| Loss | KL against what | Where it is used |
| ---- | --------------- | ---------------- |
| Next-token cross-entropy | the one-hot true token | pre-training, this session |
| Fill-in-the-middle | the same, rearranged | pre-training |
| Label smoothing | a softened one-hot | pre-training, optional |
| Perplexity | not a loss, exp of the mean CE | evaluation |
| z-loss and router z-loss | not a KL, a penalty on log Z | stability, Sections 9 and 12 |
| MoE load balance | not a KL, a penalty on routing skew | Session 14 |
| SFT | the one-hot, masked to the completion | Session 17 |
| Reward model | Bradley-Terry over a preference pair | Section 16 |
| PPO / RLHF | the reference policy | Section 17 |
| DPO | the reference policy, reward model cancelled | Section 18 |
| GRPO | the same, value network cancelled | Section 19 |
| RLVR | GRPO with a verifier in place of a reward model | Section 19 |
| IPO, KTO, ORPO, SimPO | variations on DPO's assumptions | Section 20 |
| Distillation | a teacher's distribution | Section 21 |

The unifying idea: many rows reuse the same comparison shape. The name changes when the thing being
compared against changes.

*Widget: One KL divergence with a swappable first argument. Change what you compare against and the
loss changes its name: cross-entropy, label smoothing, distillation, the RLHF penalty.*

## 23. What V5 has to decide

Session 7 factored the front door and won 93.75%. Session 9 has shown that the back door is still a
dense 536.9M matrix, and that the standard escape is unavailable to us because there is nothing to
tie to.

What the evidence already settles:

- The full-vocabulary softmax stays exact. The field tried approximating it and went the other way.
  We use fused or chunked cross-entropy, and it changes nothing about what the model learns.
- The logits tensor should not be materialised at our target context. 64 GiB at 256K is an
  implementation design problem, not a tuning problem.
- MTP is a design decision, not an experiment. Three of the four reference architectures ship it.

What is genuinely open:

| Question | What would settle it |
| -------- | -------------------- |
| A factored output head. Kronecker-style, low-rank, or pay the 536.9M. | A matched-quality ablation at proxy scale. This is ours to run and nobody has run it for a byte-codec input side. |
| MTP head count. Each head is another 536.9M dense, or another factored head. | Acceptance rate against parameter cost at our width. |
| Chunk size for the fused loss. | A memory-against-throughput sweep on the actual cluster. |
| Head stability: z-loss, soft-cap, or centering. | Cheap to test at proxy scale, and worth doing before the real run rather than after a NaN. |
| The alignment path: DPO for cost, GRPO for reasoning, or both in sequence. | What V5 is actually for, which is a Session 18 conversation informed by today. |

The practical summary: we now have a map of the main losses in the pipeline, and one architectural
hole remains at the very last layer. That is a good place to be at session nine of twenty.

## 24. The assignment

> One notebook, one loss harness, and one thing you have to get right by reading rather than by
> guessing.
>
> **Part 1: the harness**
>
> ```python
> hidden = model(tokens)
> logits = output_head(hidden)
> loss = cross_entropy(
>     logits[:, :-1].reshape(-1, vocab_size),
>     tokens[:, 1:].reshape(-1),
> )
> ```
>
> Take that and make it correct and observable. You must:
>
> - Print every tensor shape and say in one line what each dimension is.
> - Verify the shift by printing the actual token strings, inputs beside targets. Not the ids. The
>   strings. You will not catch an off-by-one in a wall of integers.
> - Mask padding and confirm the count of contributing tokens changes.
> - Pack two documents into one sequence and mask the boundary. Show the loss before and after
>   masking it, and explain the difference.
> - Compute perplexity, and show that an untrained model sits near your vocabulary size. If it does
>   not, find the bug before you go further.
> - Compare tied against untied head parameter counts on your configuration.
> - Measure peak memory for ordinary cross-entropy against a chunked version you write yourself.
>   Report both numbers and the ratio.
>
> **Part 2: one extra head**
>
> Add a second output head predicting token t+2. Report both losses separately and their sum, and
> say what happens to the second head's loss over training compared with the first. Explain what you
> see.
>
> **What to submit**
>
> A Google Colab notebook moved to GitHub that runs top to bottom, and a short write-up with the
> seven numbers from Part 1, the two losses from Part 2, and your demonstration from Part 3.
>
> One warning, the same one as last time. A target shift in the incorrect direction can produce a
> beautiful loss curve. Print the strings. Many serious training bugs live in the few lines between
> the model output and the scalar, and they do not always raise an exception.
>
> What are you submitting? GitHub README.md link.

**Note on the source, not part of the quote:** the assignment lists Part 1 and Part 2 only, then
"What to submit" asks for "your demonstration from Part 3". No Part 3 appears anywhere on the lesson
page. The gap is in the source, not in this transcription. Part 1 is also described as producing
"seven numbers", which matches its seven bullets.
