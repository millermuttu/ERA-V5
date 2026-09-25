# Session 13: Distributed Training II, Model and Pipeline Parallel

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms13nr0y6n1u8b5zs9hi/lesson
Captured: 2026-09-25

Widgets (9) saved under `widgets/`.
Videos embedded on the page: https://youtu.be/iZzAM911-sk (Studio) and https://youtu.be/i7vPv7eJ_Wc (GMeet)

## 0. Why this session exists at all

Session 12 divides storage. ZeRO takes the 16 bytes per weight and spreads the optimizer state, the
gradients and finally the weights across the copies. Every GPU still runs the whole model, and every GPU
still processes its own micro-batch from the first layer to the last.

Three things break under that arrangement, and each one is a section of Session 13.

- **ZeRO never divides activations.** One sequence of 8,192 tokens through our 30B model stores 127.5 GiB
  of intermediate results. An H100 card holds 74.5 GiB. Take ZeRO-3 to its limit, put the entire training
  state at 16/N, and that one sequence still does not fit on one card. Tensor and pipeline parallelism
  divide the activations, because they divide the computation itself.
- **ZeRO-3 needs a whole layer to fit and rebuilds it every step.** It gathers each layer's full weights
  on every GPU just before use. That costs 3 parameter-sizes of traffic per step, and it assumes one
  layer's weights and one layer's work belong on one GPU.
- **One long sequence cannot be split by data parallelism.** A 131,072-token sequence produces 2,040 GiB
  of activations. It is a single sample, so no number of data-parallel copies helps. Context parallelism
  is the answer.

Session 12 answers "our model's state does not fit". Session 13 answers "our model's computation does not
fit". Production runs use both at once: DeepSeek-V3 ran 16-way pipeline, 64-way expert parallelism and
ZeRO-1 together.

### Prelude to whole session — attempt 1

**ZeRO divides what is STORED. Tensor / Sequence / Pipeline / Context parallelism divide what is COMPUTED.**

That distinction makes almost everything fall into place.

*Imagine one giant Transformer as a school textbook.* Suppose the model is a 1,000-page textbook, and you
have 8 children/GPUs trying to study/train it. There are several different things consuming GPU memory:

```
MODEL TRAINING MEMORY
│
├── Weights          ← the actual model
├── Gradients        ← how weights should change
├── Optimizer state  ← Adam momentum, variance, etc.
│
└── Activations      ← temporary working memory
                       created while processing the tokens
```

ZeRO attacks the first three. TP / SP / PP / CP attack the computation and, importantly, activations.

#### 1. Normal Data Parallelism (attempt 2 follows below)

Before ZeRO, imagine 8 GPUs. Every GPU gets a complete copy of the textbook:

```
GPU 0: [FULL MODEL] → Batch A
GPU 1: [FULL MODEL] → Batch B
GPU 2: [FULL MODEL] → Batch C
...
GPU 7: [FULL MODEL] → Batch H
```

Each GPU processes different examples. This is great for speed, but terrible for memory because every GPU
stores weights, gradients and Adam state — all duplicated 8 times.

#### 2. ZeRO-1

ZeRO says: *"Why is every GPU storing the same Adam optimizer state?"* So it divides optimizer state.

```
        Weights  Gradients  Optimizer
GPU 0   FULL     FULL       1/8
GPU 1   FULL     FULL       1/8
GPU 2   FULL     FULL       1/8
...
GPU 7   FULL     FULL       1/8
```

But computation hasn't changed. Every GPU still runs Layer 1, Layer 2, Layer 3 … Layer 96 on its own batch.

> ZeRO-1 = memory trick. Not model-compute splitting.

#### 3. ZeRO-2

Then ZeRO says: *"Why are we duplicating gradients too?"*

```
        Weights  Gradients  Optimizer
GPU 0   FULL     1/8        1/8
GPU 1   FULL     1/8        1/8
...
GPU 7   FULL     1/8        1/8
```

Again, every GPU still computes the entire model.

```
GPU 0: Layer 1 → 2 → 3 → ... → 96
GPU 1: Layer 1 → 2 → 3 → ... → 96
```

Different micro-batches, same complete model computation.

#### 4. ZeRO-3

Now ZeRO says: *"Why even duplicate the weights?"* At rest:

```
GPU 0: 1/8 weights
GPU 1: 1/8 weights
GPU 2: 1/8 weights
...
GPU 7: 1/8 weights
```

This is where the confusion usually starts. You might think: *"Aha! GPU 0 now computes 1/8 of the
model."* No. When Layer 17 needs to execute:

```
GPU0 ─┐
GPU1 ─┤
GPU2 ─┤
...   ├── All-gather Layer 17's weights
GPU7 ─┘
       ↓
 FULL Layer 17
 temporarily exists
       ↓
 Every GPU computes Layer 17
 on its own micro-batch
```

Then those weights can be discarded. Next layer: gather Layer 18 → compute Layer 18 → discard.

> ZeRO-3 means: **Divide model STORAGE, not the actual matrix multiplication.**

This sentence is worth remembering.

#### Now Tensor Parallelism

Tensor Parallelism says: *"Instead of all 8 GPUs computing the entire matrix multiplication, let's split
the matrix multiplication itself."* Suppose a layer contains this enormous matrix:

```
              8192 outputs
        ┌──────────────────────┐
        │                      │
  8192  │       W MATRIX       │
 inputs │                      │
        └──────────────────────┘
```

With TP=8:

```
GPU0   GPU1   GPU2         GPU7
 ↓      ↓      ↓            ↓
1/8 W  1/8 W  1/8 W  ...  1/8 W
```

Now one token enters. Instead of GPU0 computing the whole matrix, GPU1 computing the whole matrix, …, you get:

```
                 Input
                   │
     ┌─────────────┼─────────────┐
     ↓             ↓             ↓
   GPU0          GPU1          GPU7
   1/8 W         1/8 W         1/8 W
     │             │             │
     └──────── communicate ──────┘
                   ↓
            complete result
```

That is fundamentally different from ZeRO.

```
ZeRO-3                              Tensor Parallel
weights stored divided              weights stored divided
BUT                                 AND
computation = full layer per GPU    computation = layer itself divided between GPUs
```

That's the critical distinction.

#### Sequence Parallelism

Now imagine the activation tensor looks like `4096 tokens × 8192 hidden dimensions`. Tensor Parallelism
tends to split the 8192 hidden dimensions. Sequence Parallelism says: *"For operations where tokens don't
need to interact, let's also divide the 4096 tokens."*

```
Without SP:                 With SP=8:
GPU0: all 4096 tokens       GPU0: tokens 1–512
GPU1: all 4096 tokens       GPU1: tokens 513–1024
GPU2: all 4096 tokens       GPU2: tokens 1025–1536
...                         ...
                            GPU7: tokens 3585–4096
```

For things like LayerNorm, Dropout and some residual operations, each token can largely be handled
independently. So instead of storing all those activations everywhere, each GPU holds its part. That's
why TP and SP are normally paired.

```
Tensor Parallel:   split HIDDEN DIMENSION
Sequence Parallel: split TOKEN DIMENSION for suitable operations
```

#### Pipeline Parallelism

Completely different axis. Pipeline says: *"Why does every GPU need to execute every layer?"* Suppose
there are 96 layers and 8 GPUs.

```
GPU0 → layers 1–12
GPU1 → layers 13–24
GPU2 → layers 25–36
GPU3 → layers 37–48
GPU4 → layers 49–60
GPU5 → layers 61–72
GPU6 → layers 73–84
GPU7 → layers 85–96
```

Now the model physically flows through the GPUs:

```
Tokens
  ↓
GPU0  Layers 1-12
  │ activations
  ↓
GPU1  Layers 13-24
  ↓
GPU2  Layers 25-36
 ...
  ↓
GPU7  Layers 85-96
```

This really does mean: **GPU 0 never computes layers 13–96.** Very different from ZeRO-3, where every GPU
computes `1 → 2 → 3 → ... → 96`. Pipeline parallel: GPU0 computes `1 → ... → 12`, GPU1 computes
`13 → ... → 24`, etc.

#### Context Parallelism

This one sounds similar to Sequence Parallelism but solves a different problem. Imagine you're training
with 128,000 tokens. Attention becomes enormous because every token attends to other tokens. Context
Parallel says: *"Let's divide this one giant 128K-token sequence across GPUs."*

```
With CP=8:
GPU0: tokens 0–15,999
GPU1: tokens 16,000–31,999
GPU2: tokens 32,000–47,999
...
GPU7: tokens 112,000–127,999
```

But there is a problem. GPU0's tokens need to attend to tokens living on GPU7. Therefore GPUs exchange
keys and values during attention. Conceptually:

```
GPU0: [tokens 0────16K]
         ↕ K,V
GPU1: [tokens 16K──32K]
         ↕ K,V
GPU2: [tokens 32K──48K]
         ↕
...
GPU7: [tokens 112K─128K]
```

That's Context Parallelism. Its purpose is primarily: **make extremely long sequences manageable.**

#### So put the whole thing on one picture

Your Transformer has several axes that can be cut.

```
                         TRAINING
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
       STORAGE                             COMPUTE
          │                                   │
        ZeRO               ┌────────────────┼───────────────┐
                           │                │               │
                         WIDTH            DEPTH          TOKENS
                           │                │               │
                          TP               PP              CP
                           │
                          SP
                   helps activation memory
```

More precisely:

| Technique | Child-level explanation |
| --- | --- |
| ZeRO-1 | Divide the optimizer notebooks |
| ZeRO-2 | Divide optimizer notebooks + gradients |
| ZeRO-3 | Divide optimizer + gradients + model weights while stored |
| Tensor Parallel | Divide each giant matrix multiplication |
| Sequence Parallel | Divide token activations for suitable per-token operations |
| Pipeline Parallel | Divide the model's layers |
| Context Parallel | Divide one enormous sequence across GPUs |

#### Why doesn't ZeRO-3 solve everything?

Take an absurd example. You have a 1 trillion parameter model and enough GPUs that ZeRO-3 makes the
weights fit. Great. But now you want 256K context and your activation tensor is gigantic. ZeRO says
`¯\_(ツ)_/¯` because ZeRO didn't fundamentally divide those activations. Then you may need:

```
CP → divide 256K tokens
SP → reduce duplicated activation storage
TP → divide enormous matrix calculations
PP → divide 100+ layers
```

ZeRO solves a different dimension of the problem.

#### The best mental model: five axes

Forget the implementation names temporarily. A Transformer has roughly these things you can divide:

```
WHAT CAN I DIVIDE?

1. Training state → ZeRO  (weights, gradients, optimizer)
2. Matrix width → Tensor Parallel
3. Layers / depth → Pipeline Parallel
4. Tokens / context → Context Parallel
5. Batch → Data Parallel

And Sequence Parallel is closely associated with #2:
6. Activation tokens → Sequence Parallel
   during operations where token-wise splitting works
```

That is really the entire story.

#### A concrete 64-GPU example

Suppose you have 64 H200s. You could configure `TP = 8, PP = 4, CP = 2` — `8 × 4 × 2 = 64 GPUs`. One
training sample is therefore spread across all 64 GPUs:

```
ONE MODEL

Pipeline stage 0   16 GPUs   layers 1–24
  │
  ├── CP=2
  │    ├── first half tokens
  │    └── second half tokens
  │
  └── TP=8 within each CP group

Pipeline stage 1   16 GPUs   layers 25–48
Pipeline stage 2   16 GPUs   layers 49–72
Pipeline stage 3   16 GPUs   layers 73–96
```

Now suppose you actually have 512 GPUs. You could replicate that 64-GPU model eight times (DP = 8,
Group 0 … Group 7 → 64 GPUs each). Now ZeRO can operate across those 8 data-parallel replicas:

```
DP replica 0 \
DP replica 1  \
DP replica 2   >  shard optimizer / gradients / perhaps params
...           /
DP replica 7 /
```

This is why modern large-model training can simultaneously say:

```
TP = 8
PP = 8
CP = 4
DP = 16
ZeRO / distributed optimizer enabled
Sequence Parallel = enabled
```

They're not alternatives. They're cutting completely different dimensions.

#### The one sentence I'd use in class

> ZeRO asks, "Where do I STORE the model's training state?" TP, PP and CP ask, "Which GPU actually
> COMPUTES which part of the model?"

And then:

> TP cuts a layer sideways, PP cuts the model vertically by layers, CP cuts the input sequence by
> tokens, and ZeRO cuts the duplicated training-state memory.

That distinction is the missing piece behind your table.

## 1. Introduction

A training run spreads its work across many GPUs. The simplest arrangement copies the whole model onto
every GPU and gives each copy different data. This arrangement is called data parallelism. That
arrangement stops working once a single copy of the model no longer fits on one GPU, or once the working
memory of a single long sequence no longer fits. This session covers the ways to divide the model itself,
and the sequence itself, across GPUs.

One model is used for every calculation. It has 30.2 billion parameters in 96 layers, with a hidden size
of 5,120 and 40 attention heads. An attention head is one of the parallel attention calculations inside a
layer, and each of ours works on 128 dimensions.

Training one weight requires more memory than the weight itself.

| What is stored for one weight | Bytes |
| --- | --- |
| the weight, in 16-bit format | 2 |
| its gradient, in 16-bit format | 2 |
| a 32-bit master copy of the weight | 4 |
| the optimizer's two running averages, in 32-bit format | 8 |
| total | 16 |

$$30.2 \times 10^9 \text{ weights} \times 16 \text{ bytes} = 483.2 \text{ GB} = 450.0 \text{ GiB}$$

An 80 GB card holds 74.5 GiB and a 180 GB card holds 167.6 GiB, so the training state alone needs more
than six of the smaller cards or more than two of the larger ones.

**The detail.** The parameter count follows from the shape. Our 40 attention heads share 8 sets of keys
and values, called key-value heads. Eight key-value heads of 128 dimensions give keys and values of width
1,024, which is one fifth of the query width.

$$2 \times 5120^2 + 2 \times 5120 \times 1024 = 62.9 \text{M attention parameters per layer}$$

The feed-forward network uses three matrices that connect the hidden size of 5,120 to an inner size of
16,384.

$$3 \times 5120 \times 16384 = 251.7 \text{M feed-forward parameters per layer}$$

$$(62.9 + 251.7) \text{M} \times 96 \text{ layers} = 30.2 \times 10^9$$

The embedding tables are left out of this count.

The forward pass also stores intermediate results, called activations, which the backward pass needs.
Korthikanti and colleagues published a count in 2022 of every tensor a transformer layer keeps. With
FlashAttention, which computes attention without storing the full table of attention scores, the count
comes to about 34 bytes per token per unit of hidden size. The count was made for a slightly different
layer design, so 34 is an estimate for ours.

$$8192 \text{ tokens} \times 5120 \times 34 \text{ bytes} = 1.33 \text{ GiB per layer}$$

Across 96 layers that is 127.5 GiB of activations for one sequence of 8,192 tokens.

> **Carry this forward:** the training state is 450.0 GiB, which is larger than either card, and the
> activations of one sequence are 127.5 GiB.

## 2. Terminology

Ten terms are used throughout the session, each with a number attached.

- A **GPU** is one accelerator card. This session uses two NVIDIA GPUs. The H100 holds 80 GB, which is
  74.5 GiB. The B200, from the newer Blackwell generation, holds 180 GB, which is 167.6 GiB.
- A **node** is one machine holding eight GPUs of the same kind.
- **NVLink** is the wiring that connects GPUs inside a node. H100 nodes use NVLink 4, which carries about
  450 GB per second in each direction for each GPU. B200 nodes use NVLink 5, which carries about 900 GB
  per second in each direction for each GPU.
- **InfiniBand** is the network cable that connects nodes. Both kinds of node have one 400 Gb/s network
  card per GPU, and each card carries $400 \div 8 = 50$ GB per second in each direction.
- A **micro-batch** is the group of sequences processed in one forward and backward pass. In this session
  a micro-batch is one sequence of 8,192 tokens.
- An **all-reduce** takes a tensor from every GPU and returns the combined result to every GPU. Arranged as
  a ring of $N$ GPUs, each GPU sends a total of $2(N-1)/N$ times the size of the tensor, which is 1.75
  times at $N = 8$.
- An **all-gather** collects one slice from each GPU and gives every GPU the complete tensor. A
  **reduce-scatter** combines the tensors and gives each GPU one slice of the result. A ring all-reduce is
  a reduce-scatter followed by an all-gather.
- An **all-to-all** has every GPU send a different piece to every other GPU.
- A **point-to-point send** moves a tensor from one GPU to one other GPU.
- A **degree** is the number of GPUs a form of parallelism divides its work across, written for example
  as TP = 8.

> **Carry this forward:** one NVLink 4 connection carries 9 times the traffic of one InfiniBand card, and
> one NVLink 5 connection carries 18 times.

## 3. Tensor Parallelism

A single weight matrix can be too large for one GPU. Tensor parallelism cuts that matrix into slices and
gives each GPU one slice. Every GPU multiplies its own slice at the same time, and the GPUs combine their
results before the next part of the layer begins.

The combining step is where the cost lies. It happens inside every layer, for every token, so the GPUs
exchange data constantly throughout the forward and backward passes.

**The detail.** The standard method was published as Megatron-LM in 2019, and it pairs two kinds of split
so that each block needs only one combining step.

The feed-forward network has three matrices. Two of them expand the hidden size from 5,120 to 16,384, and
both are split by columns, so each GPU produces its own share of the 16,384 outputs with no need to talk
to the others. The third matrix brings the size back to 5,120 and is split by rows, so each GPU consumes
exactly the share the first two produced. The outputs of the row-split matrix are then summed with one
all-reduce.

Attention follows the same pattern. The query, key and value projections are split by attention head,
which is a column split. The output projection is split by rows and followed by one all-reduce.

Each layer therefore performs two all-reduces in the forward pass and two in the backward pass.

| TP degree | training state per GPU |
| --- | --- |
| 1 | 450.0 GiB |
| 2 | 225.0 GiB |
| 4 | 112.5 GiB |
| 8 | 56.25 GiB |

The traffic follows from the size of one activation tensor.

$$8192 \text{ tokens} \times 5120 \times 2 \text{ bytes} = 83.9 \text{ MB}$$

At TP = 8, each of the four all-reduces moves 1.75 times that tensor, in every one of the 96 layers.

$$4 \times 83.9 \text{ MB} \times 1.75 \times 96 = 56.4 \text{ GB per sequence}$$

| link | time to move 56.4 GB |
| --- | --- |
| NVLink 4, inside an H100 node | 0.125 s |
| NVLink 5, inside a B200 node | 0.063 s |
| InfiniBand, each GPU in a different node | 1.13 s |

That figure applies to each sequence. A data-parallel copy that processes 32 sequences in one step moves
32 times as much, about 1,805 GB. Tensor parallelism is kept inside a single node because its traffic
grows with every token processed. Megatron-LM's authors measured this on servers with eight network cards
each, and their guidance is to use tensor parallelism up to the number of GPUs in one server. PyTorch's
asynchronous tensor parallelism reduces the waiting by running part of this communication during the
matrix multiplications. In torchtitan's measurements on 256 GPUs, it raised throughput for Llama 3.1 70B
by 12.59%.

> **Carry this forward:** tensor parallelism divides every layer's weights by the degree, and its traffic
> of 56.4 GB per sequence confines it to NVLink.

*Widget 1 (`widget_1_tensor_split.html`): Tensor parallelism. One feed-forward block split across GPUs,
with the first matrix divided by columns and the second by rows. Raising the degree adds traffic per
sequence, and the link toggle shows how long that traffic takes on NVLink 4, NVLink 5 and InfiniBand.*

## 4. Sequence Parallelism

Tensor parallelism splits the large matrices, and some small operations in each layer still run on the
full sequence on every GPU. Layer normalization, which rescales each token's values, is one of them.
Dropout, which sets random values to zero during training, is another. The activations those operations
store are therefore held in full on every GPU in the group.

Sequence parallelism splits those operations along the sequence. Each GPU keeps its share of the tokens
for them, which divides the stored activations by the tensor-parallel degree.

**The detail.** Megatron-LM added this in 2022. Each all-reduce is replaced by an all-gather before the
attention or feed-forward block and a reduce-scatter after it. A ring all-reduce is a reduce-scatter
followed by an all-gather, so the pair carries about the same volume as the all-reduce it replaces.

| TP degree | training state | activations, one 8,192-token sequence | total per GPU |
| --- | --- | --- | --- |
| 2 | 225.0 GiB | 63.8 GiB | 288.8 GiB |
| 4 | 112.5 GiB | 31.9 GiB | 144.4 GiB |
| 8 | 56.25 GiB | 15.9 GiB | 72.2 GiB |

At TP = 8 the total is 72.2 GiB against an H100 card of 74.5 GiB. The remaining 2.3 GiB must also hold
communication buffers and the memory the framework reserves, which is too little in practice. A B200 card
of 167.6 GiB holds the same total with 95.4 GiB to spare. It also holds the TP = 4 total of 144.4 GiB,
with 23.2 GiB to spare. Section 9 shows how the optimizer's share of the state is reduced further.

The name "sequence parallelism" is also used elsewhere for splitting a long sequence across GPUs for
attention. This session calls that technique context parallelism and covers it in Section 7.

> **Carry this forward:** sequence parallelism divides the stored activations by the tensor-parallel
> degree at the same communication volume.

## 5. Pipeline Parallelism

A model can also be divided by layers. Pipeline parallelism places the first group of layers on one GPU,
the next group on the next GPU, and so on. Each GPU is called a stage. A sequence passes forward through
the stages like an item on an assembly line, and its gradients pass back through them in reverse.

Each stage holds only its own layers, so the training state is divided by the number of stages.

**The detail.** With 96 layers the stages divide evenly at every power of two up to 32.

| PP degree | layers per stage | training state per GPU | activations of one sequence per stage |
| --- | --- | --- | --- |
| 4 | 24 | 112.5 GiB | 31.9 GiB |
| 8 | 12 | 56.25 GiB | 15.9 GiB |
| 16 | 6 | 28.1 GiB | 8.0 GiB |

A stage sends one activation tensor to the next stage for each micro-batch, and receives one gradient
tensor of the same size back.

$$8192 \times 5120 \times 2 \text{ bytes} = 83.9 \text{ MB per micro-batch per boundary}$$

InfiniBand moves 83.9 MB in 1.7 milliseconds. A step at PP = 8 with 32 micro-batches crosses 7 boundaries
in each direction.

$$2 \times 83.9 \text{ MB} \times 7 \times 32 = 37.6 \text{ GB per step}$$

That traffic travels between neighboring stages only. Pipeline parallelism can span nodes because each
send is small and goes to a single neighbor.

> **Carry this forward:** pipeline parallelism divides the model by layers, and its 83.9 MB sends are
> small enough to cross InfiniBand.

## 6. Pipeline Schedules

A pipeline has a waiting problem. When the first micro-batch enters stage 0, every later stage has
nothing to do until that micro-batch reaches it. The same happens in reverse at the end of the step. The
time a stage spends waiting is called the bubble.

The order in which stages process forward and backward passes is called the schedule, and it decides how
much memory each stage needs. Adding micro-batches shrinks the bubble.

**The detail.** Take $p$ stages and $m$ micro-batches. The first micro-batch needs $p - 1$ time slots to
reach the last stage, so every stage waits $p - 1$ slots and works $m$ slots.

$$\text{share of the step spent waiting} = \frac{p - 1}{m + p - 1}$$

| stages | 8 micro-batches | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| 4 | 27.3% | 15.8% | 8.6% | 4.5% |
| 8 | 46.7% | 30.4% | 17.9% | 9.9% |
| 16 | 65.2% | 48.4% | 31.9% | 19.0% |

Published papers usually write the bubble as $(p - 1)/m$, which divides waiting time by working time. At
8 stages and 32 micro-batches that form gives 21.9%, and the share of the step gives 17.9%. The two
describe the same run.

Three schedules are in common use.

- **All forward, then all backward.** Every micro-batch completes its forward pass before any backward
  pass begins. Stage 0 must therefore hold the activations of all $m$ micro-batches at once.
- **1F1B.** After a short warm-up, each stage alternates one forward pass with one backward pass. Stage 0
  then holds at most $p$ micro-batches at a time. The bubble is unchanged.
- **Interleaved 1F1B.** Each GPU holds $v$ smaller chunks of layers from different parts of the model. The
  bubble shrinks to $(p - 1)/(v \cdot m + p - 1)$, and the number of sends grows $v$ times. Stage 0 also
  holds more activations.

**Zero-bubble schedules.** A team at Sea AI split each backward pass into two parts, the gradient for the
layer's input and the gradient for its weights. The weight part can wait, so it fills slots that would
otherwise sit idle. Their ZB-H1 schedule cuts the bubble to a third of 1F1B's at the same peak memory.
Their ZB-H2 schedule removes the bubble, holds up to $2p - 1$ micro-batches at stage 0, and checks each
optimizer step so that a bad step can be rolled back.

| schedule, 8 stages, 32 micro-batches | micro-batches held at stage 0 | activation memory at stage 0 | waiting share |
| --- | --- | --- | --- |
| all forward, then all backward | 32 | 510.0 GiB | 17.9% |
| 1F1B | 8 | 127.5 GiB | 17.9% |
| interleaved 1F1B, 2 chunks per GPU | more than 8 | above 127.5 GiB | 9.9% |

> **Carry this forward:** more micro-batches shrink the bubble, and 1F1B caps each stage's activations at
> the number of stages.

*Widget 2 (`widget_2_pipeline_schedule.html`): Pipeline schedules. A grid of stages against time slots,
filled by the chosen schedule and counted cell by cell. Switching the schedule changes the memory held at
stage 0 and leaves the waiting share unchanged.*

## 7. Context Parallelism

Some training sequences are very long. A sequence of 131,072 tokens produces sixteen times the
activations of an 8,192-token sequence, which is far more than any GPU holds.

Context parallelism divides the tokens of one sequence across GPUs. Most of a layer processes each token
independently, so each GPU can handle its own share. Attention is the difficult part, because each token
must look at earlier tokens that live on other GPUs.

**The detail.** The activation figure from Section 1 scales directly with sequence length.

$$131072 \times 5120 \times 34 \text{ bytes} \times 96 \text{ layers} = 2040 \text{ GiB}$$

| CP degree | tokens per GPU | activations per GPU | with TP = 8 as well |
| --- | --- | --- | --- |
| 2 | 65,536 | 1,020.0 GiB | 127.5 GiB |
| 4 | 32,768 | 510.0 GiB | 63.8 GiB |
| 8 | 16,384 | 255.0 GiB | 31.9 GiB |
| 16 | 8,192 | 127.5 GiB | 15.9 GiB |

Three methods solve the attention problem.

**Ring Attention** connects the GPUs in a ring. Each GPU computes attention between its own queries and
the block of keys and values it currently holds, then passes that block to its neighbor. After $c - 1$
passes every GPU has seen every block. At CP = 8 each block holds 16,384 tokens.

$$2 \times 8 \text{ KV heads} \times 128 \times 16384 \times 2 \text{ bytes} = 67.1 \text{ MB}$$

Seven passes in each of 96 layers move 45.1 GB per sequence in the forward pass. A pass can run while the
receiving GPU computes attention on the block it already holds. That overlap hides the pass only when
computing attention on one block takes longer than sending the block.

**All-gather context parallelism** is the method Meta used for Llama 3's long sequences. Each GPU keeps
the queries for its own tokens. In every layer the GPUs run one all-gather of the keys and values, so
every GPU holds them for the whole sequence. Each GPU then computes attention between its own queries and
those full keys and values. The Llama 3 report gives two reasons for the choice. A full set of keys and
values makes any attention mask easy to apply, including the mask that keeps packed documents apart. Keys
and values are also small with 8 key-value heads, and attention work grows with the square of the
sequence length, so the all-gather takes a small share of the step. At CP = 8 the all-gather moves the
same 45.1 GB per sequence in the forward pass as the ring.

**Ulysses** uses an all-to-all to regroup the work. Before attention, each GPU holds some of the tokens
for every head. After the all-to-all, each GPU holds every token for some of the query heads. Attention
runs locally for those heads, and a second all-to-all restores the original split. The Ulysses degree
must divide the number of query heads. Our 40 query heads allow degrees of 2, 4, 5, 8, 10, 20 and 40, and
a degree of 16 is impossible. When the degree exceeds our 8 key-value heads, the key-value heads are
copied so that every GPU holds at least one.

Causal attention makes the work uneven when each GPU holds one continuous piece of the sequence. A token
attends only to earlier tokens, so the GPU holding the first piece has the least attention work and the
GPU holding the last piece has the most. At CP = 8 the last GPU does 15 times the attention work of the
first. **Head-tail load balancing** cuts the sequence into $2c$ chunks and gives each GPU one early chunk
and one late chunk, which evens the work across GPUs. Llama 3 trained with this split.

The methods suit different links. Ulysses sends all-to-all traffic, which runs well inside one NVLink
domain. Ring passes go only to a neighbor, which suits the slower links between nodes. A hybrid runs
Ulysses inside each node and a ring across nodes.

> **Carry this forward:** context parallelism divides a long sequence across GPUs, and attention completes
> by passing key-value blocks around a ring, gathering them at once, or regrouping by head.

*Widget 3 (`widget_3_context_parallel.html`): Context parallelism. One sequence of 131,072 tokens divided
across GPUs, shown as a ring, as an all-gather, and as a regroup by query head. The load panel compares
one continuous piece per GPU with the head-tail split.*

## 8. Communication Overhead

Every form of parallelism adds traffic. The forms differ in volume and in the pattern of sending, and the
pattern decides which wires can carry the traffic.

**The detail.** One copy of our parameters in 16-bit format is $30.2 \times 10^9 \times 2 = 60.4$ GB.
Data parallelism all-reduces the gradients once per step.

$$60.4 \text{ GB} \times 1.75 = 105.7 \text{ GB per step, at 8 data-parallel copies}$$

| form | pattern | volume | grows with |
| --- | --- | --- | --- |
| data parallelism, 8 copies | all-reduce of gradients | 105.7 GB per step | parameter count |
| tensor parallelism, TP = 8 | four all-reduces of activations per layer | 56.4 GB per sequence | tokens |
| pipeline parallelism | point-to-point send between neighbors | 83.9 MB per micro-batch per boundary | tokens |
| context parallelism, ring, CP = 8, 131,072 tokens | point-to-point pass of key-value blocks | 45.1 GB per sequence, forward | tokens |

| form | through one NVLink 4 connection | through one NVLink 5 connection | through one InfiniBand card |
| --- | --- | --- | --- |
| data parallelism, per step | 0.235 s | 0.117 s | 2.11 s |
| tensor parallelism, per sequence | 0.125 s | 0.063 s | 1.13 s |
| pipeline parallelism, per send | 0.19 ms | 0.093 ms | 1.7 ms |
| context parallelism, forward pass per sequence | 0.100 s | 0.050 s | 0.90 s |

Those times assume all of the traffic crosses one link. NCCL, the library that runs these collectives,
spreads traffic between nodes across all eight network cards in a node and chooses among several
algorithms, so real times between nodes are shorter.

All four can overlap with computation to some degree. Data-parallel gradients are sent during the
backward pass as each layer finishes. Pipeline sends happen between stages. Ring passes can run during
attention on the current block. In the backward pass, the tensor-parallel all-reduce runs while the GPU
computes the weight gradients. In the forward pass, asynchronous tensor parallelism splits the
communication into pieces that run alongside the matrix multiplications. The context-parallel figures
cover the forward pass, and the backward pass sends key-value gradients as well.

> **Carry this forward:** tensor parallelism sends traffic inside every layer for every sequence, which is
> why it needs the fastest link.

## 9. Topology-Aware Placement

A cluster has fast wiring inside each node and slower cables between nodes. Topology-aware placement
assigns each form of parallelism to a link fast enough to carry its traffic.

The forms are nested from the most demanding to the least. Tensor parallelism is innermost, followed by
context parallelism, then pipeline parallelism, then data parallelism on the outside.

**The detail.** Take 64 GPUs in 8 nodes. Tensor parallelism at degree 8 fills each node over NVLink. Data
parallelism at degree 8 spans the nodes over InfiniBand.

| per GPU | GiB |
| --- | --- |
| training state at TP = 8 | 56.25 |
| activations at TP = 8 with sequence parallelism | 15.9 |
| total | 72.2 |

That total leaves too little headroom on an H100 card of 74.5 GiB. The 12 bytes per weight for the master
copy and the two running averages can also be divided across the 8 data-parallel copies, because each
copy needs only its own share to perform the update. The training state then falls to 19.3 GiB.

$$19.3 + 15.9 = 35.3 \text{ GiB per GPU}$$

Megatron-LM's guidance is to use tensor parallelism up to the number of GPUs in one server, and to add
pipeline parallelism across servers. Our model's 8 key-value heads also keep tensor parallelism at 8 or
below, because a larger degree would need copies of those heads.

Most clusters connect their nodes with a **rail-optimized network**. Each node has one network card per
GPU. Network card 0 of every node connects to one switch, network card 1 of every node connects to the
next switch, and so on. Each of those paths is called a rail. Pipeline and data-parallel partners usually
sit at the same position in different nodes, so their traffic stays on a single rail.

A B200 node has the same eight GPUs and eight network cards, so the same placement applies. Its NVLink 5
connection carries 18 times the traffic of one network card, which is twice the ratio inside an H100
node. Its larger cards also allow tensor parallelism at degree 4, because the Section 4 total at TP = 4 is
144.4 GiB. Section 11 describes a run on hardware where NVLink was only about three times faster than
InfiniBand, and what that ratio did to its layout.

> **Carry this forward:** tensor parallelism stays on NVLink inside each node, and a rail-optimized network
> carries the traffic between nodes.

*Widget 4 (`widget_4_placement.html`): Topology-aware placement. 64 GPUs in 8 nodes, with GPU k of every
node wired to rail switch k. The node toggle switches between H100 nodes on NVLink 4 and B200 nodes on
NVLink 5. The tensor-parallel view draws each group as an outline, and a degree of 16 makes a group cross
the network. The data-parallel view lights the partners of GPU 0 and the rails their traffic uses.*

## 10. Production Layouts

Large runs combine every form of parallelism at once. Meta published the layout used to train Llama 3
405B, which shows the nesting from Section 9 at a scale of sixteen thousand GPUs.

MFU, or model FLOPs utilization, is the fraction of the GPUs' arithmetic capacity that the run actually
used.

**The detail.** The run used up to 16,384 H100 GPUs connected by RoCE, an Ethernet network at 400 Gb/s.

| GPUs | TP | CP | PP | DP | sequence length | MFU |
| --- | --- | --- | --- | --- | --- | --- |
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 38% |

Each row multiplies to its GPU count, and the last row is $8 \times 16 \times 16 \times 8 = 16384$. The
dimensions nest as tensor, context, pipeline, then data parallelism, with tensor parallelism kept inside
one server.

The two rows at 16,384 GPUs compare the two sequence lengths. The 131,072-token configuration, with
16-way context parallelism, ran at 38% MFU against 41%. Data parallelism also fell from 128 to 8 between
those rows. The long-sequence row used all-gather context parallelism, with each sequence cut into 32
chunks for its 16 context-parallel GPUs.

> **Carry this forward:** a production run nests tensor, context, pipeline and data parallelism in that
> order, and long-context training costs a few points of utilization.

## 11. DeepSeek-V3 (SELF STUDY)

DeepSeek-V3 is a mixture-of-experts model. Each layer contains many small feed-forward networks called
experts, and each token uses only a few of them. Spreading the experts across GPUs is called expert
parallelism.

V3 is notable for one layout decision. It trained without tensor parallelism, and the reason is its
hardware.

**The detail.** V3 has 671B parameters, of which 37B are active for each token. It trained on 2,048 H800
GPUs with 8 in each node. Its report gives NVLink at about 160 GB per second and InfiniBand at 50, a ratio
of about 3.2.

| dimension | degree |
| --- | --- |
| pipeline parallelism | 16 |
| expert parallelism | 64, across 8 nodes |
| data parallelism | with optimizer state divided across copies |
| tensor parallelism | none |

The report describes tensor parallelism as costly and states that memory optimizations made it
unnecessary. DeepSeek's companion paper on hardware links the decision to the H800's reduced NVLink
bandwidth.

**Node-limited routing** caps each token at 4 nodes. InfiniBand carries the token to the GPU at the same
position inside each of those nodes, and NVLink carries it on to the experts inside that node. A token
reaches an average of 3.2 experts per node, so this arrangement supports up to 13 experts per token at
the communication cost of the 8 that V3 uses.

**DualPipe** is V3's pipeline schedule. Micro-batches enter the pipeline from both ends at once, and
expert and pipeline communication runs while the GPU computes. DualPipe keeps two copies of the model's
parameters so that every GPU can run its part of the model for both directions. It splits each backward
pass into two parts: the gradient for the layer's input, which the previous stage needs soon, and the
gradient for the layer's weights, which can wait.

| schedule | bubble | parameters per device | activations per device |
| --- | --- | --- | --- |
| 1F1B | $(PP - 1)(F + B)$ | 1x | PP |
| ZB1P, a zero-bubble 1F1B | $(PP - 1)(F + B - 2W)$ | 1x | PP |
| DualPipe | $(PP/2 - 1)(F\&B + B - 3W)$ | 2x | PP + 1 |

Here $F$ is forward time, $B$ is backward time, $W$ is the time for the weight-gradient part, and $F\&B$
is the time when a forward and backward pass overlap. A variant called DualPipeV gives the same bubble on
half the devices.

V3 computed the forward and both backward matrix multiplications of its linear layers in 8-bit floating
point. The embedding, output head, the gating that picks each token's experts, normalization and
attention stayed at higher precision. Activations were scaled in tiles of 1 by 128 values and weights in
blocks of 128 by 128. Training took $2664\text{K} + 119\text{K} + 5\text{K} = 2788\text{K}$ H800 GPU
hours.

Kimi K2, a model of over one trillion parameters, used 16-way pipeline parallelism with interleaved 1F1B,
16-way expert parallelism, and data parallelism with optimizer state divided across copies. Its report
states that DualPipe was rejected because it doubles the memory for parameters and gradients.

> **Carry this forward:** when NVLink is only about three times faster than InfiniBand, a large run can
> drop tensor parallelism and rely on pipeline and expert parallelism with communication overlapped onto
> computation.

*Widget 5 (`widget_5_dualpipe.html`): DualPipe. One pipeline fed from one end beside one fed from both
ends. In the second pipeline, each GPU holds two stages, one for each direction. Step through the slots
and count how long each pipeline takes to give every GPU work.*

## 12. DeepSeek-V4 (SELF STUDY)

DeepSeek-V4 comes in two sizes, Pro and Flash. Its report describes several parallelism problems created
by the model's design, together with the solution to each. The report gives no cluster size, GPU type,
GPU hours or parallelism degrees.

**The detail.**

| | V4-Pro | V4-Flash |
| --- | --- | --- |
| total parameters | 1.6T | 284B |
| active per token | 49B | 13B |
| layers | 61 | 43 |
| hidden size | 7,168 | 4,096 |
| training tokens | 33T | 32T |

V4 removed V3's limit on how many nodes a token can reach, and the report states that the parallelism
strategy was redesigned to match. The accompanying communication kernel runs expert traffic during
computation, and DeepSeek measured it at 1.50 to 1.73 times faster for general inference and up to 1.96
times for latency-sensitive work. Those figures come from inference and reinforcement-learning rollout.

**A hybrid division of optimizer state for Muon.** Muon is an optimizer that computes each update from a
whole weight matrix. Dividing a matrix into pieces across GPUs, which is how optimizer state is usually
divided, would break that requirement. V4 assigns each dense matrix whole to one GPU, using a balancing
algorithm so that every GPU receives a similar load, and it caps the size of each group. The report puts
the extra memory needed to even out the groups at under 10%. Expert weights are flattened and divided
evenly across every GPU, since each expert is updated on its own. Gradients are rounded to 16-bit format
and summed locally in 32-bit format after an all-to-all.

**Two-stage context parallelism.** V4 uses compressed sparse attention, in which each token attends to a
selected set of compressed blocks of keys and values. It groups keys and values into blocks of $m$
consecutive entries, and each compressed entry reads its own block together with the $m$ entries just
before it. Dividing a sequence across GPUs, with $s$ entries on each GPU, creates two problems.

The first problem is uneven length. Each training sample is packed from several sequences, each sequence
is compressed on its own, and any final group shorter than $m$ is dropped. The compressed length on each
GPU is therefore usually below $s/m$, and it differs from one GPU to the next.

The second problem is the boundary. The entries that a compression step needs can straddle the boundary
between two neighboring GPUs.

1. **Stage one.** Each GPU sends its last $m$ uncompressed entries to the next GPU. The receiving GPU
   compresses them together with its own $s$ entries and pads its output to a fixed length of $s/m + 1$
   compressed entries.
2. **Stage two.** An all-gather collects the compressed entries from every GPU, and a fused operation
   arranges them into the full set of $cp \times s/m$ entries with the padding at the end.

The same two stages serve V4's other attention layers, which compress more heavily.

V4 also widens its residual stream, the vector that carries each token from one layer to the next, into
four parallel copies, a design called manifold-constrained hyper-connections. That widening adds pipeline
traffic. Fused kernels, selective recomputation and an adjusted overlapped 1F1B schedule together hold
the extra cost to 6.7% of a stage's time.

> **Carry this forward:** V4 shows that architecture choices create parallelism problems, including an
> optimizer that needs whole matrices and a compression that needs whole blocks.

*Widget 6 (`widget_6_two_stage_cp.html`): Two-stage context parallelism. Four GPUs holding key-value
entries, with compression blocks drawn over them at small illustrative sizes. Stage one hands the
boundary entries across, and stage two gathers the compressed entries into one ordered set.*

## 13. DeepSeek-V4.1-Flash (SELF STUDY)

DeepSeek released V4.1-Flash on 10 September 2026. Its upper 20 layers compute their keys and values from
the output of a single lower layer, and many of its attention layers share keys, values and index results
with later layers. Both designs save a large amount of memory when the model is serving requests.

Sharing between layers creates a problem during training. A pipeline cut can place the layer that
produces the shared state on one stage and the layers that read it on another stage.

**The detail.** V4.1-Flash has 552B parameters in its main network and a further 196B in lookup tables.
It activates 8B parameters per token while reading a prompt and 16B while generating. Its published
configuration lists 40 layers, a hidden size of 5,120, and 384 routed experts plus one shared expert, with
6 routed experts active per token.

The design is called a causal encoder-decoder and is derived from an earlier design called YoCo. Every
layer is causal, so each token attends only to earlier tokens. The upper 20 layers project their keys and
values from the output of layer 20, each with its own projection weights.

The report describes three measures that make this train across pipeline stages.

- **Shadow indexers.** The indexer is the small module that picks which compressed blocks each token
  attends to. Each stage that needs the shared attention components keeps a lightweight executable copy,
  and one stage owns the parameters.
- **Shared state on the existing sends.** The shared tensors travel on the point-to-point path between
  stages, divided in the same way as the context-parallel split.
- **Lifetime tracking per micro-batch.** Each shared tensor stays in memory until its last reader
  finishes, across the forward pass, recomputation and the backward pass.

The lookup tables are called Engram. They are tables looked up by short sequences of up to four tokens
long, divided into two modules. Their tables and projections are stored in 8-bit floating point. The two
modules sit at layers 1 and 14, counting from zero, and the report states that this placement balances
memory across the training pipeline stages.

V4.1-Flash trained on 45T tokens with the batch held at 100.6 million tokens. Its sparse attention trained
from the start at 64K tokens, and the sequence length was extended to one million at 34T tokens. The
report gives no cluster size, GPU hours or parallelism degrees.

> **Carry this forward:** sharing state between layers turns a pipeline cut into a question of which stage
> owns each tensor and how long each tensor must stay in memory.

*Widget 7 (`widget_7_pipeline_cut.html`): Shared state across pipeline stages. Forty layers drawn as a
column, with the output of layer 20 feeding the keys and values of layers 21 to 40. The stage control
redraws the cuts and marks every link that crosses a stage.*

## 14. Choosing a Layout

Every form covered so far divides something. A run picks one degree for each of them, and the degrees
multiply to the number of GPUs.

$$\text{GPUs} = TP \times CP \times PP \times DP$$

The degrees are chosen in a fixed order, because the memory that one form saves decides whether the next
form is needed at all.

**The detail.** The order below runs from the cheapest decision to the most expensive.

1. **Count the two numbers.** For our model that is a training state of 450.0 GiB and 127.5 GiB of
   activations for one sequence of 8,192 tokens.
2. **Set tensor parallelism to the number of GPUs in one node.** That is 8, and our 8 key-value heads cap
   it at 8 as well. Sequence parallelism runs at the same degree. A GPU now holds 56.25 GiB of state and
   15.9 GiB of activations, which is the 72.2 GiB from Section 4.
3. **Divide the optimizer state across the data-parallel copies.** The 12 bytes per weight for the master
   copy and the two running averages are needed on one GPU only during the update. At 8 copies the state
   falls to 19.3 GiB, and a GPU holds 35.3 GiB. This step adds no communication, so it is always taken
   when there is more than one copy.
4. **Add context parallelism when one sequence stops fitting.** The activations grow with the sequence
   length, and at 131,072 tokens they reach 2,040 GiB for one sequence. Context parallelism is the only
   form that divides a single sample.
5. **Add pipeline parallelism when the state still does not fit.** Pipeline stages divide the state by the
   number of stages. Our model at 35.3 GiB per GPU does not need them, and a 400B model on the same
   hardware would.
6. **Give the remaining GPUs to data parallelism.** With 64 GPUs and tensor parallelism at 8, that leaves
   8 copies.
7. **Check the global batch.** The global batch is the data-parallel degree multiplied by the micro-batch
   and the number of accumulation steps. The token budget for the run fixes the global batch, so a large
   data-parallel degree forces a small micro-batch.
8. **Check the bubble if there is a pipeline.** The waiting share from Section 6 is $(p-1)/(m+p-1)$, so 8
   stages need about 32 micro-batches to hold it near 18%.
9. **Measure the step time.** The arithmetic produces candidate layouts, and the measured throughput
   chooses between them. Section 10 gives two layouts on the same hardware that reached 43% and 38% of the
   GPUs' arithmetic capacity.

Take one node of 8 B200 GPUs, where each card holds 167.6 GiB. Tensor parallelism is 8, so there are no
data-parallel copies, and steps 3 and 6 return nothing. Context parallelism would have to share the same 8
GPUs with tensor parallelism, which raises the state per GPU by as much as it lowers the activations, so
step 4 returns nothing as well. The sequence length alone decides.

| sequence length | activations per GPU | state per GPU | total | fits 167.6 GiB |
| --- | --- | --- | --- | --- |
| 8,192 | 15.9 GiB | 56.25 GiB | 72.2 GiB | yes |
| 16,384 | 31.9 GiB | 56.25 GiB | 88.1 GiB | yes |
| 32,768 | 63.8 GiB | 56.25 GiB | 120.0 GiB | yes |
| 65,536 | 127.5 GiB | 56.25 GiB | 183.8 GiB | no |
| 131,072 | 255.0 GiB | 56.25 GiB | 311.3 GiB | no |

One node trains our model up to 32,768 tokens. Longer sequences need more nodes, which is what makes
steps 3, 4 and 6 useful. Two nodes hold a 65,536-token sequence at 162.7 GiB per GPU. A 131,072-token
sequence needs 4 nodes with context parallelism at 2, which puts 162.7 GiB on each GPU.

> **Carry this forward:** the layout is decided by memory in a fixed order, tensor parallelism first and
> data parallelism last, and the measured step time settles what is left.

*Widget 8 (`widget_8_layout_choice.html`): Choosing a layout. One 30.2B model on nodes of eight B200 GPUs.
Pick a sequence length and a node count to see the degrees the rules produce, the memory on each card,
and which forms are switched on.*

## 15. V5 Decisions

Three things are settled by this session. The model uses 96 layers, so pipeline stages divide it evenly.
Tensor parallelism stays inside a node, because it communicates inside every layer and Megatron-LM's
measurements found it best kept within one server. Sequence parallelism always accompanies tensor
parallelism, because the activation budget in Section 4 depends on it.

Four questions remain open.

| Question | What would settle it |
| --- | --- |
| TP 8 with data parallelism across nodes, or a pipeline added as well? | A measured step time for both on our cluster, with activations and communication buffers included. |
| Which context parallelism for the long-context stage? | The target sequence length, the attention masks the data needs, and the Ulysses degrees available once TP = 8 leaves 5 query heads on each GPU. |
| Which pipeline schedule, if a pipeline is used? | The activation memory of 1F1B against the smaller bubble of the interleaved schedule, measured at our micro-batch count. |
| TP 4 or TP 8 on B200 nodes? | A measured step time for both, since the TP = 4 total of 144.4 GiB fits a B200 card of 167.6 GiB. |

## 16. Reversibility

Every section so far has divided the activations across more GPUs. A reversible network removes most of
them instead. The forward pass keeps the state that enters the stack and the state that leaves it, and
the backward pass rebuilds every layer's activations as it needs them, by running the layer's update in
reverse.

The memory this saves does not depend on the number of layers. A 96-layer model and a 20-layer model
store the same two boundary states.

**The detail.** The method used here was published in November 2025 by Gal, Eliasof, Turek, Ascher,
Treister and Haber, as *Reversing Large Language Models for Efficient Training and Fine-Tuning*. It treats
the layer index as a step in time and advances the residual stream with a rule that can be run in both
directions. Write $p^{\ell}$ for the residual stream entering layer $\ell$, and $f_{\theta_\ell}$ for the
whole transformer block, attention followed by the feed-forward network. The midpoint rule is

$$p^{\ell+1} = p^{\ell-1} + 2h \cdot f_{\theta_\ell}(p^{\ell})$$

where $h$ is a fixed step size. Reading that line from right to left recovers the state two layers back.

$$p^{\ell-1} = p^{\ell+1} - 2h \cdot f_{\theta_\ell}(p^{\ell})$$

The backward pass therefore walks down the stack, and at every step it reconstructs the input it needs
from the output it already holds. Nothing in between has to be kept.

An ordinary block adds its output to its own input, as $p^{\ell} = p^{\ell-1} + f_{\theta_\ell}
(p^{\ell-1})$. That rule cannot be reversed, because recovering $p^{\ell-1}$ would need
$f_{\theta_\ell}(p^{\ell-1})$, which is computed from the state being recovered. The reversible rule adds
the block's output to the state two layers back, and it evaluates the block at the state in between,
which is the state the backward pass already holds.

This is not the same as recomputation, which Section 1 counted at 34 bytes per token per unit of hidden
size. Recomputation stores the input of every group of layers and runs those layers forward again during
the backward pass. A reversible stack stores no layer inputs at all, and it rebuilds them by running the
rule backward.

The cost is compute. Each layer's block runs once in the forward pass and once more during the backward
pass. The authors estimate the overhead at 30% to 50% of a step, on the reasoning that a backward pass
already costs more than a forward pass, so one extra forward evaluation is a small addition. They measured
throughput at a hidden size of 512 with 8 attention heads, and at 96 layers the reversible model reached
114.49 samples per second against 56.96 for the standard model. That gain comes from the batch size the
saved memory allows, which was 874 against 52.

Two constraints come with the method.

- **The forward pass must be deterministic.** The backward pass reconstructs the activations by computing
  the same block again, and a random dropout mask would make the reconstruction differ from what the
  forward pass used. The Lightning LM report treats dropout of zero as a correctness requirement for this
  reason.
- **The rule is only marginally stable.** The paper's analysis requires the step size and the blend
  coefficient to sit in a narrow range, and it reports that reversible networks can be difficult to train
  outside it.

The paper's own memory result is a batch-size measurement. On an 80 GB H100 its standard model fitted a
batch of 26 and its reversible model fitted 257, which is a factor of 9.88, and the same ratio of about 10
held on five different cards.

> **Carry this forward:** a reversible stack stores the boundary states and rebuilds the rest during the
> backward pass, which makes activation memory independent of depth at the cost of about a third more
> compute per step.

*Widget 9 (`widget_9_reversibility.html`): Reversibility against the seven forms. One 30.2B model on one
node of eight B200 GPUs. Switch between stored activations and reversibility at three sequence lengths,
and read what each form of parallelism is doing in that state.*

## 17. What Reversibility Changes

Reversibility does not make memory free. It moves the constraint. The activations stop being the term that
decides the layout, and the weights and the optimizer state become that term instead. Those are the things
ZeRO divides, so the decisions in Section 14 change order.

**The detail.** Apply the rule to our own model. A reversible stack stores the state that enters the stack
and the state that leaves it, which is $2 \times \text{tokens} \times 5120 \times 2$ bytes, and it needs
the working memory of the one layer it is currently rebuilding, which is the 1.33 GiB per layer from
Section 1.

| sequence | stored activations | reversible activations |
| --- | --- | --- |
| 8,192 | 127.5 GiB | 1.5 GiB |
| 32,768 | 510.0 GiB | 5.9 GiB |
| 131,072 | 2,040.0 GiB | 23.8 GiB |

The 96 in the calculation disappears, and what remains grows only with the number of tokens.

Take the single node from Section 14, eight B200 GPUs with tensor parallelism at 8. The state per GPU is
56.25 GiB in both columns.

| sequence | stored, total per GPU | reversible, total per GPU |
| --- | --- | --- |
| 8,192 | 72.2 GiB | 56.4 GiB |
| 32,768 | 120.0 GiB | 57.0 GiB |
| 131,072 | 311.3 GiB | 59.2 GiB |

One node stops at 32,768 tokens when the activations are stored. The same node holds 131,072 tokens when
they are rebuilt, and the card still has 108 GiB free.

Each form of parallelism is affected differently.

- **Tensor parallelism is unchanged.** It divides the weight matrices, and the weights are still there.
- **Sequence parallelism now divides the boundary states**, which are small. Its original job was the
  stored activations.
- **Pipeline parallelism loses one of its two reasons.** It divided the state by layer and it divided the
  activations by layer, and the second reason is gone.
- **Context parallelism is needed at much longer sequences.** The length at which one sequence stops
  fitting moves out by about two orders of magnitude.
- **The ZeRO stages become more important.** They divide the weights, the gradients and the optimizer
  state, which is what now fills the card.

One interaction needs care. Under ZeRO-3 a GPU holds only a slice of each layer's weights, and the
backward pass gathers them once to compute the gradients. A reversible backward pass reads those weights a
second time, when it rebuilds the activations, so the gathering has to happen there as well. The Lightning
LM report states the principle directly: the recompute path is a second forward pass, and every parameter
that the forward pass gathers has to be gathered again for it.

Lightning LM is the evidence that this works at scale. It trained a 120B sparse mixture-of-experts model
with 20 layers, a hidden size of 4,096 and a sequence length of 8,192, on a single node of eight GPUs, with
a measured peak of about 256 GB per GPU. Its report gives 130.9 GB per GPU for the 9B model at a sequence
length of 8,192, and estimates the same run at 550 to 650 GB per GPU with stored activations. On eight 40
GB A100s, its 2B model at 4,096 tokens trained with a batch of 32 under reversibility, and a batch above 8
did not fit without it. Its integrator ran with a step size of 0.25 and a blend coefficient of 0.5, and the
report states that these values have to be set explicitly because the library defaults differ.

Two limits are worth stating plainly. The comparison numbers above 40 GB are estimates of what stored
activations would have cost, because those runs were never made. Reversibility also slows a run down
whenever memory is not the binding constraint, and the Lightning LM report gives its own 2B model on 80 GB
cards as the case where the standard path was faster.

> **Carry this forward:** reversibility trades compute for memory that does not grow with depth, which
> moves the binding term from activations to the training state and changes which form of parallelism is
> needed first.

## 18. Assignment

> Train a 20M LLM for 50M tokens on Google Colab (or anything else of your choice). Fix Batch size that
> you can run. Train again with Reversibility (report which variant worked for you, mid-point, euler, etc)
> Train again with Reversibility, but push it to the maximum batch size.
>
> Report final loss, speed (token/s), memory peak and other findings.
