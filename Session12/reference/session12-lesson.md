# Session 12: Distributed Training I, Data Parallel and ZeRO

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms12nq9x5m0t7a4yr8gh/lesson
Captured: 2026-09-18

Widgets (8) saved under `widgets/`, inline scripts collapsed into `widgets-scripts-dump.txt`.
Videos embedded on the page: https://youtu.be/LvovhEZUQ70 and https://youtu.be/OS9qBfyNB2Q

## 1. Introduction

A single graphics card has a fixed amount of memory. The largest cards available to us hold 80 GB.
Our model has about 30 billion numbers in it, and every one of those numbers has to be stored
somewhere while the model is being trained. This session is about what happens when the thing you
are training is larger than the machine you are training it on.

Start with the arithmetic, because it settles the question quickly. Each weight in the model carries
more than just itself. It carries a record of how it should change, a high-accuracy copy of itself,
and two running averages that the training procedure keeps between steps.

| What is stored for one weight | Bytes |
|---|---|
| the weight, in the 16-bit format used for arithmetic | 2 |
| its gradient, the number saying how it should change | 2 |
| a 32-bit copy of the weight, kept for accuracy | 4 |
| two running averages the optimizer keeps | 8 |
| **total** | **16** |

**The detail.** The two running averages are the optimizer's memory. One tracks the average of the
recent gradients for that weight, and the other tracks the average of their squares. Both are held
in 32-bit floating point, so they cost four bytes each. The 32-bit copy of the weight exists because
repeatedly adding very small updates to a 16-bit number loses them to rounding, so the authoritative
value is kept at higher precision and a 16-bit version is made from it for the arithmetic.

Multiply that out for V5, which is planned at 27 to 30 billion parameters. Every calculation in this
session uses 30 billion.

```
30 × 10^9 weights × 16 bytes = 480 GB = 447.0 GiB
```

That is six 80 GB cards to hold the model, before a single calculation is performed. Activations,
which are the intermediate results the forward pass produces, are on top of that. So the model must
be spread across many cards, and this session covers how the work is divided and what the division
costs.

> **Carry this forward:** training one weight costs 16 bytes, and 30 billion weights therefore cost
> 447 GiB before any work begins.

## 2. Terminology

Seven words are used throughout the session, and each is given here with a number attached.

A **GPU** is one graphics card, and it is the unit of compute we buy. In this session every GPU has
80 GB of memory, which is 74.5 GiB.

A **node** is one physical machine holding several GPUs, almost always eight.

**World size** is the total number of GPUs taking part in a training run. A run on four nodes has a
world size of 32.

A **process rank** is the index of one GPU within that set, numbered from 0. In a world size of 32
the ranks run from 0 to 31. This word is used elsewhere in mathematics for the number of independent
rows in a matrix, and that meaning is unrelated to this one.

An **interconnect** is the wiring that carries data between GPUs. Inside a node the wiring is called
NVLink and carries roughly 450 GB per second. Between nodes it is a network cable, usually
InfiniBand, and carries roughly 50 GB per second.

A **collective** is an operation that every GPU in the run performs together, at the same time, on
data that each of them holds a piece of.

**P** is the size of one complete copy of the model's parameters, measured in bytes. For a 30 billion
parameter model held in 16-bit format, P is 60 GB. Communication in this session is always measured
in multiples of P, so a cost of 2P means 120 GB crosses the wire for each GPU on every step.

> **Carry this forward:** P is 60 GB for our model, and every communication cost in this session is a
> multiple of it.

## 3. Data Parallelism

The simplest way to use eight GPUs is to give each of them a complete copy of the model and a
different portion of the data. Each GPU reads its own examples, runs them through its own copy, and
produces its own gradients. Those gradients differ from each other, because each GPU saw different
text.

The copies then have to be brought back into agreement. Every GPU sends its gradients to every other
GPU, all of them compute the average, and each applies that same average to its own copy. Because
they started identical and applied an identical update, they remain identical. This arrangement is
called data parallelism.

**The detail.** The averaging step is what makes the arrangement correct. Averaging the gradients
from eight GPUs, each of which processed 32 sequences, produces exactly the gradient that a single
GPU would have produced from all 256 sequences at once. The distributed run is therefore
mathematically identical to a single-GPU run on a batch eight times larger. This is the property that
lets a training recipe move from one GPU to many without changing what the model learns.

The batch the optimizer sees is the product of three numbers.

```
global batch = sequences per GPU × GPUs × accumulation steps
```

Accumulation steps are repeats of the forward and backward pass before the weights are updated, used
when the desired batch is larger than what fits in memory at once.

> **Carry this forward:** data parallelism gives every GPU a full copy of the model and keeps those
> copies identical by averaging the gradients on every step.

*Widget: Data parallelism. Four GPUs holding identical weights and different samples. Step through
split, gradients, average and update. The readout gives the largest difference between any two weight
copies, which stays at 0.000 while averaging is on.*

## 4. Collective Operations

The averaging step above has a name, and there are three operations of its kind that appear
throughout this session. Each one involves every GPU at once.

**All-reduce** combines a value from every GPU and gives the combined result back to all of them.
Averaging gradients is an all-reduce.

**Reduce-scatter** combines the values in the same way and gives each GPU only one slice of the
answer. Eight GPUs each end up holding one eighth of the averaged result.

**All-gather** is the reverse. Each GPU starts with one slice and ends with the complete set,
assembled from everyone's slices.

**The detail.** A reduce-scatter followed by an all-gather produces exactly what an all-reduce
produces. This equivalence is the reason the next two sections work. It also explains the cost. A
ring all-reduce is implemented internally as precisely those two phases, so each GPU sends about one
copy of the data during the first phase and receives about one copy during the second, giving a total
of 2P.

| Operation | Each GPU starts with | Each GPU ends with |
|---|---|---|
| all-reduce | a full set of values | the combined full set |
| reduce-scatter | a full set of values | one slice of the combined set |
| all-gather | one slice | the full set |

> **Carry this forward:** a reduce-scatter followed by an all-gather is an all-reduce, and the pair
> costs the same as the whole.

*Widget: The three collective operations. A four by four grid of GPUs and chunks, stepped through
reduce-scatter and then all-gather. The result panel beside it shows what all-reduce produces, and the
final step matches it exactly.*

*Widget: The ring all-reduce. The same four GPUs arranged in a ring, with each one sending only to its
right neighbour. Step forward and back through all six sends and watch each chunk accumulate one
contribution at a time.*

## 5. The Cost of Communication

Sending data between GPUs takes time, and that time is not free. While a GPU is waiting for numbers
to arrive, it is not computing. The question for any distributed run is whether the waiting is small
compared with the work.

Our model gives P of 60 GB, so data parallelism moves 120 GB per GPU on every step. Wiring inside a
single machine carries that in about a quarter of a second. A network cable between machines carries
the same 120 GB in about two and a half seconds.

| Path | 2P, which is 120 GB | 3P, which is 180 GB |
|---|---|---|
| NVLink, inside one node | 0.27 s | 0.40 s |
| InfiniBand, between nodes | 2.40 s | 3.60 s |

**The detail.** Whether those numbers matter depends entirely on how long the computation itself
takes. A step processing one million tokens on 64 H100 cards takes about 7.1 seconds of compute at a
realistic utilisation. The same step on 64 B200 cards takes about 3.1 seconds, because the cards are
faster.

| | compute per step | 2P over InfiniBand | communication as a fraction of compute |
|---|---|---|---|
| 64 x H100 | 7.10 s | 2.40 s | 34% |
| 64 x B200 | 3.12 s | 2.40 s | 77% |

That last column is communication divided by compute, which is the standard test for whether a run is
limited by its arithmetic or by its wiring. If none of the transfer is hidden, the step takes the sum
of the two, so 9.50 seconds on H100 and 5.52 seconds on B200.

Faster GPUs raise the ratio of communication to compute. The volume stays at 120 GB while the compute
it hides behind gets shorter. This is the most important consequence of buying newer hardware, and it
is why the arrangement of the run matters more to V5 than it did to earlier and smaller runs.

The ratio is still below 1 in both rows, so a transfer that runs entirely during the computation is
still fully hidden. Section 10 covers how that is arranged, and the rise from 34 percent to 77 percent
is what turns that arrangement into a requirement.

> **Carry this forward:** communication cost is judged as a fraction of step time, and that fraction
> grows as the cards get faster.

*Widget: The cost of communication. Compute and network drawn as two bars on an axis pinned from 0 to
10 seconds. The card control moves only the compute bar. The link control moves only the network bar.*

## 6. ZeRO Stages 1, 2 and 3

Look again at what data parallelism stores. Every GPU holds the full 16 bytes for every weight, and
the copies are identical. Eight GPUs therefore hold eight identical copies of 447 GiB of state, which
is 3,576 GiB of memory to store 447 GiB of information.

Most of that is never needed in eight places. During the update, each GPU applies the same average to
the same weights and produces the same answer, so seven of the eight are repeating work already being
done. The idea is to give each GPU responsibility for one slice of the weights, let it keep only the
state for that slice, and have it fetch anything else it needs when it needs it. Every GPU receives a
different slice, so all eight hold the same amount of state and none of them sits idle.

That idea is called ZeRO, which stands for Zero Redundancy Optimizer. It was published by Microsoft in
2019. It describes an arrangement of storage, and Section 9 covers the software that implements it.

**The detail.** ZeRO is applied in three stages, and each stage removes one more class of duplicated
state.

**Stage 1** splits the optimizer state, which is the 32-bit weight copy and the two running averages,
twelve of the sixteen bytes. Each GPU keeps one slice and updates only that slice. The updated slices
are then shared out so every GPU has the current weights again.

**Stage 2** splits the gradients as well. A GPU only ever needs the gradients for the slice of weights
it is responsible for updating, so the others are discarded as soon as they have been sent where they
are needed.

**Stage 3** splits the weights themselves. Each GPU stores one slice of the model. When the forward
pass reaches a layer, the GPUs collect that layer's weights from each other, use them, and discard
them again immediately.

| | bytes per weight, 8 GPUs | 30B model per GPU | communication |
|---|---|---|---|
| data parallelism | 16.00 | 447.0 GiB | 2P |
| ZeRO-1 | 5.50 | 153.7 GiB | 2P |
| ZeRO-2 | 3.75 | 104.8 GiB | 2P |
| ZeRO-3 | 2.00 | 55.9 GiB | 3P |

Stages 1 and 2 return ten of the sixteen bytes at the communication volume data parallelism was
already paying. The reason is the equivalence from Section 4. Data parallelism performs an all-reduce,
which is internally a reduce-scatter and an all-gather. Stages 1 and 2 perform those same two phases
and simply keep the intermediate slice instead of discarding it. Stage 3 adds a further all-gather of
the weights in the forward pass and again in the backward pass, which raises the total from 2P to 3P.

> **Carry this forward:** stages 1 and 2 cost nothing extra in communication, and stage 3 costs half as
> much again.

*Widget: ZeRO stages. Eight GPUs and eight weights, where GPU k owns weight k. Step through the stages
and count the solid cells in one GPU box. The division shown beneath the figure produces the bytes per
weight from that count.*

## 7. The Memory Ladder

Applying the table above to our own model narrows the choice to two arrangements.

| | 8 GPUs | 16 GPUs | 32 GPUs | 64 GPUs |
|---|---|---|---|---|
| data parallelism | 447.0 GiB | 447.0 GiB | 447.0 GiB | 447.0 GiB |
| ZeRO-1 | 153.7 GiB | 132.7 GiB | 122.2 GiB | 117.0 GiB |
| ZeRO-2 | 104.8 GiB | 80.3 GiB | 68.1 GiB | 62.0 GiB |
| ZeRO-3 | 55.9 GiB | 27.9 GiB | 14.0 GiB | 7.0 GiB |

A card holds 74.5 GiB. Data parallelism and ZeRO-1 never fit, at any number of GPUs. Both leave the
weights and the gradients replicated on every card, which is four bytes per weight. Four bytes across
30 billion weights is 111.8 GiB, and that figure is the same on one card as on a thousand. ZeRO-2 fits
from 32 GPUs upward. ZeRO-3 fits from 8.

That floor is set by the model size alone. Four bytes per weight fills a 74.5 GiB card exactly at 20
billion parameters, so a 20B model sits on the boundary and our 30B model sits past it. Dragging the
model size in the widget below crosses that line.

**The detail.** Those figures cover the training state only. Activations are additional, and their
size is set by the batch size and the sequence length. Room has to be left for them, which pushes the
practical threshold above what the table alone suggests. The common technique for controlling
activation memory is to discard intermediate results during the forward pass and recompute them during
the backward pass, at a cost of roughly 30 percent more compute.

> **Carry this forward:** at 30 billion parameters the choice is between ZeRO-2 on many GPUs and ZeRO-3
> on fewer, because the arrangements below them do not fit at all.

*Widget: The memory wall. GiB per GPU against GPU count on log axes, with one line per arrangement and
the 74.5 GiB card drawn as a rule. A line is saturated where it fits. Two of the four never reach it.*

## 8. Offload to CPU and NVMe

A GPU is not the only memory in the machine. A node has system memory, usually far more of it than the
GPUs have, and it has solid-state storage beyond that. State that is needed rarely can be kept in those
slower places and brought in when required.

The optimizer state is the natural candidate, because it is twelve of the sixteen bytes and it is
touched exactly once per step. Holding it in system memory removes it from the GPU entirely.

**The detail.** There are two versions of this. The simpler one parks the state in system memory and
copies it to the GPU for the update. The more effective one performs the update on the CPU as well, so
the state never moves and the GPU is freed of that work altogether. The cost is the link between the
CPU and the GPU, which is PCIe. It carries roughly 60 GB per second, which is far below NVLink and
comparable to a network cable. Offload therefore converts a memory problem into a bandwidth problem. It
earns its place in a run whose binding constraint is GPU memory.

> **Carry this forward:** offload buys GPU memory by spending PCIe bandwidth, and it helps only when
> memory is the binding constraint.

## 9. FSDP2 and DeepSpeed

Sections 6 to 8 described an idea. Two pieces of software implement it, and the choice between them is
a practical one.

**DeepSpeed** is the library Microsoft released alongside the ZeRO paper. It is configured with a
single JSON file that names the stage and the options, and it has the most complete support for offload
to system memory and to storage.

**FSDP2** stands for Fully Sharded Data Parallel, version 2. It is PyTorch's own implementation,
shipped as part of the framework, and it is the recommended path from PyTorch 2.6 onward. It
corresponds to ZeRO stage 3.

**The detail.** FSDP2 is applied by calling `fully_shard()` on parts of the model, which replaces the
older approach of wrapping the whole model in a class. Each parameter is split along its first
dimension and represented as a DTensor, which is a tensor that knows which part of itself lives on
which GPU. That representation is what allows the compiler to work on a sharded model, which the
earlier version did not support.

| | ZeRO stages | offload | integration |
|---|---|---|---|
| DeepSpeed | 1, 2 and 3 | system memory and storage | external library, JSON configuration |
| FSDP2 | stage 3 | system memory | built into PyTorch, composes with the compiler |

> **Carry this forward:** ZeRO is the design, and DeepSpeed and FSDP2 are two implementations of it with
> different strengths.

## 10. Overlapping Communication with Compute

Section 5 measured communication as a share of step time. That share can be reduced without sending
less data, by arranging for the sending to happen while the GPU is busy with something else.

The backward pass makes this possible. It works from the last layer of the model to the first, so the
gradients for the last layer are finished long before the pass reaches the first layer. Those gradients
can begin their journey immediately, while the earlier layers are still being computed.

**The detail.** Gradients are collected into buckets, and a bucket is sent as soon as it fills, part way
through the pass. The bucket size sets a balance.

| Bucket size | Effect |
|---|---|
| smaller | transfers begin earlier in the pass, giving more room to overlap |
| larger | the fixed cost of starting a transfer is paid less often |

Production settings are given in bytes, and a few hundred megabytes is a common choice. The widget below
groups whole layers into each bucket so that twelve of them fit on one screen.

There is a third effect that appears only on fast hardware. Smaller buckets keep helping while the link
drains them as fast as they arrive. Past that point the transfers queue behind each other and the step
time climbs again. On an H100 step the smallest bucket is still the best one. On a B200 step the best
bucket holds two layers, and going below that makes the run slower.

Stage 3 requires the same treatment in the other direction. The weights for the next layer are gathered
while the current layer is still being computed, so the arrival is complete by the time it is needed.

The measurement that tells you whether any of this is working is the share of the transfer time that
finished before the backward pass ended. At a bucket of two layers on an H100 step that share is 83
percent.

> **Carry this forward:** communication that happens during computation costs nothing, and arranging for
> it is a matter of configuration.

*Widget: Overlapping communication with compute. One step on a pinned axis, with twelve layer blocks
above and the transfers they trigger below. Drag the bucket size down and the transfers move under the
computation. The card control shows where that stops helping.*

## 11. Precision and What Blackwell Changes

Blackwell is the current generation of NVIDIA hardware, following the generation the H100 belongs to.
The B200 and GB200 are Blackwell cards. Their arithmetic units can multiply 8-bit floating point numbers
directly in hardware, which the previous generation could not do at the same granularity.

An 8-bit number holds far less detail than a 16-bit one, so it cannot be used naively. The technique
that makes it work is to store a shared scale factor alongside a small block of numbers. MXFP8 is the
Blackwell-native form of this, using blocks of 32 values with one shared 8-bit exponent.

**The detail.** The effect on this session's arithmetic is smaller than it first appears.

| | bytes per weight | 30B model |
|---|---|---|
| 16-bit weights and gradients | 16.00 | 447.0 GiB |
| 8-bit weights and gradients | 14.06 | 392.9 GiB |

Moving the weights and gradients to 8 bits removes two of the sixteen bytes and adds back a small
amount. Each block of 32 values carries its own shared scale byte, which is 0.0625 bytes for every
parameter across the two tensors. The 32-bit copy and the two running averages account for twelve of the
sixteen and are unaffected, because the update arithmetic still needs the accuracy. The reduction in
stored state is 12.1 percent.

The gains from 8-bit arithmetic appear in three other places. Matrix multiplication runs faster,
activation memory falls, and the volume crossing the interconnect falls with it. TorchTitan reports
pre-training up to 41 percent faster for a large mixture-of-experts model on B200 using MXFP8, and loss
curves over 1,500 steps that match 16-bit training.

One part of the model is left at higher precision. The attention softmax amplifies small errors in its
input, because the exponential turns a modest gap between two scores into a very large ratio. Keeping it
in 32-bit costs almost nothing, because softmax is limited by memory bandwidth and not by arithmetic. The
operations held at high precision are the operations where high precision is close to free.

> **Carry this forward:** 8-bit arithmetic reduces the stored state by 12.1 percent, and its real
> contribution is to compute speed, activation memory and communication volume.

*Widget: Precision. Sixteen squares, one per byte, grouped by what sets them. Two toggles change the
matmul format and the optimizer format. The four bars beneath compare the saving in stored state against
the saving everywhere else.*

## 12. What V4 Ran

The previous run, LightningLM v0.1, used DeepSpeed at ZeRO stage 2. The configuration is worth reading
because it records a set of decisions made under real constraints.

| Setting | Value |
|---|---|
| stage | 2 |
| arithmetic format | bf16 |
| sequences per GPU | 2 |
| accumulation steps | 2 |
| global batch | 32, which resolves to 8 GPUs |
| peak learning rate | 3e-4 with 500 warmup steps |
| weight decay | 0.0 |
| gradient clipping | 1.0 |
| bucket sizes | 2e8 bytes |

**The detail.** Three of those entries are the interesting ones. `overlap_comm` was enabled, which is the
technique of Section 10. `round_robin_gradients` was enabled, and it appears in the filename of the
configuration as an out-of-memory fix, which records that the run hit a memory ceiling and was rescued by
changing the order in which gradient buckets were assigned. Weight decay was set to zero, which was a
deliberate choice for that architecture.

A stage 3 configuration exists in the same repository, with offload of both the optimizer state and the
parameters to system memory. It was verified on four 16 GB cards and it never ran the production model.

> **Carry this forward:** the previous run reached stage 2 on eight GPUs and never needed stage 3, and our
> model is large enough that this is no longer the case.

## 13. V5 Decisions

Three things follow from the arithmetic in this session.

Our model does not fit under data parallelism or under ZeRO-1 at any world size, so the starting point is
ZeRO-2 from 32 GPUs or ZeRO-3 from 8. Communication is measured as a fraction of step time and logged from
the first step, because Section 5 shows that fraction rising as the hardware gets faster. Gradient
bucketing and overlap are enabled from the start, since they reduce the fraction without reducing the
volume.

Four questions remain open.

| Question | What would settle it |
|---|---|
| ZeRO-2 on 32 GPUs, or ZeRO-3 on 8? | A measured step time for both on our real architecture, with activation memory included. |
| How many GPUs per node, and how many nodes? | Section 5 shows a nine-fold difference between the two interconnects, so the answer follows from how much traffic can be kept inside a node. |
| Is 8-bit arithmetic committed from the start? | A short run in bf16 and in MXFP8 on the same architecture, comparing loss and step time. This commits us to Blackwell hardware. |
| Does any state go to system memory? | Whether the run is memory-bound or communication-bound once the stage is chosen. |

## 14. Assignment

> Work with your agents and create a simple 32 virtual GPUs (can be your CPU threads or Colab GPU). Then
> write a demo model that runs on top of these. Simulate ZeRO1, ZeRO2, and ZeRO3. Show how the memory and
> computation chanages.
>
> Submit your ipynb notebook, and GitHub Repo link with a detailed README that explains that YOU have
> understood these concepts (and not your agent).
