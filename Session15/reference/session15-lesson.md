# Session 15: Stability, Debugging and Live Monitoring

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms15ns2a8p3w0d7bu1kl/lesson
Captured: 2026-10-26

Widgets (8) saved under `widgets/`.
Videos embedded on the page: https://www.youtube.com/embed/U5wpgmPZ1GU (Studio),
https://www.youtube.com/embed/zWspB4o40zc (GMeet)

## 0. Why this session exists

A large training run lasts for weeks. During that time, three kinds of things can go wrong.

```
                         A TRAINING RUN
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
      THE MODEL           THE MACHINE        THE CODE AND DATA
     loss spikes          a GPU fails       a weight loads wrong
     divergence          a network link      a setting is wrong
     experts die          a job hangs          a batch is bad
          │                    │                    │
          ▼                    ▼                    ▼
    some are loud        most are loud       many are silent
```

> A run must watch itself, because nobody can watch every number for weeks.

This session answers four questions.

- Why does a run become unstable, and how do we prevent it?
- How do we find a problem automatically, and how do we recover from it?
- What changes when some parts of the model are frozen?
- What must a live dashboard show, and how do we build one that stays fast?

## 1. Introduction

This session uses one model for all its numbers. A team trained an open model of exactly this shape
in 2026, and its report gives real data for stability, failures and monitoring.

| part | value |
|---|---|
| layers | 50 |
| hidden size | 2,560 |
| attention | 48 query heads, 4 key-value heads, 128 dimensions each, with QK-norm |
| attention pattern | 4 sliding-window layers, then 1 full layer, repeated |
| experts per layer | 384 routed and 1 shared, each 512 wide |
| experts used per token | 6 routed and the shared expert |
| vocabulary | 128,000 |
| total parameters | 78.1B |
| active parameters | 3.46B |

The team trained the model on 768 B200 GPUs in 96 nodes.

| training fact | value |
|---|---|
| sequence length | 16,384 tokens |
| sequences per step | 4,608 |
| tokens per step | 75.5M |
| steps for 20T tokens | 264,750 |
| time per step | about 6 s |
| pre-training time | 21 days |
| training state, 16 bytes per weight | 1,164 GiB |

The team also recorded every interruption of the run.

| failure fact | value |
|---|---|
| unplanned interruptions | 38 in 377,000 GPU hours |
| one interruption, on average, every | 12.9 hours |
| checkpoint interval | 250 steps, about 25 minutes |
| mean time to recover | 45.3 minutes |
| total time lost | 28.7 hours, 5.6% of the run |

A larger run had more failures. Llama 3 405B trained on 16,384 GPUs and had 466 interruptions in 54
days. Automation handled almost all of them, and only 3 needed a person.

> **Note:** our model has 78.1B weights and uses 3.46B for each token, and on 768 GPUs it stops about
> once every 13 hours.

## 2. Terminology

The **loss** is the number that training makes smaller. The **held-out loss** is the loss on text that
the model does not train on.

The **grad norm** is the size of all the gradients together, measured as one number.

A **loss spike** is a sudden rise in the loss that goes away again. A **divergence** is a rise that
does not go away. The loss stays high, or it becomes NaN.

**NaN** means "not a number". It appears when a calculation divides by zero or overflows. One NaN
spreads to every weight in one step.

A **checkpoint** is a saved copy of the weights and the optimizer state. A **rollback** restarts the
run from an earlier checkpoint.

A **rolling mean** is the mean of the last $N$ values. An **EMA**, or exponential moving average,
is a mean that gives more weight to recent values. A **z-score** tells how many standard deviations a
value is from the mean.

**MaxVio** measures expert imbalance. It is the load of the busiest expert minus the mean load,
divided by the mean load. A value of 0 means that the load is perfectly even.

A **dead expert** receives no tokens.

A **frozen parameter** is a weight that training does not change.

**Silent data corruption** is a wrong result from hardware that reports no error.

A **straggler** is a GPU that works correctly but slowly. A **hang** is a job that stops and reports no
error.

An **Xid** is an error code that an NVIDIA GPU driver writes when the GPU has a fault.

> **Note:** a spike goes away, a divergence stays, and a silent failure gives no sign at all.

## 3. A Healthy Run

A healthy loss curve is not smooth. Each batch has different text, so the loss moves up and down a
little at each step.

```
loss
 │╲
 │ ╲
 │  ╲___
 │      ‾‾‾╲___               small up-and-down noise at each step
 │             ‾‾‾‾╲______
 │                        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 └────────────────────────────────────────────── step
```

> Noise is normal, and a monitor must learn its normal size before it raises an alarm.

**The detail.** Four traces describe a healthy run together.

| trace | healthy behaviour |
|---|---|
| loss | decreases fast, then slowly, with noise |
| grad norm | stays in a narrow band after warmup |
| held-out loss | decreases with the training loss |
| learning rate | follows the schedule exactly |

Some planned events change the training loss. A curriculum changes the mix of data at set points
in training. When the new data is harder, the training loss goes up. This rise is not a problem. The
Lightning LM report from 2026 states the rule: the held-out loss, measured on a fixed set of text,
shows if a curriculum change is working. The training loss alone cannot show this.

> **Note:** watch the held-out loss to tell a planned change from a real problem.

## 4. Loss Spikes and Divergence

A spike and a divergence look the same at the start. The difference shows only later.

```
SPIKE                                  DIVERGENCE
loss                                   loss
 │      ▲                               │      ▲‾‾‾‾‾‾‾‾‾‾‾‾‾‾  stays high, or NaN
 │     ╱ ╲                              │     ╱
 │──────╯   ╲──────── recovers          │──────╯
 └────────────────── step               └────────────────── step
```

> A spike costs a few steps, and a divergence costs the run unless we act.

**The detail.** Spikes are common at large scale.

- **PaLM 540B, 2022.** The loss spiked about 20 times, although the run used gradient clipping. Smaller
  models of the same family did not spike.
- **OPT-175B, 2022.** The team restarted the run by hand at least 35 times in two months. Some restarts
  were for divergence and some were for hardware.
- **OLMo 2, 2025.** The team found that spikes in the grad norm occur more often than spikes in the
  loss. A large grad-norm spike often comes before a loss spike.

PaLM found an important fact about spikes. The team took a batch that caused a spike and gave it to the
model at an earlier checkpoint. The spike did not occur. Thus, a spike needs a specific batch and a
specific state of the model at the same time.

> **Note:** a spike needs both a batch and a model state, so the same data can be safe at one step and
> dangerous at another.

## 5. Why Runs Diverge

Research has found six main causes.

```
1. attention logits grow    → attention looks at only one token → gradients explode
2. output logits drift      → the softmax over the vocabulary becomes unstable
3. the AdamW epsilon is wrong → updates become too small or too large
4. the embeddings are too small → gradients spike in the first layers
5. low precision overflows  → numbers become too large for the format
6. a bad batch meets a weak state → one large update pushes the weights too far
```

> Most divergences start when one group of numbers grows without a limit.

**The detail.**

1. **Attention logits.** An attention logit is $q \cdot k / \sqrt{d}$. When the query and key weights
   grow, the logits grow, and the softmax gives almost all of its weight to one token. Researchers at
   Google first saw this in a 22B vision model in 2023. Kimi K2's team saw the maximum logit go above
   1,000 in a mixture-of-experts model, and the run then had loss spikes. The team that trained our model
   shape removed QK-norm in a small test model, and that model diverged early and did not recover.
2. **Output logits.** The logits of the final layer can drift to very large negative values late in
   training. The z-loss keeps them near zero. It adds $10^{-4} \cdot (\log Z)^2$ to the loss, where $Z$
   is the sum of the exponentials of the logits.
3. **AdamW epsilon.** AdamW divides each update by $\sqrt{v} + \epsilon$. When the epsilon is large
   compared with the gradients, the updates become too small. OLMo 2 changed epsilon from $10^{-5}$ to
   $10^{-8}$, and its grad norm settled faster and stayed lower. Our model uses $10^{-15}$.
4. **Small embeddings.** A study in 2023 found that very small embedding values cause large gradients
   in the layer normalization after them. One fix is to stop weight decay on the embeddings. OLMo 2 used
   this fix. Our model also starts its embeddings at a standard deviation of 1 and does not decay them.
5. **Low precision.** An FP8 number has a small range. FP8 with delayed scaling sets its scale from
   past values, so a sudden large value overflows. A study in 2024 trained a 7B model in FP8, and it
   diverged after 220B tokens. DeepSeek-V3 avoided this. It set the scale for each small block of 128
   values at the time of use. Our model trained in BF16.
6. **A bad batch and a weak state.** OLMo 2 found that batches with long repeated text often caused
   spikes. It removed documents that contain a sequence of 32 or more repeated n-grams. This reduced the
   spikes, and it did not remove all of them.

> **Note:** each cause has a measured fix, and the most important fix for attention is QK-norm.

## 6. Preventing Divergence

Prevention costs less than recovery. The frontier reports that had no divergence all used prevention.

| fix | what it limits | evidence |
|---|---|---|
| QK-norm | attention logits | a small test model without it diverged; OLMo 2, Qwen3, GLM-4.5 use it |
| QK-Clip | attention logits, for designs where QK-norm cannot apply | Kimi K2 capped logits at 100 and trained 15.5T tokens with zero loss spikes |
| soft-capping | logits, with $c \cdot \tanh(x/c)$ | Gemma 2 uses a cap of 50 for attention and 30 for the output |
| z-loss | output logits | PaLM, OLMo |
| gradient clipping | the size of one update | almost every run uses a limit of 1.0 |
| no weight decay on embeddings | small embeddings | OLMo 2 |
| a long warmup | early large updates | Llama 3 used 8,000 warmup steps |
| a smaller early batch | early instability | Llama 3 started at 4M tokens per step and later used 16M |
| a careful initialization | gradient spikes | OLMo 2's new initialization cut its grad-norm spike score from 0.40 to 0.03 |

> Two layers of protection are better than one: limit the cause, and then limit the update.

**The detail.** Our model's team tested the attention fixes directly.

| test | result |
|---|---|
| no QK-norm | diverged early and did not recover |
| QK-norm removed only from the full-attention layers | the grad norm spiked by several orders of magnitude |
| QK-Clip at 100 in place of QK-norm | the grad norm showed spikes |
| no QK-norm, learning rate ten times smaller | stable, and the final loss was higher |

QK-norm applies a normalization to each query and key before the dot product. The size of the logits
then cannot grow with the weights. The full-attention layers need it most, because they look at the
whole sequence.

*Widget: Attention logits (`widget_1_logits`). The largest attention logit over training steps. Select
no fix, QK-norm, QK-Clip or soft-capping, and see where the loss spikes start.*

> **Note:** QK-norm, z-loss, clipping and a long warmup prevent most divergences before they start.

## 7. Instability in a Mixture-of-Experts Model

A mixture-of-experts model has all the problems of a standard model. It also has problems in its
router.

```
ROUTER PROBLEMS
router logits grow            → the router z-loss limits them
outliers inside experts       → routing makes them larger
the router drifts             → a few experts take all the tokens
the bias overrides the router → the experts in that layer stop mattering
```

> The balance numbers can look healthy while a layer has stopped using its experts.

**The detail.**

**Outliers and routing.** DeepSeek-V4's team traced its loss spikes to very large values in the
mixture-of-experts layers, and found that routing made them worse. A rollback stopped each spike for a
short time, and the spikes came back. The team added two fixes. It limits the SwiGLU values in each
expert to the range $-10$ to $10$. After a detected spike, it computes the routing with slightly older
weights for a short period, and then it switches back.

**Router drift.** In MiMo-V2.6's reinforcement learning, a trainable router changed the expert load in 20
steps. The busiest expert went from 6 to 16 times the mean load. The team froze the router, and the load
stayed flat.

**A silent routing fault in our model shape.** Our model balances its experts with a bias that changes
which experts are selected. The team found a fault in the first two layers.

| metric for layers 0 and 1 | value | what it means |
|---|---|---|
| MaxVio, mean over all 50 layers | 0.31 | the model looks balanced |
| dead experts | moderate | nothing looks wrong |
| top-k intervention rate | 98.5% | the bias changed almost every choice of the router |
| selected top-k mass | close to $6/384$ | the chosen experts had almost no router preference |

The intervention rate is the share of choices that the bias changes. In these two layers it reached
98.5%, which is almost $1 - 6/384 = 98.4\%$. The router had a strong preference, and the bias overrode
it almost completely.

The team then tested the layers directly. It replaced the chosen experts with random experts, and then it
removed the routed experts completely. The loss did not change for layers 0 and 1. Those two layers did
not use their routed experts at all. A test run of 327.6B tokens did not show the fault, and the team did
not fix it in the final model.

The team found the fault with five metrics for each layer.

| metric | what it measures |
|---|---|
| dead-expert fraction | experts that receive no tokens |
| router entropy | how strongly the router prefers some experts |
| top-k intervention rate | how often the bias changes the router's choice |
| intrinsic top-k mass | the router's score for its own first choices |
| selected top-k mass | the router's score for the experts that were actually used |

> **Note:** monitor what the router wants and what the model does, for each layer, and not only the
> average balance.

## 8. Detecting Divergence Automatically

A program must check every step, because a person cannot. Each rule compares a new value with the
values before it.

```
grad norm
 │                              ▲   ← z-score 6.2: alarm
 │  · ·  ·   · ·               ╱│
 │ ·  · ·  ·  · · ·  ·        ╱ │       band = EMA ± 2.5 standard deviations
 │═════════════════════════════
 └──────────────────────────────── step
```

> A good rule finds every real spike and raises few false alarms.

**The detail.** Teams publish four kinds of rule.

| rule | source | how it works |
|---|---|---|
| NaN check | Megatron-LM, PyTorch | stop at once if the loss or a gradient is NaN or infinite |
| spike score | OLMo 2 | a value that is 7 or more standard deviations from the mean of the last 1,000 values |
| EMA z-score | ZClip, 2025 | an EMA of the grad norm with $\alpha = 0.97$; a z-score above 2.5 is a spike |
| ratio to the maximum | Megatron-LM | a loss larger than 10 times the largest loss seen before |

The z-score of a new value $x$ is

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ and $\sigma$ are the moving mean and standard deviation. A fixed limit, for example "grad
norm above 5", does not work for a whole run. The normal grad norm early in training is very different
from the normal grad norm late in training.

Megatron-LM also has a tool that checks if a spike is real. When a step looks wrong, the tool runs the
same step again on the same GPU. If the result is different, the fault was temporary. If the result is
the same, the tool runs the step on a different GPU. A different result there shows a hardware fault in
the first GPU. Google's Gemini 2.5 team used a similar replay. It replayed about 0.25% of steps, and 6%
of those replays found real hardware corruption.

*Widget: Spike detector (`widget_2_detector`). A grad-norm trace with real spikes and normal noise.
Select a fixed limit, the 7-sigma rolling rule or the EMA z-score, and count the spikes found and the
false alarms.*

> **Note:** compare each value with its recent history, and stop at once on a NaN.

## 9. Recovering from a Divergence

When a run diverges, the team goes back to a checkpoint before the problem.

```
checkpoints:   C1        C2        C3        C4
step:      ────●─────────●─────────●─────────●────  ▲ spike
                                   │
               roll back to C3 ────┘ , skip some batches, continue
```

> Recovery needs a good checkpoint, so the checkpoint interval is a stability decision.

**The detail.** Teams use four methods.

| method | source | result |
|---|---|---|
| roll back about 100 steps and skip 200 to 500 batches | PaLM | it worked for each of about 20 spikes |
| roll back and decrease the learning rate | OPT-175B | the run recovered |
| roll back and change the method for a short time | DeepSeek-V4 | a rollback alone did not stop the spikes from coming back |
| fix a code difference and restart from the divergence | OLMo 2 | two versions of the z-loss code gave different results |

A checkpoint can also be bad. In the Lightning LM run, the step-3009 checkpoint was damaged during the
save. The run continued from step 2909, the last good checkpoint. Thus, a team must keep more than one
checkpoint.

The checkpoint interval is a balance. Frequent saves cost time. Rare saves lose more steps after each
failure. Young's formula from 1974 gives a good interval:

$$\text{interval} \approx \sqrt{2 \times \text{save time} \times \text{mean time between failures}}$$

Our model's run had one failure every 12.9 hours, which is 775 minutes. If one save takes 1 minute, the
formula gives $\sqrt{2 \times 1 \times 775} = 39$ minutes, which is about 390 steps of 6 s. The team
saved every 250 steps, which is about 25 minutes.

*Widget: Rollback (`widget_3_rollback`). A run with checkpoints and a spike. Select the checkpoint
interval and the rollback point, and read the lost steps and the time cost.*

> **Note:** keep several good checkpoints, and set the interval from the save time and the failure
> rate.

## 10. Why Parameters Are Frozen

Sometimes a team keeps some weights fixed and trains only the others.

```
FROZEN                           TRAINED
router during RL    ███████████  experts, attention
8-bit expert base   ███████████  small low-rank adapters on top
vision encoder      ███████████  the language model
old layers          ███████████  new layers added on top
```

> Freezing protects a part that already works, or it saves the memory for its optimizer.

**The detail.** Four reasons appear in recent work.

| reason | example | result |
|---|---|---|
| stability | MiMo-V2.6 froze the router during RL | the expert load stayed flat |
| memory | QLoRA trains adapters on a frozen 4-bit base | a 65B model fine-tunes on one 48 GB GPU |
| memory, at pre-training scale | Lightning LM froze an 8-bit expert base under rank-16 adapters | Adam state fell from about 800 GB to about 18 GB |
| protect old knowledge | LLaMA Pro added 8 new blocks to 32 frozen blocks | it beat full fine-tuning on both general and domain scores |

A frozen weight has no gradient, no master copy and no optimizer state. With 16 bytes for each trained
weight, freezing a weight saves 14 of those 16 bytes. Only the 2-byte weight stays.

> **Note:** a frozen weight keeps only its 2 bytes, so freezing saves most of the training memory for
> that weight.

## 11. Frozen-Layer Constraints

Freezing creates rules that the training code must obey. If the code breaks a rule, the error is often
silent.

```
CONSTRAINT                                   WHAT GOES WRONG IF WE IGNORE IT
optimizer groups: trainable weights only     a frozen weight still gets weight decay
FSDP1: one wrapped unit = all frozen or      the program stops with an error
       all trainable
old-style checkpointing needs an input       a block silently gets no gradients
  that requires a gradient
the grad norm leaves out frozen weights      old clip limits and alarms no longer fit
unfreeze or merge → re-warm the LR           the run diverges
merge adapters into a 4-bit base             rounding changes the model's answers
```

> A freezing bug does not crash the run, so we must test for it.

**The detail.**

**Optimizer groups.** Build the optimizer only from weights that require a gradient. Megatron-LM does
this, and it makes no master copy for a frozen weight. If a weight is frozen after the optimizer is
built, an old zero gradient can stay on it, and AdamW can still apply weight decay to it.

**Sharding.** FSDP version 1 joins many weights into one flat tensor. It stops with an error when one
flat tensor mixes frozen and trainable weights. The fix is to wrap frozen parts separately. FSDP
version 2 shards each weight on its own and removes this limit. DeepSpeed's ZeRO optimizers skip frozen
weights and save them separately in the checkpoint.

**Activation checkpointing.** The older "reentrant" form of activation checkpointing needs at least one
input that requires a gradient. A block after a frozen embedding has no such input. The block then gets
no gradients, and PyTorch prints only a warning. The fix is the non-reentrant form, or a hook that marks
the embedding output as requiring a gradient.

**The grad norm.** The grad norm counts only the weights with a gradient. When a large part freezes, the
grad norm changes its size. The clip limit and the alarm bands must change with it.

**Unfreezing and merging.** ReLoRA merges its adapters into the base weights at regular intervals. Its
test showed that the run diverges after a merge unless the learning rate warms up again over 50 to 100
steps. A merge into a 4-bit base also adds rounding error.

**Lightning LM's frozen base.** The frozen 8-bit expert base created two problems. First, a rank-16
adapter can follow only a small part of the change in a weight matrix. The routed experts needed 74%
of their spectrum to hold 90% of their energy. To work around this limit, the team merged the adapters
into the base every few steps. That method worked below 1B, and it diverged at 120B. Second, the team
froze the embeddings during sharp changes in the data mix. The trainable parts then had to absorb the
whole change. The report's fix is to change the data mix slowly when parts are frozen.

*Widget: Frozen parameters (`widget_4_frozen`). Our model with parts that you can freeze. Select the
parts, and read the memory for each weight, the optimizer state, and the zero-gradient count for each
part.*

> **Note:** freeze with care, because the optimizer, the sharding, the recompute and the grad norm all
> change.

## 12. Proving That Freezing Works

Two facts must be true after freezing. The frozen weights must not change. The trainable weights must
receive gradients.

```
CHECK                                         WHEN
trainable weight with no gradient?            after the first backward pass
frozen weight with a gradient?                after the first backward pass
zero-gradient count for each part             every step, on the dashboard
hash of the frozen weights                    at every checkpoint
```

> Measure the freezing directly, because the loss does not show it.

**The detail.** Megatron-LM can count the gradients that are exactly zero. MiMo-V2.6's public training
dashboard shows this count for each part of the model: the experts, the other layers, the vocabulary,
the encoders and the multi-token-prediction head. MiMo's earlier report gives two rules. A rising count
in a mixture-of-experts model means that the expert balance is becoming worse. A falling count late in
fine-tuning means that the model is overfitting.

A hash is a short number calculated from all the bytes of a tensor. Two tensors with the same hash are
the same. Calculate the hash of the frozen weights at step 0, and again at each checkpoint. A different
hash means that something changed a frozen weight.

> **Note:** count the zero gradients for each part, and hash the frozen weights at each checkpoint.

## 13. Hardware and Systems Failures

Most interruptions in a large run come from the machine.

| cause in Llama 3, 419 unexpected interruptions | share |
|---|---|
| faulty GPU | 30.1% |
| GPU memory (HBM3) | 17.2% |
| software bug | 12.9% |
| network switch or cable | 8.4% |
| host maintenance | 7.6% |
| other causes, for example NICs, GPU SRAM, NCCL timeouts, power | 22.4% |
| silent data corruption | 1.4% |

Our model's run had 38 interruptions: 10 from GPU or node hardware, 9 from slow storage reads, 2 from
bad network links, 1 from a checkpoint save timeout, and 16 with no clear cause.

> On a large cluster, a failure is a normal event, so recovery must be automatic.

**The detail.** Four tools find most hardware problems.

| tool | what it finds |
|---|---|
| DCGM health checks and Xid alerts | a GPU that fails, overheats or has memory errors |
| `dcgmi diag` before the run | a bad GPU before training starts |
| the PyTorch NCCL flight recorder | which GPU stopped a collective operation and caused a hang |
| step timing for each GPU | a straggler that slows all the other GPUs |

PyTorch stops a collective operation after 10 minutes with no progress. Silent data corruption is rare,
about 1 fault in 1,000 devices in Meta's fleet. It is dangerous because a bad gradient looks like a
normal gradient.

*Widget: Failure rate (`widget_5_failures`). Select the number of GPUs and the checkpoint interval. Read
the interruptions per day and the time lost, from our model's measured failure rate.*

> **Note:** check the GPUs before the run, alert on Xid errors, and keep a flight recorder on for hangs.

## 14. Silent Failures

A loud failure stops the run. A silent failure lets the run continue with a wrong model.

```
LOUD      the run crashes, or the loss becomes NaN             found in minutes
PARTIAL   something degrades, and a check in the code finds it found in hours
SILENT    every cheap check passes                             found late, or never
```

> A finite loss and a clean checkpoint do not prove that the run is correct.

**The detail.** The Lightning LM report sorted its failures into these three classes. Its most dangerous
failure was a silent checkpoint misload. A converter wrote weights under the wrong names. The load did
not check the names, so it skipped those weights and left them random. Five cheap checks passed:

- the file existed,
- the file had the right size,
- the load returned with no error,
- the parameter count was correct,
- training started, and the loss was finite.

The only sign was the loss at step one: 2.95 with the bad load, against 2.55 with the correct load. The
fix was a strict load that stops on any missing or wrong name.

Other teams report the same class of failure.

| failure | what showed it |
|---|---|
| every tensor-parallel GPU had the same random seed (SmolLM3) | weak results after about 1T tokens, and a full restart |
| gradient accumulation computed the loss wrongly (many trainers, 2024) | the accumulated loss did not match the full-batch loss |
| the data mix used 57% where the plan said 8% (Lightning LM) | it was found late, from the console |
| two layers did not use their routed experts (our model shape) | five routing metrics for each layer |

> **Note:** for each important assumption, log a number that proves it.

*Widget: A silent misload (`widget_6_misload`). The loss at the first steps for a correct load and for a
load that left some weights random. Read where the difference sits against the normal noise.*

## 15. A Debugging Method

When a problem appears, a fixed order of steps finds the cause faster.

```
1. Is it real?           run the same step again on the same GPU, then on another GPU
2. Is it the hardware?   a different result on another GPU points to the first GPU
3. Is it the data?       give the batch to an earlier checkpoint
4. Where in the model?   read the statistics for each layer
5. Is it the code?       compare with a reference run or a small model
6. Is the fix real?      repeat the test after the change
```

> Make the problem small and repeatable before you try to fix it.

**The detail.** Each step has a published example.

- **Rerun.** Megatron-LM's tool and Gemini's replay find hardware faults this way.
- **Data.** PaLM showed that a spiking batch was safe at an earlier checkpoint. OLMo 2 read the spiking
  batches and found repeated text.
- **Layer statistics.** Kimi K2 tracked the largest attention logit for each head, and this found the
  layers with the problem.
- **Reference.** DeepSeek-V3 checked that its FP8 run stayed within 0.25% of a BF16 run.
- **Evidence.** The Lightning LM report gives every claim a grade: artifact, checkpoint, log,
  transcript or recollection. A claim with a log is stronger than a claim from memory. This habit tells a
  team which facts it can trust.

> **Note:** confirm the problem, localize it, then fix it, and grade the evidence for each step.

## 16. The Live Dashboard

A live dashboard shows the health of the run to everyone, at every step.

```
┌───────────────── MODEL ─────────────────┐ ┌──────────────── MIXTURE OF EXPERTS ───────────────┐
│ loss · held-out loss · MTP loss         │ │ MaxVio per layer · dead experts · top-10 share   │
│ grad norm · spike score · LR            │ │ intervention rate · intrinsic and selected mass  │
│ update-to-weight ratio · max attn logit │ │ expert load heat map (every 100 steps)           │
└─────────────────────────────────────────┘ └──────────────────────────────────────────────────┘
┌──────────── FROZEN PARTS ───────────────┐ ┌──────────────── SYSTEM AND HARDWARE ──────────────┐
│ zero-gradient count per part            │ │ tokens/s · MFU · step time · memory · data wait  │
│ frozen-weight hash check                │ │ GPU temperature · Xid errors · ECC · NVLink      │
└─────────────────────────────────────────┘ └──────────────────────────────────────────────────┘
```

> A dashboard is useful only if its graphs load fast and keep every spike visible.

**The detail: alerts.** These are starting values for V5. The team will adjust them after the first days.

| signal | alert |
|---|---|
| loss or gradient is NaN or infinite | stop at once |
| loss or grad norm, spike score | page at 7 standard deviations from the last 1,000 steps |
| grad norm, EMA z-score | warn above 2.5 |
| largest attention logit, any layer | warn above 100 |
| MaxVio, any layer | warn above 1.0 for 1,000 steps |
| dead experts, any layer | warn above 0 for 1,000 steps |
| intervention rate, any layer | warn above 90% |
| zero-gradient count, a trainable part | warn on a sudden rise |
| frozen-weight hash | stop on any change |
| tokens per second | warn on a drop of 10% for 100 steps |
| GPU Xid 79, 94 or 95, or an ECC error | page, and remove the node |

**The detail: why large graphs load slowly.** Our model's run has about 300,000 steps. A screen is
about 2,000 pixels wide. A graph of the whole run must therefore put about 150 steps into each pixel.

If each pixel shows only the mean, a spike disappears. Take 1,000 steps where the loss is 2.0, and one
step where it is 8.0. The mean is 2.006. The real spike is four times the normal value.

The fix is to keep five numbers for each bucket of steps: the minimum, the maximum, the first value,
the last value and the mean. A method called M4 proved in 2014 that the minimum, maximum, first and last
values for each pixel column draw a line chart with no visible error.

**The detail: the design.** The data goes into ClickHouse, a database that can use the training step as
its key. A table for each level stores the five numbers for every 10, 100, 1,000 and 10,000 steps. The
extra levels add only about 11% more rows. A graph of the whole run reads the 1,000-step table, which
has about 300 rows for each series. Grafana reads ClickHouse through its ClickHouse plugin.

| data for our run | series | logged every | points |
|---|---|---|---|
| scalars: loss, MTP loss, grad norm, LR, MFU and others | 30 | 1 step | 9.0M |
| per-layer statistics: grad norm, update ratio, attention logit | 150 | 10 steps | 4.5M |
| per-layer routing metrics | 350 | 10 steps | 10.5M |
| zero-gradient count for each part | 6 | 1 step | 1.8M |
| load of every expert | 19,200 | 100 steps | 57.6M |
| rollup tables | | | +11% |
| total | | | about 93M points, 1.5 GB before compression |

If the run logged the load of every expert at every step, it would need 5.8 billion points. A summary
at every step and the full load every 100 steps keep the data small.

**The detail: the public dashboard.** Grafana's shared dashboards are read-only. A viewer cannot write a
query, and only the saved panel queries run. These rules keep the public dashboard safe:

- Run a separate Grafana for public use.
- Give it a ClickHouse user that can only read the rollup tables.
- Use a Grafana version with the fix for CVE-2026-21722, which affected annotations on public
  dashboards.
- Limit the number of requests at the web proxy, because query caching is part of Grafana Enterprise.

*Widget: Downsampling (`widget_7_downsample`). A loss curve of 300,000 steps with one spike, drawn on
2,000 pixels. Select the mean, the minimum and maximum, or M4, and see which methods keep the spike.*

*Widget: The live dashboard (`widget_8_dashboard`). A replay of a simulated run on the four panels.
Incidents occur, and the alerts that find them light up.*

> **Note:** log summaries at every step, store rollups that keep the minimum and maximum, and give the
> public a read-only copy.

## 17. Reference Runs (SELF STUDY)

| run | spikes or divergence | main stability method |
|---|---|---|
| PaLM 540B, 2022 | about 20 spikes | rollback and skip batches |
| OPT-175B, 2022 | many restarts | rollback, lower learning rate, tighter clipping |
| Llama 3 405B, 2024 | few spikes, no intervention | long warmup, small early batch, FP32 accumulation |
| DeepSeek-V3, 2024 | no rollback | fine-grained FP8 scaling, bias-based balancing |
| OLMo 2, 2025 | spike score reduced | QK-norm, z-loss, new initialization, epsilon $10^{-8}$, data filter |
| Kimi K2, 2025 | zero spikes in 15.5T tokens | QK-Clip at 100 |
| DeepSeek-V4, 2026 | spikes tied to MoE outliers | SwiGLU limits, routing with older weights after a spike |
| MiMo-V2.6, 2026 | router drift in RL | frozen router, zero-gradient counts per part |
| our model shape, 2026 | silent routing fault in two layers | QK-norm, five routing metrics per layer |

## 18. V5 Decisions

Some choices follow directly from this session.

- **Prevention.** QK-norm in every attention layer, z-loss, gradient clipping at 1.0, no weight decay on
  embeddings, and a long warmup.
- **Detection.** A NaN check, the spike score and the EMA z-score on the loss and the grad norm.
- **MoE health.** The five routing metrics and MaxVio for each layer.
- **Freezing.** Optimizer groups from trainable weights only, zero-gradient counts for each part, and a
  hash of frozen weights at each checkpoint.
- **Dashboard.** ClickHouse with step rollups, a private Grafana for the team, and a read-only public
  Grafana.

Some questions remain open.

| question | what will settle it |
|---|---|
| What checkpoint interval? | The measured save time and the failure rate of our cluster. |
| Which alert limits? | The noise of the first days of the run. |
| Who answers an alert, and when? | The roles that the lab assigns for the run. |
| Which parts will V5 freeze, and when? | The plan for growth, fine-tuning and RL. |

## 19. Capstone: The Training Lab

The session has no separate assignment; the capstone below is the work. Quoted in full, since it is the
spec.

> From Saturday 24 October, our class works as a training lab. Together, we prepare the main training run
> of our own open Mixture-of-Experts model. The training run starts on Saturday 28 November.
>
> Large labs train such models with many teams. Each team owns one part of the work, and each team has
> fixed dates. One late team can stop the whole run. We work in the same way.
>
> **The goal**
>
> The model must be:
>
> - **Very good at programming.** This is the first priority in every decision.
> - **Good at agentic work.** The model reads a repository, makes a plan, calls tools, reads the results
>   and fixes the code.
> - **Able to understand Indian languages.**
>
> **The plan**
>
> | Date | Event |
> |---|---|
> | Sat 24 Oct | Team work starts. The data diagnosis task starts. |
> | Tue 27 Oct | The first 150 labelled rows of your data diagnosis are due. |
> | Sat 31 Oct | Gate 1. The full data diagnosis is due. |
> | 5 to 11 Nov | Break. There is no work in this week. |
> | Sat 14 Nov | Gate 2. The tokenizer and the model configuration are frozen. |
> | Sat 21 Nov | Gate 3. Dry run 1 ends. |
> | 24 to 26 Nov | Dry run 2 |
> | Sat 28 Nov | Gate 4. Go or no-go. The training run starts. |
>
> All due times are 23:59 IST.
>
> **Read this first**
>
> 1. Your team has no link to your final result. The team that you get does not change your final result
>    in this course.
> 2. Every student starts with the data diagnosis task. All students do it first, in every team.
> 3. The 3-day rule. A task is due in 3 days. 90% of it must be done, or you must show proof that you are
>    on target. If not, you leave the capstone, and a student from the waitlist takes your place. Data
>    tasks get 7 days.
> 4. The data diagnosis has a hard date: Saturday 31 October. If you do not finish it on time, you are out
>    of the capstone.
>
> **The data diagnosis task**
>
> The data that we use was cleaned by the groups that released it. We do not trust that cleaning without
> a check. In V4, a "cleaning" step damaged Indian-language text. This time, you are the doctor.
>
> - You get one sample pack: about 500 documents from one source, in one language or one programming
>   language. Two students read each pack.
> - You read each document. You find each disease: for example invisible characters, ghost chat markers,
>   broken encoding, wrong language labels, bad translation, secrets in code, or damage from earlier
>   cleaning.
> - You give evidence for each disease: the document, the exact characters, and a hex dump.
> - You propose a medicine: a rule that fixes or removes the disease.
> - You run a harm test: the medicine must not change or drop more than 1% of a set of good documents.
>   ZWJ and ZWNJ are part of Indian scripts. A medicine must never remove them.
>
> The TSAI Lab runs the approved medicines on the full data. You do not run cleaning on the full data,
> and you do not need cloud keys.
>
> Everything that you need is in the lab repository: https://github.com/theschoolofai/era-v5-lab
>
> - `diagnosis/DISEASES.md`: the list of diseases and the label sheet
> - `diagnosis/tools/view_pack.py`: the viewer, which shows hidden characters
> - `diagnosis/templates/`: the label and medicine templates
> - `diagnosis/submissions/README.md`: how to submit with a pull request
>
> Your pack ID and your download link are on the lab plan sheet. The link is valid for 7 days.
>
> **Your teams**
>
> The teams are on the lab plan sheet. The sheet shows the timeline, the members of each team, and the
> work of each week. Each team has its own chat channel on Axiom.
>
> A student can be in up to three teams, with a staggered workload. The sheet shows your teams.
>
> | # | Team | The work |
> |---|---|---|
> | 1 | Data A: English, Code, Math, Science | Diagnose the data, own the gold sets, check the cleaned shards |
> | 2 | Data B: Indian Languages and Code-adjacent Text | The same work, for Indian languages and for text about code |
> | 3 | Tokenizer | Build a tokenizer that beats the best current tokenizers on Indian languages, with no loss on code and English |
> | 4 | KE2 (Kronecker Embedding 2) | Prove the Kronecker code at the input and at the output of the model |
> | 5 | Architecture and Experiments | Test the main design choices on small models, and freeze the configuration |
> | 6 | Performance and Kernels | Make each part of the training step faster, with the same results |
> | 7 | Infrastructure | The cluster, the checkpoints, the automatic restart, and the dry runs |
> | 8 | Monitoring and Records | The dashboards, the alerts, and the records for the next model |
> | 9 | Evaluation | The benchmark harness, and the firewall that keeps benchmark data out of training |
> | 10 | Post-training and Agentic | The instruction data, the RL data, and the agent environments |
> | 11 | Release | The model card, the weekly updates, and the launch announcement |
> | 12 | Quantization and Serving | Cheap and fast evaluation of each checkpoint, and serving the final model |
>
> **Rules for all teams**
>
> - Speed changes must be correct. Each speed change must pass a test against the reference code.
> - Each claim needs evidence. Publish your results on GitHub, with a log, a checkpoint or a file.
> - Use legal data only.
> - Benchmark data never trains the model.
> - Each team has one lead. The leads meet TSAI once each week.
