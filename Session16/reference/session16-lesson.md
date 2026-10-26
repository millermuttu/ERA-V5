# Session 16: Scaling Laws and Compute Planning

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms16nt3b9q4x1e8cv2lm/lesson
Captured: 2026-10-26

Widgets (8) and figures (8) saved under `widgets/` — the figures are embedded diagrams with no caption
of their own, the widgets are the interactive ones.
Videos embedded on the page: https://www.youtube.com/embed/AGs8EkpOqcw (Studio),
https://www.youtube.com/embed/C2FRJeq8UX8 (GMeet)

## 1. The Question Before Every Run

Before a lab starts a large training run, it has a fixed amount of compute. The GPUs are booked for a
fixed number of days. After that, the money is spent. The lab must answer four questions before the
first step.

*Figure: the four questions of a fixed budget (`fig_1_budget`).*

A large run is too expensive to repeat. So we cannot learn these answers from the large run itself. We
learn them from many small runs, and then we predict the large run. A **scaling law** is the formula that
makes this prediction. It tells how the loss changes when the model size, the data or the compute grows.

This session teaches the full method. You count the compute. You choose the model size and the token
count. You check that the data is sufficient. You choose the learning rate and the batch size from a
formula. Then you test the forecast on small models before you trust it.

Terms that we use in this session.

| term | meaning |
|---|---|
| FLOP | one floating-point operation, for example one multiply or one add |
| FLOP/s | FLOPs per second, the speed of a GPU |
| compute, $C$ | the total FLOPs of a training run |
| parameters, $N$ | the weights of the model |
| tokens, $D$ | the number of training tokens that the model reads |
| MoE | Mixture-of-Experts: a model in which each token uses only some of the weights |
| active parameters | the weights that do work for one token |
| total parameters | all the weights of the model, in memory |
| TPP | tokens per parameter, $D / N$; for an MoE model, tokens per active parameter |
| epoch | one full pass over the unique data |
| BPB | bits per byte: the held-out loss, measured per byte of text |

BPB does not depend on the tokenizer. So it compares two models that use different tokenizers. This
matters for us, because we build our own tokenizer for Indian languages.

> **Note:** a scaling law lets small runs answer the questions of a large run.

## 2. Counting FLOPs

The first number of every plan is the compute. For a transformer, one simple formula gives a good
estimate.

$$C \approx 6 \cdot N \cdot D$$

- $C$ is the training compute in FLOPs.
- $N$ is the number of parameters that do work on each token.
- $D$ is the number of training tokens.

The factor 6 has a simple source. Each parameter does one multiply and one add for each token in the
forward pass. That is 2 FLOPs. The backward pass costs about twice the forward pass. That is 4 more
FLOPs. The total is 6 FLOPs per parameter per token.

```
forward pass        backward pass
  2 FLOPs     +       4 FLOPs      =   6 FLOPs
per parameter      per parameter      per parameter
  per token          per token          per token
```

For an MoE model, $N$ is the **active** parameter count. Each token goes to only a few experts. The other
experts do no work for that token. Use this table.

| what you plan | which count to use |
|---|---|
| compute, $C = 6ND$ | active parameters |
| memory on the GPUs | total parameters |

**The detail: the attention term.** The $6ND$ formula counts the weight matrices. It does not count the
work of attention itself. That work grows with the sequence length. DeepSeek LLM uses a fuller count of
FLOPs per token.

$$M = 72 \cdot n_{layer} \cdot d_{model}^2 + 12 \cdot n_{layer} \cdot d_{model} \cdot l_{seq}$$

Then $C = M \cdot D$. Here $n_{layer}$ is the number of layers, $d_{model}$ is the hidden size and
$l_{seq}$ is the sequence length. The second term is the attention term. At a short sequence it is
small. At a long sequence it becomes a large part of the compute.

*Widget: FLOP counter (`widget_1_flops`). Set the layers, the hidden size, the sequence length, the
active fraction and the tokens. Compare $6ND$ with the fuller count, and see how large the attention
share becomes at a long context.*

> **Note:** $6ND$ is a good first estimate, with $N$ as the active parameters, and attention adds more at
> a long context.

## 3. The Budget Equation

The $6ND$ formula tells how much compute a model needs. The hardware tells how much compute we have. A
second equation gives that number.

$$C_{available} = \text{GPUs} \times \text{days} \times \text{peak FLOP/s} \times \text{MFU} \times \text{goodput}$$

The days are converted to seconds. One day has 86,400 seconds.

Two terms in this equation need a definition.

- **MFU** (model FLOPs utilization) is the fraction of the peak GPU speed that the model actually uses.
  The rest is lost to memory traffic, communication and small kernels.
- **Goodput** is the fraction of the wall-clock time that produces progress. The rest is lost to
  failures, restarts and work that a rollback discards.

```
  GPUs × days × peak     the hardware that you rent
× MFU                    how well the code uses each GPU
× goodput                how much of the time is not lost
─────────────────
= the compute that the model really receives
```

> Each term is a multiplier. A low MFU removes as much compute as a low GPU count.

**Check the equation on a real run.** Microsoft AI trained MAI-Base-2 in 2026 and published its numbers.

| fact | value |
|---|---|
| model | MoE, 80B active, 2.8T total |
| GPUs | 16,000 GB200 |
| time | 38 days |
| peak per GPU | 2.5 PFLOP/s BF16 |
| MFU | 21% |
| goodput | 89% |
| tokens | 50T |

Do the two calculations by hand.

$$6 \cdot N \cdot D = 6 \times 80 \times 10^9 \times 50 \times 10^{12} = 2.4 \times 10^{25} \text{ FLOPs}$$

$$16{,}000 \times 2.5 \times 10^{15} \times 0.21 \times 38 \times 86{,}400 = 2.76 \times 10^{25}$$

$$2.76 \times 10^{25} \times 0.89 = 2.45 \times 10^{25} \text{ FLOPs}$$

The two numbers agree to about 2%. The model side and the hardware side give the same compute.

Real frontier runs have an MFU of about 20% to 40% and a goodput of about 90%. Llama 3 405B had an MFU of
38% to 43% in BF16 and an effective training time above 90%. In a 54-day period, it had 466
interruptions.

**The cost of failure.** The equation also prices each loss of time.

- On a 38-day run, a rollback of 1 day costs about 2.6% of the budget.
- A goodput of 89% means that 11% of the GPU time produced no progress. At 16,000 GPUs, that is about
  1,750 GPUs idle on average.

*Widget: Budget calculator (`widget_2_budget`). Set the GPU count, the peak speed, the MFU, the goodput
and the days. Load the MAI-Base-2 check, and compare the hardware compute with $6ND$.*

> **Note:** compute is a product of five terms, and MFU and goodput matter as much as the GPU count.

## 4. Chinchilla and the Epoch Refit

We now know the compute $C$. Next we split it between model size $N$ and tokens $D$. At a fixed compute,
a larger model reads fewer tokens. Which split gives the lowest loss?

In 2022, Hoffmann and colleagues answered this question in the Chinchilla paper. They fit a loss formula
with three terms.

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- $E$ is the loss that no model can remove. It is the noise of the text itself.
- $A / N^{\alpha}$ is the loss from a model that is too small.
- $B / D^{\beta}$ is the loss from data that is too little.

*Figure: IsoFLOP profiles (`fig_2_isoflop`).*

They used three methods. In the first, they fixed the model size and varied the tokens. In the second,
they used IsoFLOP profiles: at each fixed compute, they trained models of many sizes and found the size
with the lowest loss. In the third, they fit the formula above.

The result: the best $N$ and the best $D$ both grow with about $C^{0.5}$. Scale parameters and tokens
equally. The rule of thumb is about **20 tokens per parameter**.

They proved it with an experiment. Chinchilla had 70B parameters and read 1.4T tokens. Gopher had 280B
parameters and read 300B tokens. Both used the same compute. Chinchilla was better.

**The Epoch refit.** In 2024, Besiroglu and colleagues at Epoch AI took the data from the Chinchilla
figures and fit the third method again, with 240 points.

| coefficient | Hoffmann, method 3 | Epoch refit |
|---|---|---|
| $A$ | 406.4 | 482.01 |
| $B$ | 410.7 | 2085.43 |
| $E$ | 1.6934 | 1.8172 |
| $\alpha$ | 0.3392 | 0.3478 |
| $\beta$ | 0.2849 | 0.3658 |
| implied tokens per parameter | about 70 | about 20 |

The original third method gave about 70 tokens per parameter. That disagreed with the first two methods.
The refit gives about 20. That agrees with the first two methods. So the 20 tokens per parameter rule
stands, and the original parametric fit had an error.

The refit also found a problem with the error bars. Hoffmann reported an interval of 0.454 to 0.455 for
one exponent. An interval that narrow needs about 600,000 experiments. Hoffmann had about 400 to 500.

**Always bootstrap your own fit.** To bootstrap, you fit the law many times on random resamples of your
points. The spread of the answers is your real error bar.

**The data recipe moves the answer.** DeepSeek LLM fit the same kind of law on different data. The
exponent of the model size was 0.450 on early data, 0.524 on their current data and 0.578 on
OpenWebText2. Better data puts more of the compute into the model. So "20 tokens per parameter" is a
property of the Chinchilla data recipe. Your recipe needs its own fit.

*Widget: IsoFLOP explorer (`widget_3_isoflop`). Choose a compute budget, and switch between the Hoffmann
and the Epoch coefficients. See the minimum of each curve and the tokens per parameter that it implies.*

> **Note:** at a fixed compute, about 20 tokens per parameter gives the lowest loss, and the valley
> around it is flat.

## 5. Inference-Aware Sizing and Over-Training

Chinchilla finds the lowest loss for a fixed training compute. A released model also has a second cost.
After the release, people use it, and each token that the model produces costs compute. This is the
inference cost.

Sardana and colleagues (2024) added the inference cost to the objective.

$$\text{total cost} = 6 \cdot N \cdot D + 2 \cdot N \cdot (\text{tokens served})$$

The inference term grows with $N$. A model that serves many tokens must be small. To reach the same loss,
a small model must read more tokens. With a large demand, about 1B requests, the best model is smaller
and trained longer than the Chinchilla point.

**Over-training** means training past the Chinchilla point, at a higher TPP. Sardana trained 47 models.
Their quality kept improving up to 10,000 tokens per parameter. The gain flattens at extreme ratios, and
it flattens more than a fit at normal ratios predicts.

Gadre and colleagues (2024) showed that over-training is predictable. They trained 104 models, from
0.011B to 6.9B parameters. At a fixed over-training ratio, the loss curves are parallel lines in a
log-log plot. So a small fit predicts a large over-trained run. They predicted a 1.4B model on 900B
tokens, about 640 TPP, from runs with 300 times less compute.

**What shipped models do.** The table gives tokens per active parameter for released models.

| model | active / total | tokens | tokens per active parameter |
|---|---|---|---|
| Chinchilla (2022) | 70B dense | 1.4T | 20 |
| Llama 3 405B | 405B dense | 15.6T | about 39 |
| DeepSeek-V3 | 37B / 671B | 14.8T | about 400 |
| Kimi K2 | 32B / 1.04T | 15.5T | about 480 |
| MAI-Base-2 | 80B / 2.8T | 50T | about 625 |
| DeepSeek-V4-Pro | 49B / 1.6T | more than 32T | more than 650 |
| Qwen3-235B-A22B | 22B / 235B | 36T | about 1,640 |
| Llama 3 8B | 8B dense | about 15T | about 1,900 |
| DeepSeek-V4-Flash | 13B / 284B | more than 32T | more than 2,400 |

Typical frontier MoE models sit at about 400 to 700 tokens per active parameter. Treat this range as a
typical value. Small models go to thousands of tokens per parameter.

Inference cost also changes the shape of a model. Kimi K2 chose 64 attention heads. 128 heads would add
83% more inference FLOPs at a 128k context, for a loss gain of only 0.5% to 1.2%.

*Widget: Inference-aware sizing (`widget_4_inference`). Set the tokens served and the target loss. See
the training plus inference cost for each model size, and the best tokens per parameter.*

> **Note:** heavy use pushes the best model to a smaller size and a longer training, and frontier MoE
> models use about 400 to 700 tokens per active parameter.

## 6. When Data Runs Out

A high TPP needs many tokens. Good unique data is limited, especially for some languages. When the unique
data is too small, we must repeat it. How much is a repeated token worth?

Muennighoff and colleagues (2023) answered this with 400 runs, up to 9B parameters and up to 900B tokens.

*Figure: the value of a repeated token (`fig_3_repeat`).*

| epochs of the same data | what happens |
|---|---|
| up to about 4 | the loss is nearly equal to unique data; repetition is almost free |
| up to about 16 | repetition still helps; after this, the value falls fast |
| about 40 | repetition is almost worthless |

**The detail.** They fit a formula for the effective data $D'$. $U_D$ is the unique tokens and $R_D$ is
the number of repetitions.

$$D' = U_D + U_D \cdot R^*_D \cdot \left(1 - e^{-R_D / R^*_D}\right)$$

The constant $R^*_D \approx 15.4$. After about 15 repetitions, a repeated token has lost $1/e$ of its
value. In the data-constrained regime, spend extra compute on more epochs first, and add parameters after
that. They also found that code can fill up to 50% of the data with no damage to natural-language tasks.

**Newer results move the limit (SELF STUDY).**

- **Regularization.** Kim and colleagues (2025) found that the best weight decay for many epochs is about
  30 times larger than standard practice. Their best recipe used 5.17 times less data. Lovelace and
  colleagues (2026) found that a weight decay of 1.0 cuts the overfitting from repetition by about 70%.
- **Mixtures.** Sedova and colleagues (2026) ran more than 2,000 runs. A scarce source inside a larger
  mixture can be reused 15 to 20 times. The generic data acts as a regularizer. This matters for
  Indian-language data inside a larger mixture.
- **Rephrasing.** Kimi K2 tested rephrased text. Raw text repeated for 10 epochs scored 23.76 on
  SimpleQA. One rephrasing repeated for 10 epochs scored 27.39. Ten rephrasings for one epoch scored
  28.94. Rephrasing adds unique tokens.

MAI-Base-2 also repeats data. Its vision-language data has 4.5T unique tokens and is 18% of a 50T
mixture. So that data repeats about twice.

*Widget: Repetition value (`widget_5_repeat`). Set the unique tokens and the total tokens. Choose a single
source or a source inside a mixture. See the effective tokens and the marks at 4, 16 and 40 epochs.*

> **Note:** about 4 epochs are almost free, the value falls after about 16, and it is near zero at 40.

## 7. Scaling Laws for MoE

An MoE model has more choices than a dense model. Three terms describe its shape.

- The **activation ratio** is active parameters divided by total parameters. **Sparsity** is the
  fraction of the parameters that are not active.
- **Granularity** $G$ tells how small the experts are. Krajewski and colleagues define
  $G = d_{ff} / d_{expert}$. At $G = 1$, each expert is as wide as a dense feed-forward layer. At a
  higher $G$, the experts are smaller, more of them are routed for each token, and the active parameters
  stay the same.
- **Efficiency leverage** (EL) is the compute of a dense model divided by the compute of an MoE model
  that reaches the same loss.

*Figure: granularity (`fig_6_granularity`).*

**Granularity.** Krajewski and colleagues (2024) found that the best $G$ rises with compute: 8 at about
$3 \times 10^{18}$ FLOPs, and 64 at about $6 \times 10^{24}$ to $5 \times 10^{25}$ FLOPs. Experts as wide
as the dense layer are not optimal at almost any budget. A compute-optimal MoE at $10^{20}$ FLOPs matches
a dense model with about 20 times more compute. The gap exceeds 40 times beyond $10^{25}$ FLOPs.

**Sparsity.** Abnar and colleagues (2025) found that, at a fixed compute, the best sparsity rises with the
total parameters and approaches 1 for large models.

**Efficiency leverage.** Tian and colleagues (2025, Ant Group, Ling) trained more than 300 models. EL
rises as the activation ratio falls. The lowest ratio that they tested, 0.8%, was best at every budget.
EL also grows with compute. They predict EL above 7 at $10^{22}$ FLOPs with 3.1% activation. Their model
Ling-mini-beta had 0.855B active and 17.5B total parameters. It matched a 6.1B dense model on the same 1T
tokens with over 7 times less compute.

**More experts change the token count.** Ludziejewski and colleagues (2025) found that, at a fixed
compute, more experts give fewer active parameters and more tokens. At $10^{22}$ FLOPs, a dense model was
best at 18.8B parameters and 88.6B tokens. With 32 experts, the best point was 12.2B active and 136.9B
tokens. So the best TPP for an MoE model is higher than for a dense model.

**What frontier models chose.**

| model | total / active | activation ratio |
|---|---|---|
| Qwen3-235B-A22B | 235B / 22B | 9.4% |
| DeepSeek-V3 | 671B / 37B | 5.5% |
| Kimi K2 | 1.04T / 32B | 3.1% |
| DeepSeek-V4-Pro | 1.6T / 49B | 3.1% |
| MAI-Base-2 | 2.8T / 80B | 2.9% |

A typical frontier activation is about 3% of the parameters or lower. Treat this as a typical value.

**The returns diminish.** Kimi K2 kept the active parameters fixed and added experts. At a loss of 1.5,
sparsity 48 needed 1.69 times fewer FLOPs than sparsity 8, 1.39 times fewer than 16, and only 1.15 times
fewer than 32.

**Memory and kernels set the real limit.** A lower activation ratio means more total parameters, and the
total parameters must fit in GPU memory. Small experts also run as small matrix multiplications, and the
GPU runs these slowly. The tokens must also travel between GPUs to reach their experts. At MAI, the
MoE-layer MFU rose from 24% to 35% only after kernel work. Wan and colleagues (2026) found that the
near-optimal band of designs widens with scale. So engineering limits can choose a point inside the band
at little cost.

*Widget: MoE sparsity and granularity (`widget_6_moe`). Set the activation ratio, the granularity and the
compute budget. See the efficiency leverage, the total parameters and the memory per GPU.*

> **Note:** a lower activation helps with diminishing returns, frontier models use about 3% active or
> lower, and memory and kernels set the real limit.

## 8. Hyperparameter Laws

The loss law gives the model size and the tokens. Two more choices remain: the learning rate (LR) and the
batch size. A wrong LR on the large run wastes the whole budget. So labs also fit laws for these values.

muP transfers the LR when the model gets wider. The laws in this section give the LR and the batch size
when the compute, the tokens and the steps grow. A plan uses both.

**DeepSeek LLM (2024).** They fit the laws on runs from $10^{17}$ to $2 \times 10^{19}$ FLOPs and checked
them at $10^{20}$. The batch $B$ is in tokens.

$$B_{opt} = 0.2920 \cdot C^{0.3271} \qquad \eta_{opt} = 0.3118 \cdot C^{-0.1250}$$

**Step Law (StepFun, 2025).** They trained more than 3,700 models with nearly 1M H800 GPU hours.

$$\eta(N, D) = 1.79 \cdot N^{-0.713} \cdot D^{0.307} \qquad B(D) = 0.58 \cdot D^{0.571}$$

The batch depends mainly on the tokens $D$. The law held across 16 MoE sparsity levels within 0.5%, and
across three data recipes within 0.25%.

**MAI-Base-2 (2026).** Microsoft AI searched a grid of LR and weight decay (WD) values for each small
model. They fit a quadratic surface to the loss and took its minimum. Then they fit laws over the steps
and the TPP.

$$LR^* \cdot WD^* = \frac{17.04}{\text{steps}}$$

$$\log LR^* = -1.51 + 0.175 \cdot \log \text{TPP} - 0.573 \cdot \log \text{steps}$$

The exponent on the steps is near $-0.5$. This matches a known rule: when the batch doubles, multiply the
LR by $\sqrt{2}$.

*Figure: the broad hyperparameter optimum (`fig_8_hparams`).*

All three labs find a broad optimum. The loss surface is flat near the best point. An LR error of 2 times
costs little. An LR error of 10 times costs a lot.

**Caveat.** These laws are empirical. Each lab fit them on its own data, its own architecture and its own
optimizer. The MAI laws come from its Muon recipe. Before you use any of these formulas, check them on
your own small runs.

*Widget: Hyperparameter law plotter (`widget_7_hparams`). Set the compute or the tokens, the steps and the
tokens per active parameter. Compare the LR and the batch from DeepSeek, Step Law and the MAI formulas,
with the near-optimal band.*

> **Note:** the LR falls and the batch rises with scale, and the optimum is broad.

## 9. Precision and Forecasting Downstream

**Precision (SELF STUDY).** Kumar and colleagues (2024) trained more than 465 models at different numeric
precisions. A lower precision lowers the effective parameter count. Two results matter for a plan.

- The damage from post-training quantization (PTQ) grows with $D / N$. PTQ converts a trained model to
  fewer bits. An over-trained model is more fragile under PTQ. Past some point, more data makes the
  quantized model worse.
- Below 4 bits, a model must be more than 4 times larger for the same loss scaling.

A high TPP and a low-bit deployment work against each other. If the plan has both, it needs
quantization-aware training or a higher deployment precision.

**Downstream tasks.** A loss law predicts the loss. A user cares about the benchmark scores. Llama 3 used
a two-step forecast.

*Figure: the two-step downstream forecast (`fig_7_downstream`).*

Their small runs went from $6 \times 10^{18}$ to $10^{22}$ FLOPs. At $3.8 \times 10^{25}$ FLOPs, the fit
gave 402B parameters and 16.55T tokens. They trained 405B on 15.6T. The ARC-Challenge forecast
extrapolated four orders of magnitude and only slightly under-predicted the result.

Benchmark forecasts are less reliable than loss forecasts. A 2025 meta-analysis by Lourie and colleagues
found that downstream scaling is predictable in only about 39% of cases.

| what you forecast | how much to trust it |
|---|---|
| loss or BPB | high; give an interval |
| a benchmark score | low; use a sigmoid, wide error bars and many evals |

> **Note:** forecast BPB with confidence, forecast benchmarks with wide error bars, and plan quantization
> when the TPP is high.

## 10. The Scaling Ladder

A **scaling ladder** is a set of small models of increasing size, all trained with the same recipe. Each
model is a rung. We fit the laws on the rungs, and then we extrapolate to the large run.

*Figure: a scaling ladder (`fig_4_ladder`).*

A good ladder has 5 to 8 rungs over 2 to 3 orders of magnitude of compute. A typical ladder costs a few
percent of the main run.

**The MAI ladder.** Microsoft AI names each rung by its layer count $L$. One shape rule sets the rest.

- hidden size $= (256/3) \cdot L$
- query heads $= L$, rounded to a multiple of 16
- the same TPP on every rung (MAI does not publish the value)

**Source scaling.** The large run repeats some sources. A small rung reads fewer tokens. If it reads the
same mixture with no change, it sees no repetition, and its forecast is wrong. Source scaling fixes this.
Each source repeats the same number of times on every rung. For example, a source that repeats 8 times in
a 50T run also repeats 8 times in a 5T run, over 1/10 of the unique data.

```
50T run:  source X, unique size U,       repeated 8×
 5T run:  source X, unique size U / 10,  repeated 8×
```

**The held-out rung check.** Before you trust the ladder, hold out the top rung. Fit the law on the other
rungs. Predict the top rung, with a bootstrap interval, before it finishes. Trust the extrapolation only
if the held-out rung lands inside the interval.

**Comparing two recipes.** MAI uses the **efficiency gain** (EG). EG is the ratio of the compute that a
new recipe needs to reach the eval loss of a baseline. To find it, fit BPB against training FLOPs for each
recipe on the ladder. Then read off the FLOPs at which each line reaches the same BPB. A good change moves
the whole line.

*Widget: Ladder and forecast (`widget_8_ladder`). Set the number of rungs, the noise and the held-out
rung. Switch source scaling on and off. See the fit, its bootstrap band and the held-out prediction.*

> **Note:** trust a forecast only after a held-out rung lands inside the band.

## 11. Case Study: How Microsoft Got 4 to 12 Times More Efficient

In October 2026, Microsoft AI described how it trained MAI-Base-2. They fit BPB against training FLOPs on
four held-out evals. Then they read off the FLOPs that each recipe needs to reach the BPB of
DeepSeek-V4-Pro-Base. The MAI recipe needed 4.2 to 12.3 times fewer FLOPs. Against their previous model,
MAI-Base-1, the gain was 6.3 to 12.7 times.

The post names six engineering changes.

| # | change | what MAI did | evidence in the post |
|---|---|---|---|
| 1 | better data | deduplication, quality filtering and diversity sampling with embeddings | the post gives no separate number |
| 2 | Muon replaced AdamW | Muon optimizer for the whole run | Muon cost 2.1% of the iteration time (165 ms); the estimated floor is 55 ms |
| 3 | MoE in every layer | every layer has experts; the previous design alternated | MoE-layer MFU rose from 24% to 35% after kernel work |
| 4 | activation gradient clipping | clips the gradients of the activations | the grad norm stayed tight; the benefit grows with scale; it exposed a malformed training example |
| 5 | hyperparameters from scaling laws | LR, WD and dropout from fitted laws | $LR^* \cdot WD^* = 17.04 / \text{steps}$ and the LR law of Section 8 |
| 6 | scaling experiments that copy the large run | a ladder by layer count with source scaling | each rung repeats each source as often as the full run |

The malformed example was a cut license header. A preprocessing bug had produced it. Gradient clipping and
monitoring found it. A scaling law cannot find a bug of this kind.

The post states its main lesson in four words: "Simple methods scale best."

**Caveat.** The 4.2 to 12.3 times is for the combined recipe against DeepSeek-V4-Pro-Base. It comes from
BPB-against-FLOPs fits on four evals, and the range is wide. Microsoft gives no breakdown per technique.
So we do not know how much of the gain each change gives.

> **Note:** six simple changes together made the MAI recipe 4.2 to 12.3 times more FLOP-efficient, and
> each one needs its own test.

## 12. What We Test from This

The case study gives us candidate changes. It does not tell us which change gives which gain. So we test
each one on our own ladder, with the efficiency gain as the measure.

| test | what we compare | why |
|---|---|---|
| Muon against AdamW | the two optimizers, same ladder, same data | MAI used Muon; we measure the gain on our recipe |
| activation gradient clipping | on and off, as a separate ablation | an ablation removes one change and keeps all the others; this separates the clipping gain from the Muon gain |
| data diversity sampling | sampling with embeddings, and the mixture with no such sampling | MAI used it; we measure it on code and on Indian-language data |
| the MAI LR and WD laws | the predicted LR and WD, and a small grid around them, on our small runs | the laws are empirical; we check them before we use them |
| MoE in every layer | every layer, and alternating layers | MAI raised the MoE-layer MFU only after kernel work; we measure both the loss and the speed |

Some questions remain open.

| question | what will settle it |
|---|---|
| Which TPP for the main run? | the inference target, the unique data and the ladder fit |
| Which activation ratio? | the memory per GPU and the speed of our expert kernels |
| Do the laws transfer to our tokenizer and data? | the held-out rung check |

> **Note:** we test each change alone on our ladder, and we keep only the changes that move the BPB line.

## 13. Capstone: Planning Our Run

The TSAI Lab plans the main training run with the method of this session. The plan has five steps.

*Figure: the planning steps (`fig_5_plan`).*

**A worked example.** The numbers below are an example with round values. They are not the numbers of our
run.

**Step 1: the budget.** Assume 1,000 GPUs, a peak of 1 PFLOP/s BF16 each, 30% MFU, 90% goodput and 30
days.

$$C = 1{,}000 \times 10^{15} \times 0.30 \times 0.90 \times 2.592 \times 10^{6} \text{ s} \approx 7 \times 10^{23} \text{ FLOPs}$$

**Step 2: the size.** The TPP comes from the objective.

| objective | TPP | equation | active $N$ | tokens $D$ |
|---|---|---|---|---|
| lowest training cost (Chinchilla) | 20 | $C = 6 \cdot N \cdot 20N = 120 N^2$ | about 76B | about 1.5T |
| heavy inference | 500 | $C = 3{,}000 N^2$ | about 15B | about 7.6T |

**Step 3: the data.** Assume 2T unique tokens of good data. At 500 TPP, 7.6T tokens is about 3.8 epochs.
This is inside the 4-epoch band, where repetition is almost free. A higher TPP would go past 4 epochs.
Then the plan needs stronger regularization, rephrased data, or a higher loss.

**Step 4: the sparsity.** At 3% active, 15B active parameters give about 500B total. Check the memory per
GPU and the expert-parallel layout.

**Step 5: the ladder.** Train 6 rungs from about $10^{19}$ to $10^{22}$ FLOPs, with the same TPP, the same
shape rule and source-scaled repetition. The ladder costs under about 5% of $C$. Fit the LR and the batch
on the lower rungs. Check them on the fifth rung. Predict the sixth rung before it finishes.

**Step 6: the forecast.** Write the predicted BPB of the main run, with an interval. Set the alert level
of the live run from this forecast. During the run, the loss must stay on the predicted line. A gap is a
bug signal.

**Step 7: the reserve.** One lost day is 3.3% of a 30-day budget. Keep a reserve of that size.

We also have a 16x secret. We explain it only during training, to the students who take part.

**Your part in the lab.**

- The lab plan sheet shows the timeline, your teams and the work of each week.
- The CAP chat channel on Axiom is the channel of the capstone. It has one group for each team, from G1
  to G12. Post your work and your questions in your team group.
- Week 1 is the data diagnosis. Every student starts with it. The first 150 labelled rows are due on
  Tuesday 27 October. The full diagnosis is due on Saturday 31 October. All due times are 23:59 IST.
- **The 3-day rule.** A task is due in 3 days. 90% of it must be done, or you must show proof that you
  are on target. If not, you leave the capstone, and a student from the waitlist takes your place. Data
  tasks get 7 days.

The data that you diagnose this week sets the unique tokens of the plan. The unique tokens set the
epochs, and the epochs set the TPP that we can afford. So your diagnosis is the first input to the plan.

> **Note:** the TSAI Lab sizes the run from the budget, tests the recipe on a ladder, and trusts the
> forecast only after a held-out rung lands inside the band.

## 14. Assignment

> There is no separate assignment from this session to Session 20. The capstone is the assignment. Your
> work is the work of your teams: the data diagnosis in week 1, your core team every week, and your
> rotation teams in weeks 2 and 3. The plan sheet in the CAP session gives your tasks and dates. The
> 3-day rule applies to every task.
