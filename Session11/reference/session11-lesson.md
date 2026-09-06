# Session 11: Optimizers and Learning-Rate Schedules

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms11kq8w4l9r6y3zp7ef/lesson
Captured: 2026-09-06

Widgets (12) saved under `widgets/`, inline scripts collapsed into `widgets-scripts-dump.txt`.
Videos embedded on the page: https://youtu.be/e-J51Ju-Ho0 and https://youtu.be/EUJMyNbiwTE

## 1. Introduction

Session 10 defined the training step as five lines of code, one of which was left unspecified.
`loss.backward()` computes a gradient for every weight, and `optimizer.step()` then uses those
gradients to move the weights, but *how* was never explained. This session covers that. It begins
from the observation that a gradient is incomplete information.

A gradient specifies the direction in which a weight should move. It does not specify the distance.

The gradient computed in Session 10 was 64. That value does not prescribe a movement of 64, or of
0.64, or of any particular size. It gives a direction and a relative urgency, and the distance is a
separate decision. Every method in this session is a different rule for making that decision.

| Method | How it sets the distance | Section |
|---|---|---|
| learning rate | one fixed value for every weight | 3 |
| momentum | from an average of recent gradients | 4 |
| second moment | separately per weight, from its own gradient history | 5 |
| Adam | both of the above combined | 6 |
| weight decay | adds a constant pull towards zero | 7 |
| warmup | small at first, while the estimates are unreliable | 9 |
| schedule | decreasing over the course of the run | 10 |
| batch scaling | from the number of samples averaged | 11 |
| muP | from the width of the model | 12 |
| Muon | from the shape of the weight matrix | 13 |

    Session 10: loss to gradient → Session 11: gradient to distance → Session 12: across many GPUs

The methods appear in the order in which they were developed, because each one corrects a specific
failure of the one before it.

## 2. Terminology

Five terms recur throughout the session, each given here with a representative value.

The **learning rate**, written η, is the factor that converts a gradient into a distance. For a
model of our size it usually lies near 0.0003.

A **hyperparameter** is a setting we choose before training rather than learn during it. The
learning rate is one, and the batch size from Session 10 is another.

**Optimizer state** is any quantity the optimizer stores about a weight between steps. Plain
gradient descent stores none. The two values in the Session 10 widget are optimizer state, and
Section 6 identifies them.

An **epoch** is one complete pass over the training data. Our run makes roughly one pass, so steps
rather than epochs are the useful unit.

A **schedule** is a rule that varies the learning rate over the course of training, and it is the
subject of Section 10.

> Carry this forward: the learning rate is the most consequential single number in the run, and the
> rest of this session exists to constrain how it is set.

## 3. Gradient Descent

The simplest rule that uses a gradient moves each weight against it by a fixed fraction.

    w ← w − η · g

This is gradient descent. Take one weight starting at 0, and a loss L = (w − 5)² that is smallest
when the weight reaches 5. Its gradient is 2(w − 5), the 2 coming from differentiating the square.

Call the distance still to travel d = w − 5. Substituting that gradient into the update rule and
then subtracting 5 from both sides gives the distance after one step:

    w_new     = w − η · 2(w − 5)
    w_new − 5 = (w − 5) − 2η(w − 5)
    d_new     = d · (1 − 2η)

The (w − 5) factors out and leaves a single number multiplying it, so every step shrinks or grows
the remaining distance by the same factor (1 − 2η), and the whole behaviour of the run is decided
by that one number.

That factor carries two separate pieces of information. Its sign says which side of the minimum the
weight lands on: positive and it stayed on the side it started, negative and it crossed over, which
is an overshoot. Its magnitude says whether the distance shrank or grew: below 1 and the weight is
closer than before, above 1 and it is further away. Convergence therefore needs |1 − 2η| < 1, which
for this loss means η between 0 and 1.

| η | multiplier | w over five steps |
|---|---|---|
| 0.01 | +0.98 | 0.10, 0.20, 0.29, 0.39, 0.48 |
| 0.10 | +0.80 | 1.00, 1.80, 2.44, 2.95, 3.36 |
| 0.90 | −0.80 | 9.00, 1.80, 7.56, 2.95, 6.64 |
| 1.10 | −1.20 | 11.0, −2.2, 13.6, −5.4, 17.4 |

Four different behaviours from one number. At 0.01 the weight crawls and will need several hundred
steps. At 0.10 it converges cleanly. At 0.90 it overshoots the target every step but the overshoot
shrinks, so it still converges. At 1.10 the overshoot grows instead, and the weight is further from
the target after five steps than when it started.

The same three lines work on any loss of this shape. If the gradient is c · d rather than 2d, the
multiplier comes out as (1 − ηc), where c measures how sharply the loss bends around its minimum.
We call c the **curvature**, and it was 2 in the example above. Convergence requires |1 − ηc| < 1,
so the safe range for η is set by the curvature.

A real loss does not have one curvature. It bends sharply along some directions and gently along
others, and a single η has to serve all of them. Consider

    L = ½ (20u² + v²)

which has curvature 20 along u and curvature 1 along v. Starting from (1, 1), each step multiplies
u by (1 − 20η) and v by (1 − η), read exactly as above.

| η | u multiplier | v multiplier | after five steps |
|---|---|---|---|
| 0.01 | +0.80 | 0.99 | u is converging, v has barely moved at 0.951 |
| 0.09 | −0.80 | 0.91 | u overshoots every step, v is still slow at 0.624 |
| 0.11 | −1.20 | 0.89 | u has diverged to −2.49 and is growing |

The value that suits u leaves v almost stationary, and the value that moves v makes u diverge. This
is the failure the next three sections correct.

> Carry this forward: one learning rate shared by every weight is the first assumption to break,
> because the weights do not share a curvature.

*Widget: Gradient descent in two dimensions. Raise the learning rate and the steep direction starts
to oscillate and then diverges, while the shallow direction is still crawling.*

## 4. Momentum

A weight whose gradient has kept the same sign over many steps can safely be moved further than one
whose gradient alternates. Using that fact requires a record of recent gradients whose size does not
grow with the number of steps.

An **exponential moving average** keeps a single running value and mixes each new observation into
it by a fixed fraction:

    m ← β₁ · m + (1 − β₁) · g

The fraction β₁ is a hyperparameter and is almost always 0.9. Each new gradient then contributes one
tenth while the existing average keeps nine tenths. A gradient's share is multiplied by β₁ on every
later step, so it has faded to almost nothing after about 1/(1 − β₁) = 10 steps, and that is the
sense in which m remembers the last ten gradients while storing one number.

The two directions of Section 3 produce gradients of two characteristic shapes. Along the steep
direction the sign flips on every step, because the weight lands past the minimum each time. Along
the shallow direction the sign never changes, because the weight approaches the minimum from one
side and never reaches it.

Take the cleanest version of each shape, one alternating between +1 and −1 and one holding at a
constant +0.2, five times smaller.

| step | steep direction | shallow direction |
|---|---|---|
| 1 | 0.100 | 0.020 |
| 2 | −0.010 | 0.038 |
| 3 | 0.091 | 0.054 |
| 4 | −0.018 | 0.069 |
| 5 | 0.084 | 0.082 |

The alternating gradients cancel each other while the constant one accumulates, so after five steps
the two averages are nearly equal despite a fivefold difference in the gradients themselves.
Momentum is the use of m in place of g in the update rule.

> Carry this forward: an exponential moving average cancels what alternates and accumulates what is
> consistent.

*Widget: Momentum. Enter gradients by hand and the average follows them, then apply it to the
surface from Section 3 and move β₁ from 0 to 0.99.*

## 5. Per-Parameter Learning Rates

Momentum still leaves every weight sharing one learning rate, and the weights of a language model
differ enormously in how often they receive a gradient at all. Session 7 gives the clearest case.
The embedding row for a common word receives a large gradient in almost every batch, while the row
for a rare word may receive none across a thousand batches. One learning rate cannot suit both.

The weights of a model are also called its **parameters**, and the fix is to give each parameter a
learning rate of its own, computed from its own gradient history. We keep a second exponential
moving average, this time over the squared gradient, which estimates how large that parameter's
gradients typically are:

    v ← β₂ · v + (1 − β₂) · g²

The subscript distinguishes it from β₁ above, and β₂ is almost always 0.999, so by the same
1/(1 − β₂) count this average spans about a thousand gradients rather than ten. If a parameter's
gradient holds steady at g, the average settles on g², so √v settles on |g|, which is where the two
values in the table below come from. The step is then divided by √v, and writing the update out
shows what that does:

    w ← w − (η / √v) · g

The quantity η/√v is a learning rate belonging to that parameter alone, and it is what gives this
section its name. For two parameters, one receiving a gradient of 1.0 on every step and one
receiving 0.01, after two hundred steps:

| quantity | Parameter A | Parameter B |
|---|---|---|
| gradient g | 1.0000 | 0.0100 |
| √v | 1.0000 | 0.0100 |
| its own learning rate, η/√v | 1.00 η | 100.00 η |
| resulting step, ηg/√v | 1.0000 η | 1.0000 η |

Parameter B has been given a learning rate a hundred times larger than Parameter A's, and the two
then take exactly the same step. The division removes the magnitude of the gradient and keeps only
its sign and its consistency. A parameter that has been inactive for a thousand steps therefore
takes a full-sized step when a gradient finally arrives, which is what a rare word's embedding
requires.

> Carry this forward: η/√v is a separate learning rate for every parameter, and it gives them all
> the same step size whatever their gradients.

*Widget: Per-parameter learning rates. Two parameters with gradients a hundred times apart, and the
learning rate each one is given, before and after the division.*

## 6. Adam

Adam combines the two averages. The first sets the direction of the step, the second sets its scale,
and a correction removes a bias they share.

Both averages start at zero, so their early values are pulled towards zero and read too low. Feed a
constant gradient g into the first average and after t steps it has reached m_t = g (1 − β₁ᵗ), short
of g by exactly the factor (1 − β₁ᵗ). Dividing by that factor restores it, and the same argument
applies to v. This is **bias correction**, and it is exact rather than approximate.

It matters most at the first step, because β₂ = 0.999 is much closer to 1 than β₁ = 0.9 and the two
averages are therefore pulled down by very different amounts. At t = 1 we have m₁ = 0.1 g and
v₁ = 0.001 g², so without the correction

    m₁ / √v₁ = 0.1 g / (0.0316 |g|) = 3.16

while with it the two corrections cancel and the ratio is exactly g/|g| = 1.

| at step 1, with g = 0.5 | step taken |
|---|---|
| without bias correction | 3.16 η |
| with bias correction | 1.00 η |

The complete rule is

    w ← w − η · m̂ / (√v̂ + ε)

where ε is a very small constant, usually 10⁻⁸, added only so that a weight whose gradients have all
been zero is not divided by zero. Computed by hand for five steps at η = 0.001:

| t | g | m | v | m̂ | v̂ | step | w |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.0500 | 0.000250 | 0.5000 | 0.2500 | −0.001000 | 0.999000 |
| 2 | 0.40 | 0.0850 | 0.000410 | 0.4474 | 0.2050 | −0.000988 | 0.998012 |
| 3 | 0.60 | 0.1365 | 0.000769 | 0.5037 | 0.2567 | −0.000994 | 0.997018 |
| 4 | 0.45 | 0.1678 | 0.000971 | 0.4881 | 0.2431 | −0.000990 | 0.996028 |
| 5 | 0.55 | 0.2061 | 0.001273 | 0.5032 | 0.2550 | −0.000996 | 0.995031 |

The step column follows from the two averages. Once both are corrected, m̂ is close to the recent
average gradient and √v̂ is close to its typical size, so their ratio sits near ±1 and the step sits
near ±η. The gradients range from 0.40 to 0.60, while every step falls within half a percent of
0.001, which is the learning rate itself. This is the property that made Adam the default: the
gradient sets the direction and the learning rate sets the distance, and the two decisions stop
interfering with each other.

m and v are the two values stored in the Session 10 widget.

> Carry this forward: Adam moves every weight by approximately η, whatever the magnitude of its
> gradient.

*Widget: Adam, one weight, five steps. Every intermediate value as it is computed, with bias
correction switchable so the first step moves between 1.00 and 3.16 times η.*

## 7. Weight Decay and AdamW

Weights tend to grow over a long run, and large weights make a model brittle. The established remedy
is **L2 regularization**, which adds a penalty for magnitude directly to the loss:

    L_total = L + ½ λ w²

The derivative of that penalty is λw, which joins the gradient. Under plain gradient descent it
shrinks every weight by the same fraction on each step. Under Adam it does not, because that added
λw is divided by √v̂ along with everything else.

For two weights, both sitting at 0.5, with λ = 0.1 and η = 0.001, the decoupled amount is
ηλw = 0.001 × 0.1 × 0.5 = 5.0 × 10⁻⁵, and the L2 route is that same amount divided by √v̂:

| parameter | √v̂ | shrinkage per step |
|---|---|---|
| A | 1.00 | 5.0 × 10⁻⁵ |
| B | 0.01 | 5.0 × 10⁻³ |

Parameter B is decayed a hundred times more strongly than Parameter A, for a reason unconnected to
how large it is. How much regularization a weight receives has become a function of its gradient
history.

**AdamW** applies the shrinkage separately, after the Adam step, where nothing divides it:

    w ← w − η · m̂ / (√v̂ + ε) − ηλw

Both weights above then shrink by 5.0 × 10⁻⁵.

A further consequence was established in 2025. The decoupled term makes the final weights an
exponential moving average of the updates applied along the way, with a timescale of 1/(ηλ) steps.
At η = 0.0003 and λ = 0.1 that is 1/(0.0003 × 0.1) = 33,333 steps, so the model we finish with is an
average over roughly the last thirty thousand steps of training. It follows that η and λ are not two
independent settings. Their product is the setting.

We exclude normalization scales and biases from decay, because shrinking them changes what the layer
computes rather than how large it is.

> Carry this forward: decoupled decay shrinks every weight equally, and ηλ determines how far back
> the finished model averages.

*Widget: L2 against decoupled decay. The same weight down both routes, identical while its gradients
are large and a hundred times apart when they are small.*

## 8. Optimizer Memory Cost

Session 10 counted sixteen bytes for every weight in training. Two of the five entries in that count
belong to the optimizer, and Section 6 identified them.

| What is stored | Bytes |
|---|---|
| the weight, in bf16 | 2 |
| its gradient, in bf16 | 2 |
| a full-precision copy, in fp32 | 4 |
| m, the gradient average | 4 |
| v, the squared-gradient average | 4 |
| total | 16 |

Half the training memory is optimizer state, so the choice of rule changes the bill directly. A 9B
model at 16 bytes a weight needs 9 × 10⁹ × 16 = 144 billion bytes, and dividing by 2³⁰ turns that
into gibibytes:

| Optimizer | Bytes per weight | A 9B model |
|---|---|---|
| gradient descent | 8 | 67.1 GiB |
| with momentum | 12 | 100.6 GiB |
| AdamW | 16 | 134.1 GiB |
| 8-bit AdamW | 10 | 83.8 GiB |

The 8-bit variant stores m and v in one byte each with a shared scale, using the block-scaling method
from Session 10, and saves 50 GiB on a 9B model.

> Carry this forward: the optimizer is half of the memory required to hold a model in training.

*Widget: Optimizer state memory. Vary the model size and the choice of rule, and each one crosses the
capacity of an 80 GB card at a different point.*

## 9. Learning Rate Warmup

The warmup in this section is a different thing from the mixture warmup band of Sessions 5 and 6,
which blended a change of data distribution. Here we are ramping the learning rate itself.

Section 6 established that Adam moves each weight by approximately η. That approximation depends on
how consistent the gradients have been:

| gradient behaviour | resulting step |
|---|---|
| same sign on every step | 1.000 η |
| noisy, averaging to zero | 0.281 η |

The first row is the exact case from Section 6: a gradient of unchanging sign makes m̂ and √v̂ equal,
so the ratio is 1. The second is measured rather than derived, by running the same rule on gradients
drawn independently with mean zero, where m̂ largely cancels while √v̂ does not.

Through most of a run the gradients disagree with each other and every weight takes a fraction of η.
At the start of a run they agree, because the model is randomly initialized and almost every weight
is wrong in the same direction, so every weight takes close to the full step, repeatedly.

Weights are initialized at a scale of 1/√fan-in, so that a layer's output has about the same size as
its input however wide it is. At d_model = 4,096 that is 1/√4096 = 0.0156. A full step of η = 0.0003
therefore moves a weight by

    0.0003 / 0.0156 = 0.0192

of its own size, on the first step, in a direction chosen by an untrained model. A healthy run sits
near 0.001, so the starting value is nineteen times too large. Warmup corrects this by raising η from
near zero over the first few thousand steps, during which the gradients become less correlated.

That ratio, the size of the update divided by the size of the weight, logged per layer, is the
quantity to monitor. It should stay near 10⁻³ for the whole run.

> Carry this forward: warmup is needed because early gradients are correlated, and correlated
> gradients produce Adam's largest possible step.

*Widget: Learning rate warmup. The update-to-weight ratio over 10,000 steps. The largest ratio the
run ever sees falls from 19.2e-3 without warmup to 2.83e-3 with it.*

## 10. Learning Rate Schedules

Warmup fixes the first part of the learning rate's trajectory, and the rest of it must also be
specified. Large steps early cover distance quickly and small steps late let the model settle, so
every schedule rises and then falls. They differ in the shape of the fall.

**Cosine decay** brings the learning rate from its peak to near zero along a cosine curve. It has
been the standard choice for several years and it performs well, but it carries a structural
constraint. The curve is defined in terms of the total number of steps, so the length of the run must
be fixed before the first step is taken. A run stopped early has not completed its decay, and the
model it leaves behind is worse than one trained to that shorter length deliberately.

**WSD**, for warmup, stable and decay, holds the peak rate flat for an unspecified duration and
decays only over the final few percent of the run. Because the flat phase has no predetermined end,
we can save the weights at any point, which is called a **checkpoint**, and decay separately from
there. One run therefore yields finished models at many budgets and can continue afterwards.

    warmup (~2% of steps) → stable (flat, any length) → decay (last ~10%)
                            └ checkpoint here and decay separately

A result published in 2026 goes further. Holding the learning rate constant for the whole run while
keeping an exponential moving average of the weights, which is the mechanism of Section 4 applied to
w rather than to g, produces averaged weights that match cosine at every point along the run. The
decay phase and weight averaging turn out to serve the same purpose, which is suppressing noise over
the final stretch. This has been measured at 150M and 300M parameters and not at our scale, so it
stays unconfirmed for a run of our size.

> Carry this forward: cosine needs the length of the run in advance and WSD does not, which matters
> more than any difference in final loss.

*Widget: Schedule comparison. Cosine, WSD and constant with weight averaging on one axis. Move the
stopping point and each shape yields a different quality of model.*

## 11. Batch Size and Learning Rate Scaling

Session 10 established that the global batch size is chosen independently of what fits in memory.
That choice has a consequence, because it changes the appropriate learning rate.

A gradient computed from a batch is an estimate, and averaging more samples improves it. The average
of N independent samples carries 1/√N of the noise of a single one, so against a batch of 8 the noise
is √(8/N). At V4's batch of 56 that is √(8/56) = 0.378:

| global batch | noise, relative to 8 |
|---|---|
| 8 | 1.000 |
| 32 | 0.500 |
| 56, which V4 used | 0.378 |
| 128 | 0.250 |
| 512 | 0.125 |

A more accurate gradient can be trusted over a longer step, so a larger batch supports a larger
learning rate. For gradient descent the relationship is linear, and four times the batch permits four
times the rate. For Adam it is the square root, and four times the batch permits twice the rate,
because Adam has already divided out the magnitude of the gradient and only its consistency remains.

Two separate mechanisms raise the global batch, and the learning rate distinguishes neither of them.
Session 10 built **gradient accumulation**, which runs several micro-batches one after another and
steps once at the end. **Data parallelism** runs those micro-batches at the same time on several GPUs
and averages their gradients before the step. To the optimizer these are the same change.

    global batch = micro-batch × GPUs × accumulation steps

V4 ran a micro-batch of 7 on each of 8 GPUs with no accumulation, which is where its global batch of
56 comes from.

| Change | Global batch | Learning rate under Adam |
|---|---|---|
| accumulation steps from 1 to 4 | ×4 | ×2 |
| GPUs from 8 to 32 | ×4 | ×2 |
| both together | ×16 | ×4 |

The trap is in how the gradients are combined. Data parallelism averages across GPUs, so nothing
further is required. Accumulation adds, so the sum has to be divided before the step, and Session 10
showed that dividing by the number of micro-batches instead of by the number of tokens is itself a
bug. Getting this wrong multiplies the learning rate by the accumulation count without anyone having
chosen to.

Both relationships have a limit. Past a threshold called the **critical batch size** the gradient is
already accurate enough that more samples buy almost nothing, and the extra compute is spent for no
return. Where that threshold sits is a property of the model and the data, and it has to be measured.

> Carry this forward: under Adam, four times the batch permits twice the learning rate, up to the
> critical batch size. More GPUs and more accumulation steps are the same change, because only the
> global batch enters the rule.

*Widget: Batch size and learning rate. The linear rule against the square-root rule, and the point at
which doubling the batch produces no further improvement.*

## 12. Hyperparameter Transfer (muP)

Finding the learning rate by sweeping it on a small model and applying the result to a large one
would be inexpensive, but it is not valid, because the best learning rate depends on the width of the
model. Under the standard parameterization it varies roughly in inverse proportion to width:

| width | approximate best η |
|---|---|
| 256 | 3.0 × 10⁻³ |
| 512 | 1.5 × 10⁻³ |
| 1,024 | 7.5 × 10⁻⁴ |
| 2,048 | 3.8 × 10⁻⁴ |
| 4,096 | 1.9 × 10⁻⁴ |

Carrying the value from width 256 across to width 4,096 would overstate it by a factor of sixteen.

**muP** changes how the initialization scale and the per-layer learning rates depend on width, so
that this dependence cancels. Under it the curve of loss against learning rate has its minimum at the
same value at every width, and a sweep at width 256 therefore determines the value for width 4,096.
The same transfer has been demonstrated for weight decay, at model sizes of several billion
parameters.

For this run the difference is one day of small sweeps against one estimate.

> Carry this forward: muP does not improve the model. It makes the small model's measurement valid
> for the large one.

*Widget: Learning rate transfer. Loss against learning rate at four widths, with the minima drifting
leftwards under the standard parameterization and aligning under muP.*

## 13. Matrix Optimizers: Muon, MuonClip, Hyperball

Adam treats a weight matrix as a collection of independent numbers and scales each one separately. A
matrix is not a collection of independent numbers. It maps an input vector to an output vector, and
it amplifies some input directions more than others. Those amplification factors are the matrix's
**singular values**, and when a few of them are much larger than the rest, nearly all of the matrix's
effect is confined to a few directions while the others contribute very little.

The momentum matrix has exactly this property. **Muon** replaces it with the nearest matrix whose
singular values are all equal to 1, so every direction is updated by the same amount. It stores only
m, which puts it at 12 bytes per weight against AdamW's 16.

The reported results:

| Finding | 2026 |
|---|---|
| Speedup over a well tuned AdamW | 1.4x at 0.1B, falling to 1.1x at 1.2B |
| Tokens to reach the same loss at very large batches | 10 to 15% fewer |
| Batch size at which AdamW destabilises and Muon holds | around 100M tokens |
| Kimi K2, one trillion parameters | 15.5T tokens, zero loss spikes |

The first row governs the decision. Most published two-fold speedups were measured against an AdamW
baseline that had not been tuned to the same standard. When both sides are tuned equally the
advantage is real but modest, and it decreases as the model grows.

Muon introduces a failure of its own. Making every direction equally strong lets the query and key
matrices grow without limit, and the attention scores of Session 8 then rise past 1,000, at which
point the softmax saturates and the run fails. **MuonClip** rescales the query and key matrices
directly after each update, so the correction lands on the weights rather than after the softmax.
This is the mechanism that carried Kimi K2 through 15.5 trillion tokens without a loss spike.

**Hyperball**, published in June 2026, replaces weight decay's indirect control of matrix magnitude
with a direct projection onto a fixed magnitude after every step. It reports 20 to 30% over AdamW at
1.2B, increasing with the length of the run, against approximately 10% for Muon with conventional
decay.

One constraint holds across every recipe that has been shown to work. Muon is applied only to
two-dimensional weight matrices. Embeddings, normalization scales and the output head stay on AdamW,
because they do not amplify directions and the argument does not apply to them.

> Carry this forward: a reported speedup is a statement about someone else's baseline until both
> sides have been tuned to the same standard.

## 14. V5 Decisions

This session settles four decisions. We train with AdamW and decoupled decay, excluding normalization
scales and biases from it. We warm up over the first few thousand steps, and we log the
update-to-weight ratio per layer from step one, alongside the grad norm that Session 10 put on the
dashboard. We treat ηλ as one setting rather than two. The schedule is WSD, because the run has to be
able to stop, branch and continue.

Four decisions stay open.

| Question | What would settle it |
|---|---|
| AdamW or Muon for the two-dimensional matrices? | A matched short run at our real width with both sides tuned to the same standard, taking the 1.1x figure as the prior. |
| Whether to spend a day on a muP sweep at widths 256 to 2,048. | The sweep costs about one day and fixes every hyperparameter that follows, which makes it the highest-value day available to us. |
| One learning rate, or a separate one for the output head? | The head is 536.9M weights and dense. Log its update-to-weight ratio against the body's for a thousand steps and compare. |
| Whether to decay the Kronecker factors from Session 7. | No published result exists. This is an ablation the cohort can run. |

## 15. Assignment

> Reproduce Adam by hand. Take one weight and five gradients, compute m, v, m̂, v̂ and the resulting
> step yourself, then check each against PyTorch. They should agree to several decimal places.
>
> Disable bias correction and plot the first twenty steps both ways. Report the number of steps after
> which the difference stops mattering.
>
> Log the update-to-weight ratio for every layer, and identify the step at which warmup stops
> changing it.
>
> Train the same model twice for 300 steps, once under cosine and once under WSD, and stop both at
> step 200. Report both losses and state which model you would keep.
>
> Sweep the learning rate at widths 256, 512 and 1,024, plot loss against learning rate, and mark the
> three minima. State the value you would use at width 4,096 and how confident you are in it.
>
> Tune both sides before accepting a comparison. Almost every optimizer claim that failed to
> replicate was a well tuned method measured against a badly tuned one.
