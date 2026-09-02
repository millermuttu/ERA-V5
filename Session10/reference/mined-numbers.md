# Session 10 — numbers mined from the widgets

Everything below was read out of the widget source under `widgets/`, not out of the lesson prose.
Where the lesson and a widget disagree, the disagreement is noted — the widget is the executable
version and is what an implementation gets measured against.

The two shared widgets (`shared_one_neuron.html`, `shared_training_loop.html`) are reused from
earlier sessions and carry no Session 10 baselines; they are omitted here.

---

## widget_1_step.html — one step, start to finish

Fixed network: `x = 2`, `h = w1·x`, `y = w2·h`, `loss = (y − t)²`.

| Constant | Value | Meaning |
| -------- | ----- | ------- |
| `X` | 2 | the only input |
| initial `w1` | 3 | |
| initial `w2` | 4 | |
| `B1` | 0.9 | Adam β₁ |
| `B2` | 0.999 | Adam β₂ |
| `EPS` | 1e-8 | Adam epsilon |
| `CAP` | 24 | run stops after 24 steps |
| run interval | 420 ms per step | |

Sliders:

| Slider | min | max | default | maps to |
| ------ | --- | --- | ------- | ------- |
| `lr` | 1 | 40 | 10 | learning rate = value/100, so default **0.10** |
| `t` (target) | 4 | 40 | 20 | target for input 2 |

Gradients computed: `gy = 2(y−t)`, `gw2 = gy·h`, `gh = gy·w2`, `gw1 = gy·w2·x`.
At defaults: h=6, y=24, loss=16, gy=8, gw2=48, gh=32, gw1=64 — the lesson's §4 table.

Optimiser state update as written in the widget (note the literal `0.1` and `0.001` rather than
`1−B1` and `1−B2` — numerically identical here since 1−0.9 = 0.1 and 1−0.999 = 0.001):
`m = B1·m + 0.1·g`, `v = B2·v + 0.001·g²`, bias-corrected by `1−B1ⁿ` and `1−B2ⁿ`.

## widget_2_floats.html — how a computer holds a number

Format table, as `[name, exponent bits, mantissa bits]`:

| Format | exponent | mantissa | total bits |
| ------ | -------- | -------- | ---------- |
| fp32 | 8 | 23 | 32 |
| fp16 | 5 | 10 | 16 |
| bf16 | 8 | 7 | 16 |
| fp8 E4M3 | 4 | 3 | 8 |
| fp8 E5M2 | 5 | 2 | 8 |
| fp4 E2M1 | 2 | 1 | 4 |
| custom | 6 (default) | 9 (default) | 16 |

- Bias formula: `2^(eb−1) − 1`. Subnormal scale fixed at `2^(1−bias)`.
- Rounding is **round-half-to-even** (`rne`), matching IEEE 754 default.
- All-ones exponent is reserved (infinity if mantissa 0, else "reserved pattern").
- int8 row: scale = `max(1, |x|) / 127`, value stored as `round-half-even(x/scale)`.
- Default value stored: **0.1**. Quick-pick values: `0.1`, `1e-8`, `3.14159`, `65000`.
- Custom-format sliders: exponent 2–11 (default 6), mantissa 1–23 (default 9).

The assignment's "write 0.1 in fp32, bf16, fp8 E4M3" is exactly this widget's default row.

## widget_3_nudge.html — the finite-difference gradient

Same network as widget 1, with `w2` and the target held fixed: `W2 = 4`, `X = 2`, `T = 20`.
Loss curve plotted over `w1 ∈ [1.4, 4.4]`, loss axis capped at `LMAX = 240`.
Bottom of the curve is at `w1 = 2.5` (loss 0, slope 0).

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `wt` (weight ×1000) | 1500 | 4000 | 25 | 3000 → **w1 = 3.0** |
| `nu` (nudge index) | 0 | 4 | 1 | 4 |

Nudge sizes `NUD = [0.5, 0.2, 0.05, 0.01, 0.001]` — index 4 is the default **0.001**.

Analytic slope: `2(y − 20)·W2·X = 2(8·w1 − 20)·8`. At w1 = 3 that is **64**.
The measured loss readings are rounded to 3 decimals (`Math.round(l*1000)/1000`) before the
difference is taken, which is why the widget's 0.064/0.001 reproduces the lesson exactly.
Agreement tolerance used for the "calculus gives the same number" message: `|m − exact| < 5e-4`.

## widget_4_chain.html — backprop one link at a time

Fixed `X = 2`, `W1 = 3`, `W2 = 4`; four stages, one per link.
Target slider `t`: min 4, max 40, default **20**.
Stage outputs at defaults: `gy = 8`, `gw2 = 48`, `gh = 32`, `gw1 = 64`.

## widget_5_accumulate.html — gradient accumulation

Per-example gradient: `g(i) = 2(12x − 20)·4·x` where `x = 1 + ((i−1) mod 8)·0.25`,
so the eight distinct inputs are 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75 (`W2 = 4`, `T = 20`).

| Slider | min | max | default |
| ------ | --- | --- | ------- |
| `mb` micro-batch size | 1 | 8 | 4 |
| `ac` accumulation steps | 1 | 12 | 8 |

Default global batch = 4 × 8 = **32 examples**. The claim the widget checks: accumulating the
`ac` micro-batch means and dividing by `ac` equals the mean over all `mb·ac` examples "to the last
digit" — true only because every micro-batch holds the same count, which is the setup section 8
then breaks.

## widget_6_wrong_average.html — the token-weighting bug

Fixed losses `L = [2, 2, 5]`. Default token counts `D = [4, 4, 2]` (also the reset values).

| Slider | min | max | default |
| ------ | --- | --- | ------- |
| `n0` tokens in micro-batch 1 | 1 | 8 | 4 |
| `n1` tokens in micro-batch 2 | 1 | 8 | 4 |
| `n2` tokens in micro-batch 3 | 1 | 8 | 2 |

- Correct: `Σ nᵢLᵢ / Σ nᵢ` = 26/10 = **2.6000**
- Naive: `ΣLᵢ / 3` = 9/3 = **3.0000**
- Error: (3.0 − 2.6)/2.6 = **15.4%** — the lesson's figure.
- Batch 3 holds 20.0% of the tokens but gets 33.3% of the vote, so its loss counts
  **1.67× heavier** than its tokens support (`(N/3)/n2` = 3.333/2).
- Error reads exactly 0.0% whenever all three counts are equal (`|pct| < 0.05` is the zero test).

## widget_7_underflow.html — fp16 underflow and loss scaling

Formats compared: fp32 (e8,m23), bf16 (e8,m7), fp16 (e5,m10).
Smallest positive (subnormal) floor computed as `2^(2 − 2^(e−1) − m)`:

| Format | floor |
| ------ | ----- |
| fp32 | 1.40e-45 |
| bf16 | 9.18e-41 |
| fp16 | 5.96e-08 |

bf16 reaches roughly **33 decades** further down than fp16 (`log10(fp16 floor / bf16 floor)`).
fp16 ceiling used for the overflow message: `(2 − 2⁻¹⁰)·2¹⁵ = 65504`.

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `g` gradient magnitude | −450 | 0 | 1 | −80 → **1e-8** (value is `10^(g/10)`) |
| `s` loss scale exponent | 0 | 16 | 1 | 0 → **×1** (scale is `2^s`) |

Quick button sets `g = −80, s = 10` → gradient 1e-8 lifted by **×1024**, which is the lesson's
`10⁻⁸ × 1024 = 1.02 × 10⁻⁵`. Reset button returns to `g = −80, s = 0`.
The scale ceiling matters as well as the floor: at a high enough `s` the value saturates to
infinity, "which is why real trainers halve it whenever an infinity appears".

## widget_8_block_scale.html — block-shared exponents (fp4/NVFP4)

fp4 E2M1 grid of representable magnitudes: `G = [0, 0.5, 1, 1.5, 2, 3, 4, 6]` (top value 6 = the
reason the shared exponent shifts by 2).

Shared scale: `2^(floor(log2(block max)) − 2)`, stored once as an 8-bit exponent with bias 127.

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `n` values per block | 4 | 32 | 4 | **16** |
| `o` outlier size (×/10 of the rest's max) | 10 | 120 | 5 | 15 → **1.5×** |

Bit budget: `(n × 4 + 8) / n` bits per value — at n = 16 that is **4.50 bits**, the lesson's
figure. Memory shrink vs fp32 = `n·32 / (n·4 + 8)` → **7.11×** at n = 16.
Random block is deterministic: LCG seeded `sd = 7`, `sd = (sd·1103515 + 12345) mod 2147483647`,
32 values drawn as `(r+r+r−1.5)·0.62 ± 0.02`.
Widget warns (`.warn`) once a quarter or more of the block is crushed to zero (`z*4 >= n`).

## widget_9_clip.html — gradient clipping and the grad norm

Simulation: `N = 40` steps, the spike injected at step `SP = 24`.
Baseline norm at step i: `0.30 + 0.55·exp(−i/14) + 0.22·rnd(i)`, `rnd` a deterministic
`sin(i·12.9898)·43758.5453` hash. Loss starts at 3.2, decays ×0.955 per step, and the uncapped
run adds `max(0, a[i−1] − 1.2)·0.42` one step later — that one-step lag is why the loss reacts
after the norm does.

| Slider | min | max | default | maps to |
| ------ | --- | --- | ------- | ------- |
| `cap` | 1 | 120 | 10 | cap = value/10 → **1.0** |
| `spk` spike norm | 20 | 120 | 84 | spike = value/10 → **8.4** |

At defaults: scale factor = `cap/spk` = 1.0/8.4 = **0.119**, the lesson's number.
Example gradient direction held fixed at `DIR = [0.75, 0.5, 0.4330127]` (a unit vector), so
clipping visibly changes length only, never direction.
The lesson's "spike at 24, loss reacts at 26" is the widget's default readout.

## widget_10_memory.html — 16 bytes per weight

Per-weight state, in bytes:

| Piece | bytes | share of 16 |
| ----- | ----- | ----------- |
| weight, bf16 | 2 | 12.5% |
| gradient, bf16 | 2 | 12.5% |
| master copy, fp32 | 4 | 25.0% |
| Adam m, fp32 | 4 | 25.0% |
| Adam v, fp32 | 4 | 25.0% |
| **total** | **16** | 100% |

Card size `CARD = 80` GiB, `GIB = 1073741824`.
One card fills at `FIT = 80·2³⁰/16` = **5.4B weights** — the lesson's 5.4B.
Node line drawn at 8 × 80 = **640 GiB**.

| Slider | min | max | default |
| ------ | --- | --- | ------- |
| `n` model size (×0.5B) | 1 | 240 | 18 → **9B** |

Quick picks: 4 → 2B, 18 → 9B, 40 → 20B, 240 → 120B. Chart y-ticks at 4, 32, 256, 2048 GiB.
Training state = `n × 16 / 2³⁰` GiB, reproducing 29.8 / 134.1 / 298.0 / 1788 GiB.

## widget_11_mfu.html — Model FLOPs Utilisation

| Constant | Value |
| -------- | ----- |
| `PEAK` | 989e12 FLOP/s per H100 |
| `BUD` | 100e9 tokens, the training budget used for the wall-clock comparison |
| `HEAL` | 0.45, the healthy MFU the second panel is drawn at |
| healthy band shaded | 35% to 50% |

| Slider | min | max | default | maps to |
| ------ | --- | --- | ------- | ------- |
| `n` model size | 1 | 240 | 18 | ×0.5B → **9B** |
| `t` tokens/s | 4 | 360 | 24 | ×500 → **12,000 tokens/s** |
| `g` GPUs | 1 | 64 | 8 | **8 H100** |

Formulae: `achieved = 6·N·tokens/s`, `peak = g · 989 TFLOP/s`, `MFU = achieved/peak`.
At defaults: achieved = **648 TFLOP/s**, peak = **7,912 TFLOP/s**, MFU = **8.2%** — the lesson's
table exactly. Gauge colouring: green ≥ 35%, amber ≥ 20%, red below.
Wall clock compared as `BUD/tokens per second` against the same budget at 45% MFU; the ratio is
the "four times as long" claim (default ratio ≈ 5.5×).
Anything above 100% is flagged as impossible — "the token count or the model size is wrong".

---

## Gates and thresholds worth carrying into an implementation

| Gate | Value | Source |
| ---- | ----- | ------ |
| healthy MFU band | 35%–50% | widget_11 (`HEAL = 0.45`, shaded band) |
| MFU impossible above | 100% | widget_11 |
| grad-norm clip cap, default | 1.0 | widget_9 |
| one 80 GiB card is full at | 5.4B weights | widget_10 |
| training state per weight | 16 bytes | widget_10 |
| FLOPs per weight per token | 6 | widget_11 |
| loss-scale default when enabled | ×1024 (2¹⁰) | widget_7 quick button |
| fp16 usable floor | 5.96e-08 | widget_7 |
| bf16 usable floor | 9.18e-41 | widget_7 |
| block-scale bits per value at n=16 | 4.50 | widget_8 |
| token-weighting bug, at [4,4,2]×[2,2,5] | 15.4% | widget_6 |
| finite-difference agreement tolerance | 5e-4 | widget_3 |
