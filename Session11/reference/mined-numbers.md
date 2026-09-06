# Session 11 — numbers mined out of the widgets

Read out of the inline scripts of the twelve widgets under `widgets/`. Slider ranges are given as
`min .. max (default)`. These are the baselines an implementation gets measured against.

## widget_1_descent — gradient descent in two dimensions

| Constant | Value |
|---|---|
| `STEPS` | 5 |
| start `(U0, V0)` | (1, 1) |
| curvatures `KU`, `KV` | 20, 1 |
| loss | `0.5 * (20u² + v²)` |
| contour levels `CONT` | 0.14, 0.28, 0.45, 0.65, 0.88, 1.15, 1.5, 1.95, 2.5 |
| plot window | u ∈ [−2.8, 2.8], v ∈ [−1.15, 1.3] |

- η slider: `min=5 max=130 step=1 value=90` in thousandths, i.e. **η = 0.005 .. 0.130, default 0.090**.
- Multipliers displayed: u by `(1 − 20η)`, v by `(1 − η)`.
- Divergence ceiling stated by the widget: past **η = 0.100** the factor (1 − 20η) drops below −1 and
  u grows without limit.
- Steps for v to fall below 0.100: `ceil(log(0.1)/log(1 − η))`.

## widget_1b_surface — the anisotropic surface

- Curvature ratio slider `r`: `min=1 max=50 step=1 value=20` — the ratio KU/KV.
- Contour amplitudes `AV` = 0.2, 0.4, 0.6, 0.8, 1.0; ellipse aspect = √r.
- Convergence test in code: multiplier 0 → converges in 1 step; |m| ≥ 1 → never; otherwise
  `ceil(log(0.1)/log|m|)` steps to reach a tenth.

## widget_2_momentum — exponential moving average

| Constant | Value |
|---|---|
| oscillating gradient sequence `OSC` | +1, −1, +1, −1, +1 |
| steady gradient sequence `STD` | 0.2, 0.2, 0.2, 0.2, 0.2 |
| number of hand-entered gradients `NG` | 5 |
| curvatures `KU`, `KV` | 20, 1 |
| `ETA` for the trajectory | 0.010 |
| trajectory steps `NS` | 24 |
| start | (1, 1) |

- β slider: `min=0 max=95 step=5 value=90` in hundredths → **β = 0.00 .. 0.95, default 0.90**.
- EMA: `m ← β·m + (1 − β)·g`; the trajectory divides the step by `(1 − β)`.
- Lesson table (β = 0.9, five steps): steep 0.100, −0.010, 0.091, −0.018, 0.084;
  shallow 0.020, 0.038, 0.054, 0.069, 0.082.

## widget_2b_directions — the two gradient shapes

- `STEPS` = 5, `KU` = 20. Fixed illustration, no controls; shows the alternating vs. constant
  gradient patterns and their running averages side by side.

## widget_3_per_parameter — per-parameter learning rates

| Constant | Value |
|---|---|
| Parameter A gradient `GA` | 1.0 |
| `B2` | 0.999 |
| steps `T` | 200 |
| `ETA` (shown as multiples of η) | 1 |

- Parameter B gradient slider: `min=-3 max=0 step=0.05 value=-2` as a log10 exponent →
  **g_B = 10⁻³ .. 10⁰, default 10⁻² = 0.01**.
- `v` after T steps: `(1 − β₂^T)·g²`; `v̂ = v/(1 − β₂^T)`, so **√v̂ = |g|** exactly.
- Division toggle (`div` checkbox) switches between step `η·g` and step `η·g/√v̂`.
- With the division on, both parameters take the same step 1.0000 η; B's own learning rate is
  100.00 η against A's 1.00 η.

## widget_4_adam — Adam by hand, five steps

| Constant | Value |
|---|---|
| `ETA` | 1e-3 |
| `B1` | 0.9 |
| `B2` | 0.999 |
| `EPS` | 1e-8 |
| `W0` | 1.0 |
| `N` | 5 |

Three preset gradient sequences (`<select>` options, default index 0):

| # | sequence |
|---|---|
| 0 | 0.50, 0.40, 0.60, 0.45, 0.55 |
| 1 | 0.0050, 0.0040, 0.0060, 0.0045, 0.0055 |
| 2 | 0.50, −0.40, 0.60, −0.45, 0.55 |

- Bias-correction checkbox (`bc`). Without it the first step is **3.16 η**, with it **1.00 η**.
- Preset 0 reproduces the lesson table, ending at w = 0.995031 after five steps.

## widget_5_decay — L2 against decoupled decay

| Constant | Value |
|---|---|
| `W0` | 0.5 |
| `LAM` (λ) | 0.1 |
| `ETA` | 1e-3 |
| steps `N` | 200 |

- √v̂ slider: `min=-2301 max=301 step=1 value=-2000`, read as `10^(value/1000)` →
  **√v̂ = 0.005 .. 2.0, default 0.01**.
- Decoupled shrinkage per step: `η·λ·w = 5.0e-5`, independent of √v̂.
- L2 shrinkage per step: `η·λ·w/√v̂`, so the ratio between the two routes is `1/√v̂`.
- Per-step retention factors: L2 `1 − ηλ/√v̂`, decoupled `1 − ηλ`.

## widget_6_warmup — update-to-weight ratio

| Constant | Value |
|---|---|
| `W0` (init scale) | `1/√4096` = 0.015625 |
| `PEAK` (η) | 3e-4 |
| `FLOOR` (noisy-gradient step, fraction of η) | 0.281 |
| total steps `N` | 10,000 |
| correlation decay `TAU` | 1200 |
| weight-growth factor `GROW` | 4.2 |
| weight-growth half-life `THALF` | 1500 |
| healthy band `LO`, `MID`, `HI` | 5e-4, 1e-3, 2e-3 |

- Warmup-length slider: `min=0 max=4000 step=50 value=2000` steps.
- Warmup toggle on/off; `lr(n) = PEAK · min(1, n/W)` when on, `PEAK` when off.
- Gradient consistency: `cons(n) = FLOOR + (1 − FLOOR)·exp(−(n−1)/TAU)` — 1.000 η at step 1 falling
  to 0.281 η once gradients decorrelate.
- Weight magnitude: `wmag(n) = W0·(1 + (GROW−1)·t/(t + THALF))`, t = n−1.
- Ratio plotted: `cons(n)·lr(n)/wmag(n)`. Peak **19.2e-3 without warmup, 2.83e-3 with the default
  2,000-step warmup**. Target band is 10⁻³.

## widget_7_schedules — cosine, WSD, constant + weight averaging

| Constant | Value |
|---|---|
| `PEAK` | 3e-4 |
| axis length `AX` | 100,000 steps |
| samples `NS` | 140 |
| EMA smoothing on the loss trace | `a = 2/13` |

- Stop-step slider: `min=1 max=100 value=45`, in thousands → **stop at 1k .. 100k steps, default 45k**.
- Cosine total-length slider: `min=20 max=100 step=5 value=100`, in thousands → **planned run length
  20k .. 100k steps, default 100k**.
- `cosine(t,T) = PEAK · 0.5 · (1 + cos(π t/T))`, zero past T.
- `wsd(t,T)`: warmup `W = 0.02·T`, stable until `D = 0.9·T`, then linear decay over the final
  `0.1·T`. So **warmup 2% of the run, decay the last 10%**.
- "Flat" test for the WSD readout: `|lr − PEAK| ≤ 3% of PEAK`.

## widget_8_batch — batch size and learning rate scaling

| Constant | Value |
|---|---|
| batch ladder `NS` | 8, 16, 32, 56, 64, 128, 256, 512, 1024 |
| reference batch `REF` | 8 |
| `MAXN` | 1024 |
| true gradient `(TX, TY)` | (1.7, 1.2) |
| per-sample noise `SIG` | 1.0 |
| RNG seed | 2473 |

- Batch slider: `min=0 max=8 value=3` — an index into `NS`, so **default 56, which is V4's global
  batch** (marked in the table).
- `noise(n) = √(REF/n)`; linear rule `n/REF`; square-root rule `√(n/REF)`.
- At 4× the batch: gradient descent 4.00x the learning rate, Adam 2.00x.
- Critical batch size is named but deliberately not given a value — the widget states it has to be
  measured per model and dataset.

## widget_9_transfer — learning rate transfer (muP)

| Constant | Value |
|---|---|
| widths `W` | 256, 512, 1024, 2048 |
| `TARGET` width | 4096 |
| `K` (standard parameterization) | 0.768 |
| `BASE` (muP best η, width-independent) | 3.0e-3 |
| η axis `LO` .. `HI` | 1e-4 .. 1e-2 |
| loss axis `YLO` .. `YHI` | 2.4 .. 3.9 |
| default mode / sweep width | standard parameterization, width 256 |

- Best η under the standard parameterization: `best(w) = K/w` → 3.0e-3 at 256, 1.5e-3 at 512,
  7.5e-4 at 1024, 3.75e-4 at 2048, **1.875e-4 at 4096**. Ratio 256 → 4096 is **16x**.
- Under muP: `best(w) = 3.0e-3` at every width.
- Loss model: `lmin(w) = 3.05 − 0.2·log₂(w/256)`, plus `0.30c²` below the optimum and `0.55c²`
  above it, with `c = log₁₀(η/best)` — the curve is asymmetric, punishing too-large η harder.

## widget_10_memory — optimizer state memory

Fixed cost every run pays (`FIX`), 8 bytes per weight:

| What is stored | Bytes |
|---|---|
| weight, bf16 | 2 |
| gradient, bf16 | 2 |
| master copy, fp32 | 4 |

Optimizer state on top (`OPT`):

| Optimizer | extra bytes | total | what it stores |
|---|---|---|---|
| gradient descent | 0 | 8 | no extra state |
| momentum | 4 | 12 | m in fp32 |
| AdamW (default selection) | 8 | 16 | m and v in fp32 |
| 8-bit AdamW | 2 | 10 | m and v in int8 |

- Model-size slider: `min=1 max=240 value=18`, each unit 0.5B → **0.5B .. 120B weights, default 9B**.
- `GIB` = 1073741824; card capacity `CARD` = 80e9 bytes = **74.5 GiB**.
- 9B at 16 bytes = 134.1 GiB; at 12 = 100.6 GiB; at 8 = 67.1 GiB; at 10 = 83.8 GiB.
- The widget also reports the model size at which one 80 GB card fills: `CARD/bytes-per-weight`.
- x-axis ticks: 0.5B, 2B, 9B, 20B, 120B. y-axis ticks: 4, 32, 256, 2048 GiB.
