# Session 16 — numbers mined from the widget sources

Everything below is read out of the inline `<script>` of each file under `widgets/`. The end states were
reproduced by running each widget's own code under node against a stub DOM. "Default" means the slider
value or the button lit on load. Clicks in the tables below are cumulative unless a row says otherwise.

The eight `fig_*` files are static figures (hover tooltips only); their few constants are at the end.

## widget_1_flops — 6ND against the fuller DeepSeek count

| Control | min | max | step | default |
|---|---|---|---|---|
| layers `L` (slider) | 8 | 128 | 4 | 60 |
| hidden `d` (slider) | 1,024 | 16,384 | 512 | 7,168 |
| tokens `D` (slider, T) | 1 | 60 | 1 | 15 |
| context (buttons) | 4,096 / 8,192 / 32,768 / 131,072 / 1,048,576 | | | **4,096** |
| active fraction (buttons) | 3% / 5.5% / 10% / 100% dense | | | **3%** |

Formulas: `N = 12·L·d²` (active), total `= N / active`, `6ND`, `M = 72·L·d² + 12·L·d·l`, full `= M·D`,
attention share `= l / (6d + l)`. Colour gate on the share: green < 10%, amber < 30%, red ≥ 30%.

Default model (60 × 7,168, 15T): N = 37.0B active, 1.23T total at 3%, 6ND = 3.33e24.

| Context | M per token | full count | full / 6ND | attention share |
|---|---|---|---|---|
| 4,096 | 2.43e11 | 3.65e24 | 1.10 | 8.7% (green) |
| 8,192 | 2.64e11 | 3.96e24 | 1.19 | 16.0% (amber) |
| 32,768 | 3.91e11 | 5.87e24 | 1.76 | 43.2% (red) |
| 131,072 | 8.98e11 | 1.35e25 | 4.05 | 75.3% |
| 1,048,576 | 5.63e12 | 8.45e25 | 25.38 | 96.1% |

The share depends only on `l` and `d`; the active fraction changes only the total-parameter line.

## widget_2_budget — the budget equation

| Slider | min | max | step | default (example) | MAI-Base-2 preset |
|---|---|---|---|---|---|
| GPUs | 1,000 | 32,000 | 1,000 | 4,000 | 16,000 |
| peak PFLOP/s | 0.5 | 5 | 0.1 | 2.5 | 2.5 |
| MFU % | 5 | 60 | 1 | 30 | 21 |
| goodput % | 50 | 100 | 1 | 90 | 89 |
| days | 1 | 120 | 1 | 30 | 38 |

`C = GPUs × peak·1e15 × MFU × days·86,400 × goodput`, compared with `C6 = 6 × 80e9 × 50e12 = 2.40e25`.
Gate: green within ±5% of C6, red below, amber above.

| Preset | seconds | C | lost share `1 − MFU·goodput` | C / C6 | tokens at 80B active |
|---|---|---|---|---|---|
| example (default) | 2.59e6 | 7.00e24 | 73.0% | 0.292 (red) | 14.6T |
| MAI-Base-2 | 3.28e6 | 2.45e25 | 81.3% | 1.023 (green, "about 2%") | 51.1T |

MAI note computes idle GPUs as `16,000 × 11% = 1,760` (the prose says about 1,750).

## widget_3_isoflop — Chinchilla coefficients and the flat valley

| Fit | A | B | E | α | β |
|---|---|---|---|---|---|
| Hoffmann 2022 (`H`) | 406.4 | 410.7 | 1.6934 | 0.3392 | 0.2849 |
| Epoch refit 2024 (`E`, default) | 482.01 | 2085.43 | 1.8172 | 0.3478 | 0.3658 |

`N_opt = G·(C/6)^(β/(α+β))`, `G = (αA / βB)^(1/(α+β))` — G = 0.120, exponent 0.513 (Epoch); G = 1.300,
exponent 0.456 (Hoffmann). Valley = sizes whose loss is within 0.01 of the minimum. Compute buttons
1e21 / 1e22 / **1e23** / 5.88e23 (Chinchilla) / 1e24 / 1e25.

| C | Epoch N_opt | Epoch TPP | Epoch valley (×) | Hoffmann N_opt | Hoffmann TPP | Hoffmann valley (×) |
|---|---|---|---|---|---|---|
| 1e21 | 2.78B | 21.6 | 1.58B–4.89B (3.1) | 2.21B | 34.0 | 1.24B–3.98B (3.2) |
| 1e22 | 9.05B | 20.4 | 4.52B–18.1B (4.0) | 6.34B | 41.5 | 3.16B–12.8B (4.0) |
| 1e23 (default) | 29.4B | 19.2 | 12.6B–68.9B (5.5) | 18.1B | 50.7 | 7.93B–42.0B (5.3) |
| 5.88e23 | 73.0B | 18.4 | 26.9B–197B (7.3) | 40.7B | 59.2 | 15.8B–107B (6.8) |
| 1e24 | 95.9B | 18.1 | 33.7B–271B (8.1) | 51.9B | 62.0 | 19.3B–142B (7.3) |
| 1e25 | 312B | 17.1 | 86.5B–1.11T (12.9) | 148B | 75.7 | 45.8B–493B (10.8) |

Epoch TPP drifts from about 22 down to about 17; Hoffmann climbs from 34 to 76. The valley widens with
compute.

## widget_4_inference — inference-aware sizing (Sardana)

Uses the Epoch coefficients. For target loss `Lt`, `D(N)` is solved from the loss law; total cost
`6ND + 2NT`. Minimum size `N_floor = (A / (Lt − E))^(1/α)`. Tokens-served buttons none / 1T / **10T** /
100T / 1,000T; target loss **2.00** / 2.10 / 1.95. Over-training warning when TPP > 10,000.

At `Lt = 2.00`:

| Tokens served | best N | TPP | total FLOPs | compute-optimal N | smaller by | saving |
|---|---|---|---|---|---|---|
| none | 46.9B | 19 | 2.47e23 | 46.9B | — | — |
| 1T | 34.4B | 36 | 3.26e23 | 46.9B | 1.4× | 4% |
| 10T (default) | 20.0B | 151 | 7.62e23 | 46.9B | 2.3× | 36% |
| 100T | 12.7B | 883 | 3.39e24 | 46.9B | 3.7× | 65% |
| 1,000T | 9.61B | 5,256 | 2.21e25 | 46.9B | 4.9× | 76% |

At 1,000T served: `Lt = 2.10` gives 2.51B at 13,660 TPP (warning fires), `Lt = 1.95` gives 26.3B at 2,553
TPP (compute-optimal 117B).

## widget_5_repeat — Muennighoff effective data

`R*_D = 15.4`. Epochs `E = D/U`, `R_D = E − 1`, `D' = U + U·15.4·(1 − e^(−R_D/15.4))`, value of the
repeats `= (D' − U)/(D − U)`. Unique buttons 0.25 / 0.5 / **1** / 2 / 4.5 T; total 5 / **15** / 30 / 50 T;
mode **single source** / inside a mixture.

Single-source gates: ≤ 4 green, ≤ 16 amber, ≤ 40 red "falls fast", > 40 red "almost worthless". Mixture
gates (Sedova 2026): ≤ 15 green, ≤ 20 amber, > 20 red.

| U | D | epochs | D' | repeat value |
|---|---|---|---|---|
| 1T | 15T (default) | 15.0 | 10.2T | 66% |
| 1T | 50T | 50.0 | 15.8T | 30% |
| 4.5T | 50T (MAI-like) | 11.1 | 37.9T | 73% |
| 2T | 50T | 25.0 | 26.3T | 51% |
| 2T | 5T | 2.5 | 4.86T | 95% |
| 0.25T | 5T | 20.0 | 2.98T | 57% (mixture: amber, edge of 15–20) |

## widget_6_moe — efficiency leverage (Ling, Tian et al. 2025, Eq. 13)

Constants: `FA = 1.23, FD = −0.0761, FGAM = 0.0167, FBET = −0.117, ASTART = 0.0163, AMAX = 5.28e16`.

- `â(a) = 1 / (1/(a + 1/(1/ASTART − 1/AMAX)) + 1/AMAX)`
- `α(C) = FA + FD·log10 C`, `h(G) = FGAM·(log2 G)² + FBET·log2 G`
- `EL = â^(α(C) + h(G))` — log bases are the widget's choice (paper does not state them)
- sizing: `TPP = 500`, `N_act = sqrt(C / (6·500))`, `N_tot = N_act / a`, memory `= N_tot × 16 B / 64 GPUs`,
  limit `LIM = 160 GB` per GPU

Buttons: activation 25% / 9.4% / 5.5% / **3.1%** / 1.5% / 0.8%; G 2 / 4 / 8 / **12** / 24 / 48; compute
1e21 / **1e22** / 1e23 / 1e24 / 1e25. Above 6% active → amber "leaves leverage unused".

At G = 12, 1e22 (N_act = 1.8B):

| Activation | EL | total params | GB per GPU |
|---|---|---|---|
| 25% | 2.4× | 7.3B | 1.8 |
| 9.4% | 4.2× | 19.4B | 4.9 |
| 5.5% | 5.6× | 33.2B | 8.3 |
| 3.1% (default) | **7.2×** | 58.9B | 15 |
| 1.5% | 9.5× | 122B | 30 |
| 0.8% | 11.2× | 228B | 57 |

Halving activation from 6.2% to 3.1% gives 1.39× more EL for 2× the total parameters.

At 3.1%, 1e22: G = 2 → 5.3×, G = 8 → 7.2×, G = 12 → 7.2×, G = 48 → 5.8× (peak near 8–12).

At 3.1%, G = 12: 1e21 → 5.7×, 4.7 GB; 1e24 → 11.5×, 589B total, 147 GB; 1e25 → 14.5×, 1.86T total,
**466 GB, over the 160 GB limit (red)**.

Frontier marks: Qwen3-235B 9.4%, DeepSeek-V3 5.5%, Kimi K2 / DS-V4-Pro 3.1%, MAI-Base-2 2.9%.

## widget_7_hparams — three LR and batch laws

`N = sqrt(C/(6·TPP))`, `D = TPP·N`.

| Law | LR | Batch (tokens) |
|---|---|---|
| DeepSeek LLM | `0.3118·C^−0.125` | `0.2920·C^0.3271` |
| Step Law | `1.79·N^−0.713·D^0.307` | `0.58·D^0.571` |
| MAI-Base-2 | `exp(−1.51 + 0.175 ln TPP − 0.573 ln steps)` | `D / steps`; WD `= 17.04/(steps·LR)` |

Band = geometric mean of the three LRs, ×½ to ×2. Buttons: compute 1e20 … 1e25 (default **1e23**); TPP
20 / **100** / 400 / 700; steps 25k / 50k / **100k** / 200k / 400k.

| Setting | DeepSeek LR / B | Step Law LR / B | MAI LR / B / WD | all within 2×? |
|---|---|---|---|---|
| 1e23, TPP 100, 100k (default) | 4.16e-4 / 9.74M | 5.78e-4 / 4.77M | 6.75e-4 / 12.91M / 0.25 | yes (2.73e-4 to 1.09e-3) |
| 1e20 | 9.86e-4 / 1.02M | 2.35e-3 / 664k | 6.75e-4 / 408k | no: Step Law out |
| 1e25 | 2.34e-4 / 43.94M | 2.27e-4 / 17.77M | 6.75e-4 / 129.10M | no: MAI out |
| 1e23, TPP 20 | 4.16e-4 | 2.54e-4 / 3.01M | 5.09e-4 / 5.77M / 0.33 | yes |
| 1e23, TPP 700 | 4.16e-4 | 1.56e-3 / 8.32M | 9.49e-4 / 34.16M / 0.18 | no: DeepSeek out |
| 1e23, TPP 100, 25k steps | — | — | 1.49e-3 / 51.64M / 0.46 | no: MAI out |
| 1e23, TPP 100, 400k steps | — | — | 3.05e-4 / 3.23M / 0.14 | yes |

The MAI LR does not depend on C, only on TPP and steps.

## widget_8_ladder — ladder, bootstrap band and source scaling

Ground truth `BPB = 0.60 + rep × 0.90 × 10^(−0.18(lgC − 19))`, `rep = REP = 1.25` with source scaling,
1.0 without. The large run (`TGT = 1e24`) always uses `rep = 1.25`, so its result is **0.742**. Rungs are
spaced evenly from 1e19 to 1e22, with multiplicative noise `exp(σ·z)` on the excess over 0.60. Fit is
log-linear on `BPB − 0.60`. 400 bootstrap resamples (`NB`); band = 5th–95th percentile ± `TOL = 0.005`.
Seeds: rungs `1000 + n`, bootstrap `77 + 13n + hold`.

Buttons: rungs 4 / 5 / **6** / 7 / 8; noise low 0.01 / **medium 0.03** / high 0.06; hold out **top** /
second rung; source scaling **on** / off.

| Setting | fit | held-out measured vs band | forecast at 1e24 (band) | 0.742 inside? |
|---|---|---|---|---|
| default (6, 0.03, top, on) | 0.60 + 1.139·(C/1e19)^−0.185 | 0.934 in 0.891–0.938 | 0.735 (0.713–0.752) | yes |
| source scaling off | 0.60 + 0.912·…^−0.185 | 0.867 in 0.832–0.871 | 0.708 (0.690–0.722) | **no, forecast too low** |
| 4 rungs | ^−0.171 | 0.918 in 0.902–0.960 | 0.754 (0.725–0.778) | yes |
| 5 rungs | ^−0.170 | 0.924 in 0.895–0.953 | 0.754 (0.720–0.772) | yes |
| 7 rungs | ^−0.173 | 0.926 in 0.903–0.953 | 0.750 (0.723–0.772) | yes |
| 8 rungs | ^−0.174 | 0.930 in 0.903–0.948 | 0.749 (0.725–0.768) | yes |
| noise low | ^−0.182 | 0.927 in 0.910–0.932 | 0.739 (0.728–0.748) | yes |
| noise high | ^−0.191 | 0.943 in 0.866–0.946 | 0.728 (0.694–0.757) | yes |
| hold out second rung | ^−0.180 | 1.015 in 0.987–1.031 (at 2.5e21) | 0.743 (0.717–0.753) | yes |

Turning source scaling off is the only setting that breaks the forecast: the held-out rung still passes,
but the large run lands above the band.

## Figures

| File | Constants |
|---|---|
| `fig_1_budget` | the four questions: N, D, batch B, learning rate η |
| `fig_2_isoflop` | Epoch coefficients; budgets 1e21, 1e22, 1e23, 1e24; flat valley drawn at 1e23 |
| `fig_3_repeat` | `R* = 15.4`; D′ at 4 / 16 / 40 epochs = 3.73 / 10.59 / 15.18 epochs of unique data (93% / 66% / 38% of nominal) |
| `fig_4_ladder` | seed 13; truth `0.40 + 0.75·(C/1e19)^−0.10`; 5 fit rungs from 1e19 to 1e21 every half decade, held-out rung 10^21.5, target 1e24, noise 0.6% |
| `fig_5_plan` | five steps: budget, size (`D = TPP·N`), check data (≤ 4 epochs), ladder of 5–8 rungs (`LR*·WD* = 17.04/steps`), held-out check |
| `fig_6_granularity` | G = 1 compared with fine-grained experts at the same active width |
| `fig_7_downstream` | small runs 6e18 to 1e22; `loss = 1.60 − 0.15(log10 C − 19)`; `acc = 0.25 + 0.72/(1 + e^((loss−0.95)/0.12))`; at 3.8e25: loss 0.613, accuracy 92.9% |
| `fig_8_hparams` | DeepSeek laws over 1e17–1e25 (fit range 1e17–2e19 shaded); extra loss `0.02·l² + 0.06·l⁴`, `l = log10(LR/LR*)`: 0.0023 at 2× or ½×, 0.080 at 10× or 0.1× (labelled illustrative) |
