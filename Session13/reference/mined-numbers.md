# Session 13 — numbers mined from the widget sources

Everything below is read out of the inline `<script>` of each widget under `widgets/`. These are the
baselines an implementation gets measured against; several of them appear nowhere in the lesson prose.

Constants shared across the session: the 30.2e9-parameter model, 96 layers, hidden 5,120, FFN inner
16,384, 40 query heads / 8 KV heads of 128 dims. Training state is 16 bytes/param, so 450.0 GiB whole and
56.25 GiB at TP = 8. Activations are `tokens × 5120 × 34 bytes` per layer (1.33 GiB per layer at 8,192
tokens), so 127.5 GiB for one 8,192-token sequence. One activation tensor is `8192 × 5120 × 2` = 83.9 MB.
Links are NVLink 4 = 450e9, NVLink 5 = 900e9, InfiniBand = 50e9 bytes/s each way. The ring all-reduce
factor is `2(n−1)/n`. The GiB divisor is 1073741824. Card caps: H100 `80e9/GiB` = 74.5, B200 = 167.6.

## widget_1_tensor_split

| Name | Value |
|---|---|
| `STATE` | 450.0 GiB (divided by degree) |
| `TENSOR` | `8192*5120*2` bytes |
| `LAYERS`, `ARS` (all-reduces per layer, fwd+bwd) | 96, 4 |
| `FFN` | 16384 |
| degree buttons | 1, 2, 4, 8 — default `t=8` |
| link buttons | NVLink 4 / NVLink 5 / InfiniBand — default `link=0` (NVLink 4) |
| time axis `AXMAX` | pinned at 1.5 s |

`traffic(n) = 4 × TENSOR × 2(n−1)/n × 96`, `secs = traffic / BW[link]`. At TP = 8 that is 56.4 GB, and
0.125 s / 0.063 s / 1.13 s on the three links. The readout also shows the time on the other class of link
(InfiniBand when on NVLink, NVLink 4 when on InfiniBand).

| TP | traffic per sequence | NVLink 4 | NVLink 5 | InfiniBand |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 32.2 GB | 0.072 s | 0.036 s | 0.64 s |
| 4 | 48.3 GB | 0.107 s | 0.054 s | 0.97 s |
| 8 | 56.4 GB | 0.125 s | 0.063 s | 1.13 s |

## widget_2_pipeline_schedule

| Name | Value |
|---|---|
| `SEQ`, `HID`, `ACT`, `LAYERS` | 8192, 5120, 34, 96 |
| schedules | `afab` (all forward, then all backward), `1f1b` — default `1f1b` |
| stages `p` | 4, 8 — default 4 |
| micro-batches `m` | 4, 8, 16 — default 8 |
| memory axis `MMAX` | 510 GiB |
| grid width `MAXCOL` | 46 slots |

The grid is built slot by slot. AFAB puts F of micro-batch j on stage i at slot `i+j`, and B at
`(m+p−1) + j + (p−1−i)`. Non-interleaved 1F1B: stage i runs `min(p−1−i, m)` warm-up forwards, then
alternates F/B, then drains. Each op is placed at the earliest slot where its dependency (F from the
previous stage, B from the next stage) finished. `count()` reads everything off the drawn grid.
Idle share = empty cells / `(p × T)`. "Held at stage 0" = the running peak of F−B on stage 0.
Memory = `peak × perMicro(p)`, with `perMicro(p) = 8192 × 5120 × 34 × (96/p)` bytes. The interleaved
comparison readout is hard-coded at v = 2: `(p−1)/(2m+p−1)`.

Reachable readouts (the share is the same for both schedules; only memory moves):

| p | m | waiting share | interleaved v=2 | 1F1B stage-0 memory | AFAB stage-0 memory |
|---|---|---|---|---|---|
| 4 | 4 | 42.9% | 27.3% | 127.5 GiB | 127.5 GiB |
| 4 | 8 | 27.3% | 15.8% | 127.5 GiB | 255.0 GiB |
| 4 | 16 | 15.8% | 8.6% | 127.5 GiB | 510.0 GiB |
| 8 | 4 | 63.6% | 46.7% | 63.8 GiB | 63.8 GiB |
| 8 | 8 | 46.7% | 30.4% | 127.5 GiB | 127.5 GiB |
| 8 | 16 | 30.4% | 17.9% | 127.5 GiB | 255.0 GiB |

The lesson's 8-stage / 32-micro-batch row (17.9%, 510.0 vs 127.5 GiB) is not reachable in the widget,
because `m` stops at 16.

## widget_3_context_parallel

| Name | Value |
|---|---|
| `SEQ`, `HID`, `BPT`, `LAYERS` | 131072, 5120, 34, 96 |
| `QH`, `KVH`, `HD`, `KVB` | 40, 8, 128, 2 bytes |
| `TP` (second activation bar) | 8 |
| `CARD` | `80e9/GiB` = 74.5 GiB (H100 line on the chart) |
| views | `ring`, `gather`, `head` — default `ring` |
| degree `c` | 2, 4, 8, 16 — default 8 |
| activation chart | log axis 10 → 3000 GiB, ticks 10/30/100/300/1000/3000 |

Formulas: `tokens = SEQ/c`. `actGiB = SEQ×HID×34×96/c/GiB`. `block = 2×KVH×HD×tokens×2`.
`ringFwd = block × (c−1) × 96`. `fullKV = 2×8×128×SEQ×2` = 536.9 MB.
`agFwd = fullKV × (c−1)/c × 96`, which equals `ringFwd`. The by-head view is valid when `QH % c == 0`, and
KV heads are copied when `KVH/c < 1`.

| c | tokens/GPU | act (CP only) | act (with TP 8) | KV block | ring = all-gather fwd traffic | by-head valid | causal load, last/first |
|---|---|---|---|---|---|---|---|
| 2 | 65,536 | 1,020.0 GiB | 127.5 GiB | 268.4 MB | 25.8 GB | yes (20 q-heads, 4 kv) | 3× |
| 4 | 32,768 | 510.0 GiB | 63.8 GiB | 134.2 MB | 38.7 GB | yes (10, 2) | 7× |
| 8 | 16,384 | 255.0 GiB | 31.9 GiB | 67.1 MB | 45.1 GB | yes (5, 1) | 15× |
| 16 | 8,192 | 127.5 GiB | 15.9 GiB | 33.6 MB | 48.3 GB | **no** (2.5 heads; kv 0.5, copied) | 31× |

Causal load panel: a contiguous piece j costs `j + 0.5` piece-units, so last/first = `(c−1+0.5)/0.5 =
2c−1`. Head-tail uses `2c` chunks, with GPU j taking chunk j and chunk `2c−1−j`. Every GPU then costs
`(j+0.5) + (2c−1−j+0.5) = 2c` chunk-units, drawn as the flat bar at height `c/2`.

## widget_4_placement

| Name | Value |
|---|---|
| `GPUS`, `PER` (GPUs per node) | 64, 8 |
| `TENSOR` | `8192×5120×2` (the constants are named `HID=8192, SEQ=5120` in the source, swapped but the same product) |
| `ARS`, `LAYERS` | 4, 96 |
| nodes | `h100` NVLink 4 450e9 / `b200` NVLink 5 900e9 — default `h100` |
| `NIC` | 50e9 |
| `GRAD` | 60.4e9 (full 16-bit gradients) |
| TP degree | 2, 4, 8, 16 — default 8 |
| view | `tp` / `dp` — default `tp` |
| time axis `AXMAX` | 1.5 s |

`d = 64/t` data-parallel copies. In the TP view the link is the node's NVLink when `t ≤ 8`. At `t = 16` it
falls to the NIC (flagged red, with the note "needs copies of the 8 key-value heads"). The time is
`tpTraffic(t)/bw`. In the DP view each GPU holds `GRAD/t` of gradients and moves `GRAD/t × 2(d−1)/d` through
one NIC. DP partners of GPU 0 are the GPUs with `i % t == 0`.

| t | d | TP link | TP time/seq | DP bytes/GPU/step | DP time (one NIC) |
|---|---|---|---|---|---|
| 2 | 32 | NVLink | 0.072 s (H100) / 0.036 s (B200) | 30.20 GB → 58.5 GB | 1.17 s |
| 4 | 16 | NVLink | 0.107 / 0.054 s | 15.10 GB → 28.3 GB | 0.57 s |
| 8 | 8 | NVLink | 0.125 / 0.063 s | 7.55 GB → 13.2 GB | 0.264 s |
| 16 | 4 | NIC 50 GB/s | 1.21 s | 3.78 GB → 5.7 GB | 0.113 s |

## widget_5_dualpipe

| Name | Value |
|---|---|
| stages `pp` | 4, 8, 16 — default 8 |
| default slot | 3 |
| slot axis `AXMAX` | 15 |

Stage i first has work at slot `i` when fed from one end, and at slot `min(i, pp−1−i)` when fed from both
ends. Slots to give every GPU work:

| pp | one end | both ends |
|---|---|---|
| 4 | 3 | 1 |
| 8 | 7 | 3 |
| 16 | 15 | 7 |

This is the `(PP−1)` vs `(PP/2−1)` bubble factor from the DualPipe table in the lesson.

## widget_6_two_stage_cp

| Name | Value |
|---|---|
| `CP` (GPUs) | 4 |
| `S` (entries per GPU) | 12 |
| stages | problem / stage one / stage two — default problem |
| block size `m` | 2, 4 — default 4 |

`perGpu = S/m`, `fixedLen = S/m + 1`, `finalCount = CP×S/m`, `pad = CP×fixedLen − finalCount = CP`
(padding shown only from stage one on). Stage one sends the last `m` uncompressed entries (`S−m … S−1`) to
the next GPU, so each receiver compresses `S + m` entries.

| m | per GPU | fixed length | final count | padding |
|---|---|---|---|---|
| 2 | 6 | 7 | 24 | 4 |
| 4 | 3 | 4 | 12 | 4 |

## widget_7_pipeline_cut

| Name | Value |
|---|---|
| `L` (layers) | 40 |
| `PROD` (layer whose output feeds the upper KV) | 20 |
| stages `N` | 1, 2, 4, 8 — default 4 |

Cuts sit at `k × L/N`, and layer i is on stage `count(i > cut)`. For readers 21…40 the widget counts the
reader stages ≠ owner and the crossing links. Lifetime: start = owner stage, end = `2N − 1 − minReaderStage`,
held = `end − start + 1` of `2N` forward+backward stage-slots.

| N | owner stage | reader stages | links crossing | held (of 2N) |
|---|---|---|---|---|
| 1 | 0 | none | 0 | 2 of 2 |
| 2 | 0 | 1 | 20 | 3 of 4 |
| 4 | 1 | 2, 3 | 20 | 5 of 8 |
| 8 | 3 | 4–7 | 20 | 9 of 16 |

## widget_8_layout_choice

| Name | Value |
|---|---|
| `P` | 30.2e9 |
| `CAP` | 167.6 GiB (B200) |
| `TP` | 8 (fixed) |
| `ACT0`, `BASE` | 127.5 GiB at 8,192 tokens |
| sequence `SEQS` | 8192, 16384, 32768, 65536, 131072 — default 8192 |
| nodes `NSET` | 1, 2, 4, 8 — default 1 |
| memory axis `AXMAX` | 320 GiB |

`optBytes(dp) = 4 + 12/dp` (ZeRO-1 over the DP copies). `state = P × optBytes / 8 / GiB`.
`act = 127.5 × (s/8192) / (8 × cp)`. `plan(s, n)` tries `cp` in 1, 2, 4, 8 (with `cp ≤ n` and `n % cp == 0`,
`dp = n/cp`) and returns the **first** that fits within 167.6 + 1e-9. If none fits, it returns the smallest
total. It never uses PP.

| seq \ nodes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 8,192 | cp1 dp1 72.2 ✓ | cp1 dp2 51.1 ✓ | cp1 dp4 40.5 ✓ | cp1 dp8 35.3 ✓ |
| 16,384 | cp1 dp1 88.1 ✓ | cp1 dp2 67.0 ✓ | cp1 dp4 56.5 ✓ | cp1 dp8 51.2 ✓ |
| 32,768 | cp1 dp1 120.0 ✓ | cp1 dp2 98.9 ✓ | cp1 dp4 88.4 ✓ | cp1 dp8 83.1 ✓ |
| 65,536 | cp1 dp1 183.8 ✗ | cp1 dp2 162.7 ✓ | cp1 dp4 152.1 ✓ | cp1 dp8 146.8 ✓ |
| 131,072 | cp1 dp1 311.3 ✗ | cp2 dp1 183.8 ✗ | cp2 dp2 162.7 ✓ | cp2 dp4 152.1 ✓ |

State per DP degree: dp1 56.25, dp2 35.2, dp4 24.6, dp8 19.3 GiB.

## widget_9_reversibility

| Name | Value |
|---|---|
| `P`, `CAP`, `TP` | 30.2e9, 167.6 GiB, 8 |
| `ACT0`, `BASE`, `D`, `LAYER0` | 127.5 GiB, 8192, 5120, 1.33 GiB |
| sequence | 8192, 32768, 131072 — default 8192 |
| mode | `stored` / `rev` — default `stored` |

`state = P × 16 / 8 / GiB` = 56.25 (no ZeRO, one node). `stored = 127.5 × (s/8192) / 8`.
`rev = (boundary + layer) / 8` with `boundary = 2 × s × 5120 × 2 / GiB` and `layer = 1.33 × s/8192`.

| seq | stored act/GPU | stored total | rev act/GPU | rev total |
|---|---|---|---|---|
| 8,192 | 15.94 | 72.2 ✓ | 0.19 | 56.4 ✓ |
| 32,768 | 63.75 | 120.0 ✓ | 0.74 | 57.0 ✓ |
| 131,072 | 255.00 | 311.3 ✗ | 2.97 | 59.2 ✓ |

(The lesson's un-divided reversible figures, 1.5 / 5.9 / 23.8 GiB, are these × 8.)

Seven-form status table (`ROWS`). Pills are off / on · 8 / available / needed:

| form | stored | reversible |
|---|---|---|
| ZeRO-1 | off: one node has no DP copies | off, and it becomes the first thing to add |
| ZeRO-2 | off | off, and it divides the term that now dominates |
| ZeRO-3 | off: TP already divides the weights | **available**: the recompute path gathers weights a second time |
| tensor parallelism | on · 8 | on · 8 |
| sequence parallelism | on · 8: divides stored activations | on · 8: divides only the boundary states |
| pipeline parallelism | off: fits without dividing by layer | off: reversibility removes the depth term |
| context parallelism | off (at 131,072: **needed**, 4 nodes with CP = 2) | off: one node holds 131,072 tokens |

## Assignment baselines (from the lesson, not a widget)

Train a 20M LLM for 50M tokens. Run three times: a fixed batch with stored activations, the same batch
with reversibility (report the variant: midpoint, Euler, …), and reversibility at the maximum batch.
Report final loss, tokens/s and peak memory. Reference points from the paper and from Lightning LM:
midpoint rule `p^{l+1} = p^{l−1} + 2h·f(p^l)`, step size `h = 0.25` and blend coefficient 0.5 (set
explicitly, because library defaults differ), dropout must be 0, expected compute overhead is 30–50% of a
step, and the batch ratio is about 10× (26 → 257 on an 80 GB H100).
