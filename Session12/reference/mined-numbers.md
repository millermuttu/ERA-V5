# Session 12 — numbers mined from the widget sources

Everything below is read out of the inline `<script>` of each widget under `widgets/`. These are the
baselines an implementation gets measured against; several of them appear nowhere in the lesson prose.

Constants shared across the session: 30e9 parameters, 2 bytes per parameter in bf16, so P = 60 GB and
a data-parallel step moves 2P = 120 GB per GPU. One 80 GB card is 80e9 / 2^30 = 74.506 GiB.
GiB divisor used everywhere is 1073741824.

## widget_1_data_parallel

| Name | Value |
|---|---|
| initial weights `W0` | `[3, 5, 2, 4]` |
| learning rate `LR` | 0.1 |
| GPUs | 4, two samples each, batch of 8 |
| batches `NB` | 3, cycled from a fixed list of 24 sample vectors |
| averaging checkbox | `checked` by default |

Gradient per GPU k, component c: `w[k][c] - (a[c] + d[c]) / 2` where `a`, `d` are that GPU's two
samples. The mean over the 4 GPUs is applied as `w - LR * m` when averaging is on, otherwise each GPU
applies its own gradient. The agreement readout is the largest absolute difference between any two of
the four weight copies, printed as 0.000 below 5e-4. Sample list (24 vectors of 4):

```
[1,7,2,4] [3,5,4,6] [5,3,6,2] [1,7,2,6] [2,6,3,5] [6,2,7,1] [7,1,8,4] [3,5,4,0]
[2,4,1,8] [4,6,3,4] [6,2,5,3] [2,6,3,7] [3,5,4,6] [7,1,6,2] [8,4,9,1] [4,0,5,5]
[0,8,3,5] [2,6,5,3] [4,4,7,1] [0,8,3,5] [1,7,4,4] [5,3,8,0] [6,2,9,2] [2,6,7,0]
```

Phases stepped through: `batch drawn, split, compute gradients, average, update`.

## widget_2_collectives

4 GPUs, 4 chunks. GPU g's contribution to chunk c is `(c+1) * 10^g`, so the digit position identifies
the owner and the finished chunk is the column sum. Stage 0 is the start, stage 1 after reduce-scatter
(only GPU g==c holds the sum), stage 2 after all-gather (everyone holds every sum).

| Quantity | Value |
|---|---|
| cells sent after reduce-scatter | 12, which is G(G-1) |
| cells sent after all-gather | 24, which is 2G(G-1) |
| cells a direct all-to-all exchange would move | 48, which is G·C·(G-1) |
| match counter target | 16 of 16 |

## widget_3_ring

Same 4 GPUs and 4 chunks, same `10^g * (c+1)` seeding. Six sends total: sends 1 to 3 are
reduce-scatter and add into the destination cell, sends 4 to 6 are all-gather and overwrite it. Each
GPU only ever sends to `(g+1) % 4`. Chunk selected on send s is `(g-t) mod 4` for the first phase and
`(g+1-t) mod 4` for the second, with `t = s-1` or `s-4`. Final value of every chunk is the column sum,
which is the all-reduce answer.

## widget_4_comm_cost

| Constant | Value |
|---|---|
| `PAR` | 30e9 |
| `BPP` | 2 bytes |
| `TOK` | 1e6 tokens per step |
| `GPUS` | 64 |
| `MFU` | 0.40 |
| axis max | 10 s |
| `PEAK` H100 | 990e12 FLOP/s bf16 |
| `PEAK` B200 | 990e12 × 2.28 = 2257.2e12 FLOP/s |
| `BWS` NVLink | 450e9 B/s |
| `BWS` InfiniBand | 50e9 B/s |
| defaults | B200 card, InfiniBand link |

Formulas, verbatim:

```
volume = 2 * PAR * BPP                       = 120 GB
t_comm = volume / BWS[link]
t_comp = 6 * PAR * TOK / (GPUS * PEAK[gpu] * MFU)
ratio  = t_comm / t_comp          (the headline percentage)
share  = t_comm / (t_comp + t_comm)   (share of an unoverlapped step)
```

Resulting values: H100 compute 7.10 s, B200 compute 3.12 s, NVLink 0.27 s, InfiniBand 2.40 s.
Ratios 34% on H100 and 77% on B200; unoverlapped steps 9.50 s and 5.52 s.

## widget_5_zero_stages

8 GPUs, 8 weights, GPU k owns weight k. Four bands per weight:

| Band | Bytes | Split at stages |
|---|---|---|
| weight, bf16 | 2 | 3 |
| gradient, bf16 | 2 | 2, 3 |
| master copy, fp32 | 4 | 1, 2, 3 |
| optimizer averages | 8 | 1, 2, 3 |

`SHARD = [[], [2,3], [1,2,3], [0,1,2,3]]` over that band order. Bytes per weight is the solid cells in
one GPU box divided by 8, giving 16.00, 5.50, 3.75, 2.00. Multiplied by 30e9 and converted: 447.0,
153.7, 104.8, 55.9 GiB. The duplicated floor reported is the sum of bands no stage splits: 4 bytes at
data parallelism and ZeRO-1 (111.8 GiB), 2 bytes at ZeRO-2, 0 at ZeRO-3. Default selection is data
parallel.

## widget_6_memory_ladder

| Control | min | max | step | default |
|---|---|---|---|---|
| model size, billions | 1 | 100 | 1 | 30 |
| GPU count exponent | 0 | 10 | 1 | 3 (so 8 GPUs, range 1 to 1024) |

Bytes per weight as a function of stage and GPU count n:

```
data parallel : 16
ZeRO-1        : 4  + 12/n
ZeRO-2        : 2  + 14/n
ZeRO-3        : 16/n
```

Card rule drawn at 74.5 GiB. Y axis is log from 0.3 to 600 GiB, X axis powers of two to 1024. Two
dashed floors annotated: 4 bytes per weight replicated (data parallel and ZeRO-1) and 2 bytes per
weight replicated (ZeRO-2). `firstFit` scans powers of two up to 1024; at 30B it reports data
parallelism never, ZeRO-1 never, ZeRO-2 first at 32 GPUs and ZeRO-3 first at 8 GPUs, matching the
table in section 7 of the lesson.

## widget_7_overlap

| Constant | Value |
|---|---|
| layers `LAY` | 12 |
| gradient volume `VOL` | 120 GB |
| link bandwidth `BW` | 50 GB/s |
| per-transfer fixed cost `TAU` | 0.10 s |
| axis max | 10 s |
| bucket stops, in layers | 1, 2, 3, 4, 6, 12 |
| slider | min 0, max 5, step 1, default 1 (so 2 layers) |
| H100 compute per step | 7.10 s |
| B200 compute per step | 3.12 s |

Schedule model, verbatim:

```
n_buckets   = LAY / k
t_layer     = t_compute / LAY
gb          = VOL * k / LAY
t_bucket    = gb / BW + TAU
network_tot = n_buckets * t_bucket
step        = max(t_compute, network_tot) + t_bucket
hidden      = 1 - (step - t_compute) / network_tot
```

Bucket i becomes ready at `t_layer * k * i` and starts at `max(ready, previous end)`, which is where
queueing appears on the B200. At k = 2 on an H100 step the hidden share is 83 percent. The second
panel is the stage-3 prefetch: forward layer L*i* overlaps the all-gather of L*(i+1)*.

## widget_8_precision

| Constant | Value |
|---|---|
| master copy `MASTER` | 4 bytes, fixed at every setting |
| Adam moments `MOM` | 4 bytes each in FP32, 2 bytes each in BF16 |
| matmul format `MM` | 2 bytes in BF16, 1 byte in FP8 |
| MXFP8 block `BLOCK` | 32 values sharing one scale byte |
| baseline `BASE` | 16 bytes |
| defaults | BF16 matmul, FP32 moments |

```
scale(p) = p ? 1/32 : 0
total    = MASTER + 2*MOM[a] + 2*MM[p] + 2*scale(p)
```

So 16.00 bytes at bf16/fp32 and 14.0625 bytes with the FP8 matmul, a 12.1 percent cut in stored state;
the block scale contributes 0.0625 of that, being one byte per 32 values on each of the two
tensors. BF16 moments drop a further 4 bytes. The four comparison bars are: stored state 12.1% less,
peak matmul rate 100% more, activation bytes 50% less, communication bytes 50% less.
