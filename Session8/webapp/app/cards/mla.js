// Concept 21 — Multi-head latent attention.
// Built from docs/research/mla.md. Four things the research settled and this card is shaped by.
// The compression is not "store fewer numbers and decompress": the up-projection is absorbed into
// the query and the output matrices, so the cached latent is an object the query has been reshaped
// to talk to directly — an identity, checkable to floating point, and the reason the saving costs no
// arithmetic. RoPE breaks that identity, and not by a little: the term that will not move past the
// rotation is 87.5% the size of the term itself, so the fix is a separate uncompressed position
// channel rather than a correction. The paper's own headline numbers are a 236B mixture-of-experts
// against a dense 67B with FP8 weights and a 6-bit cache, so the card leads with the controlled
// ratios from its appendix instead. And the comparison the paper asserts — "equal to GQA with only
// 2.25 groups, but stronger" — is never run anywhere in it; the app can run it, because GQA and MLA
// are both linear compressions of the same map.
import { el, slider, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt, dot } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { rope } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const D = CONFIG.D;
const DR = DH / 2; // the paper's own ratio: the decoupled key is half a head wide
const R = rope({ dims: DH });
// The decoupled channel is its own width, so its rotation ladder is built for that width.
const RR = rope({ dims: DR });

// DeepSeek-V2's own configuration, for the arithmetic this app's four heads cannot show.
const V2 = { heads: 128, dh: 128, layers: 60, dc: 512, dr: 64 };

/**
 * Jacobi eigendecomposition of a small symmetric matrix. Needed because the honest version of this
 * mechanism is the *best* rank-d_c factorisation of the block's own key/value map — a trained pair
 * of projections cannot beat it, so every error this card reports is a lower bound on the real one.
 */
function jacobi(input, sweeps = 60) {
  const n = input.length;
  const A = input.map((r) => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - sn * akq;
          A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk;
          A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq;
          V[k][q] = sn * vkp + c * vkq;
        }
      }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return { vals: order.map((i) => A[i][i]), vecs: order.map((i) => Float64Array.from(V.map((r) => r[i]))) };
}

/** The joint compression of one block: down onto d_c directions, up to the full keys and values. */
const factorisations = new Map();
function factorise(wb, block) {
  if (factorisations.has(block)) return factorisations.get(block);
  const M = [];
  for (let i = 0; i < D; i++) M.push(Float64Array.from([...wb.k[i], ...wb.v[i]]));
  const G = Array.from({ length: D }, (_, i) =>
    Float64Array.from({ length: D }, (_, j) => {
      let s = 0;
      for (let c = 0; c < 2 * D; c++) s += M[i][c] * M[j][c];
      return s;
    })
  );
  const { vals, vecs } = jacobi(G);
  // The full up-projection, one row per direction: what that direction contributes to every key and
  // every value in the block. Truncating to d_c rows is the compression.
  const up = vecs.map((u) => {
    const row = new Float64Array(2 * D);
    for (let c = 0; c < 2 * D; c++) {
      let s = 0;
      for (let i = 0; i < D; i++) s += u[i] * M[i][c];
      row[c] = s;
    }
    return row;
  });
  const total = vals.reduce((a, b) => a + b, 0);
  const kept = [];
  let acc = 0;
  for (let r = 0; r < D; r++) {
    acc += vals[r];
    kept.push(acc / total);
  }
  const out = { down: vecs, up, kept, M };
  factorisations.set(block, out);
  return out;
}

/** `forward`'s latent hook: rebuild every key and value in the block from a d_c-wide vector. */
const latentHook = (dc) => (normed, wb, block) => {
  const { down, up } = factorise(wb, block);
  const K = [];
  const V = [];
  for (const h of normed) {
    const c = Float64Array.from({ length: dc }, (_, j) => {
      let s = 0;
      for (let i = 0; i < D; i++) s += h[i] * down[j][i];
      return s;
    });
    const full = new Float64Array(2 * D);
    for (let j = 0; j < dc; j++) {
      const cj = c[j];
      for (let d = 0; d < 2 * D; d++) full[d] += cj * up[j][d];
    }
    K.push(full.subarray(0, D));
    V.push(full.subarray(D, 2 * D));
  }
  return { K, V };
};

/** How much of the block's key/value map a compression of this rank destroys. */
const mlaError = (kept, dc) => Math.sqrt(Math.max(0, 1 - kept[Math.min(dc, D) - 1]));

/** GQA is a compression of the same map — averaging a fixed grouping of the heads. */
function gqaError(M, groups) {
  const per = CONFIG.HEADS / groups;
  let num = 0;
  let den = 0;
  for (let i = 0; i < D; i++) {
    for (const off of [0, D]) {
      for (let g = 0; g < groups; g++) {
        for (let d = 0; d < DH; d++) {
          let mean = 0;
          for (let h = g * per; h < (g + 1) * per; h++) mean += M[i][off + h * DH + d];
          mean /= per;
          for (let h = g * per; h < (g + 1) * per; h++) {
            const x = M[i][off + h * DH + d];
            num += (x - mean) * (x - mean);
          }
        }
      }
    }
    for (let c = 0; c < 2 * D; c++) den += M[i][c] * M[i][c];
  }
  return Math.sqrt(num / den);
}

const klDiv = (p, q) => {
  let d = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) d += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12));
  return d;
};

export function mlaCard(root, m) {
  let dc = 16;
  let withRope = true;
  let decoupled = true;

  root.appendChild(
    prose({
      problem:
        "Concept 15 bought a smaller cache by making heads share, and this paper measures what that costs on hard benchmarks at matched parameters and matched data: eight groups is four points of MMLU and five of C-Eval below full multi-head attention, and one shared head is worse again. The reason is that sharing works along the wrong axis. Heads are a grouping fixed before any data arrives, and averaging within a group throws away whatever distinguished the members — there is no reason the thing worth keeping should line up with the boundaries between heads.",
      mechanism:
        "Stop compressing along the head axis and compress the map instead. One small vector per token is produced from the layer's input, and every key and every value in the layer — for all heads — is reconstructed from it. That vector is the entire cache. The part that makes this cheap rather than merely small is an identity: the matrix that reconstructs a key can be moved onto the query side instead, and the matrix that reconstructs a value onto the output side, so at generation time no key and no value is ever built. The cached object is not a compressed key waiting to be expanded; it is something the query has been reshaped to talk to directly. Rotary position ruins that reassociation, so position is given its own small uncompressed channel — one vector per token shared by every head — and the score becomes a content term against the latent plus a position term against that.",
    })
  );

  root.appendChild(
    el("div", { class: "formula", text: "c_t = W^DKV h_t      k_t = W^UK c_t      q·k = q·(W^UK c) = (W^UKᵀ q)·c      cache = (d_c + d_h^R)·layers" })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------- 1. the absorption
  const dcSlider = slider({
    label: "the latent's width d_c",
    min: 1,
    max: D,
    step: 1,
    value: dc,
    format: (v) => `${v} numbers`,
    oninput: (v) => ((dc = v), render()),
  });
  const absorbRead = readout([
    { key: "naive", label: "score, building the key first" },
    { key: "absorbed", label: "score, never building a key" },
    { key: "gap", label: "the difference between them" },
    { key: "touched", label: "numbers read per cached token" },
  ]);
  const absorbNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the identity that makes the compression free" }),
      el("div", { class: "ctrls" }, [dcSlider.node]),
      absorbRead.node,
      absorbNote,
    ])
  );

  // ------------------------------------------------- 2. what rotation does to it
  const ropeToggle = toggle({
    label: "rotate the key",
    value: true,
    onchange: (v) => ((withRope = v), render()),
  });
  const splitToggle = toggle({
    label: "give position its own channel",
    value: true,
    onchange: (v) => ((decoupled = v), render()),
  });
  const ropeRead = readout([
    { key: "truth", label: "the score the model should get" },
    { key: "anyway", label: "the score if you absorb anyway" },
    { key: "commutator", label: "size of the term that will not move past the rotation" },
    { key: "fixed", label: "the decoupled score — a different function, and absorbable" },
  ]);
  const ropeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the rotation has to be moved out of the way" }),
      el("div", { class: "ctrls" }, [ropeToggle, splitToggle]),
      ropeRead.node,
      ropeNote,
    ])
  );

  // ---------------------------- 3. matched budget against the head-sharing family
  const budgetCurve = curveView({
    xLabel: "numbers cached per token, per layer",
    yLabel: "share of the key/value map destroyed",
    ariaLabel: "how much of the same map each compression destroys, against what it costs to cache",
  });
  const budgetRead = readout([
    { key: "mla", label: "this latent loses" },
    { key: "gqa", label: "head-sharing at the same budget loses" },
    { key: "klMla", label: "and on your sentence — the latent" },
    { key: "klGqa", label: "the same budget, shared heads" },
  ]);
  const budgetNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the comparison the paper asserts and never runs" }),
      budgetCurve.node,
      budgetRead.node,
      budgetNote,
    ])
  );

  // ------------------------------------------------- 4. the arithmetic at both widths
  const sizeBars = barList({
    rows: [
      { key: "mha", label: "every head keeps its own", alt: true },
      { key: "gqa", label: "two groups of heads share" },
      { key: "mqa", label: "one shared head" },
      { key: "mla", label: "one latent, plus the position channel" },
    ],
  });
  const sizeRead = readout([
    { key: "here", label: "against full attention, at this app's four heads" },
    { key: "there", label: "the same, at 128 heads" },
    { key: "groups", label: "the head-sharing this is worth" },
    { key: "measured", label: "quoted: measured at 250B, everything else held fixed" },
  ]);
  const sizeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what it saves, and why a four-head toy understates it" }),
      sizeBars.node,
      sizeRead.node,
      sizeNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The cache stops scaling with the head count entirely — one latent per token per layer serves every head, where sharing still pays per group",
        "At 250 billion parameters with everything but the attention held fixed, the measured cache goes from 860.2K elements per token to 34.6K, a factor of 24.9, and the benchmark scores go up rather than down",
        "It costs no extra arithmetic at generation time, because the reconstruction matrices are folded into the query and output projections and no key or value is ever materialised",
        "At a matched cache budget it preserves far more of the same map than head-sharing does, because it may choose its own subspace where sharing is stuck with the boundaries between heads",
        "The paper's own ablation puts a number on what it replaces: eight-group sharing costs 4.0 points of MMLU and 5.2 of C-Eval against full attention at matched parameters and data",
      ],
      givesUp: [
        "Rotary position cannot go through the compression at all — it has to be given a separate uncompressed channel, so the mechanism routes around the position scheme every other card on this timeline builds on",
        "The width of the latent is fixed before any data arrives. The previous concept had just argued that what to keep should be decided by the content; this decides how much to keep in advance and wins by choosing a better basis instead",
        "Nothing in the paper ablates the mechanism's own parts — no sweep of the latent width, none of the position channel, no comparison of joint against separate compression, and the query compression is silently dropped in the smaller model with no measurement either way",
        "The headline numbers are not the mechanism's: 42.5% training cost is the mixture of experts, and the 93.3% cache and 5.76× throughput compare a 236B sparse model against a dense 67B with FP8 weights and a 6-bit quantised cache stacked on top",
        "The cached object is no longer a key and a value. Anything that manipulates a cache from outside — evicting by index, renumbering positions, sharing a prefix between requests — has to be re-derived for an object that only this model's absorbed matrices can read",
        "Its own claim to beat full attention rests on four benchmarks at two scales, single runs, and one of the eight cells goes the other way",
      ],
      chooseWhen:
        "When serving is the constraint and you are training the model yourself — this is a decision made at pre-training time and cannot be retrofitted to existing weights the way head-sharing can. It is the strongest answer on this timeline to a cache that grows with batch and context, and it is worth the most exactly where head counts are high, which is where the modern large models are.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The running memory needed to hold a conversation drops by a factor of twenty-five at large scale, and the model's scores went up rather than down",
        "Reading from the smaller memory costs no extra work, because the model is rebuilt so it can read the compressed form directly",
        "It squeezes along a better direction than the earlier method, which was stuck squeezing along a boundary that was never the right one",
        "The bigger the model, the more it saves — which is the opposite of most savings",
      ],
      cons: [
        "The usual way of telling the model where a word sits does not survive the squeeze, so position needs a separate uncompressed lane of its own",
        "How hard to squeeze is decided once, before any text arrives, rather than by what the text turns out to contain",
        "The paper never tests its own settings — no experiment shows that the sizes it chose are the right ones",
        "The famous savings figures compare two models that differ in several ways at once, not just in this mechanism",
      ],
      verdict:
        "The answer to a memory bill that head-sharing could only reduce by paying in quality: stop squeezing along the boundary between heads, which was never where the information was, and squeeze the whole thing onto a small learned subspace instead. Its cleverest part is not the compression but the rearrangement — the reconstruction is folded into the query, so the small thing in memory is read directly and nothing is ever unpacked. The price is that position has to be smuggled past it in a lane of its own, and that the size of the squeeze is fixed before the first word arrives.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    dcSlider.set(dc);

    const base = forward(tokens);
    // Running the compressed model is also what builds the factorisation, so read it back after.
    const compressed = forward(tokens, { latent: latentHook(dc) });
    const { down, up, kept, M } = factorisations.get(0);

    const head = 0;
    const h0 = base.trace[0].heads[head];
    const normed = base.trace[0].input;
    const q = h0.Q[T - 1];
    const src = 3 % T;

    // The latent for one token, and this head's slice of the up-projection.
    const c = Float64Array.from({ length: dc }, (_, j) => {
      let s = 0;
      for (let i = 0; i < D; i++) s += normed[src][i] * down[j][i];
      return s;
    });
    const WUK = up.slice(0, dc).map((row) => row.subarray(head * DH, head * DH + DH));
    const k = new Float64Array(DH);
    for (let j = 0; j < dc; j++) for (let d = 0; d < DH; d++) k[d] += c[j] * WUK[j][d];

    flow.update({
      tokens,
      head: { ...compressed.trace[0].heads[head], emb: compressed.trace[0].input },
      weights: compressed.trace[0].heads[head].weights,
      out: compressed.trace[0].heads[head].out,
      top: compressed.top,
      query: T - 1,
      opts: { qkvBadge: `cached: ${dc} + ${DR} numbers per token, for all ${CONFIG.HEADS} heads` },
    });
    flowNote.innerHTML = `The picture is concept 1's and every stage still happens — but the Key and Value columns are no longer what is kept. <strong>What generation holds is one ${dc}-number vector per token, for all ${CONFIG.HEADS} heads together</strong>, plus ${DR} more carrying position. The keys and values you see drawn are reconstructed from it, and at generation time they are not even reconstructed: the matrices that would rebuild them are folded into the query and the output projection instead. What is drawn here is what the arithmetic means, not what a served model computes.`;

    // --- 1. the absorption identity
    const naive = dot(q, k) / Math.sqrt(DH);
    const qc = Float64Array.from({ length: dc }, (_, j) => {
      let s = 0;
      for (let d = 0; d < DH; d++) s += WUK[j][d] * q[d];
      return s;
    });
    const absorbed = dot(qc, c) / Math.sqrt(DH);
    absorbRead.update({
      naive: fmt(naive, 9),
      absorbed: fmt(absorbed, 9),
      gap: Math.abs(naive - absorbed).toExponential(2),
      touched: `${dc} instead of ${2 * DH * CONFIG.HEADS}`,
    });
    absorbNote.className = "note good";
    absorbNote.innerHTML = `Two ways to compute one score between the last word and word ${src}. The first rebuilds the key from the latent and takes the dot product with the query, which is what the equations say. The second moves the reconstruction onto the query — one matrix multiply per <em>step</em>, not per cached token — and then takes the dot product with the latent itself, never building a key at all. They agree to <strong>${Math.abs(
      naive - absorbed
    ).toExponential(
      2
    )}</strong>, which is floating point. <strong>This is the whole reason the compression is free rather than merely small.</strong> A scheme that stored a compressed key and expanded it would pay the expansion once per cached token per step; this pays nothing, because the query has been reshaped to talk to the compressed form directly. Note the last readout: at ${dc} numbers the mechanism reads ${dc} per cached token where full attention reads ${
      2 * DH * CONFIG.HEADS
    }, and the second number is the one that grows with the head count.`;

    // --- 2. rotation
    const kRot = R.rotate(k, src);
    const qRot = R.rotate(q, T - 1);
    const truth = dot(qRot, kRot) / Math.sqrt(DH);
    const qcRot = Float64Array.from({ length: dc }, (_, j) => {
      let s = 0;
      for (let d = 0; d < DH; d++) s += WUK[j][d] * qRot[d];
      return s;
    });
    const anyway = dot(qcRot, c) / Math.sqrt(DH);
    // How big is the part that will not reassociate? Compare W^UK(R x) against W^UK x on the model's
    // own vectors rather than on random ones, so the number is about this model.
    let num = 0;
    let den = 0;
    for (let t = 0; t < T; t++) {
      const x = h0.K[t];
      const xr = R.rotate(x, src);
      for (let j = 0; j < dc; j++) {
        let a = 0;
        let b = 0;
        for (let d = 0; d < DH; d++) {
          a += WUK[j][d] * xr[d];
          b += WUK[j][d] * x[d];
        }
        num += (a - b) * (a - b);
        den += b * b;
      }
    }
    const commutator = Math.sqrt(num / Math.max(den, 1e-12));
    // The decoupled fix: a content term the absorption can reach, plus a small rotated channel.
    const kR = RR.rotate(h0.K[src].subarray(0, DR), src);
    const qR = RR.rotate(h0.Q[T - 1].subarray(0, DR), T - 1);
    const contentTerm = dot(qc, c);
    const posTerm = dot(qR, kR);
    const split = (contentTerm + posTerm) / Math.sqrt(DH + DR);
    ropeRead.update({
      truth: withRope ? fmt(truth, 5) : fmt(naive, 5),
      anyway: withRope ? fmt(anyway, 5) : fmt(absorbed, 5),
      commutator: withRope ? `${fmt(100 * commutator, 1)}%` : "—",
      fixed: decoupled ? fmt(split, 5) : "not split",
    });
    ropeNote.className = "note " + (withRope && !decoupled ? "warn" : "");
    ropeNote.innerHTML = !withRope
      ? `With no rotation on the key the two scores are the same number, which is the previous panel restated: the reassociation is free when nothing sits between the query's matrix and the key's. Switch the rotation on — every model on this timeline since concept 12 has it on — and watch the two readouts come apart.`
      : `A rotation is applied to the key <em>after</em> it is built, so it sits between the query's matrix and the reconstruction matrix, and matrix multiplication does not commute. The true score here is <strong>${fmt(
          truth,
          5
        )}</strong>; reassociating anyway gives <strong>${fmt(
          anyway,
          5
        )}</strong>. <span class="warn">That is not a small error to be waved through</span> — on this model's own key vectors the part that will not move past the rotation is <strong>${fmt(
          100 * commutator,
          1
        )}%</strong> the size of the term itself, and it is a different rotation for every cached token, so there is no single matrix to fold anywhere. ${
          decoupled
            ? `The fix is not a correction, it is a separate lane: leave the compressed content channel with no position in it at all, and carry position in <strong>${DR} extra numbers per token</strong> that every head shares and that are cached uncompressed. The score becomes a content term the absorption can reach plus a position term it does not need to — <strong>${fmt(
                split,
                5
              )}</strong> here. <strong>That number is not meant to match the one above it</strong>, and it does not: this is a different function, not an approximation of rotating the key. The paper does not repair the broken reassociation, it designs a model in which the reassociation is never obstructed — which is why the decoupled channel is part of the architecture and not a decoding trick. That is the first time on this timeline a mechanism has had to route <em>around</em> rotary position rather than build on it.`
            : `Turn the second switch on to see the fix.`
        }`;

    // --- 3. matched budget
    const pts = [];
    for (let r = 1; r <= D; r++) pts.push([r + DR, mlaError(kept, r)]);
    const gqaPts = [];
    for (let g = 1; g <= CONFIG.HEADS; g *= 2) gqaPts.push([2 * g * DH, gqaError(M, g), ""]);
    budgetCurve.update({
      points: pts,
      xRange: [0, 2 * DH * CONFIG.HEADS + 4],
      yRange: [0, 1.12],
    });
    budgetCurve.setDots(gqaPts, [0, 2 * DH * CONFIG.HEADS + 4], [0, 1.12]);
    // The head-sharing setting whose cache is closest to this latent's, for a like-for-like readout.
    const budget = dc + DR;
    let bestG = 1;
    for (let g = 1; g <= CONFIG.HEADS; g *= 2) if (Math.abs(2 * g * DH - budget) < Math.abs(2 * bestG * DH - budget)) bestG = g;
    const klMla = klDiv(base.probs, compressed.probs);
    const klGqa = klDiv(base.probs, forward(tokens, { kvGroups: bestG }).probs);
    budgetRead.update({
      mla: `${fmt(100 * mlaError(kept, dc), 1)}% at ${budget}`,
      gqa: `${fmt(100 * gqaError(M, bestG), 1)}% at ${2 * bestG * DH}`,
      klMla: fmt(klMla, 3),
      klGqa: fmt(klGqa, 3),
    });
    budgetNote.className = "note";
    budgetNote.innerHTML = `Both mechanisms are compressions of <em>one object</em> — the map from a layer's input to all its keys and all its values — so they can be put on one axis. The line is the latent at every width; the grey dots are one shared head, two groups, and every head keeping its own. The vertical axis is the share of that map each one destroys. At ${budget} numbers this latent loses ${fmt(
      100 * mlaError(kept, dc),
      1
    )}% where the nearest head-sharing setting loses ${fmt(
      100 * gqaError(M, bestG),
      1
    )}% for ${2 * bestG * DH}${
      mlaError(kept, dc) < gqaError(M, bestG)
        ? " — less damage for the same money or less"
        : ", so at this width the latent is the cheaper and the worse of the two; read the line, not one point"
    }. <strong>Wherever the two cost the same, the line sits below the dots</strong>, and the latent's range also runs below where head-sharing can go at all: one shared head is the floor at ${
      2 * DH
    } numbers, and the latent keeps going. The reason is structural rather than lucky — sharing must average a <em>fixed grouping</em> of heads, while the latent may choose whichever subspace carries the most, and the best subspace of a given size is by construction at least as good as any grouping of that size. This is the paper's own claim, that its cache <em>“is equal to GQA with only 2.25 groups, but its performance is stronger than MHA”</em>, run as arithmetic — and the paper runs it nowhere. <span class="warn">Two cautions.</span> These weights are random, so the map's spectrum is nearly flat and a trained one would be far more compressible, which understates the latent. And on the reader's own sentence the model's next-word distribution moves by ${fmt(
      klMla,
      3
    )} against ${fmt(klGqa, 3)} for the matched sharing — same direction, but an untrained model's output is arbitrary, so read the line above and not this pair.`;

    // --- 4. the arithmetic
    const mha = 2 * DH * CONFIG.HEADS;
    const here = { mha, gqa: 2 * 2 * DH, mqa: 2 * DH, mla: dc + DR };
    const ofBar = mha;
    sizeBars.update({
      mha: { value: here.mha, of: ofBar, text: `${here.mha} numbers` },
      gqa: { value: here.gqa, of: ofBar, text: `${here.gqa} numbers` },
      mqa: { value: here.mqa, of: ofBar, text: `${here.mqa} numbers` },
      mla: { value: here.mla, of: ofBar, text: `${here.mla} numbers` },
    });
    const v2mha = 2 * V2.heads * V2.dh * V2.layers;
    const v2mla = (V2.dc + V2.dr) * V2.layers;
    sizeRead.update({
      here: `${fmt(mha / here.mla, 1)}×`,
      there: `${fmt(v2mha / v2mla, 1)}×`,
      groups: `${fmt((V2.dc + V2.dr) / (2 * V2.dh), 2)} groups`,
      measured: "860.2K → 34.6K",
    });
    sizeNote.className = "note warn";
    sizeNote.innerHTML = `The bars are this app, per token per layer, at the latent width you have chosen: <strong>${fmt(
      mha / here.mla,
      1
    )}×</strong> smaller than every head keeping its own. Now the same arithmetic at the paper's own model — 128 heads, 128 wide, 60 layers, a 512-number latent and a 64-number position channel: <strong>1,966,080 numbers per token becomes 34,560</strong>, which is <strong>${fmt(
      v2mha / v2mla,
      1
    )}×</strong>, and is worth exactly ${fmt(
      (V2.dc + V2.dr) / (2 * V2.dh),
      2
    )} groups of head-sharing — the paper's own caption, reproduced. <strong>The saving is proportional to the head count, so a four-head toy understates this mechanism more than any other on the timeline.</strong> The controlled measurement to quote is the appendix's, not the abstract's: at 250B parameters with everything but the attention held fixed, 860.2K elements per token becomes 34.6K and the benchmark scores go <em>up</em>. The abstract's 93.3% and 5.76× compare a 236B sparse model against a dense 67B, with FP8 weights and a six-bit quantised cache stacked on the throughput figure — none of which is this mechanism.`;
  }

  return { update: render, unmount: () => {} };
}
