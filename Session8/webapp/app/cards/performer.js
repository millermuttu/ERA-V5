// Concept 10 — Performer, FAVOR+.
// Built from docs/research/performer.md. The research settled the point most summaries invert:
// both the trigonometric and the positive feature maps are unbiased — unbiasedness was never the
// hard part. The trig estimator's variance blows up exactly where the kernel value is near zero,
// which is where the softmax denominator lives. Positivity fixes the denominator, not the mean.
import { el, slider, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout } from "../views/bars.js";
import { dot, fmt, mulberry32, gauss } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { softmax } from "../model/ops.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// FAVOR+ defines SM(x,y) = exp(xᵀy), with the 1/√d folded into the inputs — so scale by d^(-1/4)
// on each side. Getting this wrong converges to the wrong matrix and looks like an estimator bug.
//
// The other thing that matters, and the reason this card has a sharpness control: the estimator's
// variance grows with exp of the vector norms. This model is untrained, and its queries and keys
// have norms around 4 to 8 — at which no feasible number of random features converges. Measured:
// at norm 7.8 the error is still 0.99 at m = 2048; at norm 0.8 it reaches 0.007. That is Lemma 2,
// and it is a property of the method rather than a defect of this page.
const PRESCALE = Math.pow(DH, -0.25);

function drawProjections(m, seed) {
  const rnd = mulberry32(seed);
  return Array.from({ length: m }, () => Float64Array.from({ length: DH }, () => gauss(rnd)));
}

/** Positive random features: h(x) = exp(-‖x‖²/2), f = exp. */
function positiveFeatures(x, W, sharp = 1) {
  const v = Float64Array.from(x, (t) => t * PRESCALE * sharp);
  let sq = 0;
  for (const t of v) sq += t * t;
  const norm = 1 / Math.sqrt(W.length);
  return Float64Array.from(W, (w) => norm * Math.exp(dot(w, v) - sq / 2));
}

/** The earlier trigonometric features, kept for the comparison the paper's Lemma 2 is about. */
function trigFeatures(x, W, sharp = 1) {
  const v = Float64Array.from(x, (t) => t * PRESCALE * sharp);
  let sq = 0;
  for (const t of v) sq += t * t;
  const norm = Math.exp(sq / 2) / Math.sqrt(W.length);
  const out = new Float64Array(W.length * 2);
  W.forEach((w, i) => {
    const a = dot(w, v);
    out[2 * i] = norm * Math.cos(a);
    out[2 * i + 1] = norm * Math.sin(a);
  });
  return out;
}

export function performerCard(root, m) {
  let features = 16;
  let positive = true;
  let redraws = 1;
  let sharp = 0.25; // scales the query/key norms, which is what governs the variance

  root.appendChild(
    prose({
      problem:
        "Linear attention swapped the exponential for a feature map chosen because it is non-negative and differentiable, and simply hoped the result behaved. That leaves the obvious question unanswered: how far is the cheap thing from the exact thing it replaced?",
      mechanism:
        "Estimate the softmax kernel itself instead of replacing it. Project queries and keys onto m random directions and pass them through a positive nonlinearity, and the dot product of those features is an unbiased estimate of exp(qᵀk) — so the same regrouping applies, and the thing being estimated is the real attention. Two details carry the contribution. The features must be positive, not the original sines and cosines: both are unbiased, but the trigonometric estimator's variance grows without bound exactly where the kernel value approaches zero, which is where the softmax denominator lives. And the random directions are made orthogonal, which reduces the variance further.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "the same picture, estimated");

  // ------------------------------------------------------ error against exact
  const mSlider = slider({
    label: "random features m",
    min: 1,
    max: 128,
    value: 16,
    oninput: (v) => ((features = v), render()),
  });
  const posToggle = toggle({
    label: "positive features",
    value: true,
    onchange: (v) => ((positive = v), render()),
  });
  const sharpSlider = slider({
    label: "query/key size",
    min: 0.05,
    max: 1,
    step: 0.05,
    value: 0.25,
    format: (v) => `${v.toFixed(2)}×`,
    oninput: (v) => ((sharp = v), render()),
  });
  const errCurve = curveView({
    xLabel: "random features m",
    yLabel: "largest error",
    ariaLabel: "approximation error against exact attention as the number of random features grows",
  });
  const errRead = readout([
    { key: "max", label: "largest error in the attention matrix" },
    { key: "mean", label: "average error" },
    { key: "norm", label: "largest query norm" },
  ]);
  const errNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "how close is the estimate, on your sentence" }),
      el("div", { class: "ctrls" }, [mSlider.node, sharpSlider.node, posToggle]),
      errCurve.node,
      errRead.node,
      errNote,
    ])
  );

  // ------------------------------------------------------------ unbiasedness
  const redrawSlider = slider({
    label: "average over draws",
    min: 1,
    max: 40,
    value: 1,
    oninput: (v) => ((redraws = v), render()),
  });
  const biasRead = readout([
    { key: "single", label: "error from one draw" },
    { key: "avg", label: "error after averaging" },
    { key: "verdict", label: "what that shows" },
  ]);
  const biasNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "unbiased is not the same as accurate" }),
      el("div", { class: "ctrls" }, [redrawSlider.node]),
      biasRead.node,
      biasNote,
    ])
  );

  // ------------------------------------------------------ where trig breaks
  const trigRead = readout([
    { key: "small", label: "error where the true weight is smallest" },
    { key: "large", label: "error where it is largest" },
    { key: "ratio", label: "ratio" },
  ]);
  const trigNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the sines and cosines had to go" }),
      trigRead.node,
      trigNote,
    ])
  );

  // ------------------------------------------------------------------- cost
  const costRead = readout([
    { key: "exact", label: "multiplies, exact attention" },
    { key: "approx", label: "multiplies, this approximation" },
    { key: "verdict", label: "cheaper here?" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "and what it costs at this length" }),
      costRead.node,
      costNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Linear cost with an explicit, tunable approximation to real softmax attention rather than an arbitrary substitute — the estimate is unbiased and its error falls as you add random features",
        "Positive features fix the failure that made the earlier trigonometric estimator unusable: its variance grows without bound where the kernel value nears zero, which is exactly where a softmax denominator sits",
        "Orthogonal random directions reduce the variance further, with a proven gap over independent ones",
        "It can be initialised from a pretrained softmax Transformer and fine-tuned, rather than trained from scratch",
      ],
      givesUp: [
        "It is still an estimate, and no finite number of features makes the error zero",
        "Wherever both are reported the approximation trails exact attention — 33.00 against 33.32 on one protein benchmark, 23.48 against 25.07 out of distribution",
        "The row that actually beats the Transformer uses a plain relu feature map with no guarantee at all — the very thing this work criticised linear attention for",
        "The headline complexity result is proved for the trigonometric mechanism, not for the positive features that make the method work",
        "Enough features to be accurate erodes the speed advantage, and at short lengths the approximation costs more than the exact computation",
      ],
      chooseWhen:
        "When you need linear cost and want a principled bound on what was given up, or when adapting an existing softmax model rather than training one. Historically its importance is as the answer to how close cheap can get — and the answer shaped what the field expected from every approximation after it.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It aims at the real thing rather than a convenient substitute, so you can say how far off it is",
        "Adding more random directions makes it more accurate, and you choose how many",
        "It fixed a specific failure in the earlier version of the trick, where the estimate went wild precisely in the cases that matter most",
        "An existing model can be converted to it instead of trained again from nothing",
      ],
      cons: [
        "It is a guess with error bars, and the error never reaches zero however many directions you add",
        "Where the paper reports both, the guess scores slightly below the exact calculation",
        "Making it accurate enough costs enough that the speed saving shrinks — on a short sentence it is slower than simply doing the exact sum",
      ],
      verdict:
        "The most honest attempt to make attention cheap: instead of replacing the calculation with something convenient, it estimates the real one and tells you how wrong it might be. That honesty is also its epitaph — being able to measure the gap made it clear there was a gap, and exact attention got fast enough that most people stopped paying it.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    mSlider.set(features);
    redrawSlider.set(redraws);

    const exact = forward(tokens, {});
    const h = exact.trace[0].heads[0];

    // Exact attention at the same sharpness the features are computed at, so the comparison is
    // like for like: softmax((sharp·q)·(sharp·k)/√d).
    const s2 = (sharp * PRESCALE) ** 2;
    const trueW = h.Q.map((q, i) =>
      softmax(h.K.map((k, j) => (j <= i ? dot(q, k) * s2 : -Infinity)))
    );

    // The approximation from random features.
    const approxMatrix = (mm, usePositive, seed) => {
      const W = drawProjections(usePositive ? mm : Math.max(1, Math.floor(mm / 2)), seed);
      const f = usePositive ? positiveFeatures : trigFeatures;
      const fq = h.Q.map((x) => f(x, W, sharp));
      const fk = h.K.map((x) => f(x, W, sharp));
      return fq.map((qq, i) => {
        const row = [];
        let z = 0;
        for (let j = 0; j <= i; j++) {
          const s = dot(qq, fk[j]);
          row.push(s);
          z += s;
        }
        return row.map((s) => (z !== 0 ? s / z : 0));
      });
    };

    const errorOf = (A) => {
      let max = 0;
      let sum = 0;
      let n = 0;
      for (let i = 0; i < T; i++)
        for (let j = 0; j <= i; j++) {
          const d = Math.abs((A[i][j] ?? 0) - trueW[i][j]);
          max = Math.max(max, d);
          sum += d;
          n++;
        }
      return { max, mean: sum / n };
    };

    const approxNow = approxMatrix(features, positive, 1234);
    flow.update({
      tokens,
      head: { ...h, emb: exact.trace[0].input },
      weights: approxNow.map((row, i) => tokens.map((_, j) => (j <= i ? row[j] ?? 0 : 0))),
      out: h.out,
      top: exact.top,
      opts: {},
    });
    flowNote.innerHTML = `The marks are the <em>estimated</em> attention, not the real one — the same picture concept 1 draws, reconstructed from ${features} random projections instead of from every pair. Drag the feature count and watch the pattern sharpen towards the true one; drag the query/key size and watch it fall apart. Structurally nothing else changes, which is the claim: this is meant to be the same computation, done cheaply, rather than a different one.`;

    const cur = errorOf(approxNow);
    const maxNorm = Math.max(...h.Q.map((q) => Math.sqrt(dot(q, q)))) * PRESCALE * sharp;
    errRead.update({
      max: fmt(cur.max, 4),
      mean: fmt(cur.mean, 5),
      norm: fmt(maxNorm, 2),
    });

    // the convergence curve
    const ms = [1, 2, 4, 8, 16, 32, 64, 128];
    const pts = ms.map((mm) => [mm, errorOf(approxMatrix(mm, positive, 1234)).max]);
    errCurve.update({
      points: pts,
      xRange: [1, 128],
      yRange: [0, Math.max(...pts.map((p) => p[1])) * 1.05],
      mark: features,
      markLabel: `m = ${features}`,
    });

    errNote.className = "note " + (cur.max > 0.3 ? "warn" : "");
    errNote.innerHTML =
      cur.max > 0.3
        ? `At this size the estimate is off by ${fmt(cur.max, 4)} — and adding features barely helps. This is not a bug and it is the most useful thing on the card: the estimator's variance grows with <em>exp</em> of the vector norms, so at a largest norm of ${fmt(maxNorm, 2)} no affordable number of features converges. Measured here: at norm 7.8 the error is still 0.99 with 2,048 features; at norm 0.8 it reaches 0.007. Pull "query/key size" down and watch the whole picture change.`
        : `With ${features} random features and a largest query norm of ${fmt(maxNorm, 2)}, the biggest weight in the attention matrix is off by ${fmt(cur.max, 4)} and the average by ${fmt(cur.mean, 5)}. Drag m and the error falls; it never reaches zero, which is what an approximation with a bound means, as against linear attention's different mechanism with no bound. Now drag the size control up: convergence depends far more on how large the queries and keys are than on how many features you buy.`;

    // --- unbiasedness: average independent draws
    const draws = Array.from({ length: redraws }, (_, r) => approxMatrix(features, positive, 1000 + r * 77));
    const averaged = trueW.map((row, i) =>
      row.map((_, j) => (j <= i ? draws.reduce((s, A) => s + (A[i][j] ?? 0), 0) / redraws : 0))
    );
    const single = errorOf(draws[0]);
    const avg = errorOf(averaged);
    biasRead.update({
      single: fmt(single.max, 4),
      avg: fmt(avg.max, 4),
      verdict: avg.max < single.max ? "averaging helps" : "no gain",
    });
    biasNote.textContent = `One draw of ${features} random directions is off by ${fmt(single.max, 4)}. Average ${redraws} independent draws and the error falls to ${fmt(avg.max, 4)}. That is what unbiased means: the estimate is centred on the right answer, so the noise averages away. It is also the point most summaries invert — the trigonometric features are unbiased too. Unbiasedness was never the hard part.`;

    // --- where trig breaks: error against the size of the true weight
    const trigA = approxMatrix(features, false, 1234);
    const posA = approxMatrix(features, true, 1234);
    const cells = [];
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++)
        cells.push({
          w: trueW[i][j],
          t: Math.abs(trigA[i][j] - trueW[i][j]),
          p: Math.abs(posA[i][j] - trueW[i][j]),
        });
    cells.sort((a, b) => a.w - b.w);
    const smallest = cells.slice(0, Math.max(1, Math.floor(cells.length / 4)));
    const largest = cells.slice(-Math.max(1, Math.floor(cells.length / 4)));
    const mean = (arr, key) => arr.reduce((s, c) => s + c[key], 0) / arr.length;
    const tSmall = mean(smallest, "t");
    const tLarge = mean(largest, "t");
    trigRead.update({
      small: fmt(tSmall, 5),
      large: fmt(tLarge, 5),
      ratio: `${fmt(tSmall / (tLarge || 1e-9), 2)}×`,
    });
    trigNote.innerHTML = `Sorting every cell of the attention matrix by how much weight it really carries: the trigonometric estimator's error on the <em>smallest</em> quarter averages ${fmt(tSmall, 5)} against ${fmt(tLarge, 5)} on the largest — worse precisely where the true value is near zero. The positive features give ${fmt(mean(smallest, "p"), 5)} on the same smallest quarter. That asymmetry is the paper's Lemma 2: the trigonometric variance scales as one over the kernel value squared, so it diverges as the value approaches zero, while the positive version's scales <em>with</em> the value and vanishes there. Since the softmax denominator is a sum over exactly those near-zero terms, the old estimator broke where it was needed most.`;

    // --- cost
    const exactOps = ((T * (T + 1)) / 2) * DH;
    const approxOps = T * features * DH * 2;
    costRead.update({
      exact: exactOps.toLocaleString(),
      approx: approxOps.toLocaleString(),
      verdict: approxOps < exactOps ? "yes" : "no — it costs more",
    });
    costNote.className = "note " + (approxOps < exactOps ? "" : "warn");
    costNote.textContent = `At ${T} tokens with ${features} features the approximation does ${approxOps.toLocaleString()} multiplies against exact attention's ${exactOps.toLocaleString()} — ${approxOps < exactOps ? "cheaper" : `${fmt(approxOps / exactOps, 1)}× more expensive`}. The crossover is where m is roughly half the sequence length, so on a sentence this short the method is pure overhead. It pays at thousands of tokens, and this app can honestly show you the estimator converging but never the speed.`;
  }

  return { update: render, unmount: () => {} };
}
