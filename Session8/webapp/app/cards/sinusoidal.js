// Concept 2 — sinusoidal position encoding.
// Built from docs/research/sinusoidal.md. The research changed three things I would have
// written from memory, and the card states all three: the relative-offset property is a
// hypothesis, extrapolation was never measured, and sinusoids did not beat learned embeddings.
import { el, slider, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { barList, readout } from "../views/bars.js";
import { dot, fmt, mulberry32, gauss } from "../model/ops.js";
import { forward, CONFIG } from "../model/transformer.js";
import { sinusoidalVector } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

const D = CONFIG.D;

// A learned-style table for comparison — Table 3 row (E) in miniature.
const rnd = mulberry32(99);
const LEARNED = Array.from({ length: 64 }, () => Float64Array.from({ length: D }, () => gauss(rnd) * 0.35));

const scheme = (base) => ({
  add(vec, pos) {
    const pe = sinusoidalVector(pos, vec.length, base);
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i] + pe[i];
    return out;
  },
});

export function sinusoidalCard(root, m) {
  let on = true;
  let base = 10000;
  let gain = 1;
  let posA = 3;
  let learned = false;
  let perm = null;

  root.appendChild(
    prose({
      problem:
        "Concept 1 left attention permutation-equivariant: shuffle the words and every score follows the shuffle unchanged. The dot product compares content, and content alone cannot say which word came first. The same paper therefore had to hand the model position from outside.",
      mechanism:
        "Add a fixed vector to each token's embedding before the first block, built from sines and cosines whose wavelengths form a geometric progression from 2π to 10000·2π. Dimension pair i turns at its own rate, so a position is a unique pattern of phases across the width. It is added once, at the bottom of the stack, to the same vector the token identity lives in.",
    })
  );

  // ------------------------------------------------- 1. permutation equivariance
  const peToggle = toggle({ label: "add position", value: true, onchange: (v) => ((on = v), render()) });
  const shuffleBtn = el("button", {
    class: "tg",
    type: "button",
    text: "shuffle the words",
    onclick: () => {
      const n = state.tokens.length;
      perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(mulberry32(i * 7 + n)() * (i + 1));
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      render();
    },
  });
  const shuffleRead = readout([
    { key: "delta", label: "largest score change under the shuffle" },
    { key: "verdict", label: "does order matter?" },
  ]);
  const shuffleNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "does the model know what order the words came in?" }),
      el("div", { class: "ctrls" }, [peToggle, shuffleBtn]),
      shuffleRead.node,
      shuffleNote,
    ])
  );

  // ------------------------------------------------------------ 2. offset curve
  const posSlider = slider({
    label: "measure from position",
    min: 0,
    max: 20,
    value: 3,
    oninput: (v) => ((posA = v), render()),
  });
  const learnedToggle = toggle({
    label: "compare with a learned table",
    value: false,
    onchange: (v) => ((learned = v), render()),
  });
  const offset = curveView({
    xLabel: "offset k, positions apart",
    yLabel: "PE(pos) · PE(pos+k)",
    ariaLabel: "similarity between two position vectors against the offset between them",
  });
  const offsetRead = readout([
    { key: "zero", label: "at k = 0" },
    { key: "same", label: "PE(pos)·PE(pos+4) at two different pos" },
    { key: "holds", label: "same value?" },
  ]);
  const offsetNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the paper's hypothesis, as a number you can move" }),
      el("div", { class: "ctrls" }, [posSlider.node, learnedToggle]),
      offset.node,
      offsetRead.node,
      offsetNote,
    ])
  );

  // ------------------------------------------------------------------ 3. the base
  const baseSlider = slider({
    label: "the 10000 constant",
    min: 1,
    max: 5,
    step: 0.1,
    value: 4,
    format: (v) => Math.round(10 ** v).toLocaleString(),
    oninput: (v) => ((base = 10 ** v), render()),
  });
  const waveBars = barList({
    rows: Array.from({ length: 8 }, (_, i) => ({ key: "w" + i, label: `pair ${i + 1}` })),
  });
  const waveRead = readout([{ key: "fits", label: "pairs that complete a full turn inside the sentence" }]);
  const waveNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what the 10000 actually sets" }),
      el("div", { class: "ctrls" }, [baseSlider.node]),
      waveBars.node,
      waveRead.node,
      waveNote,
    ])
  );

  // ------------------------------------------------------- 4. one shared budget
  const gainSlider = slider({
    label: "position strength",
    min: 0,
    max: 2,
    step: 0.1,
    value: 1,
    format: (v) => `${v.toFixed(1)}×`,
    oninput: (v) => ((gain = v), render()),
  });
  const budgetRead = readout([
    { key: "ratio", label: "position size ÷ content size" },
    { key: "cos", label: "how much the embedding moved" },
    { key: "top", label: "what the last word attends to most" },
  ]);
  const budgetNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "position and content share one vector" }),
      el("div", { class: "ctrls" }, [gainSlider.node]),
      budgetRead.node,
      budgetNote,
    ])
  );

  // ------------------------------------------------------------- trade + plain
  root.appendChild(
    tradeBlock({
      buys: [
        "Order becomes visible to a mechanism that is otherwise blind to it, for the cost of one addition at the bottom of the stack",
        "It is a function, not a table, so a vector exists for any position — including positions never seen in training",
        "Nothing is learned, so it adds no parameters and cannot be undertrained at the far end",
      ],
      givesUp: [
        "Position is added into the same vector the token identity lives in, so the two compete for one budget",
        "The relative-offset property the paper hoped for is a hypothesis it never tested, and no experiment here shows a head using it",
        "Extrapolation is the stated reason for choosing it, and the paper never measured extrapolation — later work found sinusoids extrapolate poorly",
      ],
      chooseWhen:
        "When you want position for free and at the trained length — which is most of the time in 2017. The paper's own ablation found learned embeddings performed the same (4.92 perplexity either way), so this was never a quality decision; it was a bet on lengths nobody checked.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It gives the model a sense of word order, which it otherwise completely lacks",
        "It is a formula rather than a stored list, so it never runs out of positions",
        "It costs nothing to store and nothing to train",
      ],
      cons: [
        "The position signal is mixed into the same numbers that carry the word's meaning, so they get in each other's way",
        "The reason for choosing it — that it might handle longer texts than it was trained on — was never actually tested in the paper, and later work found it does not hold up well",
        "In the paper's own comparison it scored the same as simply learning the positions, so the choice bought no accuracy",
      ],
      verdict:
        "A neat, free way to tell the model what order the words are in, chosen over the learned alternative for a reason nobody checked at the time. It works, and every later positional method on this timeline exists because it does not work well enough once the text gets longer than the training data.",
    })
  );

  // ------------------------------------------------------------------- render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    posSlider.set(posA);
    gainSlider.set(gain);

    const posScheme = on
      ? {
          add(vec, p) {
            const pe = sinusoidalVector(p, vec.length, base);
            const out = new Float64Array(vec.length);
            for (let i = 0; i < vec.length; i++) out[i] = vec[i] + pe[i] * gain;
            return out;
          },
        }
      : null;

    const res = forward(tokens, { position: posScheme });
    const h = res.trace[0].heads[0];

    // --- 1. shuffle test, on the pre-mask scores
    const rawScores = (r) => {
      const hh = r.trace[0].heads[0];
      return hh.Q.map((q) => hh.K.map((k) => dot(q, k)));
    };
    const A = rawScores(res);
    let delta = null;
    if (perm && perm.length === T) {
      const shuffled = perm.map((i) => tokens[i]);
      const B = rawScores(forward(shuffled, { position: posScheme }));
      delta = 0;
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) delta = Math.max(delta, Math.abs(B[i][j] - A[perm[i]][perm[j]]));
    }

    shuffleRead.update({
      delta: delta === null ? "—" : delta < 1e-6 ? delta.toExponential(1) : fmt(delta, 2),
      verdict: delta === null ? "shuffle to find out" : delta < 1e-6 ? "no" : "yes",
    });
    shuffleNote.className = "note " + (delta !== null && delta < 1e-6 ? "warn" : "");
    shuffleNote.textContent =
      delta === null
        ? "Press shuffle. The words move; the question is whether any score changes as a result."
        : delta < 1e-6
        ? `Every score is identical after the shuffle — the largest difference is ${delta.toExponential(1)}, which is floating-point noise. Attention is exactly permutation-equivariant, and with position switched off the model cannot tell "the keeper hid the code" from "the code hid the keeper". This is the defect §3.5 exists to fix.`
        : `Now the same shuffle changes the scores by up to ${fmt(delta, 2)}. The position vectors moved with the words, so the model can finally tell the orderings apart.`;

    // --- 2. offset curve
    const vec = (p) => (learned ? LEARNED[Math.min(p, LEARNED.length - 1)] : sinusoidalVector(p, D, base));
    const ks = Array.from({ length: 41 }, (_, i) => i);
    const curveAt = (p0) => ks.map((k) => [k, dot(vec(p0), vec(p0 + k))]);
    const cA = curveAt(posA);
    const cB = curveAt(posA + 6);
    const all = [...cA, ...cB].map(([, y]) => y);
    offset.update({
      points: cA,
      reference: cB,
      xRange: [0, 40],
      yRange: [Math.min(...all, 0), Math.max(...all)],
      mark: 0,
      markLabel: `k = 0 · ${fmt(dot(vec(posA), vec(posA)), 3)}`,
    });

    const a1 = dot(vec(posA), vec(posA + 4));
    const a2 = dot(vec(posA + 6), vec(posA + 10));
    const holds = Math.abs(a1 - a2) < 1e-6;
    offsetRead.update({
      zero: fmt(dot(vec(posA), vec(posA)), 3),
      same: `${fmt(a1, 4)} vs ${fmt(a2, 4)}`,
      holds: holds ? "yes" : "no",
    });
    offsetNote.className = "note " + (holds ? "" : "warn");
    offsetNote.innerHTML = learned
      ? `With a learned table the two curves come apart: ${fmt(a1, 4)} against ${fmt(a2, 4)} for the same offset of 4. The relative-offset structure is simply absent — a learned table stores positions, it does not relate them. That is the property the paper hoped sinusoids would give, and the reason this card exists as its own entry.`
      : `Drag the slider and the solid curve does not move: <strong>PE(pos)·PE(pos+k) depends only on k</strong>, never on where you measure from. At k = 0 it reads ${fmt(dot(vec(posA), vec(posA)), 3)} = d_model/2 for every position, so all these vectors have the same length. This is the consequence of the paper's hypothesis — but note the paper only says it <em>hypothesized</em> the model could use this, and never showed a head that does.`;

    // --- 3. the base
    const wave = {};
    let fits = 0;
    for (let i = 0; i < 8; i++) {
      const wl = (2 * Math.PI) / (1 / Math.pow(base, (2 * i) / D));
      if (wl <= T) fits++;
      wave["w" + i] = {
        value: Math.min(wl, 200),
        of: 200,
        text: wl > 999 ? wl.toExponential(1) : fmt(wl, 1),
        label: `pair ${i + 1}`,
        tone: wl <= T ? "" : "alt",
      };
    }
    waveBars.update(wave);
    waveRead.update({ fits: `${fits} of ${D / 2}` });
    waveNote.textContent = `Each dimension pair turns at its own rate; the bar is how many tokens it takes to complete one full turn. At base ${Math.round(base).toLocaleString()} only ${fits} of the ${D / 2} pairs finish a turn inside your ${T}-token sentence — the rest are still on their first slow sweep, which is what encodes long distances. Drag the base down and more pairs cycle: position becomes sharply distinguishable nearby, and starts repeating itself far away. The paper picks 10000 and never says why.`;

    // --- 4. shared budget
    const peNorm = Math.sqrt(dot(sinusoidalVector(T - 1, D, base), sinusoidalVector(T - 1, D, base))) * gain;
    const noPos = forward(tokens, { position: null });
    const eLast = noPos.trace[0].hidden[T - 1];
    const contentNorm = Math.sqrt(dot(eLast, eLast));
    const w = h.weights[T - 1];
    const topIdx = w.indexOf(Math.max(...w));

    const withPos = res.trace[0].hidden[T - 1];
    const cos =
      dot(eLast, withPos) / (Math.sqrt(dot(eLast, eLast)) * Math.sqrt(dot(withPos, withPos)) || 1);

    budgetRead.update({
      ratio: fmt(peNorm / (contentNorm || 1), 2),
      cos: fmt(cos, 3),
      top: `“${tokens[topIdx].word}”`,
    });
    budgetNote.textContent =
      gain === 0
        ? "At zero the position term is gone entirely and you are back to the shuffle test above."
        : `The position vector is ${fmt(peNorm / (contentNorm || 1), 2)}× the size of what the token itself contributes, and the two are added into the same ${D} numbers — there is no separate channel for position. Push the strength up and watch the last word's top attention target change: past a point the model is attending to where things are rather than what they are.`;
  }

  return { update: render, unmount: () => {} };
}
