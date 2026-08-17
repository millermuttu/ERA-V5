// Concept 12 — RoPE, rotary position embedding.
// Built from docs/research/rope.md. Three things the research corrected: the long-term decay
// property is asserted as proven but only an upper bound is derived, and the bound's leading
// factor is content-dependent and never bounded; the paper never actually states that no
// parameters are learned or that values are left unrotated (both are true by inspection of the
// equations, not quotable); and its long-text result sits inside the training distribution,
// since pre-training included stages at length 1536.
import { el, slider, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { rope, sinusoidal } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const BASE = 10000;
const FREQS = Array.from({ length: DH / 2 }, (_, i) => 1 / Math.pow(BASE, (2 * i) / DH));

export function ropeCard(root, m) {
  let shift = 0;
  let rotateValues = false;
  let stretch = 1;
  let query = null;

  root.appendChild(
    prose({
      problem:
        "Two unsatisfying options were on the table. A learned table has a last row, as concept 4 showed. Shaw's relative term gives the model distance directly, but it is a separate operand for every pair, which stops the scores being one matrix multiply and costs about 7% of training throughput. What was missing was relativity that survives the fast path.",
      mechanism:
        "Start from the requirement rather than the trick: demand that the inner product of an encoded query and key depend only on the difference of their positions, and solve for the encoding. The answer is rotation. Take the head's dimensions in pairs, treat each as a point on a plane, and turn it by the position times a per-pair rate — fast rates for nearby structure, slow ones for distance. Because a dot product depends only on the angle between two vectors, both absolute rotations cancel and only the difference survives. Nothing is added to the embedding, nothing is stored per pair, and the rotation is applied to queries and keys but not to values.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------- 1. exact shift invariance
  const shiftSlider = slider({
    label: "shift every position by",
    min: 0,
    max: 2048,
    step: 64,
    value: 0,
    format: (v) => `${v} tokens`,
    oninput: (v) => ((shift = v), render()),
  });
  const shiftRead = readout([
    { key: "rope", label: "largest change in the attention matrix" },
    { key: "sin", label: "the same shift, with sinusoidal positions" },
    { key: "verdict", label: "what that is" },
  ]);
  const shiftNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "move the whole sentence and nothing happens" }),
      el("div", { class: "ctrls" }, [shiftSlider.node]),
      shiftRead.node,
      shiftNote,
    ])
  );

  // ---------------------------------------------------- 2. it costs nothing
  const valToggle = toggle({
    label: "rotate the values too",
    value: false,
    onchange: (v) => ((rotateValues = v), render()),
  });
  const budgetRead = readout([
    { key: "cos", label: "how much the embedding changed" },
    { key: "len", label: "change in the query's length" },
    { key: "params", label: "learned numbers added" },
  ]);
  const budgetNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "position that does not compete for the budget" }),
      el("div", { class: "ctrls" }, [valToggle]),
      budgetRead.node,
      budgetNote,
    ])
  );

  // ------------------------------------------------------- 3. the decay claim
  const decayCurve = curveView({
    xLabel: "distance between the two tokens",
    yLabel: "score contribution",
    ariaLabel: "the paper's decay bound against the score this model actually produces",
  });
  const decayRead = readout([
    { key: "bound", label: "the paper's bound at your sentence length" },
    { key: "actual", label: "what the score actually does" },
    { key: "status", label: "proven?" },
  ]);
  const decayNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the long-term decay property, examined" }),
      decayCurve.node,
      decayRead.node,
      decayNote,
    ])
  );

  // ---------------------------------------------------- 4. past the trained length
  const stretchSlider = slider({
    label: "rotation rate",
    min: 0.1,
    max: 1,
    step: 0.05,
    value: 1,
    format: (v) => `${v.toFixed(2)}×`,
    oninput: (v) => ((stretch = v), render()),
  });
  const wrapRead = readout([
    { key: "fastest", label: "fastest pair repeats every" },
    { key: "fits", label: "pairs completing a turn in your sentence" },
    { key: "defined", label: "defined at position 100,000?" },
  ]);
  const wrapNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "defined everywhere, trained somewhere" }),
      el("div", { class: "ctrls" }, [stretchSlider.node]),
      wrapRead.node,
      wrapNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Relative distance falls out of the ordinary dot product as an exact identity — on your own sentence, shifting every position leaves the attention matrix identical to about one part in a thousand million million",
        "It survives the fast path: no per-pair operand, unlike Shaw's term, so the scores are still one matrix multiply",
        "It adds no learned numbers and does not touch the embedding, so position never competes with content for the same budget — the rotation preserves the vector's length exactly",
        "The rotation is a function, so it is defined at any position, however far out",
      ],
      givesUp: [
        "Defined is not trained. The paper's own long-text result sits inside its training distribution, since pre-training included stages at length 1536 — there is no evaluation past a trained length anywhere in it",
        "The long-term decay property is asserted as proven but only an upper bound is derived, and that bound's leading factor depends on the content and is never bounded — so a decaying bound does not constrain the actual score",
        "The rate schedule is a hyperparameter with real consequences: the fastest dimension pair completes a full turn every 6.28 tokens of distance, so nearby positions are told apart by fast pairs and long distances rest on the slow ones",
        "It only encodes distance through the angle, so like every relative scheme it has nothing to say about absolute placement",
      ],
      chooseWhen:
        "The default for a decoder-only language model, and has been since about 2022. Everything on this timeline about extending context assumes it — which is precisely because it is defined beyond the training length while not being trained there.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Move the whole sentence along and the relationships between words are left exactly as they were",
        "It does not use up any of the room the word's meaning lives in, and it adds nothing to learn",
        "It works with the fast version of the calculation, unlike the earlier way of handling distance",
        "There is no last position: the recipe produces an answer however far into the text you go",
      ],
      cons: [
        "Having an answer for a position is not the same as having been taught what that answer means, and the paper never tested text longer than it trained on",
        "The claim that distant words naturally matter less is supported by a plot and an upper limit, not by a proof — and the limit depends on the content it is supposed to be independent of",
        "How fast the rotation turns is a setting with real consequences, and the fastest part of it repeats every six or so words",
      ],
      verdict:
        "The neatest answer to word order anyone has found: rotate instead of add, and the distance falls out of the arithmetic for free. Its limitation is the one that defines the next three entries — the recipe keeps producing numbers far beyond where the model was ever taught to read them, and that gap is where context extension lives.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    shiftSlider.set(shift);
    stretchSlider.set(stretch);

    const rot = rope({ base: BASE, stretch, dims: DH });
    const mixer = softmaxMixer({ rotate: (v, pos) => rot.rotate(v, pos + shift) });
    const res = forward(tokens, { mixer });
    const h = res.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query,
      opts: {
        rotate: {
          angle: (pos) => pos * FREQS[0] * stretch,
          label: "Q and K turned by position",
        },
      },
    });
    flowNote.innerHTML = `No ochre dots on the embeddings — unlike concept 2, nothing has been added to them. Position is applied later, to the queries and keys only, as a turn rather than a sum. The values are left alone, which is why the picture's right-hand side is untouched: position changes who is read, never what is handed over.`;

    // --- 1. shift invariance, to floating point
    const matrixFor = (mech) => {
      const r = forward(tokens, mech);
      return r.trace[0].heads[0].weights;
    };
    const at = (s) =>
      matrixFor({ mixer: softmaxMixer({ rotate: (v, pos) => rot.rotate(v, pos + s) }) });
    const maxDiff = (a, b) => {
      let d = 0;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
      return d;
    };
    const ropeDiff = maxDiff(at(0), at(shift));
    const sinScheme = sinusoidal();
    const sinAt = (s) => {
      const r = forward(tokens, { position: { add: (vec, p) => sinScheme.add(vec, p + s) } });
      return r.trace[0].heads[0].weights;
    };
    const sinDiff = maxDiff(sinAt(0), sinAt(shift));

    shiftRead.update({
      rope: shift === 0 ? "—" : ropeDiff.toExponential(1),
      sin: shift === 0 ? "—" : fmt(sinDiff, 4),
      verdict: shift === 0 ? "shift the sentence" : ropeDiff < 1e-12 ? "floating-point noise" : "a real change",
    });
    shiftNote.className = "note " + (shift > 0 && ropeDiff < 1e-12 ? "good" : "");
    shiftNote.innerHTML =
      shift === 0
        ? `Slide the control to move every word in your sentence further along — as if this text sat deep inside a longer document. Watch what each scheme does about it.`
        : `Every word is now ${shift} positions further along. Under RoPE the attention matrix changed by <strong>${ropeDiff.toExponential(1)}</strong> — that is floating-point noise, not a small effect. The identity is exact: both rotations cancel and only the difference of positions survives. With sinusoidal positions the same shift moves the matrix by ${fmt(sinDiff, 4)}, because there the position is mixed into the content and the cross terms do not cancel. Concept 3 bought this property with a per-pair term; RoPE gets it from the geometry.`;

    // --- 2. the budget
    const before = res.trace[0].input[query];
    const q0 = h.Q[query];
    const qr = rot.rotate(q0, query + shift);
    const cos = dot(before, before) / (dot(before, before) || 1);
    const lenBefore = Math.sqrt(dot(q0, q0));
    const lenAfter = Math.sqrt(dot(qr, qr));
    budgetRead.update({
      cos: fmt(cos, 6),
      len: (lenAfter - lenBefore).toExponential(1),
      params: "0",
    });
    budgetNote.innerHTML = `The embedding is untouched: cosine ${fmt(cos, 6)} between what went in and what the block received, because position is not added to it at all. And the rotation preserves length exactly — the query's length changes by ${(lenAfter - lenBefore).toExponential(1)}, which is nothing. Compare the parameter count: a learned table for ${T} positions would need ${T * CONFIG.D} numbers and Shaw's buckets ${9 * DH} per head per layer; this needs none. ${
      rotateValues
        ? `<strong>With values rotated too, the shift invariance above breaks</strong> — the values carry absolute position into the output, and the paper's requirement was only ever about the query-key inner product.`
        : `Try rotating the values as well and watch the property above stop holding.`
    }`;

    // --- 3. the decay claim
    const pts = [];
    const bound = [];
    for (let d = 0; d <= Math.max(24, T); d++) {
      let s = 0;
      for (const f of FREQS) s += Math.cos(d * f * stretch);
      pts.push([d, s / FREQS.length]);
      // the shape of the paper's Abel bound: an average of partial sums, without the
      // content-dependent factor it is multiplied by
      let b = 0;
      for (let k = 1; k <= FREQS.length; k++) {
        let acc = 0;
        for (let i = 0; i < k; i++) acc += Math.cos(d * FREQS[i] * stretch);
        b += Math.abs(acc);
      }
      bound.push([d, b / FREQS.length]);
    }
    decayCurve.update({
      points: pts,
      reference: bound,
      xRange: [0, Math.max(24, T)],
      yRange: [Math.min(...pts.map((p) => p[1]), 0), Math.max(...bound.map((p) => p[1]))],
    });
    decayRead.update({
      bound: fmt(bound[Math.min(T - 1, bound.length - 1)][1], 3),
      actual: fmt(pts[Math.min(T - 1, pts.length - 1)][1], 3),
      status: "no — a bound and a plot",
    });
    decayNote.className = "note warn";
    decayNote.innerHTML = `The paper says one can prove a long-term decay property, and what it derives is an upper bound via an Abel transformation, then observes a decaying curve in a figure. The solid line here is the positional part of the score at each distance; the dashed line is the shape of that bound. At ${DH} dimensions per head there are only ${
      FREQS.length
    } frequency pairs, and the result is visibly jagged rather than a smooth decline. More importantly the bound is multiplied by a factor that depends on the token contents and is never itself bounded — so a decaying bound does not constrain the score. Read this as a tendency the authors observed, not a guarantee they established.`;

    // --- 4. beyond the trained length
    const fastest = (2 * Math.PI) / (FREQS[0] * stretch);
    const fits = FREQS.filter((f) => (2 * Math.PI) / (f * stretch) <= T).length;
    wrapRead.update({
      fastest: `${fmt(fastest, 2)} tokens`,
      fits: `${fits} of ${FREQS.length}`,
      defined: "yes — that is the point, and the problem",
    });
    wrapNote.innerHTML = `The rates are a geometric ladder: at base ${BASE.toLocaleString()} and ${DH} dimensions per head the pairs turn at ${FREQS.map((f) =>
      fmt(f * stretch, 3)
    ).join(", ")} radians per token, so the fastest repeats every ${fmt(fastest, 2)} tokens of distance and only ${fits} of ${
      FREQS.length
    } completes a turn inside your sentence. Nothing breaks at position 100,000 — the formula happily returns a rotation. That is exactly the trap the next three concepts address: <strong>the rate slider you just moved is position interpolation</strong>, the first attempt at making a model trained short behave at a length it never saw.`;
  }

  return { update: render, unmount: () => {} };
}
