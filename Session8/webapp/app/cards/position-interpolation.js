// Concept 16 — Position Interpolation.
// Built from docs/research/position-interpolation.md. The research settled four things that shape
// this card. The seam needs no change at all: `rope({stretch})` already *is* the paper's eq. 4, and
// stretch = L/L' reproduces it to 0.00e+0. The famous ~600x is a ratio between two bounds, one of
// which the paper has just called vacuous, and it rests on a numerical floor checked at one head
// width — which fails at this app's width. Theorem 2.1's proof reverses the tangent-line inequality,
// so the published bound is too tight, and on this model's real vectors the data escapes it. And the
// paper's Figure 2 is a least-squares fit to random noise, not a measurement of any model; the
// control the paper omits makes the blow-up vanish.
import { el, slider, toggle, choice } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt, mulberry32, gauss } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { rope } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const C = 10000;
const R = rope({ base: C, dims: DH });
const FREQS = R.freqs; // {1, 0.1, 0.01, 0.001} at d_k = 8

/** Run the model with each token placed at an arbitrary position. */
const atPositions = (tokens, posOf) =>
  forward(tokens, { mixer: softmaxMixer({ rotate: (v, i) => R.rotate(v, posOf(i)) }) });

const maxDiff = (A, B) => {
  let m = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  return m;
};

/**
 * The paper's eq. 3 written out: h_j = (q_2j + i q_2j+1)(k_2j − i k_2j+1), and the score at relative
 * distance s is the real part of the sum of h_j e^{i s θ_j}. Everything the theory section says is a
 * statement about this function, so the card computes it from the model's own q and k.
 */
function coefficients(q, k) {
  const h = [];
  for (let j = 0; j < DH / 2; j++) {
    const a = q[2 * j] * k[2 * j] + q[2 * j + 1] * k[2 * j + 1];
    const b = q[2 * j + 1] * k[2 * j] - q[2 * j] * k[2 * j + 1];
    h.push({ a, b, mag: Math.hypot(a, b) });
  }
  return h;
}
const scoreAt = (h, s) => h.reduce((acc, c, j) => acc + c.a * Math.cos(s * FREQS[j]) - c.b * Math.sin(s * FREQS[j]), 0);

/** B(s) = Σ_k |A_{k+1}(s)|, the quantity the 600x rests on. Computed at any head width. */
function bOverD(d, s) {
  const half = d / 2;
  let re = 0;
  let im = 0;
  let sum = 0;
  for (let j = 0; j < half; j++) {
    const th = Math.pow(C, (-2 * j) / d);
    re += Math.cos(s * th);
    im += Math.sin(s * th);
    sum += Math.hypot(re, im);
  }
  return sum / d;
}

/**
 * Appendix C.1's recipe: fit `L` standard-normal targets with the `d` trig basis functions by least
 * squares, then evaluate outside the fitted range. The paper presents the result as evidence that
 * attention scores explode past the training window; running it at a second width shows what it is
 * really measuring.
 */
const fitCache = new Map();
function figureTwo(d, L) {
  const key = d + ":" + L;
  if (fitCache.has(key)) return fitCache.get(key);
  const rnd = mulberry32(4242);
  const half = d / 2;
  const th = Array.from({ length: half }, (_, j) => Math.pow(C, (-2 * j) / d));
  const basis = (s) => {
    const row = new Float64Array(d);
    for (let j = 0; j < half; j++) {
      row[2 * j] = Math.cos(s * th[j]);
      row[2 * j + 1] = Math.sin(s * th[j]);
    }
    return row;
  };
  // normal equations, then Gaussian elimination with partial pivoting — d is at most 128
  const M = Array.from({ length: d }, () => new Float64Array(d + 1));
  for (let s = 0; s < L; s++) {
    const row = basis(s);
    const y = gauss(rnd);
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) M[a][b] += row[a] * row[b];
      M[a][d] += row[a] * y;
    }
  }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) M[a][b] = M[b][a];
  for (let a = 0; a < d; a++) M[a][a] += 1e-9; // the basis is near-singular when d is close to L
  for (let col = 0; col < d; col++) {
    let piv = col;
    for (let r = col + 1; r < d; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const p = M[col][col] || 1e-12;
    for (let r = col + 1; r < d; r++) {
      const f = M[r][col] / p;
      if (!f) continue;
      for (let c = col; c <= d; c++) M[r][c] -= f * M[col][c];
    }
  }
  const w = new Float64Array(d);
  for (let r = d - 1; r >= 0; r--) {
    let acc = M[r][d];
    for (let c = r + 1; c < d; c++) acc -= M[r][c] * w[c];
    w[r] = acc / (M[r][r] || 1e-12);
  }
  const evalAt = (s) => {
    const row = basis(s);
    let acc = 0;
    for (let a = 0; a < d; a++) acc += row[a] * w[a];
    return acc;
  };
  const step = Math.max(1, Math.floor((2 * L) / 400));
  const pts = [];
  let inside = 0;
  let outside = 0;
  for (let s = 0; s < 2 * L; s += step) {
    const v = evalAt(s);
    pts.push([s, v]);
    if (s < L) inside = Math.max(inside, Math.abs(v));
    else outside = Math.max(outside, Math.abs(v));
  }
  const out = { pts, inside, outside, coef: Math.max(...Array.from(w, Math.abs)), L };
  fitCache.set(key, out);
  return out;
}

const WIDTHS = [8, 16, 32, 64, 128];
const FITS = [
  { value: "8:16", label: "d = 8, L = 16 — this app" },
  { value: "128:2048", label: "d = 128, L = 2048 — the paper's own setting" },
  { value: "8:2048", label: "d = 8, L = 2048 — the control the paper omits" },
  { value: "real", label: "the model's own query and key" },
];

export function positionInterpolationCard(root, m) {
  let spread = 64; // tokens between this sentence's words, inside the long document
  let usePI = true;
  let s1 = 3; // which unit interval the bound is examined on
  let corrected = false;
  let width = 128;
  let fit = "8:16";

  root.appendChild(
    prose({
      problem:
        "The previous concept's rotation is defined at every position and was trained at none of them past the window, and the concept after that measured what happens when you go there: the score can be anything. But the models that needed longer contexts already existed, already had this rotation baked in, and cost millions to train — so replacing the position scheme, which is what the distance-penalty branch asks for, means starting over. The question changed from “what should we have used?” to “what can we do to the model we already shipped?”",
      mechanism:
        "Do not send the model to a position it has never seen. Divide every position index by the extension factor before it enters the rotation, so a token at 8,000 in a model trained to 2,000 is rotated as if it sat at 2,000 — inside the range the weights know. Nothing else changes: not the frequencies, not the base, not a single weight. Then repair the squeeze with a short burst of further training. The move is available only because the rotation is a function of a real number: there is an angle for position two-and-a-half, where a lookup table has no such row.",
    })
  );

  const boundary = el("p", { class: "note warn" });
  boundary.innerHTML = `<strong>What this page cannot show, said before anything else.</strong> Every quality claim in this paper is a perplexity or a benchmark score measured <em>after</em> fine-tuning a model with billions of parameters. The model on this page is untrained and cannot be fine-tuned, and perplexity here would mean nothing. So this card demonstrates the geometry — that the rescaling lands exactly on the trained grid, what it costs at the near end, and what the theory does and does not prove — and quotes the paper for everything else, marked as quoted. The one thing a toy with random weights <em>can</em> settle is an exact identity or a counterexample, and there are four of each below.`;
  root.appendChild(boundary);

  const { flow, note: flowNote } = flowPanel(root, "the same sentence, sitting deep inside a long document");

  // ---------------------------------------------------- 1. back onto the trained grid
  const spreadSlider = slider({
    label: "tokens between each of your words",
    min: 1,
    max: 128,
    value: 64,
    format: (v) => `${v} — the sentence ends at position ${15 * v}`,
    oninput: (v) => ((spread = v), render()),
  });
  const piToggle = toggle({
    label: "rescale the positions",
    value: true,
    onchange: (v) => ((usePI = v), render()),
  });
  const gridRead = readout([
    { key: "delta", label: "how far the attention moved from the trained arrangement" },
    { key: "pos", label: "the last word sits at position" },
    { key: "seen", label: "the rotation the model actually gets" },
  ]);
  const gridNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the whole method, as one equality" }),
      el("div", { class: "formula", text: "f′(x, m) = f(x, m · L ⁄ L′)" }),
      el("div", { class: "ctrls" }, [spreadSlider.node, piToggle]),
      gridRead.node,
      gridNote,
    ])
  );

  // ------------------------------------------------------- 2. what the near end pays
  const resRead = readout([
    { key: "rad", label: "turn between neighbours, fastest pair" },
    { key: "cos", label: "how nearly parallel that makes them" },
    { key: "turns", label: "pairs still completing a full turn in your sentence" },
    { key: "top", label: "the model's next word" },
  ]);
  const resBars = barList({
    rows: Array.from({ length: DH / 2 }, (_, j) => ({ key: "p" + j, label: `pair ${j + 1}` })),
  });
  const resNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "and what the original window pays for it" }),
      resBars.node,
      resRead.node,
      resNote,
    ])
  );

  // --------------------------------------------- 3. the bound, and the line that breaks it
  const s1Slider = slider({
    label: "between trained positions",
    min: 0,
    max: 14,
    value: 3,
    format: (v) => `${v} and ${v + 1}`,
    oninput: (v) => ((s1 = v), render()),
  });
  const corrToggle = toggle({
    label: "use the corrected constant",
    value: false,
    onchange: (v) => ((corrected = v), render()),
  });
  const boundCurve = curveView({
    xLabel: "position between two trained ones",
    yLabel: "how far the score strays from the straight line",
    ariaLabel: "the actual excursion of the attention score against the paper's bound on it",
  });
  const boundRead = readout([
    { key: "actual", label: "the score's actual excursion" },
    { key: "bound", label: "what the theorem allows" },
    { key: "slack", label: "allowance ÷ actual" },
  ]);
  const boundNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the theorem, and the line of the appendix that breaks it" }),
      el("div", { class: "ctrls" }, [s1Slider.node, corrToggle]),
      boundCurve.node,
      boundRead.node,
      boundNote,
    ])
  );

  // ------------------------------------------------------ 4. the 600x at your width
  const widthSelect = choice({
    label: "head width",
    value: "128",
    options: WIDTHS.map((w) => ({ value: String(w), label: `${w}${w === 8 ? " — this app" : w === 128 ? " — the paper's model" : ""}` })),
    onchange: (v) => ((width = Number(v)), render()),
  });
  const bCurve = curveView({
    xLabel: "distance between the two tokens",
    yLabel: "the quantity the 600× rests on",
    ariaLabel: "B(s) divided by the head width, against distance, with the floor the paper's constant assumes",
  });
  const bRead = readout([
    { key: "peak", label: "at distance zero" },
    { key: "min", label: "its lowest point" },
    { key: "below", label: "how often it falls under the floor" },
    { key: "ratio", label: "the resulting ratio" },
  ]);
  const bNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the 600×, recomputed at your head width" }),
      el("div", { class: "ctrls" }, [widthSelect]),
      bCurve.node,
      bRead.node,
      bNote,
    ])
  );

  // ------------------------------------------- 5. the figure that is not a measurement
  const fitSelect = choice({
    label: "what is being fitted",
    value: "8:16",
    options: FITS,
    onchange: (v) => ((fit = v), render()),
  });
  const fitCurve = curveView({
    xLabel: "distance between the two tokens",
    yLabel: "the score",
    ariaLabel: "a score curve inside and beyond the fitted range",
  });
  const fitRead = readout([
    { key: "in", label: "largest value inside the window" },
    { key: "out", label: "largest value beyond it" },
    { key: "ratio", label: "blow-up" },
    { key: "coef", label: "largest coefficient" },
  ]);
  const fitNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the picture everyone reproduces, and the control nobody runs" }),
      el("div", { class: "ctrls" }, [fitSelect]),
      fitCurve.node,
      fitRead.node,
      fitNote,
    ])
  );

  // ---------------------------------------------- 6. what the paper measured, quoted
  const paperBars = barList({
    rows: [
      { key: "base", label: "the model as shipped, at its own length" },
      { key: "pi0", label: "rescaled to 4×, no repair training", alt: true },
      { key: "pi0b", label: "rescaled to 8×, no repair training", alt: true },
      { key: "raw", label: "no rescaling, 4× the length", alt: true },
      { key: "pi200", label: "rescaled to 4×, 200 steps", alt: true },
      { key: "pi1000", label: "rescaled to 4×, 1000 steps", alt: true },
    ],
  });
  const paperNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what the paper measured — quoted, not reproduced" }),
      paperBars.node,
      paperNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "A model that already exists can be extended 4× to 16× without touching its architecture, its weights or its position scheme — the change is one division applied to the position index",
        "The repair is genuinely short: 1,000 steps, and the passkey retrieval measurement saturates a 16× extension after 200 — against a direct fine-tuning baseline that crawls from 2,048 to 2,560 in 10,000 steps",
        "Perplexity at the new lengths improves rather than merely surviving: 7.20 at 2,048 becomes 6.77 at 32,768 on one of the two corpora",
        "It works because the rotation accepts a real number, so half-integer positions exist at all — where a learned table simply has no row",
        "Nothing about serving changes: no new weights, no new kernels, no new cache layout",
      ],
      givesUp: [
        "Resolution at the near end, and the paper concedes it: the original window's positions are forced into a narrower arc, and the benchmark cost is real — 76.1 to 73.2 on one task at 4×, and 64.7 at 16×, against prose claiming up to 2%",
        "It is not training-free. Without the repair, extending 4× reads 16.10 perplexity against the baseline's 7.20, and 8× reads 112.13",
        "Every frequency is divided by the same factor, including the slow ones that never completed a turn anyway — the fast pairs that carry word order pay the same tax as the ones that had room to spare",
        "The theory is weaker than it is quoted as being: the 600× is a ratio between two bounds, one of which this same paper calls vacuous, and the interpolation bound's proof reverses an inequality",
        "The gains do not extend monotonically — on one corpus the 32,768 models are worse at 32,768 than at 16,384, which the text does not mention",
      ],
      chooseWhen:
        "When you own a rotation-based model and need a longer window this quarter rather than next year. It is the first entry on the branch that bends a finished model rather than training a different one — and the crack it leaves, that one factor is applied to every frequency alike, is what the next two concepts open.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "An existing model can be taught to read four to sixteen times more text, with a short burst of extra training rather than a new one",
        "The change is a single division applied to the position counter — no new parts, no new memory, nothing to re-engineer for serving",
        "On long documents it does not merely cope, it gets better: the model's score at the longest setting beats its own score at the length it was built for",
      ],
      cons: [
        "Squeezing the positions makes neighbouring words look more alike than they used to, and tasks that depend on exact word order lose real accuracy",
        "It still needs training afterwards — without it the model is badly degraded, just not destroyed",
        "The same squeeze is applied to every scale of position, including the ones that had plenty of room, which is a blunt instrument",
        "The mathematical argument it is famous for is much weaker than its reputation, and part of the proof is simply wrong",
      ],
      verdict:
        "The first idea on this timeline that treats a trained model as something you can bend rather than replace: divide every position by the amount you are stretching by, and the model is never asked about a position it has not seen. It works, cheaply, and the honest cost is at the other end — everything the model used to read at close range is now packed into a smaller arc, and the words it has to tell apart got harder to tell apart.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    spreadSlider.set(spread);
    s1Slider.input.max = Math.max(1, T - 2);
    s1Slider.set(Math.min(s1, T - 2));

    const base = atPositions(tokens, (i) => i);
    const baseA = base.trace[0].heads[0].weights;

    // --- 1. the sentence, spread out, with and without the rescaling
    const posOf = (i) => (usePI ? i : i * spread);
    const res = atPositions(tokens, posOf);
    const moved = maxDiff(res.trace[0].heads[0].weights, baseA);
    const h0 = res.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...h0, emb: res.trace[0].input },
      weights: h0.weights,
      out: h0.out,
      top: res.top,
      query: T - 1,
      opts: {
        rotate: {
          angle: (i) => posOf(i) * FREQS[0],
          label: usePI ? "rescaled back onto the trained range" : "rotated at its real position",
        },
      },
    });
    flowNote.innerHTML = usePI
      ? `Every word of your sentence now sits ${spread} tokens apart inside a longer document — the last one at position ${
          15 * spread
        } — and the dials are turned by the <em>rescaled</em> position. Because the rescaling divides by exactly the spacing, each dial lands on the angle it had when the words were adjacent, and the attention matrix is the trained one to the last bit. Switch the rescaling off and watch the dials spin away.`
      : `The same sentence at the same real positions, rotated as they actually are. The dials have spun far past anything the model saw in training, and the attention pattern has changed by <strong>${fmt(
          moved,
          4
        )}</strong> — on weights that are probabilities, so that is close to the largest change arithmetically available. This is what “the model cannot read past its window” looks like inside the mechanism.`;

    gridRead.update({
      delta: usePI ? moved.toExponential(2) : fmt(moved, 4),
      pos: String(15 * spread),
      seen: usePI ? `as if position ${15}` : `position ${15 * spread}`,
    });
    gridNote.className = "note " + (usePI ? "good" : "warn");
    gridNote.innerHTML = usePI
      ? `<strong>${moved.toExponential(
          2
        )}.</strong> Not small — zero. Divide the position index by the extension factor and the rotation is bit-for-bit the one the model was trained on, whatever the spacing; drag the slider to 128 and it stays zero. That equality is the entire method, and it is available only because the rotation is a function of a real number rather than a lookup into a table of trained rows. The paper's own sentence: <em>“we reduce position indices from [0, L′) to [0, L) to match the original range of indices before computing RoPE.”</em>`
      : `Without the rescaling the same sentence at the same places moves the attention by ${fmt(
          moved,
          4
        )}. The paper measures this at scale as perplexity going from 7.20 to above 1,000 — its own words for that are <em>“comparable to untrained models”</em>. Turn the rescaling back on.`;

    // --- 2. the near end
    // The rescaling a real deployment applies is 1/g on *every* input, including short ones, so
    // measure it where it hurts: the sentence at ordinary positions, rotated as if compressed.
    const near = atPositions(tokens, (i) => i / spread);
    const nearMoved = maxDiff(near.trace[0].heads[0].weights, baseA);
    const turnsAt = (g) => FREQS.filter((f) => ((T - 1) * f) / g >= 2 * Math.PI).length;
    const bars = {};
    FREQS.forEach((f, j) => {
      const wl = (2 * Math.PI) / (f / spread);
      bars["p" + j] = {
        value: Math.min(1, (T - 1) / wl),
        of: 1,
        text: `${wl < 1e4 ? fmt(wl, 1) : wl.toExponential(1)} tokens`,
        label: `pair ${j + 1} — a full turn takes`,
        tone: (T - 1) / wl >= 1 ? "" : "alt",
      };
    });
    resBars.update(bars);
    resRead.update({
      rad: `${fmt(FREQS[0] / spread, 4)} rad`,
      cos: fmt(Math.cos(FREQS[0] / spread), 4),
      turns: `${turnsAt(spread)} of ${FREQS.length}`,
      top: `“${near.top[0].word}” at ${fmt(near.top[0].p * 100, 1)}%`,
    });
    resNote.className = "note " + (turnsAt(spread) === 0 ? "warn" : "");
    resNote.innerHTML = `An extended model applies the same division to <em>every</em> input, so a short prompt is squeezed too. At ${spread}× the fastest dimension pair now turns <strong>${fmt(
      FREQS[0] / spread,
      4
    )}</strong> radians between neighbouring words instead of 1.0000, which makes two adjacent tokens ${fmt(
      Math.cos(FREQS[0] / spread),
      4
    )} aligned on that pair instead of ${fmt(
      Math.cos(FREQS[0]),
      4
    )} — nearly parallel where they used to be nearly at right angles. ${
      turnsAt(spread) === 0
        ? "<strong>No pair completes a full turn inside your sentence any more.</strong>"
        : `${turnsAt(spread)} pair still completes a full turn inside your sentence.`
    } The attention on the ordinary 0…15 positions moves by ${fmt(
      nearMoved,
      4
    )} and the model's next word changes. That is the geometry behind the paper's own concession — <em>“Position Interpolation forces position encodings in original context window to reside in a much narrower region, which may negatively affect the language model's performance”</em> — and behind its benchmark column, which loses 76.1 → 73.2 → 69.8 → 64.7 on one task as the extension goes 1× → 4× → 8× → 16×. Those numbers are measured after repair training, on a model two hundred million times this one's size; this page supplies the mechanism, not the consequence.`;

    // --- 3. the bound
    const hb = coefficients(base.trace[0].heads[0].Q[T - 1], base.trace[0].heads[0].K[0]);
    const mh = Math.max(...hb.map((c) => c.mag));
    const s2 = s1 + 1;
    const aL = (s) => {
      const lam = (s - s1) / (s2 - s1);
      return (1 - lam) * scoreAt(hb, s1) + lam * scoreAt(hb, s2);
    };
    const coefPaper = DH / (8 * Math.log(C));
    const coefCorrected = 1 / (2 * (1 - Math.pow(C, -4 / DH)));
    const coef = corrected ? coefCorrected : coefPaper;
    const bound = (s) => mh * coef * (s - s1) * (s2 - s);
    const pts = [];
    const ref = [];
    let worst = 0;
    for (let k = 0; k <= 100; k++) {
      const s = s1 + k / 100;
      const dev = Math.abs(scoreAt(hb, s) - aL(s));
      worst = Math.max(worst, dev);
      pts.push([s, dev]);
      ref.push([s, bound(s)]);
    }
    const allow = (mh * coef) / 4;
    boundCurve.update({
      points: pts,
      reference: ref,
      xRange: [s1, s2],
      yRange: [0, Math.max(worst, allow) * 1.1],
      mark: s1 + 0.5,
      markLabel: "the worst case the theorem bounds",
    });
    boundRead.update({
      actual: fmt(worst, 4),
      bound: fmt(allow, 4),
      slack: `${fmt(allow / (worst || 1e-9), 2)}×`,
    });
    boundNote.className = "note " + (allow < worst ? "warn" : "good");
    boundNote.innerHTML = corrected
      ? `With the corrected coefficient the allowance is ${fmt(
          allow,
          4
        )} against an actual excursion of ${fmt(
          worst,
          4
        )}, and the solid curve sits under the dashed one where it belongs. The correction is not a matter of taste: the appendix's step replaces an exact geometric sum by <em>d ⁄ (4 ln c)</em>, justified by the claim that <em>c<sup>x</sup> ≤ 1 + x ln c</em> — which is the tangent-line inequality for a convex function written backwards, and false for every x. The exact coefficient is 1 ⁄ (8(1 − c<sup>−4/d</sup>)), which is 1.15× larger at the paper's head width and <strong>4.65× larger at this one</strong>. At the paper's scale the mistake costs its conclusion nothing; here it is the difference between a bound that holds and one that does not.`
      : `The solid curve is how far the real score strays from the straight line between two trained positions; the dashed one is the theorem's allowance for that. <strong>The curve leaves the band.</strong> ${fmt(
          worst,
          4
        )} against an allowed ${fmt(
          allow,
          4
        )} — the published bound is violated on this model's own vectors, and not by rounding. The cause is one line of the appendix, and the toggle above fixes it. Note what is <em>not</em> being claimed: the theorem's shape is right, and at the paper's own head width the same error costs only 15%. It is the quotable constant that is not established by the proof, not the conclusion.`;

    // --- 4. the 600x
    const step = 8;
    const bPts = [];
    let bMin = Infinity;
    let below = 0;
    let n = 0;
    for (let s = 0; s <= 4096; s += step) {
      const v = bOverD(width, s);
      bPts.push([s, v]);
      if (s > 0) {
        bMin = Math.min(bMin, v);
        if (v < 1) below++;
        n++;
      }
    }
    const peak = (width + 2) / 8;
    bCurve.update({
      points: bPts,
      reference: bPts.map(([s]) => [s, 1]),
      xRange: [0, 4096],
      yRange: [0, peak * 1.05],
      mark: null,
    });
    const ratioLo = 2 * 32 * Math.log(C) * bMin;
    const ratioHi = 2 * 32 * Math.log(C) * peak;
    bRead.update({
      peak: fmt(peak, 3),
      min: fmt(bMin, 4),
      below: `${((below / n) * 100).toFixed(1)}% of distances`,
      ratio: `${Math.round(ratioLo)}× … ${Math.round(ratioHi)}×`,
    });
    bNote.className = "note " + (below / n > 0.2 ? "warn" : "");
    bNote.innerHTML = `The headline number is two bounds divided by each other, and this curve is the floor the division assumes. At the paper's head width the assumption holds: the quantity never drops below 1 across the whole range, the peak is exactly ${fmt(
      (128 + 2) / 8,
      2
    )}, and the ratio comes out between 658× and 9,579× — so <em>“at least ∼600×”</em> is fair at the width it was checked. Move the selector to this app's width and the floor is gone: below 1 for ${((below / n) * 100).toFixed(
      0
    )}% of distances at width ${width}. The paper checks exactly one width, and the caption of its own plot concedes that the quantity <em>“decays with s”</em>. Two further things worth knowing before quoting the number: the two bounds measure different quantities — one is how far an interpolated score strays from a chord, the other is how large a score can be — and the extrapolation side is the very bound this paper had just called <em>“too loose … and thus vacuous”</em> two pages earlier. Beating a bound you have declared vacuous is a statement about bounds.`;

    // --- 5. figure two
    if (fit === "real") {
      const maxS = 2048;
      const env = hb.reduce((a, c) => a + c.mag, 0);
      const rp = [];
      let inW = 0;
      let outW = 0;
      for (let s = 0; s <= maxS; s += 4) {
        const v = scoreAt(hb, s);
        rp.push([s, v]);
        if (s < T) inW = Math.max(inW, Math.abs(v));
        else outW = Math.max(outW, Math.abs(v));
      }
      fitCurve.update({
        points: rp,
        reference: rp.map(([s]) => [s, env]),
        xRange: [0, maxS],
        yRange: [-env * 1.05, env * 1.05],
        mark: T,
        markLabel: "your sentence ends here",
      });
      fitRead.update({
        in: fmt(inW, 3),
        out: fmt(outW, 3),
        ratio: `${fmt(outW / (inW || 1e-9), 2)}×`,
        coef: fmt(env, 2),
      });
      fitNote.className = "note good";
      fitNote.innerHTML = `The same curve computed from this model's real query and key rather than from fitted noise. It ripples, it does not explode, and it cannot: the sum of the coefficient magnitudes is a hard ceiling on the score at <em>every</em> distance, real or integer, inside the window or far outside it — that is the dashed line, and the curve never touches it. Across every head, block and pair in this model the worst out-of-window blow-up is about nine times, not sixteen thousand. The catastrophic-extrapolation <em>fact</em> is well established — the paper measures perplexity above 1,000 on a real model — but the picture usually offered as the explanation is not a measurement of any model.`;
    } else {
      const [fd, fl] = fit.split(":").map(Number);
      const f = figureTwo(fd, fl);
      const cap = Math.min(f.outside, Math.max(f.inside * 40, f.outside));
      fitCurve.update({
        points: f.pts,
        reference: null,
        xRange: [0, 2 * fl],
        yRange: [-cap * 1.05, cap * 1.05],
        mark: fl,
        markLabel: "the fitted range ends here",
      });
      fitRead.update({
        in: fmt(f.inside, 3),
        out: f.outside > 1e4 ? f.outside.toExponential(2) : fmt(f.outside, 2),
        ratio: `${f.outside / f.inside > 1e4 ? (f.outside / f.inside).toExponential(1) : fmt(f.outside / f.inside, 1)}×`,
        coef: f.coef > 1e3 ? f.coef.toExponential(2) : fmt(f.coef, 2),
      });
      const blowUp = f.outside / f.inside;
      fitNote.className = "note " + (blowUp > 10 ? "warn" : "good");
      fitNote.innerHTML =
        fit === "8:2048"
          ? `<strong>This is the control the paper does not run, and it is the whole point.</strong> Same recipe, same code, same basis — only now the basis has 8 functions to fit 2,048 random targets instead of 128. The blow-up is ${fmt(
              blowUp,
              1
            )}×, the coefficients are of order ${fmt(
              f.coef,
              2
            )}, and there is no catastrophe at all. So what the picture measures is what happens when a rich basis is fitted to noise over a short range: the coefficients grow enormous, the curve cancels itself to stay small inside the fitted region, and the cancellation stops the moment you step outside. The paper half-admits this in a sentence it does not follow up — that regularising the fit makes the extrapolated values comparable to the ones inside.`
          : `The paper's own recipe, run here: draw random targets, fit them by least squares with the rotation's own sine and cosine basis over the training range, then keep evaluating past the end. Inside, the curve stays around ${fmt(
              f.inside,
              2
            )}; outside it reaches ${
              f.outside > 1e4 ? f.outside.toExponential(2) : fmt(f.outside, 1)
            }, a blow-up of ${
              blowUp > 1e4 ? blowUp.toExponential(1) : fmt(blowUp, 1)
            }× with fitted coefficients up to ${
              f.coef > 1e3 ? f.coef.toExponential(2) : fmt(f.coef, 1)
            }. It reproduces, and it is genuinely alarming. Now change the selector to the third option — same code, one number different — before deciding what it proves.`;
    }

    // --- 6. the paper's own column
    const P = { base: 7.2, pi0: 16.1, pi0b: 112.13, raw: 1000, pi200: 7.12, pi1000: 6.95 };
    const of = 120;
    paperBars.update({
      base: { value: P.base, of, text: "7.20" },
      pi0: { value: P.pi0, of, text: "16.10" },
      pi0b: { value: P.pi0b, of, text: "112.13" },
      raw: { value: of, of, text: "above 1,000" },
      pi200: { value: P.pi200, of, text: "7.12" },
      pi1000: { value: P.pi1000, of, text: "6.95" },
    });
    paperNote.className = "note";
    paperNote.innerHTML = `<strong>Quoted from the paper, not computed here</strong> — a 7-billion-parameter model on a book corpus, lower is better. Read the middle three bars together: rescaling with no repair at all is <em>degraded</em>, at 16.10 against the model's own 7.20; not rescaling at the same length is <em>destroyed</em>, above 1,000, which the paper describes as <em>“comparable to untrained models”</em>. The gap between those two bars is the argument of the entire method, and the last two show what 200 and 1,000 steps of repair buy on top. The other measurement worth carrying: on a retrieval test, this method reaches the full 8,192 after <strong>200</strong> steps, while ordinary fine-tuning at the same length crawls from 2,048 to 2,560 in <strong>10,000</strong>. Two cautions the paper leaves to the reader — that retrieval test counts a length as reached at a 20% success rate, and one of its rows dips to 18,432 in the middle of an otherwise perfect run.`;
  }

  return { update: render, unmount: () => {} };
}
