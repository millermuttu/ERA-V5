// Concept 13 — ALiBi, attention with linear biases.
// Built from docs/research/alibi.md. The research settled the thing everyone gets wrong: the
// paper defines extrapolation as the perplexity curve not blowing up, and its own Appendix B.2
// says the model "might not be using contexts longer than the ones it was trained on". Under a
// stride-1 sliding-window evaluation the apparent gain disappears entirely. This card says so.
import { el, slider, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt } from "../model/ops.js";
import { forward, CONFIG } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { alibi, alibiSlopes } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const SLOPES = alibiSlopes(CONFIG.HEADS);

export function alibiCard(root, m) {
  let head = 0;
  let showAll = true;
  let query = null;

  root.appendChild(
    prose({
      problem:
        "RoPE can be evaluated at any position, but nobody had shown it works there. This paper measured it, and the answer was bad: a model trained at 1,024 tokens scored 19.34 perplexity at its training length and 106.99 at 16,024 with rotary positions, and 453.32 with sinusoidal ones. Position schemes were being chosen for an extrapolation nobody had tested — including, as concept 2 records, the original Transformer's.",
      mechanism:
        "Do not encode position at all. Subtract a penalty proportional to the distance between the two tokens straight from the score, with a different fixed slope per head, so some heads look only nearby and others look far. Nothing is learned, nothing is added to the embedding, and the term is not divided by the square root of the head width. The slopes are a geometric sequence chosen by hand: it starts at 2 to the power of minus eight over the head count, steps by that same ratio, and therefore always ends at one over 256.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------------- the four slopes
  const allToggle = toggle({
    label: "show every head",
    value: true,
    onchange: (v) => ((showAll = v), render()),
  });
  const headSlider = slider({
    label: "head",
    min: 0,
    max: CONFIG.HEADS - 1,
    value: 0,
    format: (v) => `${v + 1} — slope ${SLOPES[v]}`,
    oninput: (v) => ((head = v), render()),
  });
  const slopeCurve = curveView({
    xLabel: "distance back, in tokens",
    yLabel: "penalty subtracted from the score",
    ariaLabel: "the distance penalty each head applies, one curve per head",
  });
  const spanBars = barList({
    rows: Array.from({ length: CONFIG.HEADS }, (_, h) => ({ key: "h" + h, label: `head ${h + 1}` })),
  });
  const slopeRead = readout([
    { key: "slopes", label: "the slopes, by the paper's rule" },
    { key: "near", label: "weight this head puts on the nearest four words" },
    { key: "ratio", label: "nearest word ÷ furthest word" },
  ]);
  const slopeNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "four heads, four different reaches" }),
      el("div", { class: "ctrls" }, [headSlider.node, allToggle]),
      slopeCurve.node,
      spanBars.node,
      slopeRead.node,
      slopeNote,
    ])
  );

  // ------------------------------------------- what a monotone penalty costs
  const costRead = readout([
    { key: "need", label: "content advantage a distant word needs, to tie" },
    { key: "dist", label: "at this distance" },
    { key: "vs", label: "in the narrowest head" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the price of a penalty that only ever grows" }),
      costRead.node,
      costNote,
    ])
  );

  // --------------------------------------------- what extrapolation means
  const extrapRead = readout([
    { key: "definition", label: "what the paper means by extrapolating" },
    { key: "sliding", label: "under a stride-1 sliding window, 512 → 3072" },
    { key: "same", label: "at the training length, against sinusoidal" },
  ]);
  const extrapNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what “trains short, tests long” actually claims" }),
      extrapRead.node,
      extrapNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Perplexity stops exploding past the training length: a model trained at 1,024 and evaluated at 16,024 stays near its training score, where sinusoidal goes from 19.34 to 453.32 and rotary to 106.99",
        "No parameters at all, no embedding change, and within about 1% of training speed and 3% of inference speed",
        "One subtraction before softmax, and the per-head slopes give a spread of receptive fields for free",
        "Because nothing is stored per position, there is no table to run off the end of and no rotation to recalibrate",
      ],
      givesUp: [
        "The penalty is monotone, so a genuinely important word far back is fought by the mechanism itself — in the narrowest head it must win on content by nearly four nats simply to tie with its neighbour",
        "The paper's own appendix says the model may not be using the longer context: under a stride-1 sliding-window evaluation the apparent gain vanishes and the curve is flat",
        "At the training length itself it is slightly behind sinusoidal — 9.79 against 9.71 at one scale, and the same ordering at two others",
        "The slopes came from a brief manual exploration of about ten sets, and making them trainable did not extrapolate well",
        "No notion of position beyond distance: no periodicity, no structure, nothing a head can key on other than how far away something is",
      ],
      chooseWhen:
        "When you need the perplexity curve to stay flat past the training length and you can accept a recency bias as the model's view of the world. Frontier models mostly went the other way — keep RoPE and recalibrate it — which is the branch the next three concepts follow.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Text longer than anything the model was trained on stops breaking it",
        "It costs nothing: no extra numbers to learn, no extra memory, and no measurable slowdown",
        "Different parts of the model naturally end up looking at different distances, from very close by to far away",
      ],
      cons: [
        "The further back a word is, the more the mechanism pushes against reading it — so something important said long ago has to fight to be noticed",
        "The authors' own tests suggest it is not really making use of the extra text, it is just not falling over",
        "On text of the length it was trained for, it is very slightly worse than what it replaced",
        "It knows only how far away something is, and nothing else about where it sits",
      ],
      verdict:
        "The cheapest possible answer to the length problem: assume nearby matters more, and let that assumption grow with distance. It keeps the model standing on text far longer than it was trained on — but standing is not the same as reading, and the paper is unusually honest that it may only be the first.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    headSlider.set(head);

    const scheme = alibi({ heads: CONFIG.HEADS });
    const res = forward(tokens, { mixer: softmaxMixer({ bias: scheme.bias }) });
    const h = res.trace[0].heads[head];

    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query,
      opts: {},
    });
    flowNote.innerHTML = `Nothing has been added to the embeddings and nothing rotated — the left of the picture is exactly the baseline's. The only change is a number subtracted from each score as the distance grows, and you can see it in the matrix: weight leans towards the diagonal. This is head ${head + 1}, whose slope is ${SLOPES[head]}; step through the heads and the lean changes with them.`;

    // --- the four curves
    const maxD = Math.max(24, T);
    const curveFor = (hh) => Array.from({ length: maxD + 1 }, (_, d) => [d, -d * SLOPES[hh]]);
    slopeCurve.update({
      points: curveFor(head),
      reference: showAll ? curveFor(CONFIG.HEADS - 1) : null,
      xRange: [0, maxD],
      yRange: [-maxD * SLOPES[0], 0],
      mark: T - 1,
      markLabel: `your sentence ends here`,
    });

    // attention mass on the nearest four, per head, from the real run
    const bars = {};
    for (let hh = 0; hh < CONFIG.HEADS; hh++) {
      const w = res.trace[0].heads[hh].weights[T - 1];
      const near = w.slice(Math.max(0, T - 4)).reduce((a, b) => a + b, 0);
      bars["h" + hh] = {
        value: near,
        of: 1,
        text: `${(near * 100).toFixed(1)}%`,
        label: `head ${hh + 1} · slope ${SLOPES[hh]}`,
        tone: hh === head ? "" : "alt",
      };
    }
    spanBars.update(bars);

    const wq = res.trace[0].heads[head].weights[T - 1];
    const nearest = wq[T - 1];
    const furthest = wq[0] || 1e-12;
    slopeRead.update({
      slopes: SLOPES.map((s) => (s >= 0.01 ? s : s.toExponential(1))).join("  "),
      near: `${(wq.slice(Math.max(0, T - 4)).reduce((a, b) => a + b, 0) * 100).toFixed(1)}%`,
      ratio: `${fmt(nearest / furthest, 1)}×`,
    });
    slopeNote.innerHTML = `The slopes are not tuned per model: the rule is 2^(−8/heads) as both the first value and the ratio, so with ${CONFIG.HEADS} heads they are ${SLOPES.join(", ")} and the last head always lands on 1/256 whatever the head count. Head 1 is a near-sighted head — it puts ${(res.trace[0].heads[0].weights[T - 1].slice(Math.max(0, T - 4)).reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of its weight on the last four words alone, against a uniform share of ${((4 / T) * 100).toFixed(1)}%. Head ${CONFIG.HEADS} barely leans at all. That spread <em>is</em> the mechanism: a fixed set of receptive fields, imposed rather than learned. The paper tried learning the slopes and reports it did not extrapolate well, and the set it uses came from a brief manual exploration of about ten candidates.`;

    // --- the price of monotonicity
    const d = T - 1;
    costRead.update({
      need: `${fmt(d * SLOPES[0], 2)} nats`,
      dist: `${d} words back`,
      vs: `${fmt(d * SLOPES[CONFIG.HEADS - 1], 3)} nats in head ${CONFIG.HEADS}`,
    });
    costNote.textContent = `In the narrowest head, a word ${d} back starts ${fmt(d * SLOPES[0], 2)} nats behind its neighbour before any content is considered — it has to be that much more relevant merely to tie. That is the cost of a penalty that only ever grows: the mechanism cannot be told that this particular distant word matters, because it does not look at the words at all. Concept 3's relative term could express that, because it multiplied against the query; this deliberately cannot, which is what makes it free.`;

    // --- what extrapolation means
    extrapRead.update({
      definition: "the perplexity curve does not blow up",
      sliding: "17.98 → 18.30, flat",
      same: "9.79 against 9.71 — slightly behind",
    });
    extrapNote.className = "note warn";
    extrapNote.innerHTML = `This is the part worth being careful about. The paper defines extrapolation as continuing to perform well as the input grows past the training length — a statement about the perplexity curve, not about comprehension. Its own Appendix B.2 says the model <em>“might not be using contexts longer than the ones it was trained on”</em>, and explains the apparent gain as fewer tokens suffering the early-token curse. Under a stride-1 sliding-window evaluation the same model reads 17.98 at 512 tokens and 18.30 at 3,072 — flat, drifting slightly worse — while the headline protocol made it look like an improvement from 19.73 to 18.40. The gain was the measurement, not the model.`;
  }

  return { update: render, unmount: () => {} };
}
