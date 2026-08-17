// Concept 15 — grouped-query attention.
// Built from docs/research/gqa.md. Three things the research settled. The paper has two separable
// contributions and most retellings collapse them: a recipe for converting a finished multi-head
// checkpoint, and the grouped middle of the axis. The famous "eight groups because a serving node
// has eight accelerators" argument is not in this paper at all — it is Llama 2's, two months later,
// and the card attributes it there. And the app's own serving figure, the 6.44 GB quoted since
// concept 7, is already a grouped configuration; the multi-head version of that model is 51.5 GB.
import { el, slider, choice, svg } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { cacheBytes, SERVING, GB } from "../model/cost.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const cos = (a, b) => {
  const d = dot(a, b);
  return d / (Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b)) || 1e-12);
};

/** Mean over every head, block and token of a per-vector comparison against the multi-head run. */
function fidelity(res, base, pick) {
  let c = 0;
  let rel = 0;
  let n = 0;
  for (let b = 0; b < CONFIG.BLOCKS; b++) {
    for (let h = 0; h < CONFIG.HEADS; h++) {
      const now = pick(res.trace[b].heads[h]);
      const was = pick(base.trace[b].heads[h]);
      for (let i = 0; i < now.length; i++) {
        c += cos(now[i], was[i]);
        let num = 0;
        let den = 0;
        for (let d = 0; d < DH; d++) {
          num += (now[i][d] - was[i][d]) ** 2;
          den += was[i][d] ** 2;
        }
        rel += Math.sqrt(num) / Math.sqrt(den || 1e-12);
        n++;
      }
    }
  }
  return { cos: c / n, rel: rel / n };
}

const tv = (p, q) => {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += Math.abs(p[i] - q[i]);
  return s / 2;
};

const POOLS = [
  { value: "mean", label: "mean-pool the group" },
  { value: "select", label: "keep the first head" },
  { value: "random", label: "start the head fresh" },
];
const KVH = [64, 32, 16, 8, 4, 2, 1];

export function gqaCard(root, m) {
  let groups = 2;
  let pool = "mean";
  let kvHeads = 8;

  root.appendChild(
    prose({
      problem:
        "Multi-query attention cut the cache by the number of heads, and the measurement came with a quality cost and, this paper reports, training runs that spiked and diverged. Nobody had asked what sits between one shared key/value head and one per query head. There is a second problem, and it is not an architecture problem at all: the models that needed the cheaper cache already existed. A finished checkpoint cost millions to train, and nobody wants to pay for it twice to get a faster version of the same model.",
      mechanism:
        "Two levers. First, divide the query heads into groups and give each group one key/value head — so the two familiar designs become the ends of a single axis, one group being multi-query and one group per head being multi-head, and everything interesting sits in between. Second, build that shared head out of the ones you already trained by averaging their projection matrices, then repair the model with a small fraction of the original pre-training. The number of query heads never changes; only the count of distinct keys and values does, and the mixing arithmetic is untouched.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "the same journey, with some heads sharing what they read");

  // ------------------------------------------------------- the conversion bench
  const gSelect = choice({
    label: "groups",
    value: "2",
    options: [
      { value: String(CONFIG.HEADS), label: `${CONFIG.HEADS} — one per head, multi-head` },
      { value: "2", label: "2 — the grouped middle" },
      { value: "1", label: "1 — one for all, multi-query" },
    ],
    onchange: (v) => ((groups = Number(v)), render()),
  });
  const poolSelect = choice({
    label: "how the shared head is built",
    value: "mean",
    options: POOLS,
    onchange: (v) => ((pool = v), render()),
  });
  const convRead = readout([
    { key: "kcos", label: "how much of the original key survives" },
    { key: "krel", label: "how far it moved" },
    { key: "vcos", label: "same, for the value" },
    { key: "tv", label: "how far the prediction moved" },
  ]);
  const convBars = barList({
    rows: POOLS.map((p) => ({ key: p.value, label: p.label })),
  });
  const convNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "three ways to build the shared head, measured" }),
      el("div", { class: "ctrls" }, [gSelect, poolSelect]),
      convRead.node,
      convBars.node,
      convNote,
    ])
  );

  // -------------------------------------------------------- the grouping picture
  const matWrap = el("div", { class: "gridwrap" });
  const matNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what “groups” looks like — the heads compared with each other" }),
      matWrap,
      matNote,
    ])
  );

  // ------------------------------------------------- how much the pooling costs
  const predRead = readout([
    { key: "before", label: "how alike the heads in a group were" },
    { key: "after", label: "how much of each survives the averaging" },
    { key: "ideal", label: "what perfectly unrelated heads would give" },
  ]);
  const predNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the cost of averaging depends on what you are averaging" }),
      predRead.node,
      predNote,
    ])
  );

  // ------------------------------------------------------------- the slope
  const kvSlider = slider({
    label: "key/value heads at serving scale",
    min: 0,
    max: KVH.length - 1,
    value: KVH.indexOf(8),
    format: (v) => `${KVH[v]}${KVH[v] === 64 ? " — multi-head" : KVH[v] === 8 ? " — the paper's choice" : KVH[v] === 1 ? " — multi-query" : ""}`,
    oninput: (v) => ((kvHeads = KVH[v]), render()),
  });
  const slopeCurve = curveView({
    xLabel: "conversation length, in tokens",
    yLabel: "cache for one conversation, in GB",
    ariaLabel: "key/value cache against context length, at the chosen number of key/value heads and at multi-head",
  });
  const slopeRead = readout([
    { key: "here", label: "at 32,768 tokens" },
    { key: "vs", label: "against multi-head and multi-query" },
    { key: "same", label: "multi-head hits that same figure at" },
  ]);
  const slopeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a slope, not a wall" }),
      el("div", { class: "ctrls" }, [kvSlider.node]),
      slopeCurve.node,
      slopeRead.node,
      slopeNote,
    ])
  );

  // ------------------------------------------------------------ the retrofit
  const upRead = readout([
    { key: "alpha", label: "extra pre-training the paper used" },
    { key: "cost", label: "what that cost for the largest model" },
    { key: "knee", label: "where the returns stop" },
  ]);
  const upNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the part that is not an architecture at all" }),
      upRead.node,
      upNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The interior of the axis: on the paper's largest model, grouped-into-eight scores 47.1 average against multi-head's 47.2 and multi-query's 46.6 — it recovers 0.5 of the 0.6-point gap",
        "And it costs almost nothing in speed: 0.28 seconds per sample against multi-query's 0.24 and multi-head's 1.51, so 5.4× faster than multi-head for a tenth of a point",
        "A finished multi-head checkpoint can be converted rather than retrained — average the key/value projections within each group and pre-train for 5% of the original steps",
        "Converted-and-grouped is reported as already reasonable before any repair at all, where the fully collapsed version needs the repair to be usable",
        "The shared head survives partitioning: with one key/value head and eight serving chips the single head is replicated on all of them, and the saving comes back eightfold",
      ],
      givesUp: [
        "It is an interpolation, and an interpolation cannot leave the segment — the cache is still linear in context and in batch, just with a smaller constant",
        "The quality claim rests on one model at one size on seven dev sets, and the paper's own limitations section says the metric it used does not tell the whole story",
        "No comparison with training a grouped model from scratch exists — the paper says so outright, so “as good as multi-head” is a claim about a converted model only",
        "The choice of eight groups is read off an inference-speed curve and justified as a favourable middle ground; the systems argument everyone repeats for it is from a different paper",
        "The instability it cites is observed and not measured: frequent loss spikes, immediate divergence, no counts, and the authors state they stopped investigating",
        "It is an encoder-decoder result, and the grouped conversion is not applied to encoder self-attention at all",
      ],
      chooseWhen:
        "Whenever generation memory is the constraint and quality still matters — which is why it became the default. Pick the number of groups from how you serve the model rather than from the model itself: enough that no group has to be duplicated across chips.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It recovers most of the quality lost by having every head share one summary, while keeping nearly all of the speed",
        "It can be applied to a model that has already been trained, by averaging parts of it together and then briefly continuing training — a small fraction of what the original cost",
        "The number of groups is a dial, so it can be set to match the machines the model will actually run on",
      ],
      cons: [
        "The memory still grows with the length of the conversation and the number of users — it grows more slowly, and that is all",
        "The evidence is one model at one size, judged by a scoring method the authors themselves call flawed",
        "Nobody has checked whether a model built this way from the start would be better, and the paper says so",
        "The famous reason for choosing eight groups comes from a different paper published two months later",
      ],
      verdict:
        "The previous concept asked every head to share one set of notes and paid for it. This one notices that the choice was never binary: hand out one set of notes per group of heads, and pick the group count to suit the machines you serve on. The second half is the part that spread fastest — you do not have to train a new model to get this, you can average the parts of the one you already own and briefly continue training. What it does not do is change the shape of the problem: the memory still grows with every word, only more slowly.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;

    const base = forward(tokens);
    const res = forward(tokens, groups === CONFIG.HEADS ? {} : { kvGroups: groups, kvPool: pool });
    const perGroup = CONFIG.HEADS / groups;
    const h = res.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query: T - 1,
      opts: { kvShared: groups < CONFIG.HEADS },
    });
    flowNote.innerHTML =
      groups === CONFIG.HEADS
        ? `Every head has its own key and value — this is concept 1's picture exactly, and it is one end of the axis this card is about. Change the group count above and watch the K and V bands collapse into shared copies while the Q band stays untouched: the number of query heads never changes, only how many distinct things they read from.`
        : `${CONFIG.HEADS} query heads, ${groups} key/value head${groups > 1 ? "s" : ""} — ${perGroup} query heads to a group. The Q band is untouched and every head still computes its own attention row; what collapsed is the K and V bands, which now hold ${groups} distinct vector${
            groups > 1 ? "s" : ""
          } per token instead of ${CONFIG.HEADS}. That is the whole mechanism, and the mixing arithmetic does not change by a single multiply — only what has to be kept.`;

    // --- the conversion bench
    const kf = fidelity(res, base, (x) => x.K);
    const vf = fidelity(res, base, (x) => x.V);
    convRead.update({
      kcos: groups === CONFIG.HEADS ? "1.0000 — nothing shared yet" : fmt(kf.cos, 4),
      krel: groups === CONFIG.HEADS ? "0.0000" : fmt(kf.rel, 4),
      vcos: groups === CONFIG.HEADS ? "1.0000" : fmt(vf.cos, 4),
      tv: groups === CONFIG.HEADS ? "0.0000 — same model" : fmt(tv(res.probs, base.probs), 4),
    });

    const bars = {};
    for (const p of POOLS) {
      const r = groups === CONFIG.HEADS ? base : forward(tokens, { kvGroups: groups, kvPool: p.value });
      const f = groups === CONFIG.HEADS ? { cos: 1 } : fidelity(r, base, (x) => x.K);
      bars[p.value] = {
        value: Math.max(0, f.cos),
        of: 1,
        text: fmt(f.cos, 4),
        label: p.label,
        tone: p.value === pool ? "" : "alt",
      };
    }
    convBars.update(bars);
    convNote.className = "note";
    convNote.innerHTML =
      groups === CONFIG.HEADS
        ? `With one key/value head per query head there is nothing to build: every head keeps what it had. Drop the group count to see the three conversions separate.`
        : `The bars are how much of each head's original key direction survives the conversion, and the ordering is the paper's: averaging the group beats keeping one of them, which beats starting fresh. Be careful about what that proves here. The average is by definition the point closest to the vectors it replaces, so “averaging stays nearest” is arithmetic rather than a result — what this page can add is the <em>size</em> of the loss, and how it grows as the groups get larger. The paper's finding is the step this page cannot take: that staying nearest also <em>finishes</em> best after the repair training. Its own comparison was made after that repair, on a real trained model, and the margins live in a figure whose numbers are not printed anywhere in the text.`;

    // --- the grouping picture
    const K = res.trace[0].heads.map((x) => x.K);
    const cell = 62;
    const cells = [];
    for (let a = 0; a < CONFIG.HEADS; a++) {
      for (let b = 0; b < CONFIG.HEADS; b++) {
        let c = 0;
        for (let i = 0; i < T; i++) c += cos(K[a][i], K[b][i]);
        c /= T;
        cells.push(
          svg("rect", {
            x: 30 + b * cell,
            y: 22 + a * cell,
            width: cell - 3,
            height: cell - 3,
            rx: 3,
            fill: c >= 0 ? "#4FC58C" : "#E0693D",
            "fill-opacity": (0.08 + Math.abs(c) * 0.8).toFixed(3),
          }),
          svg("text", {
            x: 30 + b * cell + (cell - 3) / 2,
            y: 22 + a * cell + (cell - 3) / 2 + 4,
            "text-anchor": "middle",
            class: "gridlab",
            text: c.toFixed(3),
          })
        );
      }
      cells.push(
        svg("text", { x: 24, y: 22 + a * cell + cell / 2, "text-anchor": "end", class: "gridlab", text: "h" + (a + 1) }),
        svg("text", { x: 30 + a * cell + (cell - 3) / 2, y: 16, "text-anchor": "middle", class: "gridlab", text: "h" + (a + 1) })
      );
    }
    matWrap.replaceChildren(
      svg(
        "svg",
        {
          viewBox: `0 0 ${30 + CONFIG.HEADS * cell + 10} ${22 + CONFIG.HEADS * cell + 6}`,
          width: 30 + CONFIG.HEADS * cell + 10,
          height: 22 + CONFIG.HEADS * cell + 6,
          class: "grid-svg",
          role: "img",
          "aria-label": "how similar each head's keys are to every other head's, after the conversion",
        },
        cells
      )
    );
    matNote.innerHTML =
      groups === CONFIG.HEADS
        ? `Each square is how similar one head's keys are to another's, averaged over your sentence. Off the diagonal everything is near zero: four heads, four unrelated views of the same words. That is what multi-head attention buys and what the next two settings spend.`
        : groups === 1
        ? `One group: every square is 1.000. Every head is reading an identical key, and the only thing still telling them apart is its own query. This is the picture concept 7 reported as a single number.`
        : `This is the picture that only exists in the middle, and it is why this concept needed its own card. The matrix has gone <strong>block-diagonal</strong> — 1.000 inside a group because those heads now read the same key, and near zero across groups because those two heads still see different things. Some heads share and some do not; that is exactly what a group is, and a single averaged number cannot show it.`;

    // --- the predictor
    if (groups === CONFIG.HEADS) {
      predRead.update({ before: "—", after: "—", ideal: "—" });
      predNote.textContent = "Nothing is being averaged at this setting.";
    } else {
      let before = 0;
      let n = 0;
      for (let b = 0; b < CONFIG.BLOCKS; b++) {
        for (let g = 0; g < groups; g++) {
          for (let x = g * perGroup; x < (g + 1) * perGroup; x++) {
            for (let y = x + 1; y < (g + 1) * perGroup; y++) {
              for (let i = 0; i < T; i++) {
                before += cos(base.trace[b].heads[x].K[i], base.trace[b].heads[y].K[i]);
                n++;
              }
            }
          }
        }
      }
      before /= n || 1;
      predRead.update({
        before: fmt(before, 4),
        after: fmt(kf.cos, 4),
        ideal: fmt(1 / Math.sqrt(perGroup), 4),
      });
      predNote.className = "note warn";
      predNote.innerHTML = `Averaging costs whatever the things being averaged disagree about. Before the conversion, two heads inside the same group here are alike to the tune of <strong>${fmt(
        before,
        4
      )}</strong> — that is, not at all: they are essentially at right angles. Average ${perGroup} vectors at right angles and the mean keeps ${fmt(
        1 / Math.sqrt(perGroup),
        4
      )} of each by simple geometry, and the measurement lands at ${fmt(kf.cos, 4)}, a little below because the heads differ in length as well as direction. <strong>This is the worst case, and it is the honest frame for this page:</strong> the weights here are untrained noise, so nothing has ever given two heads a reason to resemble each other. The paper averages heads inside a trained model, which have had every reason to become correlated — and reports that the grouped conversion is already reasonable before any repair at all. The floor of that argument is what this page can show; the paper's own result is the part it cannot.`;
    }

    // --- the slope
    kvSlider.set(KVH.indexOf(kvHeads));
    const at = (kv, tok) => cacheBytes({ ...SERVING, kvHeads: kv, tokens: tok }) / GB;
    const maxTok = 65536;
    const pts = Array.from({ length: 33 }, (_, i) => [(i * maxTok) / 32, at(kvHeads, (i * maxTok) / 32)]);
    const ref = Array.from({ length: 33 }, (_, i) => [(i * maxTok) / 32, at(64, (i * maxTok) / 32)]);
    slopeCurve.update({
      points: pts,
      reference: ref,
      xRange: [0, maxTok],
      yRange: [0, at(64, maxTok)],
      mark: 32768,
      markLabel: "the lesson's configuration",
    });
    const here = at(kvHeads, 32768);
    const equal = Math.round((32768 * kvHeads) / 64);
    slopeRead.update({
      here: `${fmt(here, 3)} GB`,
      vs: `${fmt(at(64, 32768), 2)} GB and ${fmt(at(1, 32768), 3)} GB`,
      same: `${equal.toLocaleString()} tokens`,
    });
    slopeNote.className = "note";
    slopeNote.innerHTML = `Solid line is the chosen setting, dashed is multi-head. Two things to take from it. The first is the size of the win: at the serving configuration this app has quoted since concept 7 — 48 layers, 128 numbers per head, 32,768 tokens, one conversation — eight key/value heads costs <strong>${fmt(
      at(8, 32768),
      3
    )} GB</strong> against multi-head's ${fmt(at(64, 32768), 2)} GB. <strong>That 6.44 GB figure is a grouped configuration.</strong> The app has been quoting this paper's answer as its default since before this card existed, which is a fair picture of how completely this became the standard. The second is the shape: the line still goes through the origin. Grouping divides the slope by eight; it does not bend it. Multi-head reaches ${fmt(
      here,
      3
    )} GB at ${equal.toLocaleString()} tokens and this setting reaches it at 32,768 — the same wall, ${(64 / kvHeads).toFixed(
      0
    )}× further away. The two families that come next attack the other factors in that product: the width of what is stored per token, and the number of positions stored at all.`;

    // --- the retrofit
    upRead.update({
      alpha: "5% of the original steps",
      cost: "about 600 chip-days",
      knee: "little more past 10%",
    });
    upNote.innerHTML = `The half of this paper that is not an architecture. You do not train a grouped model — you take the multi-head one you already own, average its key and value projections inside each group, and continue pre-training on the same data with the same settings for a twentieth of the original steps. Because the projection is linear, averaging the matrices and averaging the vectors they produce are the same operation, which is why the bench above is the paper's conversion and not an analogy for it. The authors report that the grouped conversion is already reasonable with no repair at all, while the fully collapsed one needs the repair to be usable — and on this page the untrained analogue of that gap is the prediction moving ${fmt(
      tv(forward(tokens, { kvGroups: 2, kvPool: "mean" }).probs, base.probs),
      3
    )} at two groups against ${fmt(
      tv(forward(tokens, { kvGroups: 1, kvPool: "mean" }).probs, base.probs),
      3
    )} at one. That is a statement about how far the conversion moves the model, not about quality; this model is untrained and has no quality to measure. The idea underneath is the one worth carrying forward: a trained model's structure can be edited and repaired for a fraction of what it cost, which turns “what should we have trained?” into “what can we convert?”`;
  }

  return { update: render, unmount: () => {} };
}
