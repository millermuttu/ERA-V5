// Concept 1 — scaled dot-product attention, multi-head.
// Built from docs/research/transformer.md. Everything here is computed by the model in
// app/model/, on whatever sentence is in the box at the top.
import { el, svg, slider, toggle } from "../lib/dom.js";
import { attentionGrid, heat } from "../views/grid.js";
import { flowView } from "../views/flow.js";
import { barList, readout } from "../views/bars.js";
import { dot, softmax, fmt } from "../model/ops.js";
import { DH, CONFIG } from "../model/transformer.js";
import { state, baseline } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

const STAGES = [
  ["Q × K", "Every query meets every key. One dot product per pair — no scaling, no mask, nothing else yet."],
  ["scores", "Those dot products laid out as a matrix: row i is what query i thinks of every key."],
  ["scale", "Divide by √d_k. The paper's reason is variance: q·k has variance d_k, so this returns it to 1."],
  ["mask", "Anything later than the current token is set to −∞, so the model cannot read its own future."],
  ["softmax", "Each row becomes weights that are positive and sum to one — a distribution over the past."],
  ["Σ wV", "Those weights combine the value vectors. The result is the token's new representation."],
];

export function transformerCard(root, m) {
  let stage = 5;
  let query = null;
  let head = 0;
  let scaled = true;
  let masked = true;
  let averageHeads = false;

  root.appendChild(
    prose({
      problem:
        "Attention had been bolted onto recurrent encoder-decoders since 2014, but the recurrence stayed, and recurrence is sequential: token t waits for t−1. Long sequences trained slowly because the hardware sat idle.",
      mechanism:
        "Delete the recurrence, keep the attention. Each token is projected into a query, a key and a value; every query meets every key by dot product; the scores are scaled, masked, and softmaxed into weights; the weights combine the values. Several heads do this at once on different slices of the width. Because none of that knows what order the tokens came in, position has to be added separately — which is concept 2.",
    })
  );

  // ------------------------------------------------------------- the picture
  const flow = flowView();
  const flowNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "one token's journey, on your sentence" }),
      flow.node,
      flowNote,
    ])
  );

  // ---------------------------------------------------------------- stage walk
  const stageBtns = STAGES.map(([name], i) =>
    el("button", {
      class: "stage" + (i === stage ? " is-on" : ""),
      type: "button",
      text: `${i + 1}. ${name}`,
      onclick: () => {
        stage = i;
        for (const [k, b] of stageBtns.entries()) b.classList.toggle("is-on", k === i);
        render();
      },
    })
  );

  const stageNote = el("p", { class: "note" });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()) });
  const rowBars = barList({ rows: [] });
  const rowWrap = el("div", { class: "rowpanel" });

  const panelStages = el("section", { class: "panel" }, [
    el("div", { class: "panel-title", text: "the six stages, on your sentence" }),
    el("div", { class: "stages" }, stageBtns),
    stageNote,
    el("div", { class: "twoup" }, [grid.node, rowWrap]),
  ]);
  root.appendChild(panelStages);

  // ------------------------------------------------------------------- scaling
  const scaleToggle = toggle({ label: "divide by √d_k", value: true, onchange: (v) => ((scaled = v), render()) });
  const maskToggle = toggle({ label: "causal mask", value: true, onchange: (v) => ((masked = v), render()) });
  const scaleRead = readout([
    { key: "var", label: "score variance" },
    { key: "ratio", label: "variance ratio" },
    { key: "max", label: "largest weight in the row" },
    { key: "entropy", label: "row entropy" },
    { key: "future", label: "weight on the future" },
  ]);
  const scaleNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why divide by the square root of d_k" }),
      el("div", { class: "ctrls" }, [scaleToggle, maskToggle]),
      scaleRead.node,
      scaleNote,
    ])
  );

  // --------------------------------------------------------------------- heads
  const headSlider = slider({
    label: "head",
    min: 0,
    max: CONFIG.HEADS - 1,
    value: 0,
    format: (v) => `${v + 1} of ${CONFIG.HEADS}`,
    oninput: (v) => ((head = v), render()),
  });
  const avgToggle = toggle({
    label: "average all heads into one",
    value: false,
    onchange: (v) => ((averageHeads = v), render()),
  });
  const headsStrip = el("div", { class: "headstrip" });
  const headsNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "four heads, four different pictures" }),
      el("div", { class: "ctrls" }, [headSlider.node, avgToggle]),
      headsStrip,
      headsNote,
    ])
  );

  // ---------------------------------------------------------------------- cost
  const costBars = barList({ rows: [{ key: "reads", label: "key–query pairs scored" }] });
  const costRead = readout([
    { key: "path", label: "steps between any two tokens" },
    { key: "seq", label: "sequential steps to train" },
    { key: "recurrent", label: "a recurrent layer would need" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what it costs, and what it bought" }),
      costBars.node,
      costRead.node,
      costNote,
    ])
  );

  // ------------------------------------------------------------- trade + plain
  root.appendChild(
    tradeBlock({
      buys: [
        "Every position computes at once, so training uses the whole sequence in parallel instead of walking it",
        "Any token can reach any other in a single step, however far apart — Table 1's O(1) path length",
        "Dot products are a matrix multiply, which is the operation accelerators are fastest at",
      ],
      givesUp: [
        "Every token is compared with every other: the score matrix is T×T, so per-layer cost grows with the square of the context",
        "Generating keeps every earlier key and value to avoid recomputing them — a cache that grows with the conversation",
        "Attention alone has no idea what order the tokens came in; position must be injected from outside",
      ],
      chooseWhen:
        "Always, as the thing everything else is measured against — and at short context it is usually still the right answer outright: exact, well understood, and the fastest kernels in the world are written for it.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Every word can look directly at every other word, however far back, in one step",
        "The whole sentence is processed at once instead of one word after another, so training is much faster",
        "It is simple enough to be made very fast on the hardware we actually have",
      ],
      cons: [
        "Doubling the length of the text roughly quadruples the work, because every word compares itself with every word",
        "While writing a reply it has to remember something about every word so far, and that memory keeps growing",
        "On its own it cannot tell which word came first — the order has to be handed to it separately",
      ],
      verdict:
        "This is the machine everything else on this timeline is trying to make cheaper. It reads beautifully and it does not scale: the cost grows with the square of how much you give it, and the memory grows with every word it writes. Nothing after this replaces it — they all chip away at one of those two bills.",
    })
  );

  // -------------------------------------------------------------------- render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (T === 0) return;
    // Not the last token: it has no future to mask, so the mask toggle would look inert.
    if (query === null || query >= T) query = Math.max(0, Math.floor(T / 2));
    headSlider.set(head);

    const res = baseline();
    const h = res.trace[state.block]?.heads[head] ?? res.trace[0].heads[head];
    const heads = (res.trace[state.block] ?? res.trace[0]).heads;

    // Stage-by-stage, recomputed here so each stage shows its own numbers.
    const rawRow = tokens.map((_, j) => (j <= query ? dot(h.Q[query], h.K[j]) : null));
    const scaledRow = rawRow.map((v) => (v === null ? null : v / Math.sqrt(DH)));
    const useRow = scaled ? scaledRow : rawRow;
    const maskedRow = useRow.map((v, j) => (v === null ? -Infinity : masked && j > query ? -Infinity : v));
    const openRow = useRow.map((v, j) => (v === null ? (j > query ? dot(h.Q[query], h.K[j]) / (scaled ? Math.sqrt(DH) : 1) : -Infinity) : v));
    const weightsRow = softmax(masked ? maskedRow : openRow);

    // The grid always shows the finished pattern; the row panel shows the current stage.
    const gridWeights = tokens.map((_, i) => {
      const row = tokens.map((_, j) => {
        if (masked && j > i) return -Infinity;
        return dot(h.Q[i], h.K[j]) / (scaled ? Math.sqrt(DH) : 1);
      });
      return softmax(row);
    });
    grid.update({ tokens, weights: gridWeights, query });

    flow.update({
      tokens,
      head: { ...h, emb: (res.trace[state.block] ?? res.trace[0]).input },
      weights: gridWeights,
      out: h.out,
      top: res.top,
      query,
      opts: { readable: masked ? null : null },
    });
    flowNote.innerHTML = `Every word on the left becomes three things — a <strong style="color:#5b7fdb">query</strong> asking, a <strong style="color:#E0693D">key</strong> advertising, and a <strong style="color:#4FC58C">value</strong> waiting to be handed over. The dot grid is who reads whom: the row for “${tokens[queryIndex()].word}” is picked out, and each mark's size is how much weight that pair got. The bars on the right are what the model expects next — noise, because nothing here is trained, but computed noise. The dashed line over the top is the residual: whatever attention decides, the token keeps a copy of itself.`;

    stageNote.textContent = STAGES[stage][1];
    renderRow({ tokens, rawRow, scaledRow, maskedRow, weightsRow, h });

    // --- the scale argument, measured
    const pairs = [];
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) pairs.push(dot(h.Q[i], h.K[j]));
    const mean = pairs.reduce((a, b) => a + b, 0) / pairs.length;
    const varRaw = pairs.reduce((s, x) => s + (x - mean) ** 2, 0) / pairs.length;
    const varUsed = scaled ? varRaw / DH : varRaw;
    const live = weightsRow.filter((w) => w > 0);
    const entropy = -live.reduce((s, w) => s + w * Math.log(w), 0);
    const future = weightsRow.reduce((s, w, j) => s + (j > query ? w : 0), 0);

    scaleRead.update({
      var: fmt(varUsed, 2),
      ratio: scaled ? `÷ ${DH} = d_k` : "× 1",
      max: fmt(Math.max(...weightsRow), 3),
      entropy: fmt(entropy, 3),
      future: future > 1e-9 ? fmt(future, 3) : "0",
    });

    scaleNote.className = "note " + (scaled ? "" : "warn");
    scaleNote.textContent = scaled
      ? `Scaled: variance ${fmt(varUsed, 1)}, and the row keeps a spread of weights (entropy ${fmt(entropy, 2)}). Dividing every score by √d_k divides the variance by exactly d_k = ${DH}. Turn it off to see what the paper is warning about.`
      : `Unscaled: variance ${fmt(varUsed, 1)}, ${DH}× larger, and the row has collapsed towards one-hot — largest weight ${fmt(Math.max(...weightsRow), 3)}, entropy ${fmt(entropy, 2)}. That is softmax pushed into the region where its gradients nearly vanish, which is exactly why the √d_k is there.` +
        (masked ? "" : " The mask is also off, so weight is landing on words that have not happened yet.");

    // --- heads
    renderHeads(heads, tokens);

    // --- cost
    const scored = (T * (T + 1)) / 2;
    costBars.update({
      reads: { value: scored, of: T * T, text: `${scored} of ${T * T}`, label: `pairs scored (T = ${T})` },
    });
    costRead.update({ path: "1", seq: "1", recurrent: `${T}` });
    costNote.innerHTML = `At ${T} tokens the layer scores ${scored} pairs. Double the sentence and that roughly quadruples — the T² the rest of this timeline is spent attacking. What it bought is the other two bars: any token reaches any other in <strong>one</strong> step, and the whole sequence trains in <strong>one</strong> pass instead of ${T} sequential ones. A recurrent layer had those the other way round.`;
  }

  function renderRow({ tokens, rawRow, scaledRow, maskedRow, weightsRow, h }) {
    const T = tokens.length;
    const cells = [];
    const val = (j) => {
      if (stage === 0 || stage === 1) return rawRow[j];
      if (stage === 2) return scaledRow[j];
      if (stage === 3) return maskedRow[j];
      return weightsRow[j];
    };

    const maxAbs = Math.max(
      ...tokens.map((_, j) => {
        const v = val(j);
        return v === null || !Number.isFinite(v) ? 0 : Math.abs(v);
      }),
      1e-9
    );

    for (let j = 0; j < T; j++) {
      const v = val(j);
      const dead = v === null || v === -Infinity;
      const w = dead ? 0 : Math.abs(v) / maxAbs;
      cells.push(
        el("div", { class: "cellrow" }, [
          el("span", { class: "cw", text: tokens[j].word }),
          el("span", { class: "track" }, [
            el("span", {
              class: "fill" + (v < 0 ? " neg" : ""),
              style: { width: (w * 100).toFixed(1) + "%" },
            }),
          ]),
          el("span", { class: "cv", text: dead ? (stage === 3 ? "−∞" : "—") : stage === 5 ? fmt(v, 3) : fmt(v, 2) }),
        ])
      );
    }

    // Recomputed from the weights actually on screen, so the toggles move it too.
    let out = null;
    if (stage === 5) {
      out = new Float64Array(DH);
      for (let j = 0; j < T; j++) {
        const w = weightsRow[j];
        if (!w) continue;
        for (let d = 0; d < DH; d++) out[d] += w * h.V[j][d];
      }
    }
    rowWrap.replaceChildren(
      el("div", { class: "rowhead", text: `row for “${tokens[queryIndex()].word}”` }),
      el("div", { class: "cellrows" }, cells),
      out
        ? el("div", { class: "outvec" }, [
            el("span", { class: "ro-lab", text: "output vector (first 8 of " + DH + ")" }),
            el("span", { class: "vec", text: Array.from(out.slice(0, 8), (x) => fmt(x, 2)).join("  ") }),
          ])
        : null
    );
  }

  const queryIndex = () => Math.min(query ?? state.tokens.length - 1, state.tokens.length - 1);

  function renderHeads(heads, tokens) {
    const T = tokens.length;
    const q = queryIndex();
    const rows = averageHeads
      ? [
          {
            name: "averaged",
            w: tokens.map((_, j) =>
              heads.reduce((s, hh) => s + hh.weights[q][j], 0) / heads.length
            ),
          },
        ]
      : heads.map((hh, i) => ({ name: `head ${i + 1}`, w: hh.weights[q] }));

    headsStrip.replaceChildren(
      ...rows.map((r, i) =>
        el("div", { class: "headrow" + (!averageHeads && i === head ? " is-on" : "") }, [
          el("span", { class: "hn", text: r.name }),
          el(
            "span",
            { class: "hcells" },
            tokens.map((t, j) =>
              el("span", {
                class: "hcell",
                title: `${t.word}: ${fmt(r.w[j], 3)}`,
                style: { background: j > q ? "rgba(233,231,220,0.03)" : heat(r.w[j]) },
              })
            )
          ),
        ])
      )
    );

    const spread = heads.map((hh) => {
      const w = hh.weights[q];
      const top = w.indexOf(Math.max(...w));
      return tokens[top].word;
    });
    headsNote.textContent = averageHeads
      ? `Averaged into one head, the four separate patterns become a single blurred row. This is what the paper means by "with a single attention head, averaging inhibits this" — the subspaces stop being separate.`
      : `Each head is looking somewhere different: for “${tokens[q].word}” the four heads put their largest weight on ${spread.map((s) => `“${s}”`).join(", ")}. The paper's claim is about representation subspaces, not that any head has an interpretable job — so read the differences, not roles.`;
  }

  return { update: render, unmount: () => {} };
}
