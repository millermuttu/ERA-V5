// Concept 3 — relative position representations (Shaw et al.).
// Built from docs/research/relative-positions.md. The research forced a seam change: Shaw's term
// is q_i · w_clip(j−i) / √d_k, so bias() had to be handed the query vector. A scalar-per-offset
// hook could only ever have expressed ALiBi.
import { el, slider, toggle } from "../lib/dom.js";
import { attentionGrid } from "../views/grid.js";
import { readout, barList } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { relativeBuckets, sinusoidal } from "../model/position.js";
import { softmaxMixer } from "../model/mixers.js";
import { state, baseline } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const sinusoidalScheme = sinusoidal();

export function relativePositionsCard(root, m) {
  let k = 4;
  let queryDependent = true;
  let prepended = false;
  let query = null;

  root.appendChild(
    prose({
      problem:
        "Sinusoidal encoding tells a token where it sits in absolute terms. What a language model usually needs is how far away something is — and with an absolute signal the model has to reconstruct distance from two absolute labels, separately at every offset. Concept 2's own offset curve hints at the relationship; nothing makes the model use it.",
      mechanism:
        "Move position out of the input and into the comparison. Each query–key pair picks up a learned vector indexed by the clipped distance between them, and that vector enters the score as a dot product with the query itself — so how much positional pull a pair gets depends on what the querying token is looking for. Distances past a chosen k all share one bucket, so the number of learned vectors stays fixed however long the sequence is.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------------------ 1. clipping
  const kSlider = slider({
    label: "clip distance k",
    min: 0,
    max: 15,
    value: 4,
    oninput: (v) => ((k = v), render()),
  });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "attention with the relative term" });
  const clipRead = readout([
    { key: "distinct", label: "buckets a causal model can ever use" },
    { key: "nominal", label: "buckets it allocates" },
    { key: "collide", label: "two different distances, same bias" },
  ]);
  const clipNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "everything past k becomes the same distance" }),
      el("div", { class: "ctrls" }, [kSlider.node]),
      grid.node,
      clipRead.node,
      clipNote,
    ])
  );

  // ------------------------------------------------------------- 2. the shift
  const shiftToggle = toggle({
    label: "prepend a word at position 0",
    value: false,
    onchange: (v) => ((prepended = v), render()),
  });
  const shiftRead = readout([
    { key: "rel", label: "relative: change in the shared block" },
    { key: "abs", label: "sinusoidal: change in the same block" },
  ]);
  const shiftNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "shift the whole sentence and see what survives" }),
      el("div", { class: "ctrls" }, [shiftToggle]),
      shiftRead.node,
      shiftNote,
    ])
  );

  // --------------------------------------------------- 3. query dependence
  const qdToggle = toggle({
    label: "query-dependent (Shaw)",
    value: true,
    onchange: (v) => ((queryDependent = v), render()),
  });
  const diagBars = barList({ rows: [] });
  const qdRead = readout([
    { key: "spread", label: "spread of the bias at one fixed distance" },
    { key: "mode", label: "what the bias depends on" },
  ]);
  const qdNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "does the same distance mean the same thing to every word?" }),
      el("div", { class: "ctrls" }, [qdToggle]),
      qdRead.node,
      qdNote,
    ])
  );

  // ------------------------------------------------------------- 4. the cost
  const costRead = readout([
    { key: "params", label: "extra learned numbers" },
    { key: "matmul", label: "can the scores be one matrix multiply?" },
    { key: "mem", label: "extra memory, order" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what the per-pair term costs" }),
      costRead.node,
      costNote,
    ])
  );

  // -------------------------------------------------------- trade and plain
  root.appendChild(
    tradeBlock({
      buys: [
        "Distance enters the score directly, and a pattern learned at one offset holds when the whole sentence shifts — the bias matrix is constant along each diagonal",
        "How much positional pull a pair gets depends on what the query is looking for, not just how far apart they are",
        "Clipping past k keeps the parameter count fixed no matter how long the sequence gets",
        "Reported +0.3 BLEU English-German and +0.5 English-French on the base model, replacing sinusoids entirely",
      ],
      givesUp: [
        "The extra term is per-pair, so the scores can no longer be produced by a single matrix multiplication — the paper reports about 7% slower training and O(n²·d) extra memory",
        "Two of its four moving parts do not earn their keep: the value-side term is worth 0.0 BLEU, and any k of 2 or more scores the same",
        "Everything past the clip distance is one bucket, so genuinely long-range distance is flattened by design",
        "Under a causal mask only half the allocated buckets can ever fire",
      ],
      chooseWhen:
        "When relative distance matters more than absolute placement and the context is short enough that a per-pair term is affordable. Its real legacy is the idea, not the implementation: RoPE folds the same relativity into a rotation that survives one matrix multiply, and ALiBi throws away the query-dependence for a single subtraction.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model is told how far apart two words are, rather than having to work it out from two separate labels",
        "Move the whole sentence along and the relationships between words stay exactly the same",
        "How strongly distance matters can differ from word to word, because it depends on what the word is looking for",
        "It never needs more memory for longer text, because anything beyond a set distance is treated as simply far away",
      ],
      cons: [
        "It makes the fastest part of the computation slower — the shortcut that let the machine do it all in one go no longer applies, costing about 7% of training speed",
        "Past the chosen distance, everything is just far — six words back and sixty words back become the same thing",
        "In the paper's own tests, half the machinery made no measurable difference at all",
      ],
      verdict:
        "The right idea with an expensive first draft. Telling the model about distance rather than place is what everything later keeps; paying for it with a separate calculation for every pair of words is what everything later throws away.",
    })
  );

  // ---------------------------------------------------------------- render
  function render() {
    const tokens = prepended ? [{ word: "yesterday", id: 0, known: false }, ...state.tokens] : state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    kSlider.set(k);

    const rel = relativeBuckets({ k, dims: DH });
    const biasFn = queryDependent
      ? rel.bias
      : (i, j) => {
          // One scalar per offset: the same vector, collapsed against a fixed probe. This is what
          // the term becomes if you drop the query — which is the shape ALiBi later adopts.
          const v = rel.vector(j - i);
          return v.reduce((s, x) => s + x, 0) / Math.sqrt(DH) / 3;
        };

    const res = forward(tokens, { mixer: softmaxMixer({ bias: biasFn }) });
    const h = res.trace[0].heads[0];
    grid.update({ tokens, weights: h.weights, query });

    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query,
      opts: {},
    });
    flowNote.innerHTML = `Nothing is added to the embeddings here — no ochre dots, unlike concept 2. Position enters further right, inside the score: every pair picks up a learned vector for the distance between them, and it is multiplied against the query itself. The picture is otherwise the baseline's, which is the point: this mechanism changes what a pair scores, not what a token is.`;

    // --- 1. clipping
    const usable = Math.min(k, T - 1) + 1;
    let collide = "—";
    if (k < T - 1) {
      const i = T - 1;
      const j1 = Math.max(0, i - k - 1);
      const j2 = Math.max(0, i - k - 2);
      if (j1 !== j2) {
        const d = Math.abs(biasFn(i, j1, h.Q[i]) - biasFn(i, j2, h.Q[i]));
        collide = `${d.toFixed(6)} apart`;
      }
    }
    clipRead.update({ distinct: String(usable), nominal: String(2 * k + 1), collide });
    clipNote.className = "note " + (k === 0 ? "warn" : "");
    clipNote.textContent =
      k === 0
        ? "At k = 0 every pair shares one bucket: the bias is constant, so it carries no positional information at all. The paper's ablation puts this at 12.5 BLEU — the floor, and the same score as removing the mechanism entirely."
        : `Two keys further than ${k} back get the identical bias — the readout shows them ${collide}. Note the other two numbers: the scheme allocates ${2 * k + 1} vectors, but a decoder only ever looks backwards, so at most ${usable} of them can fire. Half the parameter budget is unreachable in a causal model, which the paper does not remark on.`;

    // --- 2. the shift
    // Compare the *positional* contribution only. Comparing finished attention weights would
    // fold in the content change caused by inserting a word, which is not what is being claimed.
    // The full pre-softmax score under each scheme. Comparing only the positional term would be
    // misleading: PE(i)·PE(j) is itself shift-invariant — that was concept 2's own finding — and
    // what actually breaks under a shift are the cross terms between content and position.
    const bias2d = (toks, useRel) => {
      const hh = forward(
        toks,
        useRel ? { mixer: softmaxMixer({ bias: biasFn }) } : { position: sinusoidalScheme }
      ).trace[0].heads[0];
      return toks.map((_, i) =>
        toks.map((_, j) =>
          j <= i ? dot(hh.Q[i], hh.K[j]) / Math.sqrt(DH) + (useRel ? biasFn(i, j, hh.Q[i]) : 0) : 0
        )
      );
    };
    const shortRel = bias2d(state.tokens, true);
    const longRel = bias2d([{ word: "yesterday", id: 0, known: false }, ...state.tokens], true);
    const shortAbs = bias2d(state.tokens, false);
    const longAbs = bias2d([{ word: "yesterday", id: 0, known: false }, ...state.tokens], false);
    const maxDiff = (a, b) => {
      let d = 0;
      for (let i = 0; i < a.length; i++)
        for (let j = 0; j <= i; j++) d = Math.max(d, Math.abs(a[i][j] - b[i + 1][j + 1]));
      return d;
    };
    shiftRead.update({ rel: maxDiff(shortRel, longRel).toFixed(6), abs: maxDiff(shortAbs, longAbs).toFixed(6) });
    shiftNote.innerHTML = `Insert a word at the front and every original word moves one place along. With relative positions the scores in the shared block are unchanged to <strong>${maxDiff(shortRel, longRel).toFixed(6)}</strong> — the relationships rode along with the words. With sinusoidal absolute positions the same block moves by ${maxDiff(shortAbs, longAbs).toFixed(3)}: every word sits at a different absolute place, and the cross terms between content and position rewrite every score. Worth knowing why that is the honest comparison — the position vectors' own dot product PE(i)·PE(j) is <em>already</em> shift-invariant, as concept 2 showed. It is what happens when position is mixed into the content that breaks.`;

    // --- 3. query dependence
    const off = -Math.min(3, T - 1);
    const diag = [];
    for (let i = -off; i < T; i++) diag.push([tokens[i].word, biasFn(i, i + off, h.Q[i])]);
    const spread = Math.max(...diag.map((d) => d[1])) - Math.min(...diag.map((d) => d[1]));
    qdRead.update({
      spread: spread.toFixed(6),
      mode: queryDependent ? "the distance and the asking word" : "the distance alone",
    });
    qdNote.className = "note " + (queryDependent ? "" : "warn");
    qdNote.textContent = queryDependent
      ? `Fix the distance at ${Math.abs(off)} and walk down the diagonal: the bias varies by ${spread.toFixed(4)} across words. Every token gets a different amount of positional pull at the same distance, because the term is a dot product with that token's own query. This is the part ALiBi later gives up.`
      : `With the query dropped, the bias at a fixed distance is the same for every word — spread ${spread.toFixed(6)}. Simpler and much cheaper, and it is roughly the shape ALiBi arrives at in 2021. What is lost is the ability of one word to care about distance more than another does.`;

    // --- 4. cost
    const perLayerHead = (2 * k + 1) * DH;
    costRead.update({
      params: `${(perLayerHead * CONFIG.HEADS * CONFIG.BLOCKS).toLocaleString()}`,
      matmul: "no",
      mem: `n² × d_k = ${(T * T * DH).toLocaleString()}`,
    });
    costNote.innerHTML = `${2 * k + 1} vectors of ${DH} numbers, per head, per layer — ${(perLayerHead * CONFIG.HEADS * CONFIG.BLOCKS).toLocaleString()} extra learned numbers in this toy model. The parameters are not the problem. The problem is the second row: because the term involves both the query and the pair, the scores stop being one clean matrix multiply. The paper's own words are that this <em>"prevents us from computing all e_ij for all pairs of positions in a single matrix multiplication"</em>, and it costs about 7% of training throughput plus O(n²·d) memory to work around. Concept 2 needed no such thing, and RoPE will get the relativity back without it.`;
  }

  return { update: render, unmount: () => {} };
}
