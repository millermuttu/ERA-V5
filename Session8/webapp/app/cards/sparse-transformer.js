// Concept 6 — Sparse Transformer, strided and fixed patterns.
// Built from docs/research/sparse-transformer.md. The research corrected the headline claim:
// the paper's four contributions include the residual/init change, recomputation and block-sparse
// kernels, and it says outright that recomputation ALONE trains dense attention at length 16,384.
// Sparsity is one of four things, and the card says so.
import { el, slider } from "../lib/dom.js";
import { attentionGrid } from "../views/grid.js";
import { readout, barList } from "../views/bars.js";
import { fmt } from "../model/ops.js";
import { forward } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

// The paper's two factorized patterns, as one predicate each.
// Note the fixed-pattern residue set: j mod l can never equal l, so the band is l−c … l−1.
const PATTERNS = {
  full: {
    label: "full",
    formula: "j ≤ i",
    head1: () => true,
    head2: () => true,
  },
  strided: {
    label: "strided",
    formula: "j ≥ max(0, i−l)   or   (i−j) mod l = 0",
    head1: (i, j, l) => i - j < l,
    head2: (i, j, l) => (i - j) % l === 0,
  },
  fixed: {
    label: "fixed",
    formula: "⌊j/l⌋ = ⌊i/l⌋   or   j mod l ≥ l−c",
    head1: (i, j, l) => Math.floor(j / l) === Math.floor(i / l),
    head2: (i, j, l, c) => j % l >= l - c,
  },
};

export function sparseTransformerCard(root, m) {
  let pattern = "strided";
  let l = 4;
  let c = 1;
  let headView = "merged";
  let query = null;
  let target = null;

  root.appendChild(
    prose({
      problem:
        "Concept 1 measured the bill: every query meets every key, so the work grows with the square of the context. By 2019 that was the thing standing between transformers and long sequences — images, audio, and any text long enough to matter.",
      mechanism:
        "Do not let every query see every key. The authors trained a dense 128-layer model on images and looked at what it had learned: most layers were already sparse, and layers 19 and 20 had split themselves into a row attention and a column attention on their own. So they hard-coded that split. One head reads a local block, the other reads every l-th position, and two hops through the pair connect any pair of tokens. The pattern is chosen in advance, which is what makes it fast — and, later, what makes it the thing to beat.",
    })
  );

  // ------------------------------------------------------------- the pattern
  const patternBtns = el(
    "div",
    { class: "stages" },
    Object.entries(PATTERNS).map(([key, p]) =>
      el("button", {
        class: "stage" + (key === pattern ? " is-on" : ""),
        type: "button",
        "data-pat": key,
        text: p.label,
        onclick: () => {
          pattern = key;
          for (const b of patternBtns.children) b.classList.toggle("is-on", b.dataset.pat === pattern);
          render();
        },
      })
    )
  );
  const headBtns = el(
    "div",
    { class: "stages" },
    [
      ["head1", "head 1 only"],
      ["head2", "head 2 only"],
      ["merged", "both heads"],
    ].map(([key, label]) =>
      el("button", {
        class: "stage" + (key === headView ? " is-on" : ""),
        type: "button",
        "data-head": key,
        text: label,
        onclick: () => {
          headView = key;
          for (const b of headBtns.children) b.classList.toggle("is-on", b.dataset.head === headView);
          render();
        },
      })
    )
  );
  const lSlider = slider({
    label: "stride l",
    min: 2,
    max: 16,
    value: 4,
    oninput: (v) => ((l = v), render()),
  });
  const formula = el("div", { class: "formula" });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "attention under a fixed sparse pattern" });
  const patNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the same attention, with most of it switched off" }),
      patternBtns,
      el("div", { class: "ctrls" }, [lSlider.node]),
      headBtns,
      formula,
      grid.node,
      patNote,
    ])
  );

  // ------------------------------------------------------ where the root comes from
  const costRead = readout([
    { key: "read", label: "keys read of all causal pairs" },
    { key: "per", label: "keys per query" },
    { key: "predicted", label: "l + n/l at this stride" },
    { key: "best", label: "minimised at" },
  ]);
  const strideBars = barList({ rows: [] });
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the square root, and at your sentence's length" }),
      costRead.node,
      strideBars.node,
      costNote,
    ])
  );

  // --------------------------------------------------------------- two hops
  const probe = el("div", { class: "ctrls" });
  const hopRead = readout([
    { key: "verdict", label: "can it read it?" },
    { key: "via", label: "if not, the way round" },
    { key: "weight", label: "weight if read directly" },
  ]);
  const hopNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what a query can still reach, and how" }),
      probe,
      hopRead.node,
      hopNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Attention cost falls from every pair to roughly n·√n, which is what put images and audio at thousands of positions within reach",
        "The pattern is known before the data arrives, so it can be implemented as dense operations on blocks with fast kernels",
        "Any position still reaches any other in two hops, so nothing is formally disconnected",
        "It was motivated by evidence, not assumption: a dense model was trained and inspected first, and layers 19–20 had already factorized themselves into rows and columns",
      ],
      givesUp: [
        "The pattern is fixed in advance. The same inspection that justified sparsity also found global and data-dependent layers, and the paper states plainly that those would be impacted by a predetermined pattern",
        "It is not uniformly a win: on Enwik8 the strided pattern scored 1.13 bits per byte against dense attention's 1.00",
        "A dependency that does not fall on the grid needs an intermediary, and the intermediary has to have kept what was needed",
        "Sparsity is one of four contributions in that paper — the restructured residual block, recomputation and the block-sparse kernels do a great deal of the work, and recomputation alone trained dense attention at length 16,384",
      ],
      chooseWhen:
        "Data with real periodic or local structure that you can name in advance — images, audio, code. For text the honest reading is that this bought a cheaper computation, not a better-chosen one, and every adaptive sparse method after it is an attempt to choose rather than assume.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It cuts the work dramatically, which is what made very long inputs like images and audio possible at all",
        "The shortcut is decided in advance, so the computer can be made very fast at exactly that shape",
        "Nothing is completely cut off: any word can still reach any other, just in two steps instead of one",
        "The idea came from looking at what a normal model actually did, not from guessing",
      ],
      cons: [
        "The shortcut is a fixed guess made before seeing the text, and the authors' own inspection found layers that need to look anywhere at all",
        "It is not always better: on one text benchmark it was clearly worse than looking at everything",
        "Reaching something the pattern skipped depends on the word in between having kept what you needed",
        "The paper's headline improvements were not from the shortcut alone — several other changes in the same work did much of the lifting",
      ],
      verdict:
        "The first serious answer to attention costing too much, and an honest one: it skips work rather than choosing which work to skip. That distinction is the thread running through every sparse method that comes later.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    if (target === null || target >= T) target = 1;
    lSlider.set(l);

    const P = PATTERNS[pattern];
    const h1 = (i, j) => P.head1(i, j, l, c);
    const h2 = (i, j) => P.head2(i, j, l, c);
    const readable = (i, j) =>
      j <= i && (headView === "head1" ? h1(i, j) : headView === "head2" ? h2(i, j) : h1(i, j) || h2(i, j));

    const res = forward(tokens, { mixer: softmaxMixer({ readable }) });
    const h = res.trace[0].heads[0];
    grid.update({ tokens, weights: h.weights, query, readable });

    formula.textContent = pattern === "full" ? "j ≤ i — every earlier key" : P.formula;

    // --- counts and the U-shape
    // Count what is actually on screen: the head selection changes the picture, so it has to
    // change the number too, or the two disagree in front of the reader.
    const countFor = (stride) => {
      let n = 0;
      for (let i = 0; i < T; i++)
        for (let j = 0; j <= i; j++) {
          const a = PATTERNS[pattern].head1(i, j, stride, c);
          const b = PATTERNS[pattern].head2(i, j, stride, c);
          if (headView === "head1" ? a : headView === "head2" ? b : a || b) n++;
        }
      return n;
    };
    const causal = (T * (T + 1)) / 2;
    const read = countFor(l);
    const sqrtN = Math.round(Math.sqrt(T));
    const strides = [2, 3, 4, 6, 8, 12, 16].filter((s) => s <= Math.max(16, T));
    const counts = strides.map((s) => [s, countFor(s)]);
    const best = counts.reduce((a, b) => (b[1] < a[1] ? b : a));

    costRead.update({
      read: `${read} of ${causal} (${Math.round((read / causal) * 100)}%)`,
      per: fmt(read / T, 1),
      predicted: pattern === "full" ? "—" : fmt(l + T / l, 1),
      best: pattern === "full" ? "—" : `l = ${best[0]} here, √${T} ≈ ${sqrtN}`,
    });

    strideBars.update(
      Object.fromEntries(
        counts.map(([s, n]) => [
          "s" + s,
          { value: n, of: causal, text: `${n}`, label: `stride ${s}${s === l ? "  ←" : ""}`, tone: s === l ? "" : "alt" },
        ])
      )
    );
    if (strideBars.node.children.length !== counts.length) {
      // rebuild the rows when the stride set changes
      strideBars.node.replaceChildren();
      for (const [s, n] of counts) {
        strideBars.node.appendChild(
          el("div", { class: "bar" }, [
            el("span", { class: "bar-name", text: `stride ${s}${s === l ? "  ←" : ""}` }),
            el("span", { class: "track" }, [
              el("span", { class: "fill" + (s === l ? "" : " alt"), style: { width: ((n / causal) * 100).toFixed(1) + "%" } }),
            ]),
            el("span", { class: "val", text: String(n) }),
          ])
        );
      }
    }

    costNote.className = "note";
    costNote.innerHTML =
      pattern === "full"
        ? `Full attention reads all ${causal} causal pairs. Switch to a sparse pattern and watch this number fall — and watch the bars below find their minimum.`
        : `Each query reads about a local run of l plus every l-th position, so the count per query is roughly l + n/l — minimised when l is √n. At your sentence length that is √${T} ≈ ${sqrtN}, and the bars bottom out at stride ${best[0]} — near it rather than exactly on it, because l + n/l is a smooth approximation and the real count is integers with edge effects. <strong>The square root is arithmetic, not linguistics</strong>: it is where l + n/l is smallest, nothing to do with language. And note the scale — at ${T} tokens this saves ${100 - Math.round((read / causal) * 100)}%, while the paper ran at 12,288 tokens with stride 128, where the same pattern reads under 3% of the matrix. Short sentences understate this badly.`;

    // --- two hops
    const pick = (label, value, onChange) =>
      el("div", { class: "ctrl" }, [
        el("label", { text: label }),
        el(
          "select",
          { onchange: (e) => onChange(Number(e.target.value)) },
          tokens.map((t, i) => el("option", { value: i, text: `${i}. ${t.word}`, selected: i === value ? "" : null }))
        ),
      ]);
    probe.replaceChildren(
      pick("query", query, (v) => ((query = v), render())),
      pick("needs to read", target, (v) => ((target = v), render()))
    );

    const direct = target <= query && readable(query, target);
    let via = null;
    if (!direct && target < query) {
      for (let a = target; a <= query; a++) {
        if (readable(query, a) && readable(a, target)) {
          via = a;
          break;
        }
      }
    }
    hopRead.update({
      verdict: target > query ? "it is in the future" : direct ? "directly" : via !== null ? "in two hops" : "not at all",
      via: via !== null ? `“${tokens[via].word}”` : "—",
      weight: direct ? fmt(h.weights[query][target], 3) : "0.000",
    });
    hopNote.className = "note " + (!direct && via === null && target <= query ? "warn" : "");
    hopNote.textContent = direct
      ? `“${tokens[query].word}” reads “${tokens[target].word}” straight away, weight ${fmt(h.weights[query][target], 3)} — the head that supplies it is ${h1(query, target) ? "the local one" : "the strided one"}.`
      : target > query
      ? "That word comes later in the sentence; the causal mask hides it whatever the pattern."
      : via !== null
      ? `Not directly — the pattern skips that cell. But “${tokens[query].word}” reads “${tokens[via].word}”, and “${tokens[via].word}” reads “${tokens[target].word}”, so the information can arrive in two hops through the next layer. That is the connectivity argument the factorization rests on, and its cost is that “${tokens[via].word}” has to have kept what was needed.`
      : `No path at all in one or two hops. For a valid factorized pattern this should not happen — if you are seeing it, the stride has been pushed past what the pattern supports at this sentence length.`;
  }

  return { update: render, unmount: () => {} };
}
