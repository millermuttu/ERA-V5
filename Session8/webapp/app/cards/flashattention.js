// Concept 14 — FlashAttention.
// Built from docs/research/flashattention.md. This card is unusual and the research says why: the
// mechanism changes no arithmetic, so there is nothing new to show in the attention grid, and a
// stopwatch in a browser would be a lie — there is no HBM here to save a trip to. What the page can
// do exactly is run the paper's own online-softmax recurrence over the reader's sentence one tile at
// a time, and count elements moved under both algorithms. Two traps the research flagged and this
// file handles: a fully-masked tile gives exp(−∞ − (−∞)) = NaN and poisons the row for good, and the
// paper's B_r = min(⌈M/4d⌉, d) is a proof convenience that would cap query tiles at 8 rows here.
import { el, slider, choice, toggle } from "../lib/dom.js";
import { readout, barList } from "../views/bars.js";
import { fmt, dot } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const TILES = [1, 2, 4, 8, 16];

/**
 * Algorithm 1, lines 9–13, unmodified — except that B_r is the whole sentence rather than the
 * paper's min(⌈M/4d⌉, d), so every row can be displayed at once. Returns the state after `upTo`
 * key tiles, plus the log of what each tile did.
 */
function tiledAttention(Q, K, V, dh, Bc, upTo = Infinity, keep = null) {
  const T = Q.length;
  const scale = Math.sqrt(dh);
  const m = new Array(T).fill(-Infinity);
  const l = new Array(T).fill(0);
  const O = Array.from({ length: T }, () => new Float64Array(dh));
  const tiles = [];
  let skipped = 0;
  let dropped = 0;
  let live = 0;

  for (let t = 0, j0 = 0; j0 < T; j0 += Bc, t++) {
    if (t >= upTo) break;
    const j1 = Math.min(j0 + Bc, T);
    const log = { j0, j1, moved: 0, skippedRows: 0, dropped: 0, factor: new Array(T).fill(null), before: m.slice() };

    for (let i = 0; i < T; i++) {
      // The causal mask leaves whole rows of a tile at −∞. The real kernel skips them; taking the
      // row maximum of an all−∞ row and subtracting it would evaluate exp(−∞ − (−∞)) = NaN, and
      // line 11 would then poison ℓ for the rest of the walk.
      const hi = Math.min(j1, i + 1);
      if (hi <= j0) {
        log.skippedRows++;
        skipped++;
        continue;
      }
      // A block-sparsity mask drops whole tiles that were not masked. Same skip, different reason —
      // and unlike the one above, this one changes the answer.
      if (keep && !keep(i, j0)) {
        log.dropped++;
        dropped++;
        continue;
      }
      const s = [];
      for (let j = j0; j < hi; j++) s.push(dot(Q[i], K[j]) / scale);
      const mt = Math.max(...s);
      const p = s.map((x) => Math.exp(x - mt));
      const lt = p.reduce((a, b) => a + b, 0);

      const mNew = Math.max(m[i], mt);
      // The first tile has m = −∞ with a finite new maximum: the old contribution is zero, and
      // exp(−∞ − finite) already gives that. Guarding it explicitly keeps −∞ − −∞ out of the sum.
      const a = m[i] === -Infinity ? 0 : Math.exp(m[i] - mNew);
      const b = Math.exp(mt - mNew);
      const lNew = a * l[i] + b * lt;

      for (let d = 0; d < dh; d++) {
        let acc = 0;
        for (let j = j0; j < hi; j++) acc += p[j - j0] * V[j][d];
        O[i][d] = (a * l[i] * O[i][d] + b * acc) / lNew;
      }
      if (mNew > m[i]) log.moved++;
      log.factor[i] = a;
      m[i] = mNew;
      l[i] = lNew;
      live++;
    }
    tiles.push(log);
  }
  return { out: O, m, l, tiles, skipped, dropped, live };
}

/** Elements moved to and from HBM, counted line by line off Algorithms 0 and 1. The paper gives
 *  only asymptotics; these constants are this card's, and the panel says so. */
const standardTraffic = (N, d) => 4 * N * N + 4 * N * d;
const flashTraffic = (N, d, Bc) => 2 * N * d + Math.ceil(N / Bc) * (3 * N * d + 4 * N);

const EPS = Math.pow(2, -52);

export function flashattentionCard(root, m) {
  let Bc = 4;
  let step = null; // key tiles processed
  let query = null;
  let head = 0;
  let sparse = false;
  let N = 4096;
  let d = 64;
  let M = 50000;

  root.appendChild(
    prose({
      problem:
        "Four mechanisms in a row answered the quadratic by changing what attention computes — a fixed sparsity pattern, a window, the exponential removed, the exponential estimated. Every one of them was justified by a count of multiplications, and several of them are not actually faster on a real machine. The paper's charge is that the count was of the wrong thing: a modern accelerator has a large slow memory and a tiny fast one, roughly ten times the bandwidth and a hundred thousand times smaller, and softmax is a reduction, which makes it limited by memory traffic rather than by arithmetic. Standard attention writes the whole score matrix out to the slow memory, reads it back, writes the softmaxed copy out, and reads that back.",
      mechanism:
        "Never write the score matrix at all. Load a block of keys and values into the fast memory, compute that block's scores against every query, and fold the result into a running output — keeping two extra numbers per query row, the largest score seen so far and the running total of the exponentials. When a later block contains a bigger score, the numbers accumulated against the old maximum are rescaled by the difference before the new block is added. The output stays normalised at every step, so only one copy of it ever exists. The backward pass stores none of the big intermediates either: it recomputes them from the blocks, which costs more arithmetic and less time, because the currency is bytes.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "the same sentence, one key tile at a time");

  // -------------------------------------------------------------- the tile walk
  const bcSelect = choice({
    label: "key tile width",
    value: "4",
    options: TILES.map((b) => ({ value: String(b), label: `${b} key${b > 1 ? "s" : ""} at a time` })),
    onchange: (v) => ((Bc = Number(v)), (step = null), render()),
  });
  const stepSlider = slider({
    label: "key tiles folded in",
    min: 0,
    max: 16,
    value: 4,
    oninput: (v) => ((step = v), render()),
  });
  const rowBars = barList({
    rows: Array.from({ length: 16 }, (_, i) => ({ key: "r" + i, label: "" })),
  });
  const walkRead = readout([
    { key: "tile", label: "this tile" },
    { key: "moved", label: "rows whose running maximum moved" },
    { key: "factor", label: "the rescale applied to your row" },
    { key: "skipped", label: "row and tile pairs skipped as fully masked" },
  ]);
  const walkNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the rescale — the only thing here that is a mechanism" }),
      el("div", { class: "formula", text: "a = exp(m_old − m_new)      b = exp(m_tile − m_new)" }),
      el("div", { class: "formula", text: "l_new = a·l_old + b·l_tile      O_new = ( a·l_old·O_old + b·(P_tile · V) ) / l_new" }),
      el("div", { class: "ctrls" }, [bcSelect, stepSlider.node]),
      rowBars.node,
      walkRead.node,
      walkNote,
    ])
  );

  // ------------------------------------------------------------ the identity
  const sparseToggle = toggle({
    label: "drop tiles, as the paper's own sparse variant does",
    value: false,
    onchange: (v) => ((sparse = v), render()),
  });
  const exactRead = readout([
    { key: "gap", label: "largest disagreement with the one-shot answer" },
    { key: "ulp", label: "measured in units in the last place" },
    { key: "across", label: "across every tile width" },
    { key: "s", label: "share of live tiles kept" },
  ]);
  const exactNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "exact — and exactly how exact" }),
      el("div", { class: "ctrls" }, [sparseToggle]),
      exactRead.node,
      exactNote,
    ])
  );

  // ------------------------------------------------------------- the traffic
  const nSlider = slider({
    label: "sequence length",
    min: 4,
    max: 16,
    value: 12,
    format: (v) => String(2 ** v),
    oninput: (v) => ((N = 2 ** v), render()),
  });
  const dSelect = choice({
    label: "head width",
    value: "64",
    options: [16, 32, 64, 128].map((x) => ({ value: String(x), label: String(x) })),
    onchange: (v) => ((d = Number(v)), render()),
  });
  const mSlider = slider({
    label: "fast memory, in elements",
    min: 5000,
    max: 100000,
    step: 5000,
    value: 50000,
    format: (v) => `${(v / 1000).toFixed(0)}k ≈ ${((v * 2) / 1024).toFixed(0)} KB`,
    oninput: (v) => ((M = v), render()),
  });
  const trafficRead = readout([
    { key: "std", label: "standard, elements moved" },
    { key: "flash", label: "this algorithm" },
    { key: "ratio", label: "ratio" },
    { key: "bc", label: "tile width the paper's rule gives" },
  ]);
  const trafficNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "where the cost actually moved" }),
      el("div", { class: "ctrls" }, [nSlider.node, dSelect, mSlider.node]),
      trafficRead.node,
      trafficNote,
    ])
  );

  // ------------------------------------------------- and on this sentence, it loses
  const honestRead = readout([
    { key: "here", label: "on your sentence, at this tile width" },
    { key: "even", label: "tile width where it breaks even" },
    { key: "real", label: "tile width a real accelerator would pick here" },
  ]);
  const honestNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "on a sentence this small the whole idea is a loss" }),
      honestRead.node,
      honestNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The same answer as standard attention — this is an identity over the reals, not an approximation, and the page measures the disagreement at about one unit in the last place",
        "Memory for the attention step becomes linear in the sequence rather than quadratic: up to 20× smaller than exact baselines, and still 2× smaller than the approximate ones",
        "Measured 2–4× on the attention operation, 15% end-to-end on BERT-large against the MLPerf record, 3.5× and 3.0× on GPT-2 small and medium against HuggingFace",
        "It beats the approximate methods on their own long-range benchmark while scoring higher than all of them — 2.4× and 59.8 average against their 1.3–2.5× and 54.9–59.6",
        "Longer contexts become affordable: GPT-2 at 4K is still 30% faster than Megatron at 1K, with 0.7 better perplexity",
      ],
      givesUp: [
        "Nothing about the answer — but also nothing about the arithmetic: attention is still quadratic in multiplications after this, and the key/value cache is untouched",
        "Generality. Every variant needs its own hand-written kernel, and the authors say implementations may not transfer across accelerator generations",
        "The advantage is a property of one hardware ratio: on a slower-memory card the win is larger, on one with less fast memory it is smaller, and the paper measures both",
        "Against the best existing fused kernel at short sequences it is 4% slower, because the backward pass recomputes what that kernel stored",
        "The block-sparse variant in the same paper is approximate, is not covered by the exactness claim, and scores worse than the dense version on the benchmark it was built for",
      ],
      chooseWhen:
        "Always, for exact attention on a memory-bound accelerator — which is why it stopped being a choice and became the default implementation. Its real legacy is the bar it set: after this, an approximation has to beat an exact algorithm that is several times faster than the thing it was originally benchmarked against.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It gives exactly the same answer as before — nothing about the model changes, so there is no quality to trade away",
        "It is several times faster in practice and uses far less memory, which is what made much longer inputs affordable at all",
        "It works by never writing the big intermediate table down: the work is done in small pieces inside the chip's fast scratch memory",
      ],
      cons: [
        "It does not reduce the amount of arithmetic at all — the cost that grows with the square of the length is still there, just paid in a cheaper currency",
        "The gain depends entirely on the particular machine's memory being slow relative to its scratch space; on a different chip the numbers move, and on very short inputs it can be slower",
        "Every variation of it has to be written by hand at a very low level, which is why unusual attention patterns became expensive to try",
        "The sparse version described in the same paper is not exact, and is worse than the plain version on the very task it was designed for",
      ],
      verdict:
        "Everything before this made attention cheaper by changing what it computes. This one changes nothing about the answer and moves the cost anyway, by noticing that the expensive part was never the multiplying — it was carrying a huge table back and forth between fast and slow memory. It is the rare improvement with no trade-off in quality at all, and the price is paid somewhere else entirely: the mechanism now lives in hand-written machine code, and whatever a hand-written kernel cannot express quietly stops being an option.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    const Tc = Math.ceil(T / Bc);
    if (step === null || step > Tc) step = Tc;
    stepSlider.input.max = Tc;
    stepSlider.set(step);
    if (query === null || query >= T) query = T - 1;

    const res = forward(tokens);
    const h = res.trace[0].heads[head];
    // A band of local tiles plus the first tile — the shape a block-sparse kernel can actually
    // skip, and the same "keep the beginning, keep what is near" pattern the sparse cards used.
    const keep = sparse ? (i, j0) => j0 === 0 || i - j0 < 2 * Bc : null;
    // The walk and the row display are always dense — the sparsity toggle belongs to the panel
    // below, where its cost is the thing being measured.
    const walk = tiledAttention(h.Q, h.K, h.V, DH, Bc, step);
    const full = tiledAttention(h.Q, h.K, h.V, DH, Bc, Infinity, keep);
    const seen = Math.min(T, step * Bc);

    // --- the picture, with the columns not yet folded in greyed out
    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: walk.out,
      top: res.top,
      query,
      opts: { readable: (i, j) => j < seen },
    });
    flowNote.innerHTML = `Nothing in this picture is new — that is the point of the card. The same projections, the same scores, the same output. What the control below changes is only <em>when</em> each column of the matrix is looked at: ${
      step === 0
        ? "no keys have been folded in yet, so the output is still zero."
        : `the first ${seen} key${seen > 1 ? "s" : ""} have been folded into the running output, and the rest have not been touched.`
    } Step it to the end and the output vector is bit-for-bit what the one-shot pass produced, to within the last decimal place — measured two panels down. The greyed marks here are not pairs the mechanism cannot read, as they are on the sparse cards; they are pairs it has not read <em>yet</em>.`;

    // --- the per-row state
    const cur = walk.tiles[step - 1];
    const maxL = Math.max(...walk.l.filter(Number.isFinite), 1e-9);
    const bars = {};
    for (let i = 0; i < 16; i++) {
      if (i >= T) {
        bars["r" + i] = { value: 0, of: 1, text: "", label: "" };
        continue;
      }
      const changed = cur && cur.factor[i] !== null && cur.before[i] !== walk.m[i];
      bars["r" + i] = {
        value: walk.l[i],
        of: maxL,
        text: Number.isFinite(walk.m[i]) ? `ℓ ${fmt(walk.l[i], 2)}` : "—",
        label: Number.isFinite(walk.m[i])
          ? `${tokens[i].word} · m ${fmt(walk.m[i], 2)}${changed ? " ←" : ""}`
          : `${tokens[i].word} · not reached`,
        tone: i === query ? "" : "alt",
      };
    }
    rowBars.update(bars);

    const f = cur ? cur.factor[query] : null;
    walkRead.update({
      tile: step === 0 ? "none yet" : `keys ${cur.j0 + 1} to ${cur.j1}`,
      moved: step === 0 ? "—" : `${cur.moved} of ${T}`,
      factor: f === null || f === undefined ? "row not in this tile" : f === 0 ? "first tile — nothing to rescale" : fmt(f, 4),
      skipped: `${full.skipped} of ${T * Tc}`,
    });
    walkNote.className = "note";
    walkNote.innerHTML = `Each row keeps two numbers: the largest score it has seen and the running total of the exponentials measured against that maximum. When a later tile contains a bigger score, everything accumulated so far is measured against the wrong reference — so it is multiplied by <strong>${
      f === null || f === undefined || f === 0 ? "the rescale factor" : fmt(f, 4)
    }</strong>, which is always at most one, and shrinks the old total to fit the new maximum. That single multiplication <em>is</em> the mechanism; everything else on this card is bookkeeping around it. Watch the count of rows whose maximum moved: it starts high and falls towards zero as the walk proceeds, because a running maximum stabilises. And ${full.skipped} of the ${
      T * Tc
    } row-and-tile pairs are skipped outright — under a causal mask a whole row of a tile can be entirely masked, and asking for its maximum would subtract minus infinity from minus infinity and poison that row for the rest of the pass. The real kernel skips them; so does this. That skip is also the entire machinery of the paper's block-sparse variant, which skips tiles for a different reason.`;

    // --- the identity
    const gapOf = (r) => {
      let g = 0;
      for (let i = 0; i < T; i++) for (let x = 0; x < DH; x++) g = Math.max(g, Math.abs(r.out[i][x] - h.out[i][x]));
      return g;
    };
    const gaps = TILES.map((b) => [b, gapOf(tiledAttention(h.Q, h.K, h.V, DH, b, Infinity, keep))]);
    const here = gaps.find(([b]) => b === Bc)[1];
    let mag = 0;
    for (const row of h.out) for (const x of row) mag = Math.max(mag, Math.abs(x));
    const kept = full.live / (full.live + full.dropped);
    exactRead.update({
      gap: here.toExponential(2),
      ulp: sparse ? `${(here / (EPS * mag)).toExponential(1)} ulp` : `${(here / (EPS * mag)).toFixed(1)} ulp`,
      across: gaps.map(([b, g]) => `${b}:${g.toExponential(1)}`).join("  "),
      s: sparse ? `${(kept * 100).toFixed(0)}% — ${full.dropped} tiles dropped` : "all of them",
    });
    exactNote.className = "note " + (sparse ? "warn" : "good");
    exactNote.innerHTML = sparse
      ? `Now the same measurement with tiles dropped: the disagreement is <strong>${here.toExponential(
          2
        )}</strong>, up from about 10⁻¹⁶, a jump of some fourteen orders of magnitude — and it is not a rounding artefact, it is a different answer. This is the paper's own block-sparse variant, and §3.3 opens by saying so in as many words: it extends the method <em>to approximate attention</em>. The exactness claim, the optimality claim and the theorem all cover the dense algorithm only. Worth noticing that the pattern here has the shape it does for a machine reason rather than a modelling one — a tiled kernel can skip whole tiles and nothing finer, which is why every sparse design after this one is block-structured. And the paper's own benchmark has the sparse version scoring <em>worse</em> than the dense one on the task it was built for: 56.0 against 61.4. Turn it off to get the exactness back.`
      : `The tiled walk and the one-shot pass disagree by <strong>${here.toExponential(
          2
        )}</strong> — and that is not zero, and it will never be zero. Rescaling changes the <em>order</em> in which the terms are added, and floating-point addition is not associative, so a different order gives a different last bit. Measured against this machine's resolution it is about one unit in the last place, at every tile width. “Exact” in the paper's title is a statement about algebra: the identity holds over the real numbers, and what is left is the arithmetic's own floor. Set that next to the previous concept's estimate, whose error on this same measure sits between 0.007 and 0.99 depending on how large the query and key vectors are, and falls only as the square root of the work spent. <strong>One is about 10⁻¹⁶ and does not depend on the tile width at all; the other is between 10⁻² and 10⁰ and is the price of the speed.</strong> That comparison is this paper's whole argument.`;

    // --- the traffic
    nSlider.set(Math.log2(N));
    mSlider.set(M);
    const bcReal = Math.ceil(M / (4 * d));
    const std = standardTraffic(N, d);
    const fl = flashTraffic(N, d, bcReal);
    trafficRead.update({
      std: std.toLocaleString(),
      flash: fl.toLocaleString(),
      ratio: `${(std / fl).toFixed(2)}×`,
      bc: `${bcReal.toLocaleString()} keys, ${Math.ceil(N / bcReal)} pass${Math.ceil(N / bcReal) > 1 ? "es" : ""} over the queries`,
    });
    trafficNote.className = "note";
    trafficNote.innerHTML = `This counts <strong>elements moved, not time</strong> — the page has no fast memory to save a trip to, and any stopwatch here would be measuring the browser. The paper's claim is that the first number causes the second; this panel can only show the first, and the constants are counted off the two algorithms line by line rather than taken from the paper, which gives only the shape. Two things are worth moving the sliders for. The ratio <em>rises with the sequence length</em>, because the term that grows with the square is the one being removed. And it falls as the fast memory shrinks: the tile width is that memory divided by four times the head width, so a smaller scratchpad means more passes over the queries. The authors measured exactly that — a card with less fast memory showed less speedup, and one with slower main memory showed more. For scale, their own measurement on a real machine was 40.3 GB down to 4.4 GB, a factor of 9.2, with the runtime falling from 41.7 ms to 7.3 ms; this counter is not trying to reproduce that number, only its shape.`;

    // --- the honesty panel
    const hereStd = standardTraffic(T, DH);
    const hereFl = flashTraffic(T, DH, Bc);
    const evenBc = 0.75 * DH;
    const realBc = Math.ceil(M / (4 * DH));
    honestRead.update({
      here: `${hereFl.toLocaleString()} against ${hereStd.toLocaleString()}`,
      even: `about ${evenBc} keys`,
      real: `${realBc.toLocaleString()} keys — ${Math.round(realBc / T)}× the whole sentence`,
    });
    honestNote.className = "note " + (hereFl > hereStd ? "warn" : "");
    honestNote.innerHTML = `Run the same count on the sentence at the top of this page — ${T} tokens, ${DH} numbers per head — and at a tile width of ${Bc} the algorithm moves ${hereFl.toLocaleString()} elements against standard attention's ${hereStd.toLocaleString()}: ${
      hereFl > hereStd
        ? `<strong>${(hereFl / hereStd).toFixed(1)}× worse.</strong>`
        : `${(hereStd / hereFl).toFixed(2)}× better — but only because the tile is wide enough that there is barely any tiling left.`
    } The queries are re-read once per tile, and here the queries are ${T * DH} numbers against a score matrix of ${
      T * T
    } — not enough of a difference to pay for ${Math.ceil(T / Bc)} passes. Break-even is where four times the square of the length equals three times the same square times the head width over the tile width, which puts it at about three quarters of the head width: ${evenBc} keys. And with a real scratchpad the paper's own rule would choose a tile of ${realBc.toLocaleString()} keys — so it would not tile this sentence at all. <strong>The reader is being shown a mechanism their own model does not need.</strong> That is not a flaw in the mechanism; it is what it means for a cost to be a hardware cost. If the whole thing fits in fast memory there is nothing to arrange.`;
  }

  return { update: render, unmount: () => {} };
}
