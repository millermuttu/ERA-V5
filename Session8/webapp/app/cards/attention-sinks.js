// Concept 19 — Attention sinks / StreamingLLM.
// Built from docs/research/attention-sinks.md. The seam needed nothing: `readable(i, j)` is the
// whole cache policy in one predicate. What the research settled, and what shapes every panel:
// the collapse is a step at the exact token where the first entry is evicted, not a decay; its
// cause is arithmetic — delete a term that dominated the softmax denominator and every surviving
// weight is multiplied by the same number, which this app reproduces to five decimals; the length
// generalisation comes from renumbering positions by the cache rather than by the text, which is
// the half everyone drops; and the sink itself is a *learned* artefact, so this untrained model
// does not have one. That last fact is not an obstacle to the card, it is the card's most useful
// panel: the recipe measurably does not help here, which is the evidence that the phenomenon is
// trained rather than architectural. Nothing is faked, and where the model cannot show something
// the reader is handed the paper's number with its source attached.
import { el, slider, choice, toggle } from "../lib/dom.js";
import { readout, barList } from "../views/bars.js";
import { fmt, dot, softmax } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { rope, alibiSlopes } from "../model/position.js";
import { toyCost } from "../model/cost.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

/** The whole method, as a predicate: keep the first `sinks`, keep the last `recent`, drop the rest. */
const cachePolicy = (sinks, recent) => (i, j) => j < sinks || j >= i - recent + 1;

const R = rope({ dims: DH });
const SLOPES = alibiSlopes(CONFIG.HEADS);

const klDiv = (p, q) => {
  let d = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) d += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12));
  return d;
};

const entropyRank = (row, j) => {
  let rank = 1;
  for (let k = 0; k < row.length; k++) if (row[k] > row[j]) rank++;
  return rank;
};

export function attentionSinksCard(root, m) {
  let sinks = 4;
  let recent = 8;
  let where = "1:0"; // block:head — block 1 head 0 is the one head here with visible mass on token 0
  let byCache = true;
  let scheme = "rope";
  let boost = 0;
  let softmax1 = false;
  let budget = 12;

  root.appendChild(
    prose({
      problem:
        "Concept 8 bounded the comparisons and left the memory bill untouched: a window makes each query read fewer keys, but generation still has to hold every key and value it has produced. The obvious repair is to stop holding them — evict the oldest entry each time a new one arrives, and the cache stops growing. It does not work, and it fails in a way that looks like a bug rather than a trade-off. Perplexity does not drift upward as the window slides; it detonates, at the exact token where the first entry leaves the cache. On a 13-billion-parameter model the paper measures 5.40 against 5158.07 for the same cache size — a factor of 955 for four entries.",
      mechanism:
        "The cause is the softmax, and the fix follows from it. A softmax has no output for “attend to nothing”: the weights must sum to one even when the query wants none of what it can see, so the surplus has to be dumped somewhere. Training picks the one position every query can always see — the first — and heads learn to park their unwanted mass there. That parked mass is a large term in the denominator, so deleting it does not remove information, it multiplies every surviving weight by the same number and moves the whole distribution off the scale the model was trained on. So keep four entries at the front forever, roll the rest, and — the half most summaries drop — number the positions by where a token sits in the cache rather than where it sat in the text, so the arithmetic never leaves the range the model was trained on.",
    })
  );

  root.appendChild(
    el("div", { class: "formula", text: "softmax(x)_i = e^{x_i} / ( e^{x_1} + Σ_{j≥2} e^{x_j} )      with  x_1 ≫ x_j        cache = x sinks + y recent" })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------- 1. evict the first token
  const sinkSlider = slider({
    label: "entries kept at the front",
    min: 0,
    max: 4,
    step: 1,
    value: sinks,
    format: (v) => `${v}`,
    oninput: (v) => ((sinks = v), render()),
  });
  const recentSlider = slider({
    label: "rolling window",
    min: 2,
    max: 16,
    step: 1,
    value: recent,
    format: (v) => `${v} tokens`,
    oninput: (v) => ((recent = v), render()),
  });
  const wherePick = choice({
    label: "head",
    options: Array.from({ length: CONFIG.BLOCKS * CONFIG.HEADS }, (_, n) => {
      const b = Math.floor(n / CONFIG.HEADS);
      const h = n % CONFIG.HEADS;
      return { value: `${b}:${h}`, label: `block ${b}, head ${h}` };
    }),
    value: where,
    onchange: (v) => ((where = v), render()),
  });
  // Both bar lists are one row per token, so they are built on first render and rebuilt only when
  // the sentence changes length.
  let evictBars = null;
  let evictLen = 0;
  const evictHost = el("div", {});
  const evictRead = readout([
    { key: "lost", label: "mass this head had on what the cache drops" },
    { key: "kept", label: "share of the denominator that survives" },
    { key: "mult", label: "what every surviving weight is multiplied by" },
    { key: "check", label: "check it by hand" },
  ]);
  const evictNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "evict the first token and read the denominator" }),
      el("div", { class: "ctrls" }, [sinkSlider.node, recentSlider.node, wherePick]),
      evictHost,
      evictRead.node,
      evictNote,
    ])
  );

  // ------------------------------------------------- 2. the four ways to run it
  const fourBars = barList({
    rows: [
      { key: "dense", label: "(a) keep everything" },
      { key: "window", label: "(b) rolling window only", alt: true },
      { key: "recompute", label: "(c) window, rebuilt each token" },
      { key: "stream", label: "(d) sinks + rolling window" },
    ],
  });
  const fourRead = readout([
    { key: "bEntries", label: "(b) entries held" },
    { key: "dEntries", label: "(d) entries held" },
    { key: "gap", label: "quoted: the perplexity between them" },
    { key: "grows", label: "does the cache grow with the text?" },
  ]);
  const fourNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "four ways to run a model past its window" }),
      fourBars.node,
      fourRead.node,
      fourNote,
    ])
  );

  // --------------------------------- 3. positions in the cache, not in the text
  const numberToggle = toggle({
    label: "number positions by the cache",
    value: true,
    onchange: (v) => ((byCache = v), render()),
  });
  const schemePick = choice({
    label: "position scheme",
    options: [
      { value: "rope", label: "rotation (RoPE)" },
      { value: "alibi", label: "distance penalty (ALiBi)" },
    ],
    value: scheme,
    onchange: (v) => ((scheme = v), render()),
  });
  const stripText = el("div", { class: "formula" });
  const stripCache = el("div", { class: "formula" });
  let posBars = null;
  let posKey = "";
  const posHost = el("div", {});
  const posRead = readout([
    { key: "distText", label: "distance from the query to the sink, by the text" },
    { key: "distCache", label: "the same, by the cache" },
    { key: "tv", label: "how far apart the two attention rows are" },
    { key: "winner", label: "which key wins under each" },
  ]);
  const posNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the positions are the cache's, not the text's" }),
      el("div", { class: "ctrls" }, [numberToggle, schemePick]),
      stripText,
      stripCache,
      posHost,
      posRead.node,
      posNote,
    ])
  );

  // ------------------------------------------ 4. what this page cannot show
  const budgetSlider = slider({
    label: "cache budget",
    min: 4,
    max: 16,
    step: 1,
    value: budget,
    format: (v) => `${v} entries`,
    oninput: (v) => ((budget = v), render()),
  });
  const klBars = barList({
    rows: [0, 1, 2, 4].map((x) => ({ key: `k${x}`, label: `${x} at the front` })),
  });
  const sinkRead = readout([
    { key: "mass", label: "mean weight the first token receives here" },
    { key: "uniform", label: "what an even split would give it" },
    { key: "rank", label: "its mean rank among the keys" },
    { key: "best", label: "which split is closest to full attention" },
  ]);
  const sinkNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "this model has no sink, and that is the measurement" }),
      el("div", { class: "ctrls" }, [budgetSlider.node]),
      sinkRead.node,
      klBars.node,
      sinkNote,
    ])
  );

  // ------------------------------------------ 5. install a sink by hand
  const boostSlider = slider({
    label: "extra logit on the first token",
    min: 0,
    max: 10,
    step: 0.1,
    value: 0,
    format: (v) => `+${v.toFixed(1)}`,
    oninput: (v) => ((boost = v), render()),
  });
  const smToggle = toggle({
    label: "softmax with a 1 in the denominator",
    value: false,
    onchange: (v) => ((softmax1 = v), render()),
  });
  const boostRead = readout([
    { key: "w0", label: "weight the first token now receives" },
    { key: "share", label: "share of the denominator it holds" },
    { key: "needed", label: "extra logit that would give it half the row" },
    { key: "verdict", label: "how much worse keeping nothing at the front is" },
  ]);
  const boostNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "install a sink, and the collapse appears" }),
      el("div", { class: "ctrls" }, [boostSlider.node, smToggle]),
      boostRead.node,
      boostNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The cache stops growing: a constant x + y entries however long the stream runs, and models from 2.9B to 70B are reported streaming 4 million tokens with no degradation in fluency",
        "Nothing is trained, fine-tuned or replaced — the method is a rule about which cache entries to keep, and it works on any model with a relative position scheme",
        "Against the only baseline with acceptable quality, rebuilding the window's cache every token, it is up to 22.2× faster per token, and the gap widens with cache size because that baseline is quadratic where this is linear",
        "The four kept entries are positional, not semantic: replacing them with four line breaks recovers 5.60 against 5.40, so no particular tokens have to be chosen or understood",
        "Streaming question-answering goes from effectively zero under a plain window — 3.58% on a 7B model — to 71.34%, matching a one-shot baseline that gets the sample handed to it fresh",
      ],
      givesUp: [
        "It does not extend the context, and the paper eventually says so outright — accuracy at retrieving an answer drops to exactly zero, not gradually, the moment the answer falls out of the cache. That sentence arrived in the appendices two months after the first version, and the explicit disclaimer in the introduction six months after that",
        "Everything between the sinks and the rolling window is discarded permanently, so on a long-document benchmark the published configuration loses on five tasks of six against simply truncating the middle",
        "The recipe is written from indices alone: nothing is scored, nothing looks at what an entry contains, so it fails exactly where content would have mattered",
        "Four sinks is a safe default, not a measured optimum — on three of the four models tested a single entry recovers everything, and the 1000× collapse is the worst case, not the typical one, which is nearer 1.5×",
        "With a rotation, keys must be cached before the rotation is applied and re-rotated at every step, because their positions change as the cache rolls — a real implementation cost that the phrase “no retraining” tends to hide",
        "The trained remedies are more brittle, not less: a model pre-trained with a dedicated sink token is catastrophically worse than a vanilla one if that token is ever evicted, which the paper reports and never discusses",
      ],
      chooseWhen:
        "A long-running conversation or stream where the model must stay fluent indefinitely and nothing far back needs to be retrieved — a chat assistant that should not have to be restarted, a live transcription, an agent loop. Not for anything that has to answer from the middle of a long document: for that the entry has already been dropped, and the score is not degraded but zero.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The memory a running model needs stops growing, so it can keep going for millions of words without slowing down or filling up",
        "Nothing has to be retrained — it is a rule about which few pieces of the model's short-term memory to hold on to",
        "The pieces it holds on to do not have to be the right ones or mean anything; any four from the start of the text will do",
        "Against the only other way of getting the same quality, it is up to twenty-two times faster per word",
      ],
      cons: [
        "It does not let the model remember more. Anything that falls out of the window is gone, and questions about it are not answered badly — they are not answered at all",
        "The middle of a long document is thrown away permanently, so on tasks that need it, plain truncation does better",
        "Which pieces to keep is decided purely by where they sit, never by what they say",
        "The size of the problem it fixes varies enormously between models, and the famous thousand-fold figure is the worst case rather than the usual one",
      ],
      verdict:
        "The discovery that the obvious way to bound a growing memory — throw the oldest away — breaks the model outright, and that the reason is plumbing rather than meaning: a softmax must always add up to one, so every head needs somewhere to dump the attention it does not want, and it learns to dump it on the first word. Keep those few slots, renumber everything else as though the gap were not there, and a model runs forever. What it emphatically does not do is remember more, and the sentence saying so took six months to reach the front of the paper.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    sinkSlider.set(sinks);
    recentSlider.set(recent);
    budgetSlider.set(budget);
    boostSlider.set(boost);
    const [bi, hi] = where.split(":").map(Number);

    const sinkBias = boost > 0 ? (i, j) => (j === 0 ? boost : 0) : null;
    const mech = (x, y) => ({ mixer: softmaxMixer({ readable: cachePolicy(x, y), bias: sinkBias }) });
    const full = forward(tokens, sinkBias ? { mixer: softmaxMixer({ bias: sinkBias }) } : {});
    const streamed = forward(tokens, mech(sinks, recent));
    const windowed = forward(tokens, mech(0, recent));

    const fullHead = full.trace[bi].heads[hi];
    const streamHead = streamed.trace[bi].heads[hi];
    const q = T - 1;
    const kept = [];
    for (let j = 0; j <= q; j++) if (cachePolicy(sinks, recent)(q, j)) kept.push(j);

    flow.update({
      tokens,
      head: { ...streamed.trace[0].heads[0], emb: streamed.trace[0].input },
      weights: streamed.trace[0].heads[0].weights,
      out: streamed.trace[0].heads[0].out,
      top: streamed.top,
      query: q,
      opts: { qkvBadge: `cache: ${sinks} + ${recent}` },
    });
    flowNote.innerHTML = `The apparatus is unchanged — this concept adds no operation at all. What it changes is which keys exist to be read: the cache holds <strong>${sinks} entries at the front and the last ${recent}</strong>, and everything between them has been deleted and cannot come back. On your ${T}-token sentence that is ${
      kept.length
    } of ${q + 1} keys available to the last query. In a real stream the second number keeps growing and the first does not.`;

    // --- 1. the denominator
    const scoreRow = fullHead.scores[q];
    const expRow = scoreRow.map((x) => (Number.isFinite(x) ? Math.exp(x) : 0));
    const Z = expRow.reduce((a, b) => a + b, 0);
    const Zkept = expRow.reduce((a, b, j) => a + (kept.includes(j) ? b : 0), 0);
    const mult = Z / Zkept;
    const fullRow = fullHead.weights[q];
    const cachedRow = streamHead.weights[q];
    let lost = 0;
    for (let j = 0; j <= q; j++) if (!kept.includes(j)) lost += fullRow[j];

    if (evictLen !== T) {
      evictBars = barList({ rows: tokens.map((t, i) => ({ key: `e${i}`, label: `${i}  ${t.word}` })) });
      evictLen = T;
      evictHost.replaceChildren(evictBars.node);
    } else {
      evictHost.querySelectorAll(".bar-name").forEach((n, i) => (n.textContent = `${i}  ${tokens[i].word}`));
    }
    const evictVals = {};
    const rowMax = Math.max(...cachedRow, ...fullRow.slice(0, q + 1));
    for (let j = 0; j <= q; j++) {
      const gone = !kept.includes(j);
      evictVals[`e${j}`] = {
        value: gone ? 0 : cachedRow[j],
        of: rowMax,
        text: gone ? `dropped · was ${fmt(fullRow[j], 3)}` : `${fmt(cachedRow[j], 3)} ← ${fmt(fullRow[j], 3)}`,
        tone: gone ? "alt" : "",
      };
    }
    for (let j = q + 1; j < T; j++) evictVals[`e${j}`] = { value: 0, of: 1, text: "after the query", tone: "alt" };
    evictBars.update(evictVals);

    // The identity that is the whole collapse: pick the largest surviving weight and check it.
    let probe = kept[0];
    for (const j of kept) if (fullRow[j] > fullRow[probe]) probe = j;
    evictRead.update({
      lost: fmt(lost, 3),
      kept: `${fmt((100 * Zkept) / Z, 1)}%`,
      mult: `×${fmt(mult, 4)}`,
      check: `${fmt(fullRow[probe], 5)} × ${fmt(mult, 4)} = ${fmt(fullRow[probe] * mult, 5)}`,
    });
    evictNote.className = "note " + (lost > 0.5 ? "warn" : "");
    evictNote.innerHTML = `This head had <strong>${fmt(
      lost,
      3
    )}</strong> of its attention on keys the cache has dropped. Deleting them does not delete information evenly — it removes their terms from the softmax denominator, which falls to <strong>${fmt(
      (100 * Zkept) / Z,
      1
    )}%</strong> of what it was, so <em>every surviving weight is multiplied by the same number</em>, ${fmt(
      mult,
      4
    )}. The last readout is that identity done by hand on this row's largest survivor, and you can check it against the bar: the shape of what remains is untouched, the scale is not. The paper's sentence is <em>“removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function.”</em> <strong>On a trained model the term removed is much bigger than anything here</strong> — the first token alone <em>“often exceeding half of the total attention”</em> from the 4,096th token, in every layer above the second — so there the multiplier is about 2 on <em>every</em> weight in <em>every</em> head, and the model has never seen a distribution on that scale.`;

    // --- 2. the four ways
    const costFull = toyCost(full);
    const costWindow = toyCost(windowed);
    const costStream = toyCost(streamed);
    const held = { dense: T, window: Math.min(T, recent), recompute: Math.min(T, recent), stream: Math.min(T, sinks + recent) };
    const ofReads = Math.max(costFull.readsPerQuery, costStream.readsPerQuery, 1);
    fourBars.update({
      dense: {
        value: costFull.readsPerQuery,
        of: ofReads,
        text: `${fmt(costFull.readsPerQuery, 2)} reads · ${held.dense} held · PPL 5641`,
      },
      window: {
        value: costWindow.readsPerQuery,
        of: ofReads,
        text: `${fmt(costWindow.readsPerQuery, 2)} reads · ${held.window} held · PPL 5158`,
      },
      recompute: {
        value: costWindow.readsPerQuery,
        of: ofReads,
        text: `${fmt(costWindow.readsPerQuery, 2)} reads · rebuilt every token · PPL 5.43`,
      },
      stream: {
        value: costStream.readsPerQuery,
        of: ofReads,
        text: `${fmt(costStream.readsPerQuery, 2)} reads · ${held.stream} held · PPL 5.40`,
      },
    });
    fourRead.update({
      bEntries: `${held.window}`,
      dEntries: `${held.stream}`,
      gap: "5158.07 → 5.40",
      grows: "only (a)",
    });
    fourNote.className = "note";
    fourNote.innerHTML = `The read counts and the entry counts are this app's, on your sentence; the perplexities are quoted from the paper's Figure 1 — Llama-2-13B on the first book of PG-19, 65,000 tokens. Read rows (b) and (d) together and the card is done: <strong>same complexity, same kind of cache, ${
      held.stream - held.window
    } more entries, and a factor of 955 in perplexity</strong>. Row (a) is the one whose memory grows without limit; row (c) is the only baseline with acceptable quality and it pays for it by rebuilding the window's keys and values on every single token — quadratic where the others are linear, which is what the reported 22.2× speed-up is measured against. Not against (a), and not against (b). At a ${T}-token sentence the saving in the second column looks trivial; the point is that it is <em>constant in the length of the text</em>, and (a)'s is not.`;

    // --- 3. positions by the cache
    const cacheIndex = new Map(kept.map((j, k) => [j, k]));
    const rowUnder = (useCache) => {
      const posOf = (j) => (useCache ? cacheIndex.get(j) : j);
      const qPos = useCache ? kept.length : q;
      const qv = scheme === "rope" ? R.rotate(fullHead.Q[q], qPos) : fullHead.Q[q];
      const sc = kept.map((j) => {
        const kv = scheme === "rope" ? R.rotate(fullHead.K[j], posOf(j)) : fullHead.K[j];
        const base = dot(qv, kv) / Math.sqrt(DH);
        return scheme === "alibi" ? base - (qPos - posOf(j)) * SLOPES[hi] : base;
      });
      return softmax(sc);
    };
    const textRow = rowUnder(false);
    const cacheRow = rowUnder(true);
    let tv = 0;
    for (let k = 0; k < kept.length; k++) tv += Math.abs(textRow[k] - cacheRow[k]);
    tv /= 2;
    const shown = byCache ? cacheRow : textRow;
    const other = byCache ? textRow : cacheRow;
    stripText.textContent = `kept from the text   [${kept.join(", ")}]  ← the query sits at ${q}`;
    stripCache.textContent = `given to the model   [${kept.map((_, k) => k).join(", ")}]  ← the query sits at ${kept.length}`;

    const wantKey = kept.join(",") + "|" + tokens.map((t) => t.word).join(",");
    if (posKey !== wantKey) {
      posBars = barList({ rows: kept.map((j) => ({ key: `c${j}`, label: `${j}  ${tokens[j].word}` })) });
      posKey = wantKey;
      posHost.replaceChildren(posBars.node);
    }
    const posVals = {};
    const posMax = Math.max(...shown);
    kept.forEach((j, k) => {
      posVals[`c${j}`] = {
        value: shown[k],
        of: posMax,
        text: `${fmt(shown[k], 3)}  (${fmt(other[k], 3)} the other way)`,
        tone: shown[k] > other[k] ? "" : "alt",
      };
    });
    posBars.update(posVals);
    const argmax = (a) => a.indexOf(Math.max(...a));
    posRead.update({
      distText: `${q - 0}`,
      distCache: `${kept.length}`,
      tv: fmt(tv, 3),
      winner: `${tokens[kept[argmax(textRow)]].word} / ${tokens[kept[argmax(cacheRow)]].word}`,
    });
    posNote.className = "note " + (tv > 0.3 ? "warn" : "");
    posNote.innerHTML = `Both rows above are over <em>the same ${
      kept.length
    } key vectors</em>. The only difference is which integer the position scheme is handed: the token's index in the text, or its index in the cache. The paper's own example is <em>“if the current cache has tokens [0, 1, 2, 3, 6, 7, 8] and is in the process of decoding the 9th token, the positions assigned are [0, 1, 2, 3, 4, 5, 6, 7], rather than the positions in the original text”</em> — the gap is closed, and the sink sits at distance ${
      kept.length
    } from the query rather than ${q}. Here the two distributions differ by <strong>${fmt(
      tv,
      3
    )}</strong> in total variation and the key that wins is <strong>${
      tokens[kept[argmax(textRow)]].word
    }</strong> one way and <strong>${
      tokens[kept[argmax(cacheRow)]].word
    }</strong> the other. <strong>This, not the sinks, is where running past the training length comes from</strong>: a position index of four million is nowhere in the training distribution, an index of one thousand is at home. The paper's verdict — <em>“this method of assigning positional embedding within the cache is crucial to StreamingLLM's functionality.”</em> ${
      scheme === "rope"
        ? `And it has a price the phrase “no retraining” hides: a rotation is baked into the key, so the cache must store keys <em>“prior to introducing the rotary transformation”</em> and re-rotate every one of them at every step, because their cache positions move as the window rolls.`
        : `With a distance penalty the same renumbering is easier — the bias is computed from an index at read time, so you simply hand it the cache index. The paper's phrase for the wrong version is a <em>“jumping”</em> bias: the one you get by using the text's distances across the gap.`
    }`;

    // --- 4. no sink here
    let mass = 0;
    let rank = 0;
    let n = 0;
    for (const blk of full.trace)
      for (const head of blk.heads)
        for (let i = 4; i < T; i++) {
          mass += head.weights[i][0];
          rank += entropyRank(head.weights[i].slice(0, i + 1), 0);
          n++;
        }
    mass /= n;
    rank /= n;
    let evenSplit = 0;
    for (let i = 4; i < T; i++) evenSplit += 1 / (i + 1);
    evenSplit /= Math.max(1, T - 4);

    const refProbs = full.probs;
    const kls = {};
    let best = null;
    for (const x of [0, 1, 2, 4]) {
      const y = Math.max(1, budget - x);
      const kl = klDiv(refProbs, forward(tokens, mech(x, y)).probs);
      kls[x] = kl;
      if (best === null || kl < kls[best]) best = x;
    }
    const klMax = Math.max(...Object.values(kls), 1e-6);
    klBars.update(
      Object.fromEntries(
        [0, 1, 2, 4].map((x) => [
          `k${x}`,
          {
            value: kls[x],
            of: klMax,
            text: `${x} + ${Math.max(1, budget - x)}  ·  ${fmt(kls[x], 3)}`,
            tone: x === best ? "" : "alt",
          },
        ])
      )
    );
    sinkRead.update({
      mass: fmt(mass, 3),
      uniform: fmt(evenSplit, 3),
      rank: fmt(rank, 2),
      best: `${best} at the front`,
    });
    sinkNote.className = "note warn";
    sinkNote.innerHTML = `<strong>A sink is something training builds, and nothing here has been trained.</strong> Averaged over every head, block and query on your sentence, the first token receives <strong>${fmt(
      mass,
      3
    )}</strong> of the attention against <strong>${fmt(
      evenSplit,
      3
    )}</strong> for an even split, and its mean rank among the keys a query may read is ${fmt(
      rank,
      2
    )}. It is not a sink; this model is confidently peaky about arbitrary middles of the sentence, because random projections produce confident nonsense rather than structure. The bars are the consequence, and they are the honest inverse of the paper's own ablation: at an <em>equal</em> cache budget of ${budget} entries, spending some of it on the front against spending all of it on recency produces <strong>no ordering at all</strong> — the closest to full attention here is ${best} at the front, and moving the budget around changes which one wins. The paper's Table 2 has a clean ordering on four real models, from 3359.95 at zero to 9.59 at four. That its table and these bars disagree is the whole point: <strong>the sink is a learned artefact, not a property of attention</strong>, and a page that pretended otherwise with a suggestive picture would be lying. The next panel installs one by hand instead.`;

    // --- 5. install a sink
    const boostedRow = full.trace[bi].heads[hi].weights[q];
    const w0 = boostedRow[0];
    const share = expRow[0] / Z;
    const others = expRow.reduce((a, b, j) => a + (j === 0 ? 0 : b), 0);
    const needed = Math.log(others) - scoreRow[0];
    const sm1 = (() => {
      const finite = scoreRow.filter(Number.isFinite);
      const mx = Math.max(...finite);
      const ex = scoreRow.map((x) => (Number.isFinite(x) ? Math.exp(x - mx) : 0));
      const denom = Math.exp(-mx) + ex.reduce((a, b) => a + b, 0);
      return { row: ex.map((x) => x / denom), sum: ex.reduce((a, b) => a + b, 0) / denom };
    })();
    // "Does keeping anything at the front pay?" — the paper's own answer is that one entry does
    // nearly all of the work, so the comparison is the best front-loaded split against none.
    const bestFront = Math.min(kls[1], kls[2], kls[4]);
    // A yes/no here would report noise as a result: at a boost of zero one split beats another by
    // a few percent in whichever direction the seed happens to fall. The ratio says how much.
    const frontPays = kls[0] / Math.max(bestFront, 1e-9);
    const winsWithSinks = frontPays > 3;
    boostRead.update({
      w0: softmax1 ? `${fmt(sm1.row[0], 3)}  (row sums to ${fmt(sm1.sum, 3)})` : fmt(w0, 3),
      share: `${fmt(100 * share, 1)}%`,
      needed: needed > 0 ? `+${fmt(needed, 3)}` : "already past half",
      verdict: `×${frontPays > 99 ? frontPays.toExponential(1) : fmt(frontPays, 2)}`,
    });
    boostNote.className = "note " + (winsWithSinks ? "good" : "");
    boostNote.innerHTML = `The one way to demonstrate the causal claim on an untrained model is to stop waiting for a sink and <strong>install one</strong>. The slider adds a constant to every query's logit on the first token, in every head of every block — nothing is trained, a sink is imposed — and <strong>+${fmt(
      Math.max(needed, 0),
      3
    )}</strong> more would give it half of this row if nothing else moved. Things do move: the same constant is applied in the earlier block too, so the row it feeds this one is different, and the weight lands near the prediction rather than on it. Push the slider up and watch the panels above: the first token's weight climbs, the denominator share climbs with it, and the eviction panel's multiplier grows in step, because that multiplier <em>is</em> the reciprocal of what survives. <strong>The budget bars are the point.</strong> At a boost of zero they have no real ordering — the readout beside them, how much worse it is to keep nothing at the front, sits near 1, which is noise in whichever direction the seed happens to fall. Push the boost far enough and it acquires one, in the paper's own shape: keeping <em>nothing</em> at the front gets steadily worse while keeping <em>one</em> entry gets better, until at a budget of ${budget} a single front entry is the best split by a wide margin — which is exactly what the paper measures on three of its four models, where one initial token recovers everything and four is only a safe default. ${
      softmax1
        ? `<strong>With the extra 1 in the denominator</strong> the row now sums to ${fmt(
            sm1.sum,
            3
          )} rather than 1: the head is finally allowed to want nothing, and the surplus goes to the constant instead of to a token. The paper offers exactly this and notes it is <em>“equivalent to prepending a token with an all-zero Key and Value features”</em> — and then does not recommend it, because its own measurement is a deflation: trained from scratch with that softmax, perplexity at a zero-sink cache is 29,214 against a vanilla model's 27.87, and at four sinks it is 18.01 against 18.05. A wash where it works and far worse where it does not.`
        : `The checkbox is the other half of the paper's own suggestion: put a 1 in the denominator so the weights need not sum to one, which is the same as giving every query a phantom key with nothing in it. Turn it on and read what the row sums to.`
    }`;
  }

  return { update: render, unmount: () => {} };
}
