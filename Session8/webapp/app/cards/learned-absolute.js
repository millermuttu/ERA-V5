// Concept 4 — learned absolute position tables, and the wall at the end of them.
// Built from docs/research/learned-absolute.md. Two things the research corrected: the BERT paper
// never actually says its position embeddings are learned (that is in the released code), and
// nobody has measured the "far positions are undertrained" folklore — so neither is claimed here.
// The model is untrained, so this card demonstrates structure only, never quality.
import { el, slider } from "../lib/dom.js";
import { readout } from "../views/bars.js";
import { attentionGrid } from "../views/grid.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG } from "../model/transformer.js";
import { learnedTable, sinusoidalVector } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

const D = CONFIG.D;
const REPAIRS = {
  none: "refuse to run",
  truncate: "drop the extra words",
  clamp: "reuse the last row",
  extend: "add fresh untrained rows",
};

export function learnedAbsoluteCard(root, m) {
  let rows = 12;
  let repair = "none";
  let query = null;

  root.appendChild(
    prose({
      problem:
        "Sinusoids are hand-designed. The obvious question after concept 2 was whether a model could simply learn what each position means, the way it learns what each word means — and the Transformer paper's own ablation says the two score the same, 4.92 perplexity either way. So the field took the simpler option and learned a table.",
      mechanism:
        "Allocate one trainable vector per position — a table of shape (max positions × width) — and add row i to token i's embedding. Nothing is derived; every position is a parameter. This became the default of the BERT and GPT era, though it is worth knowing that the BERT paper itself never says its position embeddings are learned: that fact lives in the released code, where the table is created with a fixed maximum of 512 rows.",
    })
  );

  // -------------------------------------------------------------- the table
  const tableWrap = el("div", { class: "tablewrap" });
  const compareNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a stored vector per position, next to a rule per position" }),
      tableWrap,
      compareNote,
    ])
  );

  // ---------------------------------------------------------------- the wall
  const rowSlider = slider({
    label: "rows in the table",
    min: 4,
    max: 24,
    value: 12,
    oninput: (v) => ((rows = v), render()),
  });
  const repairBtns = el(
    "div",
    { class: "stages" },
    Object.entries(REPAIRS).map(([key, label]) =>
      el("button", {
        class: "stage" + (key === repair ? " is-on" : ""),
        type: "button",
        "data-repair": key,
        text: label,
        onclick: () => {
          repair = key;
          for (const b of repairBtns.children) b.classList.toggle("is-on", b.dataset.repair === repair);
          render();
        },
      })
    )
  );
  const gate = el("div", { class: "gate" });
  const wallRead = readout([
    { key: "seq", label: "words in your sentence" },
    { key: "rows", label: "rows in the table" },
    { key: "check", label: "the check the code runs" },
  ]);
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "attention under a learned table" });
  const gridHolder = el("div", {});
  const wallNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the wall at the end of the table" }),
      el("div", { class: "ctrls" }, [rowSlider.node]),
      wallRead.node,
      gate,
      repairBtns,
      gridHolder,
      wallNote,
    ])
  );

  // ------------------------------------------------------------ what repairs cost
  const repairRead = readout([
    { key: "effect", label: "what the repair did" },
    { key: "number", label: "the number that shows it" },
  ]);
  const repairNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "and what each way out actually costs" }),
      repairRead.node,
      repairNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The model decides what position means instead of being told, for the cost of one addition",
        "At the trained length it is simple, needs no special kernel, and the Transformer paper's own ablation scores it level with sinusoids — 4.92 perplexity either way",
        "Every position is independent, so nothing constrains what the model may learn about any particular one",
      ],
      givesUp: [
        "A hard wall: the table has a last row. Position 513 in a 512-row table does not exist, and no inference trick conjures it — the check is a length assertion, not a quality question",
        "It stores position rather than relating positions, so the offset structure concept 2 has is simply absent",
        "It is absolute, so it teaches placement rather than distance, and shifting a sentence rewrites every position",
      ],
      chooseWhen:
        "Fixed, known, modest context — a classifier over documents you have already decided to truncate. Session 7 ruled this out for V5 by name: a stored table cannot be extended, and long context is the whole game.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model works out for itself what each position should mean, instead of being handed a formula",
        "It is about as simple as a mechanism can be: look up a row, add it on",
        "At the length it was built for, it performed exactly as well as the formula it replaced",
      ],
      cons: [
        "There is a last row, and text longer than that simply cannot be handled — the program stops rather than doing something slightly worse",
        "Every escape route loses something real: throwing words away, giving several words the same position, or inventing positions the model has never seen",
        "Because each position is stored separately, the model learns nothing about distance — only about place",
      ],
      verdict:
        "The easiest way to give a model a sense of position, and the one that ends the most abruptly. Everything that follows on this timeline about handling longer text starts from the fact that a stored list of positions has a last entry, and a formula does not.",
    })
  );

  // ---------------------------------------------------------------- render
  function render() {
    const all = state.tokens;
    const T = all.length;
    if (!T) return;
    rowSlider.set(rows);

    const table = learnedTable({ rows });
    const over = T > rows;
    const usable = repair === "truncate" ? all.slice(0, rows) : all;

    // --- the table, drawn beside the rule it replaced
    const cell = 7;
    const rowsToDraw = Math.max(rows, T);
    const heat = (x) => {
      const v = Math.max(-1, Math.min(1, x * 1.6));
      return v >= 0
        ? `rgba(79,197,140,${(0.12 + v * 0.8).toFixed(2)})`
        : `rgba(224,105,61,${(0.12 - v * 0.8).toFixed(2)})`;
    };
    const strip = (label, getRow) =>
      el("div", { class: "tblcol" }, [
        el("span", { class: "ro-lab", text: label }),
        el(
          "div",
          { class: "tbl" },
          Array.from({ length: rowsToDraw }, (_, p) => {
            const v = getRow(p);
            return el(
              "div",
              { class: "tblrow" + (v === null ? " absent" : "") + (p < T ? " live" : "") },
              v === null
                ? [el("span", { class: "tblmissing", text: `no row ${p}` })]
                : Array.from({ length: D }, (_, d) =>
                    el("span", { class: "tblcell", style: { background: heat(v[d]) } })
                  )
            );
          })
        ),
      ]);

    tableWrap.replaceChildren(
      strip("learned table — one stored row per position", (p) => table.vector(p)),
      strip("sinusoidal — a rule, defined at every position", (p) => sinusoidalVector(p, D))
    );

    const r3 = table.vector(3);
    const r4 = table.vector(4);
    const cosLearned = r3 && r4 ? dot(r3, r4) / (Math.sqrt(dot(r3, r3)) * Math.sqrt(dot(r4, r4))) : 0;
    const s3 = sinusoidalVector(3, D);
    const s4 = sinusoidalVector(4, D);
    const cosSin = dot(s3, s4) / (Math.sqrt(dot(s3, s3)) * Math.sqrt(dot(s4, s4)));
    compareNote.innerHTML = `Two ways to answer "where am I". On the left, rows are independent stored vectors: rows 3 and 4 sit at cosine ${fmt(cosLearned, 3)} — whatever the seeded values happen to be, with no relationship built in. On the right the same two positions sit at ${fmt(cosSin, 3)}, and the banding continues forever because it is computed rather than stored. The left column stops at row ${rows}. The right one does not stop.`;

    // --- the wall
    wallRead.update({
      seq: String(T),
      rows: String(rows),
      check: over ? `${T} ≤ ${rows} → FALSE` : `${T} ≤ ${rows} → true`,
    });

    gate.className = "gate" + (over && repair === "none" ? " is-blocked" : "");
    gate.textContent = over
      ? repair === "none"
        ? `The model does not run. Your sentence is ${T} words and the table has ${rows} rows, so positions ${rows}–${T - 1} have no vector to add. This is a length assertion in the code, not a degraded answer — there is nothing to compute, so nothing is shown below.`
        : `Running with a repair applied: ${REPAIRS[repair]}.`
      : `Every word has a row. Add words past ${rows} and watch what happens.`;

    if (over && repair === "none") {
      gridHolder.replaceChildren();
      wallNote.className = "note warn";
      wallNote.textContent =
        "Nothing is drawn here on purpose. A reader who sees numbers appear concludes the model coped — it did not. Pick one of the ways out above, and then look at what each one costs.";
      repairRead.update({ effect: "—", number: "—" });
      repairNote.textContent = "Choose a repair to see what it costs.";
      return;
    }

    const position = {
      add(vec, pos) {
        let p = pos;
        if (repair === "clamp") p = Math.min(pos, rows - 1);
        if (repair === "extend" || repair === "truncate" || !over) p = pos;
        const row = repair === "extend" ? extendedRow(p) : table.vector(Math.min(p, rows - 1));
        const src = repair === "clamp" ? table.vector(Math.min(pos, rows - 1)) : row;
        if (!src) return vec;
        const out = new Float64Array(vec.length);
        for (let i = 0; i < vec.length; i++) out[i] = vec[i] + src[i];
        return out;
      },
    };
    const extendedTable = learnedTable({ rows: Math.max(rows, T), seed: 4242 });
    function extendedRow(p) {
      return p < rows ? table.vector(p) : extendedTable.vector(p);
    }

    const res = forward(usable, { position });
    const h = res.trace[0].heads[0];
    gridHolder.replaceChildren(grid.node);
    const q = Math.min(query ?? usable.length - 1, usable.length - 1);
    grid.update({ tokens: usable, weights: h.weights, query: q });

    wallNote.className = "note";
    wallNote.textContent = over
      ? `Running under "${REPAIRS[repair]}". Compare this with the same sentence one word shorter — the difference is what the repair cost you.`
      : `All ${T} words are inside the table. Nothing here is remarkable, which is the point: inside its range this mechanism is simple and works.`;

    // --- what the repair cost
    if (!over) {
      repairRead.update({ effect: "nothing to repair", number: "—" });
      repairNote.textContent = "Push the sentence past the last row to see the three ways out.";
    } else if (repair === "truncate") {
      const lost = all.slice(rows).map((t) => `“${t.word}”`).join(", ");
      repairRead.update({ effect: "words removed from the input", number: `${T - rows} dropped` });
      repairNote.className = "note warn";
      repairNote.innerHTML = `${lost} ${T - rows === 1 ? "is" : "are"} simply not in the model's input any more — no column in the attention map, no influence on anything. This is what production systems usually do, and it is worth being clear that it is not a repair at all: it is deciding the text was shorter than it was.`;
    } else if (repair === "clamp") {
      const a = forward(usable, { position }).trace[0].hidden;
      const i1 = rows - 1 + 1 < T ? rows : T - 1;
      const i2 = Math.min(T - 1, i1 + 1);
      const cos = dot(a[i1], a[i2]) / (Math.sqrt(dot(a[i1], a[i1])) * Math.sqrt(dot(a[i2], a[i2])) || 1);
      repairRead.update({ effect: "overflow words share one position", number: `positions ${i1} and ${i2} now identical` });
      repairNote.className = "note warn";
      repairNote.textContent = `Every word past the limit is told it is at position ${rows - 1}. Their position vectors are now the same vector, so the only thing distinguishing them is the word itself — the model cannot tell which of them came first. Position information for the tail of the sentence is not degraded, it is gone.`;
    } else {
      repairRead.update({ effect: "new rows appended", number: `${T - rows} never trained` });
      repairNote.className = "note warn";
      repairNote.textContent = `Rows ${rows}–${T - 1} are freshly initialised. In a trained model those rows would be noise the network has never seen at any point in training, sitting in the same space as rows it knows well — a hard seam rather than a smooth continuation. This app cannot show you the quality cost of that, because its weights are untrained to begin with; what it can show is that the new rows bear no relation to the ones above them.`;
    }
  }

  return { update: render, unmount: () => {} };
}
