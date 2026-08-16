// The attention pattern over the reader's own tokens. Mount once, update in place.
// Reused by every card whose mechanism changes which keys a query may read.
import { svg, el } from "../lib/dom.js";

const CELL = 17;
const LEFT = 104;
const TOP = 22;

export const heat = (w) =>
  w <= 0 ? "rgba(233,231,220,0.045)" : `rgba(79,197,140,${(0.08 + Math.min(1, Math.sqrt(w * 3)) * 0.87).toFixed(3)})`;

export function attentionGrid({ onPickRow, label = "attention weights" }) {
  let cells = [];
  let rowLabels = [];
  let colMark = null;
  let T = 0;

  const gridG = svg("g", {});
  const marksG = svg("g", {});
  const root = svg("svg", { class: "grid-svg", role: "img", "aria-label": label }, [gridG, marksG]);
  const node = el("div", { class: "gridwrap" }, [root]);

  function build(tokens) {
    T = tokens.length;
    while (gridG.firstChild) gridG.removeChild(gridG.firstChild);
    while (marksG.firstChild) marksG.removeChild(marksG.firstChild);
    cells = [];
    rowLabels = [];

    const W = LEFT + T * CELL + 130;
    const H = TOP + T * CELL + 26;
    root.setAttribute("viewBox", `0 0 ${W} ${H}`);
    root.setAttribute("width", W);
    root.setAttribute("height", H);

    for (let i = 0; i < T; i++) {
      const row = [];
      for (let j = 0; j < T; j++) {
        const r = svg("rect", {
          x: LEFT + j * CELL,
          y: TOP + i * CELL,
          width: CELL - 2,
          height: CELL - 2,
          rx: 2,
          fill: "rgba(233,231,220,0.03)",
        });
        gridG.appendChild(r);
        row.push(r);
      }
      cells.push(row);

      const t = svg("text", {
        x: LEFT - 8,
        y: TOP + i * CELL + 12,
        "text-anchor": "end",
        class: "gridlab",
        onclick: () => onPickRow && onPickRow(i),
      }, [tokens[i].word]);
      marksG.appendChild(t);
      rowLabels.push(t);
    }

    colMark = svg("rect", {
      x: LEFT - 3,
      y: TOP,
      width: T * CELL + 6,
      height: CELL,
      rx: 3,
      fill: "none",
      stroke: "rgba(79,197,140,0.55)",
    });
    marksG.appendChild(colMark);
    marksG.appendChild(
      svg("text", { x: 2, y: TOP + T * CELL + 20, class: "gridlab" }, ["query ↓ · key → · click a token to follow its row"])
    );

    root.onclick = (ev) => {
      const box = root.getBoundingClientRect();
      const i = Math.floor((((ev.clientY - box.top) / box.height) * H - TOP) / CELL);
      if (i >= 0 && i < T && onPickRow) onPickRow(i);
    };
  }

  /** weights: T x T (rows may be shorter); readable(i,j) false = hidden by the mechanism. */
  function update({ tokens, weights, query, readable = null }) {
    if (tokens.length !== T) build(tokens);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const hidden = j > i || (readable && !readable(i, j));
        const w = weights?.[i]?.[j] ?? 0;
        cells[i][j].setAttribute("fill", hidden ? "rgba(233,231,220,0.03)" : heat(w));
        cells[i][j].setAttribute("stroke", hidden && j <= i ? "rgba(224,105,61,0.35)" : "none");
      }
      rowLabels[i].classList.toggle("is-current", i === query);
    }
    colMark.setAttribute("y", TOP + query * CELL);
  }

  return { node, update };
}
