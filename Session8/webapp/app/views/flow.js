// The dataflow picture: one token's journey from embedding to prediction, drawn from the live
// forward pass. Numbers tell you how much a mechanism costs; this shows what it *is* — and when a
// mechanism changes the shape of the computation, the picture changes shape with it.
//
// Built in the spirit of the course's own widget, on our model. Parameterised so every concept can
// point it at what that concept changes: cells the mechanism cannot read, key/value heads shared
// between queries, or the score matrix replaced entirely by a running state.
import { el } from "../lib/dom.js";
import { dot } from "../model/ops.js";

const COL = { q: "#5b7fdb", k: "#E0693D", v: "#4FC58C" };
const W = 1360;
const PITCH = 26; // vertical space per token — the whole layout is built from this
const TOP = 92;

const G = {
  tokX: 104, embX: 114, embW: 48,
  qkvX: 244, qkvW: 44,
  barX: 320, barW: 54,
  outW: 48,
};

/** A filled band from one y-range on the left to another on the right. */
const ribbon = (x0, a0, a1, x1, b0, b1) => {
  const xm = (x0 + x1) / 2;
  return `M${x0} ${a0}C${xm} ${a0},${xm} ${b0},${x1} ${b0}L${x1} ${b1}C${xm} ${b1},${xm} ${a1},${x0} ${a1}Z`;
};

/** A vector as a row of signed cells. */
function strip(x, y, w, h, vec, colour) {
  const n = vec.length;
  const cw = w / n;
  let max = 0;
  for (const t of vec) max = Math.max(max, Math.abs(t));
  max = max || 1;
  let s = "";
  for (let i = 0; i < n; i++) {
    const a = Math.abs(vec[i]) / max;
    s += `<rect x="${(x + i * cw).toFixed(2)}" y="${y}" width="${(cw + 0.4).toFixed(2)}" height="${h}" fill="${
      vec[i] >= 0 ? colour : "#E0693D"
    }" fill-opacity="${(0.12 + a * 0.8).toFixed(2)}"/>`;
  }
  return s;
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

export function flowView() {
  const node = el("div", { class: "flowwrap" });

  /**
   * data: { tokens, head, weights, out, top, query }
   * opts: { readable, kvShared, stateMode, positionAdded, title }
   */
  function update({ tokens, head, weights, out, top, query = null, opts = {} }) {
    const n = tokens?.length || 0;
    if (!n) {
      node.innerHTML = "";
      return;
    }
    const q = query === null ? n - 1 : Math.min(query, n - 1);
    const H = TOP + n * PITCH + 96;
    const midY = TOP + ((n - 1) * PITCH) / 2;
    const tokY = (i) => TOP + i * PITCH;
    // the three magnitude lists sit side by side rather than stacked, so they cannot collide
    const listX = { Key: G.barX, Query: G.barX + 92, Value: G.barX + 184 };
    // Lay the columns out from the widths they actually occupy, so the matrix can never be drawn
    // on top of the value bars however long the sentence is.
    const step = Math.min(19, Math.max(9, 300 / n));
    const listsEnd = listX.Value + G.barW;
    const matHalf = ((n - 1) / 2) * step;
    const matX = listsEnd + 66 + matHalf;
    const outX = matX + matHalf + 40;
    const probX = outX + G.outW + 64;
    const matCx = (j) => matX + (j - (n - 1) / 2) * step;
    const matCy = (i) => midY + (i - (n - 1) / 2) * step;

    let s = "";

    // heads stacked behind, so multi-head reads as depth rather than being stated
    const boxT = midY - (n * step) / 2 - 26;
    const boxH = n * step + 52;
    for (let i = 3; i >= 1; i--)
      s += `<rect x="${matX - matHalf - 34 + 9 * i}" y="${boxT + 5 * i}" width="${2 * matHalf + 68 - 18 * i}" height="${boxH + 8 * i}" rx="10"
        fill="rgba(17,30,24,0.55)" stroke="rgba(233,231,220,0.07)"/>`;
    s += `<rect x="${matX - matHalf - 34}" y="${boxT}" width="${2 * matHalf + 68}" height="${boxH}" rx="10" fill="rgba(15,26,21,0.8)" stroke="rgba(233,231,220,0.12)"/>`;

    // residual path over the top — the thing attention is added back into
    s += `<path d="M${G.embX + G.embW} ${tokY(0) - 30}C200 ${tokY(0) - 30} 210 46 250 46L820 46C856 46 858 ${midY - 30} ${outX} ${midY - 30}"
      fill="none" stroke="rgba(233,231,220,0.22)" stroke-width="1.3" stroke-dasharray="4 4"/>
      <text x="470" y="40" class="fl-lab">residual — the token keeps itself</text>`;

    // token labels, embeddings, and the ribbons into Q/K/V
    for (let i = 0; i < n; i++) {
      const ty = tokY(i);
      const isQ = i === q;
      s += `<text x="${G.tokX}" y="${ty + 4}" text-anchor="end" class="fl-tok${isQ ? " is-q" : ""}"${
        tokens[i].known ? "" : ' opacity="0.5"'
      }>${esc(tokens[i].word)}</text>`;
      s += strip(G.embX, ty - 7, G.embW, 14, head.emb ? head.emb[i] : head.Q[i], "#4FC58C");
      if (opts.positionAdded) {
        s += `<circle cx="${G.embX + G.embW + 8}" cy="${ty}" r="3" fill="#E0A03A"/>`;
      }
      // three ribbons: this token's embedding becomes its query, key and value
      [COL.q, COL.k, COL.v].forEach((c, kk) => {
        const cy = ty + (kk - 1) * 4;
        const to = ty - 9 + kk * 6;
        s += `<path d="${ribbon(G.embX + G.embW, cy - 3, cy + 3, G.qkvX, to, to + 5)}"
          fill="${c}" fill-opacity="${isQ ? 0.45 : 0.1}"/>`;
      });
    }
    if (opts.positionAdded)
      s += `<text x="${G.embX + G.embW + 2}" y="${TOP - 30}" class="fl-lab" fill="#E0A03A">+ position</text>`;
    s += `<text x="${G.embX}" y="${TOP - 18}" class="fl-letter">embedding</text>`;

    // Q, K, V strips
    const letters = ["Q", "K", "V"];
    const vecs = [head.Q, head.K, head.V];
    for (let i = 0; i < n; i++) {
      for (let m = 0; m < 3; m++) {
        const yy = tokY(i) - 9 + m * 6;
        // Under key/value sharing every query head reads the same K and V: draw them once, joined.
        const shared = opts.kvShared && m > 0;
        s += strip(G.qkvX, yy, G.qkvW, 5, vecs[m][i], [COL.q, COL.k, COL.v][m]);
        // label the three bands in place, beside the first token, rather than stacked overhead
        if (i === 0)
          s += `<text x="${G.qkvX - 5}" y="${yy + 5}" text-anchor="end" class="fl-letter" fill="${
            [COL.q, COL.k, COL.v][m]
          }">${letters[m]}</text>`;
        if (shared && i === 0)
          s += `<text x="${G.qkvX + G.qkvW + 4}" y="${yy + 5}" class="fl-lab" fill="#E0A03A">shared</text>`;
      }
    }

    // Key / Query / Value magnitude lists
    const lists = [
      { vec: head.K, col: COL.k, lab: "Key" },
      { vec: head.Q, col: COL.q, lab: "Query" },
      { vec: head.V, col: COL.v, lab: "Value" },
    ];
    for (const L of lists) {
      const x = listX[L.lab];
      const norms = L.vec.map((r) => Math.sqrt(dot(r, r)));
      const mx = Math.max(...norms) || 1;
      s += `<text x="${x}" y="${TOP - 18}" class="fl-letter" fill="${L.col}">${L.lab}</text>`;
      for (let t = 0; t < n; t++) {
        const y = tokY(t);
        s += `<rect x="${x}" y="${y - 2.5}" width="${((G.barW * norms[t]) / mx).toFixed(1)}" height="5" rx="1.5"
          fill="${L.col}" fill-opacity="${t === q && L.lab === "Query" ? 1 : 0.5}"/>`;
      }
    }

    if (opts.stateMode) {
      // The mechanism has changed shape: no score matrix, one running state instead.
      s += `<rect x="${matX - 74}" y="${midY - 64}" width="148" height="128" rx="10"
        fill="rgba(79,197,140,0.08)" stroke="#4FC58C"/>`;
      const st = opts.stateMode.matrix;
      const cells = st.length;
      const cw = 128 / cells;
      let smax = 0;
      for (const r of st) for (const x of r) smax = Math.max(smax, Math.abs(x));
      smax = smax || 1;
      for (let a = 0; a < cells; a++)
        for (let b = 0; b < st[a].length; b++)
          s += `<rect x="${(matX - 64 + b * (128 / st[a].length)).toFixed(1)}" y="${(midY - 54 + a * cw).toFixed(1)}"
            width="${(128 / st[a].length - 1).toFixed(1)}" height="${(cw - 1).toFixed(1)}" rx="1"
            fill="${st[a][b] >= 0 ? "#4FC58C" : "#E0693D"}" fill-opacity="${(0.1 + (Math.abs(st[a][b]) / smax) * 0.85).toFixed(2)}"/>`;
      s += `<text x="${matX}" y="${midY - 74}" text-anchor="middle" class="fl-letter" fill="#4FC58C">one running state</text>`;
      s += `<text x="${matX}" y="${midY + 82}" text-anchor="middle" class="fl-lab">no scores · nothing to store per token</text>`;
    } else {
      // key columns and query rows arriving at the score matrix
      for (let j = 0; j < n; j++) {
        const ky = tokY(j);
        const cx = matCx(j).toFixed(1);
        s += `<path d="M${listX.Key + G.barW + 4} ${ky.toFixed(1)}C${(listsEnd + 20).toFixed(1)} ${ky.toFixed(
          1
        )},${cx} ${ky.toFixed(1)},${cx} ${(matCy(0) - step * 0.7).toFixed(1)}"
          fill="none" stroke="${COL.k}" stroke-width="0.8" stroke-opacity="0.25"/>`;
      }
      s += `<path d="M${listX.Query + G.barW + 4} ${tokY(q).toFixed(1)}C${(listsEnd + 30).toFixed(1)} ${tokY(q).toFixed(
        1
      )},${(matX - matHalf - 46).toFixed(1)} ${matCy(q).toFixed(1)},${(matX - matHalf - 30).toFixed(1)} ${matCy(q).toFixed(1)}"
        fill="none" stroke="${COL.q}" stroke-width="1.7"/>`;

      // the matrix itself
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          const readable = !opts.readable || opts.readable(i, j);
          const w = weights?.[i]?.[j] ?? 0;
          const r = readable ? 1.6 + Math.sqrt(w) * 5.4 : 2;
          s += `<circle cx="${matCx(j).toFixed(1)}" cy="${matCy(i).toFixed(1)}" r="${r.toFixed(2)}"
            fill="${readable ? "#79E6B0" : "rgba(224,105,61,0.35)"}"
            fill-opacity="${readable ? (i === q ? 0.95 : 0.4) : 1}"/>`;
        }
      }
      s += `<text x="${matX}" y="${(matCy(0) - step * 0.7 - 12).toFixed(1)}" text-anchor="middle" class="fl-letter">who reads whom</text>`;
      if (opts.readable)
        s += `<text x="${matX}" y="${(matCy(n - 1) + step * 0.7 + 20).toFixed(1)}" text-anchor="middle" class="fl-lab" fill="#E0693D">small marks are pairs this mechanism cannot read</text>`;
    }

    // the value pool flowing into the output
    s += `<path d="${ribbon(listX.Value + G.barW + 4, tokY(0) - 5, tokY(n - 1) + 5, outX, midY - 22, midY + 22)}"
      fill="${COL.v}" fill-opacity="0.05"/>`;

    // the output vector for the chosen token
    s += strip(outX, midY - 9, G.outW, 18, out[q], "#79E6B0");
    s += `<rect x="${outX}" y="${midY - 9}" width="${G.outW}" height="18" fill="none" stroke="rgba(233,231,220,0.2)"/>`;
    s += `<text x="${outX}" y="${midY - 16}" class="fl-letter">out</text>`;

    // the model's next-token guess
    if (top && top.length) {
      const mx = top[0].p || 1;
      s += `<text x="${probX}" y="${midY - 16}" class="fl-letter">next word, after “${esc(tokens[q].word)}”</text>`;
      top.slice(0, 6).forEach((t, i) => {
        const y = midY - 8 + i * 19;
        s += `<rect x="${probX}" y="${y}" width="${(110 * (t.p / mx)).toFixed(1)}" height="11" rx="2" fill="#4FC58C" fill-opacity="${(
          0.25 +
          0.6 * (t.p / mx)
        ).toFixed(2)}"/>`;
        s += `<text x="${probX + 116}" y="${y + 9}" class="fl-lab">${esc(t.word)}</text>`;
      });
      s += `<text x="${probX}" y="${midY - 8 + 6 * 19 + 16}" class="fl-lab">untrained: shape real,</text>
      <text x="${probX}" y="${midY - 8 + 6 * 19 + 29}" class="fl-lab">words noise</text>`;
    }

    node.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="flow" role="img"
      aria-label="the path from tokens through queries, keys and values to the attention pattern and the model's next-token guess">${s}</svg>`;
  }

  return { node, update };
}
