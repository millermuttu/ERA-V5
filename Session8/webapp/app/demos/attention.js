// The baseline, computed for real: 6 tokens, 4 dimensions, one head.
// Weights are literal so every reload shows the same numbers.
import { dot, softmax, fmt } from "../lib/mathx.js";

export const TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
const D = 4;

// Rough features: [animate, action, place, determiner]
const X = [
  [0.1, 0.0, 0.2, 1.0],
  [1.0, 0.1, 0.3, 0.0],
  [0.2, 1.0, 0.1, 0.0],
  [0.0, 0.2, 0.9, 0.1],
  [0.1, 0.0, 0.2, 1.0],
  [0.6, 0.0, 0.4, 0.0],
];

// Rows are input dims, columns output dims.
// Wq turns "I am an action" into "I am looking for something animate";
// Wk turns "I am animate" into "that is what I advertise". So `sat` finds `cat`.
const Wq = [[0, 0, 0, 0], [1.2, 0, 0, 0], [0, 0, 0.8, 0], [0, 0, 0, 0.5]];
const Wk = [[1.0, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.9, 0], [0, 0, 0, 0.6]];
const Wv = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

const proj = (x, W) => W[0].map((_, j) => dot(x, W.map((row) => row[j])));

export const Q = X.map((x) => proj(x, Wq));
export const K = X.map((x) => proj(x, Wk));
export const V = X.map((x) => proj(x, Wv));

// The whole mechanism. `masked` is the causal rule, written where the scores are made.
export function attend(masked = true) {
  const scale = Math.sqrt(D);
  const scores = Q.map((q, i) =>
    K.map((k, j) => (masked && j > i ? -Infinity : dot(q, k) / scale))
  );
  const weights = scores.map(softmax);
  const out = weights.map((w) =>
    V[0].map((_, d) => w.reduce((s, wj, j) => s + wj * V[j][d], 0))
  );
  return { scores, weights, out };
}

export function mountAttention(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">Scaled dot-product attention, computed live</div>
      <div class="ctrls">
        <button class="tg" id="a-mask" aria-pressed="true">causal mask: on</button>
        <div class="ctrl"><label for="a-row">reading token</label>
          <select id="a-row">${TOKENS.map((t, i) => `<option value="${i}"${i === 2 ? " selected" : ""}>${i}. ${t}</option>`).join("")}</select>
        </div>
      </div>
      <div id="a-out"></div>
      <p class="note" id="a-note"></p>
    </div>`
  );

  const btn = el.querySelector("#a-mask");
  const sel = el.querySelector("#a-row");
  const out = el.querySelector("#a-out");
  const note = el.querySelector("#a-note");
  let masked = true;

  function render() {
    const { scores, weights, out: ctx } = attend(masked);
    const i = Number(sel.value);
    const head = TOKENS.map((t, j) => `<th>${t}</th>`).join("");

    const rows = weights
      .map((w, r) => {
        const cells = w
          .map((x, c) => {
            const dead = scores[r][c] === -Infinity;
            const cls = dead ? "masked" : x > 0.34 ? "hot" : "";
            return `<td class="${cls}">${dead ? "0.00" : fmt(x)}</td>`;
          })
          .join("");
        return `<tr><td class="rowhead">${TOKENS[r]}</td>${cells}</tr>`;
      })
      .join("");

    const future = weights[i].reduce((s, w, j) => s + (j > i ? w : 0), 0);

    out.innerHTML = `
      <table class="num">
        <tr><th class="rowhead">weight</th>${head}</tr>${rows}
      </table>
      <p class="note">
        Row <strong>${TOKENS[i]}</strong>: scores ${scores[i].map((s) => fmt(s)).join("  ")}
        &nbsp;→&nbsp; output ${ctx[i].map((v) => fmt(v)).join("  ")}
      </p>`;

    note.className = "note " + (future > 1e-9 ? "warn" : "good");
    note.textContent =
      future > 1e-9
        ? `Weight on tokens that have not happened yet, for "${TOKENS[i]}": ${fmt(future, 3)}. That is the leak the causal mask exists to stop.`
        : `Weight on future tokens: exactly 0. Every row still sums to 1 — softmax(-infinity) = 0, so the future is not competing for it.`;
  }

  btn.onclick = () => {
    masked = !masked;
    btn.setAttribute("aria-pressed", String(masked));
    btn.textContent = `causal mask: ${masked ? "on" : "off"}`;
    render();
  };
  sel.onchange = render;
  render();
}
