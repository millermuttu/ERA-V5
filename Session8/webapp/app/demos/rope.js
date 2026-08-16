// One dimension pair, two tokens, rotated by their positions. The dot product depends on the
// angle between the arrows, so moving both tokens together leaves the score alone.
import { dot, fmt } from "../lib/mathx.js";

const THETA = 0.4; // radians per position step

const rotate = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

// Same content vector for both tokens, so anything the score sees is positional.
const CONTENT = [1, 0];

export function ropeScore(i, j, on) {
  const qv = on ? rotate(CONTENT, i * THETA) : CONTENT;
  const kv = on ? rotate(CONTENT, j * THETA) : CONTENT;
  return { qv, kv, score: dot(qv, kv), gap: i - j };
}

export function mountRope(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">Position as rotation</div>
      <div class="ctrls">
        <button class="tg" id="r-on" aria-pressed="true">RoPE: on</button>
        <div class="ctrl"><label for="r-j">key at position <span id="r-jv">2</span></label>
          <input id="r-j" type="range" min="0" max="24" step="1" value="2"></div>
        <div class="ctrl"><label for="r-i">query at position <span id="r-iv">8</span></label>
          <input id="r-i" type="range" min="0" max="24" step="1" value="8"></div>
      </div>
      <div id="r-out"></div>
      <p class="note" id="r-note"></p>
    </div>`
  );

  const on = el.querySelector("#r-on");
  const si = el.querySelector("#r-i");
  const sj = el.querySelector("#r-j");
  const out = el.querySelector("#r-out");
  const note = el.querySelector("#r-note");
  let rope = true;

  function render() {
    const i = Number(si.value);
    const j = Number(sj.value);
    el.querySelector("#r-iv").textContent = i;
    el.querySelector("#r-jv").textContent = j;
    const { qv, kv, score, gap } = ropeScore(i, j, rope);
    const ang = (Math.abs(gap) * THETA * 180) / Math.PI;

    const arrow = (v, colour, label) => {
      const [x, y] = v;
      return `<line x1="65" y1="65" x2="${(65 + x * 46).toFixed(1)}" y2="${(65 - y * 46).toFixed(1)}"
                stroke="${colour}" stroke-width="2" marker-end="url(#rh-${colour.slice(1)})"/>
              <text x="${Math.max(2, 65 + x * 58 - 12).toFixed(1)}" y="${(65 - y * 58).toFixed(1)}" fill="${colour}">${label}</text>`;
    };

    out.innerHTML = `
      <svg viewBox="0 0 360 130" width="360" height="130" class="diagram" role="img"
           aria-label="query and key vectors on a plane, separated by the rotation their positions imply">
        <defs>
          <marker id="rh-4FC58C" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill="#4FC58C"/></marker>
          <marker id="rh-E0A03A" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill="#E0A03A"/></marker>
        </defs>
        <circle cx="65" cy="65" r="46" fill="none" stroke="rgba(233,231,220,0.12)"/>
        <line x1="13" y1="65" x2="117" y2="65" stroke="rgba(233,231,220,0.08)"/>
        <line x1="65" y1="13" x2="65" y2="117" stroke="rgba(233,231,220,0.08)"/>
        ${arrow(kv, "#E0A03A", "key")}
        ${arrow(qv, "#4FC58C", "query")}
        <text x="136" y="46">distance i − j = ${gap}</text>
        <text x="136" y="68">angle between = ${rope ? ang.toFixed(1) + "°" : "0.0°"}</text>
        <text x="136" y="90" class="hi">score = ${fmt(score, 3)}</text>
      </svg>`;

    note.className = "note " + (rope ? "" : "warn");
    note.textContent = rope
      ? `Both tokens hold the same content vector, so the score is pure position. Slide both sliders by the same amount: the arrows sweep round together, the gap stays ${gap}, and the score does not move.`
      : "With rotation off, two tokens holding identical content are indistinguishable — the score is 1.000 whatever their positions. That is what the dot product knows about order on its own: nothing.";
  }

  on.onclick = () => {
    rope = !rope;
    on.setAttribute("aria-pressed", String(rope));
    on.textContent = `RoPE: ${rope ? "on" : "off"}`;
    render();
  };
  si.oninput = render;
  sj.oninput = render;
  render();
}
