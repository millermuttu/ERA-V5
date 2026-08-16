// The slider is the chronology. Entries sit at their real dates, so the gaps and the pile-ups are
// the field's own rhythm: nothing much between 2017 and 2019, then everything at once after 2023.
import { svg, el, clear } from "./lib/dom.js";
import { byDate, PRESSURES } from "./data/mechanisms.js";

const W = 1000;
const H = 74;
const PAD = 26;

const t = (iso) => new Date(iso + "T00:00:00Z").getTime();

export function mountTimeline(node, { onPick }) {
  const list = byDate();
  const lo = t(list[0].date);
  const hi = t(list[list.length - 1].date);
  const x = (iso) => PAD + ((t(iso) - lo) / (hi - lo)) * (W - PAD * 2);

  const years = [];
  for (let y = 2017; y <= 2026; y++) years.push(y);

  const ticks = years.map((y) =>
    svg("g", {}, [
      svg("line", { x1: x(`${y}-01-01`), y1: 30, x2: x(`${y}-01-01`), y2: 40, stroke: "rgba(233,231,220,0.18)" }),
      svg("text", { x: x(`${y}-01-01`), y: 54, "text-anchor": "middle", class: "tick" }, [String(y)]),
    ])
  );

  // 2023 onwards the entries pile up within a few pixels of each other. Nudge a cluster apart so
  // every one of them is visible and clickable — and so the density itself reads as density.
  const lanes = [];
  let lastX = -Infinity;
  let lane = 0;
  for (const m of list) {
    const px = x(m.date);
    lane = px - lastX < 11 ? (lane + 1) % 3 : 0;
    lanes.push(lane);
    lastX = px;
  }
  const cy = (i) => 20 + (lanes[i] === 0 ? 0 : lanes[i] === 1 ? -9 : 9);

  const dots = list.map((m, i) => {
    const pick = () => onPick(i);
    const dot = svg("circle", { cx: x(m.date), cy: cy(i), r: 5, class: "dot", "data-id": m.id });
    // A transparent, generous hit target on top of the small visible dot.
    const hit = svg(
      "circle",
      {
        cx: x(m.date),
        cy: cy(i),
        r: 11,
        class: "dot-hit",
        tabindex: "0",
        role: "button",
        "aria-label": `${m.name}, ${m.date}`,
        onclick: pick,
        onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), pick()),
        onmouseenter: () => dot.classList.add("is-hover"),
        onmouseleave: () => dot.classList.remove("is-hover"),
        onfocus: () => dot.classList.add("is-hover"),
        onblur: () => dot.classList.remove("is-hover"),
      },
      [svg("title", { text: `${m.name} · ${m.date}` })]
    );
    return { dot, hit };
  });

  const cursor = svg("line", { x1: 0, y1: 8, x2: 0, y2: 32, class: "cursor" });
  const label = svg("text", { x: 0, y: 6, "text-anchor": "middle", class: "cursor-label" });

  const chart = svg(
    "svg",
    {
      viewBox: `0 0 ${W} ${H}`,
      class: "timeline-svg",
      role: "group",
      "aria-label": "Timeline of attention mechanisms by launch date",
    },
    [
      svg("line", { x1: PAD, y1: 20, x2: W - PAD, y2: 20, stroke: "rgba(233,231,220,0.12)" }),
      ...ticks,
      ...dots.map((d) => d.dot),
      cursor,
      label,
      ...dots.map((d) => d.hit),
    ]
  );

  const scrub = el("input", {
    type: "range",
    min: 0,
    max: list.length - 1,
    step: 1,
    value: 0,
    class: "scrub",
    "aria-label": "Slide through the timeline",
    oninput: (e) => onPick(Number(e.target.value)),
  });

  clear(node);
  node.appendChild(chart);
  node.appendChild(scrub);

  return function update(index) {
    const m = list[index];
    scrub.value = index;
    cursor.setAttribute("x1", x(m.date));
    cursor.setAttribute("x2", x(m.date));
    label.setAttribute("x", Math.min(W - 90, Math.max(70, x(m.date))));
    label.textContent = PRESSURES[m.pressure];
    for (const { dot } of dots) dot.classList.toggle("is-current", dot.dataset.id === m.id);
  };
}
