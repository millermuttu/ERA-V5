// A line plot with an optional shaded region and a comparison curve underneath.
// Used by every positional concept: score against distance, a trained range shaded, and the
// baseline's curve behind the mechanism's own.
import { svg, el } from "../lib/dom.js";

const W = 640;
const H = 220;
const PAD = { l: 46, r: 16, t: 14, b: 32 };

const px = (v, lo, hi, a, b) => a + ((v - lo) / (hi - lo || 1)) * (b - a);

export function curveView({ xLabel = "", yLabel = "", ariaLabel = "curve" }) {
  const shade = svg("rect", { x: PAD.l, y: PAD.t, width: 0, height: H - PAD.t - PAD.b, fill: "rgba(79,197,140,0.07)" });
  const dead = svg("rect", { x: 0, y: PAD.t, width: 0, height: H - PAD.t - PAD.b, fill: "rgba(224,105,61,0.13)" });
  const zero = svg("line", { x1: PAD.l, x2: W - PAD.r, stroke: "rgba(233,231,220,0.12)" });
  const refPath = svg("path", { fill: "none", stroke: "rgba(233,231,220,0.28)", "stroke-width": 1.1, "stroke-dasharray": "3 3" });
  const mainPath = svg("path", { fill: "none", stroke: "#4FC58C", "stroke-width": 1.7 });
  const marker = svg("line", { y1: PAD.t, y2: H - PAD.b, stroke: "#79E6B0", "stroke-dasharray": "3 3", "stroke-width": 0.9, x1: -10, x2: -10 });
  const markerText = svg("text", { class: "curve-tip", x: -100, y: PAD.t + 10 });
  const deadText = svg("text", { class: "curve-warn", x: -100, y: H / 2 });
  const xText = svg("text", { x: W - PAD.r, y: H - 8, "text-anchor": "end", class: "curve-ax", text: xLabel });
  const yText = svg("text", { x: 4, y: PAD.t + 8, class: "curve-ax", text: yLabel });
  const dots = svg("g", {});

  const root = svg(
    "svg",
    { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "curve", role: "img", "aria-label": ariaLabel },
    [shade, dead, zero, refPath, mainPath, dots, marker, markerText, deadText, xText, yText]
  );

  const node = el("div", { class: "curvewrap" }, [root]);

  /**
   * points / reference: arrays of [x, y]. `undefined` y breaks the line (a hole in the domain).
   * band: [x0, x1] shaded as "inside training". deadFrom: x beyond which the scheme is undefined.
   */
  function update({ points, reference = null, xRange, yRange, band = null, deadFrom = null, mark = null, markLabel = "", deadLabel = "" }) {
    const [x0, x1] = xRange;
    const [y0, y1] = yRange;
    const X = (v) => px(v, x0, x1, PAD.l, W - PAD.r);
    const Y = (v) => px(v, y1, y0, PAD.t, H - PAD.b);

    const line = (pts) => {
      let d = "";
      let pen = false;
      for (const [x, y] of pts) {
        if (y === undefined || y === null || !Number.isFinite(y)) {
          pen = false;
          continue;
        }
        d += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
        pen = true;
      }
      return d;
    };

    mainPath.setAttribute("d", line(points));
    refPath.setAttribute("d", reference ? line(reference) : "");
    zero.setAttribute("y1", Y(0));
    zero.setAttribute("y2", Y(0));

    if (band) {
      shade.setAttribute("x", X(band[0]));
      shade.setAttribute("width", Math.max(0, X(band[1]) - X(band[0])));
    } else shade.setAttribute("width", 0);

    if (deadFrom !== null && deadFrom < x1) {
      dead.setAttribute("x", X(deadFrom));
      dead.setAttribute("width", Math.max(0, W - PAD.r - X(deadFrom)));
      deadText.setAttribute("x", X(deadFrom) + 8);
      deadText.textContent = deadLabel;
    } else {
      dead.setAttribute("width", 0);
      deadText.textContent = "";
    }

    if (mark !== null) {
      marker.setAttribute("x1", X(mark));
      marker.setAttribute("x2", X(mark));
      markerText.setAttribute("x", Math.min(W - 120, X(mark) + 6));
      markerText.textContent = markLabel;
    } else {
      marker.setAttribute("x1", -10);
      marker.setAttribute("x2", -10);
      markerText.textContent = "";
    }
  }

  /** Optional scatter overlay — used when the points are discrete positions, not a continuum. */
  function setDots(pts, xRange, yRange) {
    while (dots.firstChild) dots.removeChild(dots.firstChild);
    const [x0, x1] = xRange;
    const [y0, y1] = yRange;
    for (const [x, y, cls] of pts) {
      if (!Number.isFinite(y)) continue;
      dots.appendChild(
        svg("circle", {
          cx: px(x, x0, x1, PAD.l, W - PAD.r),
          cy: px(y, y1, y0, PAD.t, H - PAD.b),
          r: 2.6,
          class: "curve-dot " + (cls || ""),
        })
      );
    }
  }

  return { node, update, setDots };
}
