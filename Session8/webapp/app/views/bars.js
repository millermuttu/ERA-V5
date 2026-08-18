// Horizontal bars for anything that has a magnitude — weights in a row, costs against the
// baseline, per-stage numbers. Mount once with the labels, then update values in place.
import { el } from "../lib/dom.js";

export function barList({ rows }) {
  const built = rows.map((r) => {
    const fill = el("span", { class: "fill" + (r.alt ? " alt" : "") });
    const val = el("span", { class: "val" });
    const name = el("span", { class: "bar-name", text: r.label });
    return {
      key: r.key,
      node: el("div", { class: "bar" }, [name, el("span", { class: "track" }, [fill]), val]),
      fill,
      val,
      name,
    };
  });

  const node = el("div", { class: "bars" }, built.map((b) => b.node));

  /** values: { key: {value, of, text, label} } */
  function update(values) {
    for (const b of built) {
      const v = values[b.key];
      if (!v) continue;
      const pct = v.of ? Math.min(100, (v.value / v.of) * 100) : 0;
      b.fill.style.width = pct.toFixed(2) + "%";
      b.val.textContent = v.text ?? String(v.value);
      if (v.label) b.name.textContent = v.label;
      // An empty tone resets to the default fill — a caller that tones some rows and not others
      // must be able to clear one, or the class sticks from the previous update.
      if (v.tone !== undefined) b.fill.className = ("fill " + v.tone).trim();
    }
  }

  return { node, update };
}

/** A row of numbers with captions — for stage readouts where a bar would be noise. */
export function readout(items) {
  const built = items.map((it) => {
    const v = el("span", { class: "ro-val", text: "—" });
    return { key: it.key, node: el("div", { class: "ro" }, [el("span", { class: "ro-lab", text: it.label }), v]), v };
  });
  const node = el("div", { class: "readout" }, built.map((b) => b.node));
  return {
    node,
    update(values) {
      for (const b of built) if (values[b.key] !== undefined) b.v.textContent = values[b.key];
    },
  };
}
