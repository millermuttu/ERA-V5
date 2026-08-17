// The whole "framework". A concept mounts once and then updates in place — never innerHTML on a
// live view, because that flickers, drops focus and cancels a slider drag mid-gesture.

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  apply(node, attrs);
  add(node, children);
  return node;
}

export function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    // Listeners must be added, not set: setAttribute would stringify the function and bind nothing.
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  add(node, children);
  return node;
}

function apply(node, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else node.setAttribute(k, v);
  }
}

function add(node, children) {
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
}

export const clear = (node) => {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
};

/**
 * A labelled range control. Returns the wrapper plus a `set` that writes the readout,
 * so an update pass can move the number without rebuilding the input.
 */
export function slider({ label, min, max, step = 1, value, format = String, oninput }) {
  const out = el("span", { class: "ctrl-val", text: format(value) });
  const input = el("input", {
    type: "range",
    min,
    max,
    step,
    value,
    "aria-label": label,
    oninput: (e) => {
      const v = Number(e.target.value);
      out.textContent = format(v);
      oninput(v);
    },
  });
  const node = el("div", { class: "ctrl" }, [el("label", {}, [label + " ", out]), input]);
  return { node, input, set: (v) => ((input.value = v), (out.textContent = format(v))) };
}

export function toggle({ label, value, onchange }) {
  let on = value;
  const btn = el("button", {
    class: "tg",
    type: "button",
    "aria-pressed": String(on),
    text: `${label}: ${on ? "on" : "off"}`,
    onclick: () => {
      on = !on;
      btn.setAttribute("aria-pressed", String(on));
      btn.textContent = `${label}: ${on ? "on" : "off"}`;
      onchange(on);
    },
  });
  return btn;
}

export function choice({ label, options, value, onchange }) {
  const sel = el(
    "select",
    { "aria-label": label, onchange: (e) => onchange(e.target.value) },
    options.map((o) => el("option", { value: o.value, text: o.label, selected: o.value === value ? "" : null }))
  );
  return el("div", { class: "ctrl" }, [el("label", { text: label }), sel]);
}
