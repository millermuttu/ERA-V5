// Numeric primitives for the forward pass. Nothing here is specific to a mechanism.

export function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rnd) {
  return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
}

/** rows x cols of seeded noise, scaled — the stand-in for trained weights. */
export function randMat(rnd, rows, cols, scale) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) row[j] = gauss(rnd) * scale;
    m.push(row);
  }
  return m;
}

/** v (len rows) times W (rows x cols) -> len cols. */
export function matvec(v, W) {
  const cols = W[0].length;
  const out = new Float64Array(cols);
  for (let i = 0; i < v.length; i++) {
    const vi = v[i];
    if (vi === 0) continue;
    const row = W[i];
    for (let j = 0; j < cols; j++) out[j] += vi * row[j];
  }
  return out;
}

export function dot(a, b, off = 0, n = a.length) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[off + i] * b[off + i];
  return s;
}

export function add(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function layerNorm(v) {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length;
  let varr = 0;
  for (let i = 0; i < v.length; i++) varr += (v[i] - mean) ** 2;
  varr /= v.length;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = (v[i] - mean) * inv;
  return out;
}

export const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

/** Max-shifted, so a row that is entirely -Infinity cannot produce NaN. */
export function softmax(scores, temperature = 1) {
  let hi = -Infinity;
  for (const s of scores) if (s > hi) hi = s;
  if (!Number.isFinite(hi)) return scores.map(() => 0);
  const ex = scores.map((s) => (Number.isFinite(s) ? Math.exp((s - hi) / temperature) : 0));
  let total = 0;
  for (const e of ex) total += e;
  return ex.map((e) => e / total);
}

export const fmt = (x, p = 2) => (Number.isFinite(x) ? x.toFixed(p) : "−∞");

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
