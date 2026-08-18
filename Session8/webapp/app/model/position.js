// Positional schemes. Each is a small object the model can consume:
//   add(vec, pos)   modifies the input embedding (sinusoidal, learned table)
//   bias(i, j)      added to the score before softmax (relative buckets, ALiBi)
//   rotate(v, pos)  applied to q and k before the dot product (RoPE and its extensions)
// A scheme uses whichever hooks it needs; the mixer and the model take it from there.
import { CONFIG, DH } from "./transformer.js";
import { mulberry32, gauss, dot } from "./ops.js";

/** Vaswani §3.5: PE(pos,2i) = sin(pos / 10000^(2i/d)), PE(pos,2i+1) = cos(same). */
export function sinusoidalVector(pos, d = CONFIG.D, base = 10000) {
  const v = new Float64Array(d);
  for (let i = 0; i < d / 2; i++) {
    const w = 1 / Math.pow(base, (2 * i) / d);
    v[2 * i] = Math.sin(pos * w);
    v[2 * i + 1] = Math.cos(pos * w);
  }
  return v;
}

export const sinusoidal = ({ base = 10000, scale = 1 } = {}) => ({
  kind: "sinusoidal",
  add(vec, pos) {
    const pe = sinusoidalVector(pos, vec.length, base);
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i] + pe[i] * scale;
    return out;
  },
  vector: (pos) => sinusoidalVector(pos, CONFIG.D, base),
});

/**
 * A learned table: one trainable row per position, and nothing at all past the last row.
 * Seeded noise stands in for training. `rows` is the wall.
 */
export function learnedTable({ rows = 12, seed = 7 } = {}) {
  const rnd = mulberry32(seed);
  const table = Array.from({ length: rows }, () =>
    Float64Array.from({ length: CONFIG.D }, () => gauss(rnd) * 0.35)
  );
  return {
    kind: "learned",
    rows,
    /** Past the last row there is no vector to add — the model is off the end of its table. */
    inRange: (pos) => pos < rows,
    add(vec, pos) {
      if (pos >= rows) return vec; // nothing to add: the wall
      const out = new Float64Array(vec.length);
      for (let i = 0; i < vec.length; i++) out[i] = vec[i] + table[pos][i];
      return out;
    },
    vector: (pos) => (pos < rows ? table[pos] : null),
  };
}

/**
 * Shaw et al. eq. 5: a learned *vector* per clipped relative distance, entering the score as
 * q_i · w_clip(j−i) / sqrt(d_z) — query-dependent, which is why bias() is handed the query.
 * clip(x, k) = max(-k, min(k, x)); everything past k shares one bucket.
 */
export function relativeBuckets({ k = 4, seed = 11, dims = DH } = {}) {
  const rnd = mulberry32(seed);
  const buckets = Array.from({ length: 2 * k + 1 }, () =>
    Float64Array.from({ length: dims }, () => gauss(rnd) * 0.5)
  );
  const clip = (x) => Math.max(-k, Math.min(k, x));
  const scale = Math.sqrt(dims);
  return {
    kind: "relative",
    k,
    clip,
    buckets,
    bucketOf: (i, j) => clip(j - i) + k,
    /** The query is required: the term is a dot product, not a scalar looked up by offset. */
    bias: (i, j, q) => (q ? dot(q, buckets[clip(j - i) + k]) / scale : 0),
    vector: (d) => buckets[clip(d) + k],
  };
}

/**
 * ALiBi: a per-head slope times the distance, subtracted from the score.
 * The slopes are a geometric sequence starting at 2^(-8/n) and stepping by that same ratio, so
 * whatever the head count the last head always lands on 1/256. Not learned — the paper tried that
 * and reports it did not extrapolate well. Not divided by sqrt(d_k) either.
 */
export const alibiSlopes = (heads) =>
  Array.from({ length: heads }, (_, h) => Math.pow(2, (-8 * (h + 1)) / heads));

export const alibi = ({ heads = CONFIG.HEADS, only = null } = {}) => {
  const slopes = alibiSlopes(heads);
  return {
    kind: "alibi",
    slopes,
    bias: (i, j, _q, at = {}) => -(i - j) * slopes[only ?? at.head ?? 0],
  };
};

/**
 * RoPE: rotate dimension pairs by position times a per-pair frequency.
 *
 * `stretch` is a multiplier on the angle — one number for every pair (position interpolation), or a
 * function of the pair index when the scheme treats pairs differently (YaRN's ramp). `modulus` is
 * the length the rotation is given: RoPE's own rotation has modulus 1 and so preserves the vector's
 * length, and YaRN's attention temperature is exactly this constant set to something else, which
 * scales the q·k logit by its square without touching the softmax at all.
 */
export function rope({ base = 10000, stretch = 1, dims = DH, modulus = 1 } = {}) {
  const freqs = Array.from({ length: dims / 2 }, (_, i) => 1 / Math.pow(base, (2 * i) / dims));
  const rate = typeof stretch === "function" ? stretch : () => stretch;
  return {
    kind: "rope",
    freqs,
    rate,
    modulus,
    /** The frequency the scheme actually applies to pair i, after whatever it does to the ladder. */
    applied: (i) => freqs[i] * rate(i),
    rotate(v, pos) {
      const out = Float64Array.from(v);
      for (let i = 0; i < freqs.length && 2 * i + 1 < v.length; i++) {
        const a = pos * freqs[i] * rate(i);
        const c = Math.cos(a) * modulus;
        const s = Math.sin(a) * modulus;
        const x = v[2 * i];
        const y = v[2 * i + 1];
        out[2 * i] = x * c - y * s;
        out[2 * i + 1] = x * s + y * c;
      }
      return out;
    },
  };
}
