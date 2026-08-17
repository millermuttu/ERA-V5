// The seam. A block hands a mixer the per-token queries, keys and values for one head and gets
// back the mixed output plus whatever the views need to draw. Every mechanism in the timeline is
// a configuration of one of these two, which is why the comparison between them is fair: same
// weights, same sentence, same everything except the rule being explained.
import { dot, softmax } from "./ops.js";

/**
 * Softmax attention.
 *
 *   readable(i, j)   false hides key j from query i entirely (sparse patterns, windows, top-k)
 *   bias(i, j, q, at)  added to the score before softmax. The query vector is passed because
 *                    Shaw et al.'s relative term is q_i · w_clip(j−i), not a scalar per offset;
 *                    ALiBi ignores the query but needs `at.head`, because its slope is per head.
 *                    Note it is added after the scaling, not divided by sqrt(d_k).
 *   rotate(v, pos)   applied to q and k before the dot product (RoPE and its extensions)
 *
 * Defaults are plain causal attention — the baseline the whole app is measured against.
 */
export function softmaxMixer({ readable = null, bias = null, rotate = null } = {}) {
  return function mix(Q, K, V, dh, at = {}) {
    const T = Q.length;
    const scale = Math.sqrt(dh);
    const scores = [];
    const weights = [];
    const out = [];
    let reads = 0;

    for (let i = 0; i < T; i++) {
      const q = rotate ? rotate(Q[i], i) : Q[i];
      const row = new Array(T).fill(-Infinity);
      for (let j = 0; j <= i; j++) {
        if (readable && !readable(i, j, at)) continue;
        const k = rotate ? rotate(K[j], j) : K[j];
        row[j] = dot(q, k) / scale + (bias ? bias(i, j, q, at) : 0);
        reads++;
      }
      const w = softmax(row);
      const o = new Float64Array(dh);
      for (let j = 0; j <= i; j++) {
        if (w[j] === 0) continue;
        for (let d = 0; d < dh; d++) o[d] += w[j] * V[j][d];
      }
      scores.push(row);
      weights.push(w);
      out.push(o);
    }
    return { out, scores, weights, reads, kind: "softmax" };
  };
}

/**
 * Fixed-size recurrent state — the linear-attention family.
 *
 *   write: "add"    S += v kᵀ            accumulate, never correct
 *          "delta"  S += (v − S k) kᵀ    read first, write only the difference
 *          "gated"  S = gS + (v − gS k)kᵀ  as delta, but the past decays first
 *
 * `phi` is the feature map standing in for the exponential softmax removed. The state is dh x dh
 * whatever the sequence length, which is the entire point.
 */
/** The paper's feature map: elu(x) + 1. Non-negative everywhere, and unlike relu it has no
 *  zero-gradient region — which is the stated reason they rejected relu. */
export const elu1 = (x) => (x > 0 ? x + 1 : Math.exp(x));

export function stateMixer({ write = "add", decay = 1, phi = elu1, features = null } = {}) {
  // `features` is a vector-valued map R^dh -> R^m, which is what a random-feature method needs:
  // Performer's φ projects into m random directions, so the state is m x dh, not dh x dh. When it
  // is absent the elementwise `phi` applies and m === dh, which is the linear-attention case.
  return function mix(Q, K, V, dh, at = {}) {
    const T = Q.length;
    const map = features || ((v) => Float64Array.from(v, phi));
    const m = map(Q[0]).length;
    let S = Array.from({ length: dh }, () => new Float64Array(m));
    const out = [];
    const snapshots = [];
    const denominators = [];
    let norm = new Float64Array(m);

    for (let i = 0; i < T; i++) {
      const k = map(K[i]);
      const q = map(Q[i]);
      const g = write === "gated" ? decay : 1;

      // read what the state currently returns for this key
      const cur = new Float64Array(dh);
      for (let a = 0; a < dh; a++) cur[a] = dot(S[a], k, 0, m) * g;

      for (let a = 0; a < dh; a++) {
        const target = write === "add" ? V[i][a] : V[i][a] - cur[a];
        for (let b = 0; b < m; b++) S[a][b] = S[a][b] * g + target * k[b];
      }
      for (let b = 0; b < m; b++) norm[b] = norm[b] * g + k[b];

      // The denominator, kept as computed. Taking its absolute value would quietly rescue a
      // negative normaliser — and a negative normaliser is exactly what a feature map that is not
      // non-negative produces. The requirement is visible only if the failure is left visible.
      const o = new Float64Array(dh);
      let z = 0;
      for (let b = 0; b < m; b++) z += norm[b] * q[b];
      const safe = Math.abs(z) < 1e-9 ? (z < 0 ? -1e-9 : 1e-9) : z;
      for (let a = 0; a < dh; a++) o[a] = dot(S[a], q, 0, m) / safe;
      denominators.push(z);
      out.push(o);

      snapshots.push(S.map((r) => Float64Array.from(r)));
    }
    // A state model reads one object per token, not a growing list of keys.
    return { out, scores: null, weights: null, reads: T, state: S, snapshots, denominators, m, kind: "state" };
  };
}

/**
 * Key/value head sharing. `groups` key/value heads serve `heads` query heads; groups === heads is
 * ordinary multi-head, groups === 1 is multi-query. Returns which kv head a query head reads.
 */
export const kvHeadFor = (head, heads, groups) => Math.floor(head / (heads / groups));
