// The seam. A block hands a mixer the per-token queries, keys and values for one head and gets
// back the mixed output plus whatever the views need to draw. Every mechanism in the timeline is
// a configuration of one of these two, which is why the comparison between them is fair: same
// weights, same sentence, same everything except the rule being explained.
import { dot, softmax } from "./ops.js";

/**
 * Softmax attention.
 *
 *   readable(i, j)   false hides key j from query i entirely (sparse patterns, windows, top-k)
 *   bias(i, j, q)    added to the score before softmax. The query vector is passed because
 *                    Shaw et al.'s relative term is q_i · w_clip(j−i), not a scalar per offset;
 *                    ALiBi simply ignores it.
 *   rotate(v, pos)   applied to q and k before the dot product (RoPE and its extensions)
 *
 * Defaults are plain causal attention — the baseline the whole app is measured against.
 */
export function softmaxMixer({ readable = null, bias = null, rotate = null } = {}) {
  return function mix(Q, K, V, dh) {
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
        if (readable && !readable(i, j)) continue;
        const k = rotate ? rotate(K[j], j) : K[j];
        row[j] = dot(q, k) / scale + (bias ? bias(i, j, q) : 0);
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
export function stateMixer({ write = "add", decay = 1, phi = (x) => Math.max(x, 0) + 0.01 } = {}) {
  return function mix(Q, K, V, dh) {
    const T = Q.length;
    let S = Array.from({ length: dh }, () => new Float64Array(dh));
    const out = [];
    const snapshots = [];
    let norm = new Float64Array(dh);

    for (let i = 0; i < T; i++) {
      const k = Array.from(K[i], phi);
      const q = Array.from(Q[i], phi);
      const g = write === "gated" ? decay : 1;

      // read what the state currently returns for this key
      const cur = new Float64Array(dh);
      for (let a = 0; a < dh; a++) cur[a] = dot(S[a], k, 0, dh) * g;

      for (let a = 0; a < dh; a++) {
        const target = write === "add" ? V[i][a] : V[i][a] - cur[a];
        for (let b = 0; b < dh; b++) S[a][b] = S[a][b] * g + target * k[b];
      }
      for (let b = 0; b < dh; b++) norm[b] = norm[b] * g + k[b];

      const o = new Float64Array(dh);
      let z = 0;
      for (let b = 0; b < dh; b++) z += norm[b] * q[b];
      for (let a = 0; a < dh; a++) o[a] = dot(S[a], q, 0, dh) / (Math.abs(z) + 1e-6);
      out.push(o);

      snapshots.push(S.map((r) => Float64Array.from(r)));
    }
    // A state model reads one object per token, not a growing list of keys.
    return { out, scores: null, weights: null, reads: T, state: S, snapshots, kind: "state" };
  };
}

/**
 * Key/value head sharing. `groups` key/value heads serve `heads` query heads; groups === heads is
 * ordinary multi-head, groups === 1 is multi-query. Returns which kv head a query head reads.
 */
export const kvHeadFor = (head, heads, groups) => Math.floor(head / (heads / groups));
