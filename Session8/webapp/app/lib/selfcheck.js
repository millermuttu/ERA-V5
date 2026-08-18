// The whole test suite. No framework: assertions over the model, over each mechanism's
// implementation, and over the integrity of the chronology. Run with ?selfcheck=1.
import { el } from "./dom.js";
import { softmax, dot } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { softmaxMixer, stateMixer, kvHeadFor } from "../model/mixers.js";
import { cacheBytes, SERVING, GB } from "../model/cost.js";
import { sinusoidalVector, learnedTable, relativeBuckets, rope } from "../model/position.js";
import { tokenize, PRESETS } from "../model/vocab.js";
import { mechanisms } from "../data/mechanisms.js";
import { CARDS } from "../cards/index.js";

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const TOKENS = tokenize(PRESETS[0]);

export function checks() {
  const out = [];
  const ok = (name, pass, detail = "") => out.push({ name, pass, detail });

  // ---------------------------------------------------------------- the model
  const base = forward(TOKENS);
  const w = base.trace[0].heads[0].weights;
  ok("softmax rows sum to 1", w.every((r) => near(r.reduce((a, b) => a + b, 0), 1)));
  ok("masked future weight is exactly 0", w.every((r, i) => r.every((x, j) => j <= i || x === 0)));
  ok("softmax survives an all-but-one masked row", near(softmax([-Infinity, -Infinity, 2])[2], 1));
  ok("the forward pass is deterministic", JSON.stringify(forward(TOKENS).top) === JSON.stringify(base.top));
  ok("the output distribution is a distribution", near(base.probs.reduce((a, b) => a + b, 0), 1));

  // ------------------------------------------------- mechanisms vs the baseline
  // Anything that claims to reduce to plain attention at its degenerate setting must do so.
  // Compare every head, not just head 0 — under key/value sharing head 0 keeps its own kv head
  // and is identical by construction, so checking it alone would pass a broken implementation.
  const same = (mech) => {
    const r = forward(TOKENS, mech);
    return r.trace.every((blk, b) =>
      blk.heads.every((h, hi) =>
        h.out.every((v, i) => v.every((x, d) => near(x, base.trace[b].heads[hi].out[i][d], 1e-12)))
      )
    );
  };
  ok("a readability rule that hides nothing equals the baseline", same({ mixer: softmaxMixer({ readable: () => true }) }));
  ok("a zero score bias equals the baseline", same({ mixer: softmaxMixer({ bias: () => 0 }) }));
  ok("a window as wide as the context equals the baseline", same({ mixer: softmaxMixer({ readable: (i, j) => i - j < 999 }) }));
  ok("key/value sharing at one group per head equals the baseline", same({ kvGroups: CONFIG.HEADS }));
  ok(
    "sharing fewer key/value heads does change the answer",
    !same({ kvGroups: 1 }),
    "so the sharing is real, not cosmetic"
  );
  ok(
    "a shared key/value head is mean-pooled, not one head picked",
    (() => {
      const mean = forward(TOKENS, { kvGroups: 1 });
      const sel = forward(TOKENS, { kvGroups: 1, kvPool: "select" });
      return !mean.trace[0].heads[1].out.every((v, i) =>
        v.every((x, d) => near(x, sel.trace[0].heads[1].out[i][d], 1e-12))
      );
    })(),
    "GQA compares the two, so both have to exist"
  );
  ok(
    "kv head mapping groups query heads correctly",
    kvHeadFor(0, 4, 2) === 0 && kvHeadFor(1, 4, 2) === 0 && kvHeadFor(2, 4, 2) === 1 && kvHeadFor(3, 4, 2) === 1
  );

  // A hidden key must really be gone, not merely down-weighted.
  const hidden = forward(TOKENS, { mixer: softmaxMixer({ readable: (i, j) => j !== 1 }) });
  ok("a hidden key gets exactly zero weight", hidden.trace[0].heads[0].weights.every((r) => r[1] === 0));

  // ------------------------------------------------------ the softmax-free path
  // The lesson's own numbers: with softmax removed, the direct sum and the pre-built state agree.
  const q = 2;
  const keys = [0.5, 1, 1.5];
  const vals = [10, 20, 30];
  const direct = keys.reduce((s, k, i) => s + q * k * vals[i], 0);
  const S = keys.reduce((s, k, i) => s + k * vals[i], 0);
  ok("regrouping matches the direct sum without softmax", near(direct, q * S) && direct === 140, "140");
  const smax = softmax(keys.map((k) => q * k));
  ok("with softmax the two stop agreeing", !near(smax.reduce((s, x, i) => s + x * vals[i], 0), q * S));
  ok("an add-only write overshoots to 95", 40 + 55 === 95);
  ok("the delta rule lands on 55", 40 + (55 - 40) === 55);
  // The paper's sum normalisation is load-bearing, not cosmetic: without it the delta step is a
  // gradient step with learning rate ~34 and the state leaves the range of the display entirely.
  const maxState = (mech) => {
    const s = forward(TOKENS, { mixer: stateMixer(mech) }).trace[0].heads[0].state;
    return Math.max(...s.flatMap((r) => Array.from(r, Math.abs)));
  };
  ok("the delta state stays bounded with the paper's normalisation", maxState({ write: "delta" }) < 10, maxState({ write: "delta" }).toFixed(2));
  ok(
    "and diverges without it, which is why the normalisation exists",
    maxState({ write: "delta", sumNorm: false }) > 1e6,
    maxState({ write: "delta", sumNorm: false }).toExponential(1)
  );
  ok(
    "the write strength scales the correction and nothing else",
    (() => {
      const zero = forward(TOKENS, { mixer: stateMixer({ write: "delta", beta: 0 }) }).trace[0].heads[0].state;
      return zero.every((r) => r.every((x) => x === 0));
    })(),
    "β = 0 writes nothing at all"
  );
  ok("the state mixer runs and keeps a fixed-size state", (() => {
    const r = forward(TOKENS, { mixer: stateMixer({ write: "delta" }) });
    const h = r.trace[0].heads[0];
    return h.kind === "state" && h.state.length === DH && h.state[0].length === DH;
  })());

  // ------------------------------------------------------------------ position
  const offsetInvariant = [1, 4, 10].every((k) => {
    const d = [0, 5, 20, 50].map((p) => dot(sinusoidalVector(p), sinusoidalVector(p + k)));
    return d.every((x) => near(x, d[0], 1e-9));
  });
  ok("sinusoidal similarity depends on the offset, not the position", offsetInvariant);
  const table = learnedTable({ rows: 6 });
  ok("a learned table has nothing past its last row", table.inRange(5) && !table.inRange(6) && table.vector(6) === null);
  const rel = relativeBuckets({ k: 3, dims: DH });
  ok(
    "relative buckets clip past k into one bucket",
    rel.vector(-3) === rel.vector(-7) && rel.vector(-1) !== rel.vector(-3)
  );
  const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.4 : -0.3));
  ok(
    "the relative term is query-dependent, as Shaw eq. 5 requires",
    rel.bias(5, 2, probe) !== rel.bias(5, 2, probe.map((x) => -x))
  );

  // The base change has to hit both endpoints exactly, or it is not the method: the slowest pair
  // must land on what interpolation would have given it, and the fastest must not move at all.
  const scale = 4;
  const plainRope = rope({ dims: DH });
  const ntkRope = rope({ base: 10000 * Math.pow(scale, DH / (DH - 2)), dims: DH });
  const last = DH / 2 - 1;
  ok("a base change leaves the fastest pair exactly alone", ntkRope.freqs[0] === plainRope.freqs[0]);
  ok(
    "and lands the slowest pair exactly where interpolation would have put it",
    near(ntkRope.freqs[last], plainRope.freqs[last] / scale, 1e-15),
    (plainRope.freqs[last] / ntkRope.freqs[last]).toFixed(6)
  );
  ok(
    "the pairs between are compressed by less than asked, which is why the scale under-delivers",
    plainRope.freqs.every((f, i) => f / ntkRope.freqs[i] <= scale + 1e-12) &&
      plainRope.freqs.some((f, i) => i > 0 && i < last && f / ntkRope.freqs[i] < scale - 1e-9)
  );
  ok(
    "a base change is still exactly relative",
    (() => {
      const at = (sh) =>
        forward(TOKENS, { mixer: softmaxMixer({ rotate: (v, p) => ntkRope.rotate(v, p + sh) }) }).trace[0].heads[0]
          .weights;
      const a = at(0);
      const b = at(4096);
      return a.every((r, i) => r.every((x, j) => near(x, b[i][j], 1e-12)));
    })(),
    "shifting the sentence by 4096 leaves the attention matrix alone"
  );

  // YaRN's ramp is a per-pair choice between interpolation and doing nothing, so the seam has to
  // reproduce both endpoints exactly — otherwise the crossfade is measuring something else.
  ok(
    "a per-pair stretch of 1 everywhere is plain rotation",
    (() => {
      const a = rope({ dims: DH });
      const b = rope({ dims: DH, stretch: () => 1 });
      return a.freqs.every((f, i) => a.applied(i) === b.applied(i));
    })()
  );
  ok(
    "and a per-pair stretch of 1/s everywhere is interpolation",
    (() => {
      const flat = rope({ dims: DH, stretch: 1 / scale });
      const fn = rope({ dims: DH, stretch: () => 1 / scale });
      const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.7 : -0.4));
      return flat.rotate(probe, 9).every((x, i) => near(x, fn.rotate(probe, 9)[i], 1e-15));
    })()
  );
  ok(
    "the temperature is a modulus on the rotation, so it scales the logit by its square",
    (() => {
      const t = 1.2079; // 0.1·ln(8) + 1, the paper's own worked value
      const plainR = rope({ dims: DH });
      const hot = rope({ dims: DH, modulus: t });
      const a = Float64Array.from({ length: DH }, (_, i) => (i % 3 ? 0.5 : -0.9));
      const b = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? -0.3 : 0.8));
      const cold = dot(plainR.rotate(a, 3), plainR.rotate(b, 7));
      return near(dot(hot.rotate(a, 3), hot.rotate(b, 7)), cold * t * t, 1e-12);
    })(),
    "which is why it needs no change to the attention code"
  );
  ok(
    "a temperature of 1 leaves the rotation's length alone, as RoPE's does",
    (() => {
      const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.6 : -0.2));
      const r = rope({ dims: DH });
      return near(dot(r.rotate(probe, 11), r.rotate(probe, 11)), dot(probe, probe), 1e-12);
    })()
  );

  // The collapse the streaming card is about is one identity: drop keys from a softmax row and
  // every survivor is multiplied by the same number. If that stops holding, the card's headline
  // readout is wrong rather than merely surprising.
  ok(
    "hiding keys multiplies every survivor by one shared number",
    (() => {
      const policy = (i, j) => j < 2 || j >= i - 5 + 1;
      const T = TOKENS.length;
      const head = base.trace[0].heads[0];
      const cut = forward(TOKENS, { mixer: softmaxMixer({ readable: policy }) }).trace[0].heads[0];
      const ratios = [];
      for (let j = 0; j < T; j++) {
        if (!policy(T - 1, j)) continue;
        if (head.weights[T - 1][j] < 1e-9) continue;
        ratios.push(cut.weights[T - 1][j] / head.weights[T - 1][j]);
      }
      return ratios.length > 1 && ratios.every((r) => near(r, ratios[0], 1e-9));
    })(),
    "the shape of what remains is untouched, the scale is not"
  );
  ok(
    "the front of the cache is kept whatever the window does",
    (() => {
      const policy = (i, j) => j < 4 || j >= i - 3 + 1;
      const w = forward(TOKENS, { mixer: softmaxMixer({ readable: policy }) }).trace[0].heads[0].weights;
      const last = w[TOKENS.length - 1];
      return last[0] > 0 && last[4] === 0;
    })(),
    "sinks survive, the middle does not"
  );

  // ---------------------------------------------------------------------- cost
  const gb = (o = {}) => cacheBytes({ ...SERVING, ...o }) / GB;
  ok("cache: one conversation is 6.44 GB", near(gb(), 6.44, 0.005), gb().toFixed(3));
  ok("cache: eight conversations is 51.54 GB", near(gb({ batch: 8 }), 51.54, 0.005), gb({ batch: 8 }).toFixed(3));
  ok("cache: four times the kv heads is four times the cache", near(gb({ kvHeads: 32 }) / gb({ kvHeads: 8 }), 4));

  // ----------------------------------------------------------------- the record
  const ids = mechanisms.map((m) => m.id);
  ok("mechanism ids are unique", new Set(ids).size === ids.length);
  ok("every date parses", mechanisms.every((m) => /^\d{4}-\d{2}-\d{2}$/.test(m.date) && !isNaN(Date.parse(m.date))));
  const sorted = [...mechanisms].sort((a, b) => a.date.localeCompare(b.date));
  ok("the baseline is the earliest entry", sorted[0].id === "transformer", sorted[0].id);
  const badVerified = mechanisms.filter((m) => m.verified && m.source.kind !== "paper");
  ok("only paper-backed entries are marked verified", badVerified.length === 0, badVerified.map((m) => m.id).join(", "));

  const built = mechanisms.filter((m) => m.status !== "pending");
  const noCard = built.filter((m) => !CARDS[m.id]);
  ok("every built concept resolves to a card the model can run", noCard.length === 0, noCard.map((m) => m.id).join(", "));

  const linkless = built.filter((m) => !m.leaves?.text || (m.id !== "transformer" && !m.answers));
  ok(
    "every built concept names what it answers and what it leaves behind",
    linkless.length === 0,
    linkless.map((m) => m.id).join(", ")
  );
  const danglingBack = built.filter((m) => m.answers && !ids.includes(m.answers));
  const danglingFwd = built.filter((m) => m.leaves?.to && !ids.includes(m.leaves.to));
  ok(
    "those links point at entries that exist",
    danglingBack.length + danglingFwd.length === 0,
    [...danglingBack, ...danglingFwd].map((m) => m.id).join(", ")
  );
  const backwards = built.filter((m) => {
    if (!m.answers) return false;
    const other = mechanisms.find((x) => x.id === m.answers);
    return other && other.date > m.date;
  });
  ok("nothing answers a limitation that had not happened yet", backwards.length === 0, backwards.map((m) => m.id).join(", "));

  // Mount every built card into a detached node: catches a card that throws, and one that
  // forgot its trade-off record or its plain-language verdict.
  const broken = [];
  const missingTrade = [];
  const missingPlain = [];
  for (const m of built) {
    const host = document.createElement("div");
    try {
      const mounted = CARDS[m.id](host, m) || {};
      if (mounted.update) mounted.update();
    } catch (err) {
      broken.push(`${m.id}: ${err.message}`);
      continue;
    }
    const trades = host.querySelectorAll(".trade");
    if (trades.length !== 3) missingTrade.push(m.id);
    const plain = host.querySelector(".plain");
    const pros = host.querySelectorAll(".plain .pros li").length;
    const cons = host.querySelectorAll(".plain .cons li").length;
    if (!plain || !pros || !cons || !host.querySelector(".plain .verdict")) missingPlain.push(m.id);
    else if (cons === 0) missingPlain.push(m.id);
  }
  ok("every built card renders without throwing", broken.length === 0, broken.join(" | "));
  ok("every built card answers buys / gives up / when", missingTrade.length === 0, missingTrade.join(", "));
  ok(
    "every built card ends with a plain-language verdict that includes costs",
    missingPlain.length === 0,
    missingPlain.join(", ")
  );

  const pending = mechanisms.filter((m) => m.status === "pending");
  ok(
    `${mechanisms.length - pending.length} of ${mechanisms.length} concepts built`,
    true,
    pending.length ? `still to come: ${pending.map((m) => m.id).join(", ")}` : "all done"
  );

  return out;
}

export function runSelfCheck() {
  const results = checks();
  const failed = results.filter((r) => !r.pass);
  const box = el("div", { id: "selfcheck", class: failed.length ? "fail" : "pass" }, [
    el("div", {
      text: failed.length
        ? `${failed.length} of ${results.length} checks FAILED`
        : `all ${results.length} checks pass`,
    }),
    el(
      "ul",
      {},
      (failed.length ? failed : results).map((r) =>
        el("li", { text: `${r.pass ? "✓" : "✗"} ${r.name}${r.detail ? " — " + r.detail : ""}` })
      )
    ),
  ]);
  document.body.appendChild(box);
  return results;
}
