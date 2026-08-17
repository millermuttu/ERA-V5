// The chronology. One record per concept; the deck, the timeline slider, the README source table
// and the self-check all read this array.
//
// Dates are arXiv v1 submission dates, read from the arXiv API rather than recalled. That pass
// corrected two of them: multi-query attention is 2019-11-06, parallelizable DeltaNet 2024-06-10.
//
// A record is filled in by its own research → build pass, one concept at a time. `status: "pending"`
// means the note has not been written yet and the card is a placeholder; the self-check reports how
// many remain rather than failing the whole app while the work is in progress.
//
// Filled record:
//   id, name, date, source {label,url,kind}, verified, pressure
//   answers   the id of the earlier entry whose limitation it addresses
//   leaves    what its own cost left for later work, and the id that picked it up
//   problem, mechanism      the technical narrative
//   buys[], givesUp[], chooseWhen        the assignment's three questions
//   plain     { pros[], cons[], verdict } same trade in everyday words, no formulas, no jargon
//   config    what to hand the model, plus the controls the card exposes
//   views     which visual panels the card mounts

export const PRESSURES = {
  mechanism: "the mechanism itself",
  position: "where a token sits",
  cache: "what generation must remember",
  compute: "how many comparisons",
  state: "compressing the past",
  systems: "how it meets the hardware",
};

const pending = (id, name, date, source, pressure) => ({
  id,
  name,
  date,
  source,
  verified: source.kind === "paper",
  pressure,
  status: "pending",
});

// A concept becomes "built" when its research note is written and its card is live. The record
// carries the narrative the deck renders around the card's own panels; the card itself holds the
// prose and the trade-off blocks, so what lives here is the timeline's forward and backward links.
const built = (id, answers, leaves) => (m) =>
  m.id === id ? { ...m, status: "built", answers, leaves } : m;

const paper = (label, id) => ({ label, url: `https://arxiv.org/abs/${id}`, kind: "paper" });

export const mechanisms = [
  pending("transformer", "Scaled dot-product attention, multi-head", "2017-06-12",
    paper("Vaswani et al., Attention Is All You Need — arXiv:1706.03762", "1706.03762"), "mechanism"),

  pending("sinusoidal", "Sinusoidal position encoding", "2017-06-12",
    paper("Vaswani et al., Attention Is All You Need §3.5 — arXiv:1706.03762", "1706.03762"), "position"),

  pending("relative-positions", "Relative position representations", "2018-03-06",
    paper("Shaw et al., Self-Attention with Relative Position Representations — arXiv:1803.02155", "1803.02155"), "position"),

  pending("learned-absolute", "Learned absolute position tables", "2018-10-11",
    paper("Devlin et al., BERT — arXiv:1810.04805", "1810.04805"), "position"),

  pending("transformer-xl", "Segment recurrence across contexts", "2019-01-09",
    paper("Dai et al., Transformer-XL — arXiv:1901.02860", "1901.02860"), "state"),

  pending("sparse-transformer", "Sparse Transformer, strided and fixed patterns", "2019-04-23",
    paper("Child et al., Generating Long Sequences with Sparse Transformers — arXiv:1904.10509", "1904.10509"), "compute"),

  pending("mqa", "Multi-query attention", "2019-11-06",
    paper("Shazeer, Fast Transformer Decoding — arXiv:1911.02150", "1911.02150"), "cache"),

  pending("sliding-window", "Sliding window attention with global tokens", "2020-04-10",
    paper("Beltagy et al., Longformer — arXiv:2004.05150", "2004.05150"), "compute"),

  pending("linear-attention", "Linear attention, the kernel regrouping", "2020-06-29",
    paper("Katharopoulos et al., Transformers are RNNs — arXiv:2006.16236", "2006.16236"), "state"),

  pending("performer", "Performer, FAVOR+", "2020-09-30",
    paper("Choromanski et al., Rethinking Attention with Performers — arXiv:2009.14794", "2009.14794"), "compute"),

  pending("delta-rule", "The delta rule, fast-weight programmers", "2021-02-22",
    paper("Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers — arXiv:2102.11174", "2102.11174"), "state"),

  pending("rope", "RoPE, rotary position embedding", "2021-04-20",
    paper("Su et al., RoFormer — arXiv:2104.09864", "2104.09864"), "position"),

  pending("alibi", "ALiBi, attention with linear biases", "2021-08-27",
    paper("Press et al., Train Short, Test Long — arXiv:2108.12409", "2108.12409"), "position"),

  pending("flashattention", "FlashAttention", "2022-05-27",
    paper("Dao et al., FlashAttention — arXiv:2205.14135", "2205.14135"), "systems"),

  pending("gqa", "Grouped-query attention", "2023-05-22",
    paper("Ainslie et al., GQA — arXiv:2305.13245", "2305.13245"), "cache"),

  pending("position-interpolation", "Position interpolation", "2023-06-27",
    paper("Chen et al., Extending Context Window via Position Interpolation — arXiv:2306.15595", "2306.15595"), "position"),

  pending("ntk-aware", "NTK-aware base scaling", "2023-06-30", {
    label: "bloc97, community post on r/LocalLLaMA — no paper; refined later into YaRN",
    url: "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/",
    kind: "post",
  }, "position"),

  pending("yarn", "YaRN", "2023-08-31",
    paper("Peng et al., YaRN — arXiv:2309.00071", "2309.00071"), "position"),

  pending("attention-sinks", "Attention sinks, StreamingLLM", "2023-09-29",
    paper("Xiao et al., Efficient Streaming Language Models with Attention Sinks — arXiv:2309.17453", "2309.17453"), "cache"),

  pending("mamba", "Selective state space", "2023-12-01",
    paper("Gu and Dao, Mamba — arXiv:2312.00752", "2312.00752"), "state"),

  pending("mla", "Multi-head latent attention", "2024-05-07",
    paper("DeepSeek-AI, DeepSeek-V2 — arXiv:2405.04434", "2405.04434"), "cache"),

  pending("parallel-deltanet", "Parallelizable DeltaNet", "2024-06-10",
    paper("Yang et al., Parallelizing Linear Transformers with the Delta Rule — arXiv:2406.06484", "2406.06484"), "state"),

  pending("gated-deltanet", "Gated DeltaNet", "2024-12-09",
    paper("Yang et al., Gated Delta Networks — arXiv:2412.06464", "2412.06464"), "state"),

  pending("nsa", "Natively trainable sparse attention", "2025-02-16",
    paper("Yuan et al., Native Sparse Attention — arXiv:2502.11089", "2502.11089"), "compute"),

  pending("dsa", "DeepSeek sparse attention", "2025-09-29", {
    label: "DeepSeek-V3.2-Exp release — model card and tech report, no arXiv v1 to cite",
    url: "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp",
    kind: "release",
  }, "compute"),

  // No public source, so no date to verify. The value below is a sort key that places this entry
  // last; the card says so in as many words and never presents it as a date.
  pending("drope", "DroPE", "2026-01-01", {
    label: "LightningLM V4 cookbook, via the ERA V5 Session 8 lesson — no public source",
    url: "",
    kind: "course",
  }, "position"),
];

const LINKS = [
  built("transformer", null, {
    text: "two bills — a score matrix that grows with the square of the context, and a key/value cache that grows with every token generated",
    to: "sparse-transformer",
  }),
  built("sinusoidal", "transformer", {
    text: "position added into the same vector as content, and a relative-offset property the paper only hypothesised",
    to: "relative-positions",
  }),
  built("relative-positions", "sinusoidal", {
    text: "a per-pair term that stops the scores being one matrix multiply, at about 7% of training throughput",
    to: "rope",
  }),
  built("learned-absolute", "sinusoidal", {
    text: "a table with a last row, so text longer than training simply cannot be positioned",
    to: "rope",
  }),
  built("transformer-xl", "transformer", {
    text: "a read-only cache chosen by recency rather than learned, bounded at roughly layers times memory",
    to: "mamba",
  }),
  built("mqa", "transformer", {
    text: "a measured quality cost from making every head share one view of the past, and a cache that still grows with the conversation",
    to: "gqa",
  }),
  built("sliding-window", "sparse-transformer", {
    text: "a window, a dilation and a set of global tokens all chosen by hand before the data arrives",
    to: "nsa",
  }),
  built("linear-attention", "sliding-window", {
    text: "an add-only state that cannot correct an association it has already written, and no exact retrieval of any earlier token",
    to: "delta-rule",
  }),
  built("performer", "linear-attention", {
    text: "an estimate whose error falls only as the square root of the features, so accuracy and speed trade against each other directly",
    to: "flashattention",
  }),
  built("rope", "relative-positions", {
    text: "a rotation defined at every position but trained at none of them, and a decay property asserted rather than proven",
    to: "position-interpolation",
  }),
  built("alibi", "rope", {
    text: "a penalty that only grows with distance, and an extrapolation result its own appendix says may not mean the context is being used",
    to: "yarn",
  }),
  built("sparse-transformer", "transformer", {
    text: "a sparsity pattern fixed before the data arrives, which the authors' own inspection showed cannot reproduce global or data-dependent layers",
    to: "sliding-window",
  }),
];

for (let i = 0; i < mechanisms.length; i++) for (const f of LINKS) mechanisms[i] = f(mechanisms[i]);

export const byDate = () => [...mechanisms].sort((a, b) => a.date.localeCompare(b.date));
export const indexOf = (id) => byDate().findIndex((m) => m.id === id);
