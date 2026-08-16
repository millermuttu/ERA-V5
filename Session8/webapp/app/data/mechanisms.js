// The chronology. Single source of truth: the rail, the cards, the README source table and the
// self-check all read this array. Sorted by `date` at render time, never by hand.
//
// Record shape:
//   id          stable slug, also the anchor in the URL
//   name        what to call it on the card
//   date        ISO. The first public version of the primary source, not the conference date
//   source      { label, url, kind } — kind: 'paper' | 'post' | 'release' | 'course'
//   verified    true only after the date was read off the primary source (see tasks 6.1-6.3)
//   thread      one word: what pressure it answers. A label, never a lane drawn over the timeline
//   problem     what existed before it and where that hurt. Must point at an earlier entry
//   mechanism   how it actually works
//   buys / givesUp / chooseWhen   the three questions the assignment demands of every entry
//   demo        key into app/demos/index.js, or null
//   diagram     key into app/demos/diagrams.js, or null

export const THREADS = {
  baseline: "the mechanism itself",
  position: "where a token sits",
  cache: "what generation must remember",
  compute: "how many comparisons",
  state: "compressing the past",
  systems: "how it runs on the hardware",
};

export const mechanisms = [
  {
    id: "transformer",
    name: "Scaled dot-product attention, multi-head, sinusoidal positions",
    date: "2017-06-12",
    source: {
      label: "Vaswani et al., Attention Is All You Need — arXiv:1706.03762",
      url: "https://arxiv.org/abs/1706.03762",
      kind: "paper",
    },
    verified: true,
    thread: "baseline",
    baseline: true,
    problem:
      "Attention was not new in 2017 — it had been bolted onto recurrent encoder-decoders since 2014 to stop a whole sentence being squeezed through one hidden state. But the recurrence was still there, and recurrence is sequential: token t could not be computed until t-1 was. Long sequences trained slowly because the hardware sat idle waiting for the previous step.",
    mechanism:
      "Delete the recurrence and keep only the attention. Each token is projected three ways: a query (what am I looking for), a key (what do I contain), a value (what I hand over if chosen). Every query is compared with every key by dot product, the scores are divided by the square root of the head width to keep them in softmax's useful range, softmax turns each row into weights that sum to one, and the weighted sum of values is the token's new vector. Several heads do this in parallel on slices of the width. Because nothing in that computation knows about order, position is added to the input as a fixed sinusoid of varying frequency — and a decoder additionally adds a mask of -infinity to every future score, so softmax gives the future exactly zero and the whole sequence can still be trained in one parallel pass.",
    buys: [
      "Every position is computed at once: training parallelises across the sequence instead of walking it",
      "Any token can reach any other token in one step, however far apart",
      "Sinusoidal positions are a function, not a table, so nothing structurally forbids a longer sequence",
    ],
    givesUp: [
      "Every token is compared with every other token: the score matrix is T x T, so cost grows roughly with the square of context",
      "Generation must keep every earlier key and value to avoid recomputing them — a cache that grows with the conversation",
      "Attention alone has no idea what order the tokens came in; position has to be injected from outside",
    ],
    chooseWhen:
      "Always, as the thing everything else is measured against. At short context it is not merely the baseline, it is usually still the right answer: exact, well understood, and the fastest kernels in the world are written for it.",
    demo: "attention",
    diagram: null,
  },

  {
    id: "relative-positions",
    name: "Relative position representations",
    date: "2018-03-06",
    source: {
      label: "Shaw et al., Self-Attention with Relative Position Representations — arXiv:1803.02155",
      url: "https://arxiv.org/abs/1803.02155",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "The Transformer tells a token where it sits in absolute terms — position 8 gets the sinusoid for 8. But what a language model usually needs is not 'I am token 8', it is 'that word is six places behind me'. An absolute signal makes the model reconstruct distance from two absolute labels, and that reconstruction has to be relearned at every offset.",
    mechanism:
      "Put position into the comparison instead of into the input. Each query-key pair gets a learned embedding indexed by the distance between them, added to the score before softmax. Distances beyond a chosen window are clipped to a single bucket, so the number of learned vectors stays small no matter how long the sequence is.",
    buys: [
      "The model sees distance directly, which is what the linguistics actually depends on",
      "A pattern learned at one offset transfers when the whole sentence shifts",
      "Clipping distant buckets keeps the parameter count bounded",
    ],
    givesUp: [
      "The score computation stops being a plain matrix multiply — the extra per-pair term costs memory and breaks the fastest kernels",
      "Everything past the clip distance looks identical, so genuinely long-range distance is flattened",
    ],
    chooseWhen:
      "When relative distance matters more than absolute placement and the context is short enough that the per-pair term is affordable. Later work keeps the relative idea and drops the per-pair table — see RoPE and ALiBi.",
    demo: null,
    diagram: null,
  },

  {
    id: "learned-absolute",
    name: "Learned absolute position tables",
    date: "2018-10-11",
    source: {
      label: "Devlin et al., BERT — arXiv:1810.04805 (the era's default; GPT used the same device)",
      url: "https://arxiv.org/abs/1810.04805",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "Sinusoids are hand-designed. The obvious question after the Transformer was whether a model could simply learn what each position should mean, the same way it learns what each token means. The answer was yes, and for a while that was the default in the models everybody built on.",
    mechanism:
      "Allocate one trainable vector per position — a lookup table of shape max_positions x width — and add row i to token i's embedding. Nothing is derived; every position is a parameter learned from data.",
    buys: [
      "The model decides what position means instead of being told, and it costs one addition",
      "At the trained length it is simple, effective, and needs no special kernel",
    ],
    givesUp: [
      "A hard wall: the table has a last row. Position 513 in a 512-row table does not exist, and no amount of clever inference conjures it",
      "Positions that are rare in training get undertrained rows, so the far end of the window is weaker than the near end",
      "The table is absolute, so it teaches placement rather than distance",
    ],
    chooseWhen:
      "Fixed, known, modest context — a classifier over documents you have already decided to truncate. This is the mechanism Session 7 ruled out for V5 by name: a stored table cannot be extended, and long context is the whole game.",
    demo: null,
    diagram: null,
  },

  {
    id: "transformer-xl",
    name: "Segment recurrence across contexts",
    date: "2019-01-09",
    source: {
      label: "Dai et al., Transformer-XL — arXiv:1901.02860",
      url: "https://arxiv.org/abs/1901.02860",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "To train on a document longer than the window, you cut it into segments. With the plain Transformer, each segment starts from nothing: the first token of segment two cannot see the last token of segment one. The model is repeatedly given amnesia at an arbitrary boundary that has nothing to do with the text.",
    mechanism:
      "Cache the hidden states of the previous segment and let the current segment attend over them, with no gradient flowing back across the boundary. Since the cached states came from different absolute positions, the absolute scheme breaks — so the same work replaces it with a relative encoding that survives the reuse.",
    buys: [
      "Context reaches beyond one window without paying quadratic cost over the whole document",
      "No recomputation at evaluation time: the cached segment is reused rather than rebuilt",
      "Removes the amnesia at segment boundaries",
    ],
    givesUp: [
      "The cached segment has to be stored, which is memory that scales with how far back you keep",
      "Gradients stop at the boundary, so the model never learns a dependency longer than the graph it is trained through",
      "It forced a change of positional scheme — the mechanisms are entangled",
    ],
    chooseWhen:
      "Streaming or chunked documents where the boundary is an artefact of your batching rather than of the text. The idea comes back in this session as V4's Memory Stream, compressed all the way down to a single gated vector.",
    demo: null,
    diagram: null,
  },

  {
    id: "sparse-transformer",
    name: "Sparse Transformer, strided and fixed patterns",
    date: "2019-04-23",
    source: {
      label: "Child et al., Generating Long Sequences with Sparse Transformers — arXiv:1904.10509",
      url: "https://arxiv.org/abs/1904.10509",
      kind: "paper",
    },
    verified: true,
    thread: "compute",
    problem:
      "Full attention compares every token with every other token. At a thousand tokens that is a million scores; at ten thousand it is a hundred million. The first bill attention sends is compute, and it grows with the square of the context.",
    mechanism:
      "Do not let every query see every key. Fix a pattern in advance — each query attends to a local block plus every n-th position — and split the pattern across heads so that two sparse hops can still connect any pair of tokens. The pattern is chosen by the architect, not learned, so the sparsity is known at compile time and can be given a fast kernel.",
    buys: [
      "Attention cost drops from T^2 toward T x sqrt(T): images and audio at lengths that were previously impossible",
      "A fixed pattern is predictable, so it can be implemented as an efficient dense operation on blocks",
    ],
    givesUp: [
      "The pattern is a guess made before seeing the data. A dependency that does not fall on the grid is simply unreachable in that layer",
      "Reaching an arbitrary token may take two hops through an intermediary, and the intermediary has to have preserved what was needed",
    ],
    chooseWhen:
      "Data with genuine locality or periodic structure — images, audio, code — where you can name the pattern honestly. It is the ancestor of every later sparse method, all of which try to replace the fixed guess with something adaptive.",
    demo: null,
    diagram: null,
  },

  {
    id: "mqa",
    name: "Multi-query attention",
    date: "2019-11-06",
    source: {
      label: "Shazeer, Fast Transformer Decoding: One Write-Head is All You Need — arXiv:1911.02150",
      url: "https://arxiv.org/abs/1911.02150",
      kind: "paper",
    },
    verified: true,
    thread: "cache",
    problem:
      "The second bill. When a model generates one token at a time it keeps every earlier key and value so it need not recompute them, and with multi-head attention every head stores its own pair. That cache is private to one conversation — weights are shared across users, history is not — and at decode time the accelerator spends most of its time reading that cache rather than doing arithmetic.",
    mechanism:
      "Keep all the query heads, but give the whole layer a single shared key head and a single shared value head. Every query head asks its own question of the same stored keys and values.",
    buys: [
      "The KV cache shrinks by the number of heads — often 8x to 64x — which is the difference between fitting a long conversation and not",
      "Decoding gets faster in proportion, because decode speed is bounded by memory traffic, not by FLOPs",
    ],
    givesUp: [
      "All heads must agree on what the past looks like. Quality degrades, measurably, and the degradation grows with model size",
      "Training can become less stable at the extreme",
    ],
    chooseWhen:
      "Serving budgets so tight that cache is the binding constraint and a small quality loss is acceptable. Mostly superseded by GQA, which found the middle of this trade four years later.",
    demo: "cache",
    diagram: null,
  },

  {
    id: "sliding-window",
    name: "Sliding window attention with global tokens",
    date: "2020-04-10",
    source: {
      label: "Beltagy et al., Longformer — arXiv:2004.05150",
      url: "https://arxiv.org/abs/2004.05150",
      kind: "paper",
    },
    verified: true,
    thread: "compute",
    problem:
      "The Sparse Transformer's strided pattern suits images. Text is different: most of what a token needs is a few dozen words away, but a few tokens — the question, the instruction, a section heading — matter to everything in the document.",
    mechanism:
      "Give every token a local window of w neighbours, so cost becomes linear in T. Stack layers and the receptive field widens like a convolution: after L layers a token indirectly sees L x w. Then designate a handful of positions as global — they attend to everything and everything attends to them — to carry the information that has to cross the whole document.",
    buys: [
      "Linear cost and linear cache in context length, with a window you set rather than a pattern you hope fits",
      "The global tokens give a clean, cheap escape hatch for genuinely document-wide information",
      "Depth compounds the window, so long-range information can still flow",
    ],
    givesUp: [
      "Long-range information now travels through intermediate tokens, and whatever they did not preserve is lost",
      "You choose the window size and which tokens are global before you see the input",
      "Cache still grows with context: the window bounds the compute, not the history",
    ],
    chooseWhen:
      "Long documents with local structure, and as one layer type inside a hybrid — most production long-context stacks interleave windowed layers with a few full-attention ones.",
    demo: null,
    diagram: "sliding",
  },

  {
    id: "linear-attention",
    name: "Linear attention, the kernel regrouping",
    date: "2020-06-29",
    source: {
      label: "Katharopoulos et al., Transformers are RNNs — arXiv:2006.16236",
      url: "https://arxiv.org/abs/2006.16236",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "Sparse patterns reduce how many keys a query reads, but the query still has to read individual keys, and the cache still holds one entry per token. The deeper obstacle is softmax itself: its denominator is a sum over every score, so the weight of key 1 depends on keys 2 and 3, and nothing can be summarised until the query arrives.",
    mechanism:
      "Drop the softmax and the sum can be regrouped. Because (q.k)v = q(k v-transpose) with no normaliser tying the terms together, all the old key-value pairs fold into one matrix S = sum of v k-transpose, built as the tokens arrive, and the answer for a new query is just S q. Positivity is restored with a feature map applied to queries and keys instead of an exponential. The state is a fixed d x d matrix: after ten tokens or a million, it is the same size.",
    buys: [
      "Sequence work becomes linear, and generation becomes O(1) per token — the model is literally an RNN at inference",
      "The KV cache stops existing: one fixed-size state replaces the growing history",
    ],
    givesUp: [
      "Exact token-level lookup. The whole past is compressed into one object and different memories interfere",
      "Softmax's competition — that giving weight to one key takes it from another — is gone, and unnormalised scores can grow with context",
      "At equal parameters it has generally underperformed exact attention on recall-heavy tasks",
    ],
    chooseWhen:
      "Very long sequences where a growing cache is fatal and the task tolerates a lossy summary of the past. The rest of the linear-attention story is about repairing what removing softmax broke.",
    demo: "linear",
    diagram: null,
  },

  {
    id: "performer",
    name: "Performer, FAVOR+",
    date: "2020-09-30",
    source: {
      label: "Choromanski et al., Rethinking Attention with Performers — arXiv:2009.14794",
      url: "https://arxiv.org/abs/2009.14794",
      kind: "paper",
    },
    verified: true,
    thread: "compute",
    problem:
      "Plain linear attention swaps softmax for an arbitrary feature map and hopes the result behaves. That leaves an uncomfortable question: how far is the cheap thing from the exact thing you replaced?",
    mechanism:
      "Approximate the softmax kernel itself rather than replacing it. Random positive features give an unbiased estimate of exp(q.k), so the same regrouping applies while the quantity being estimated is the real softmax attention, with variance bounds instead of hope. More random features means a closer approximation at higher cost.",
    buys: [
      "Linear cost with an explicit, tunable approximation error to the real thing",
      "Drops into a pretrained softmax model far better than an arbitrary feature map does",
    ],
    givesUp: [
      "It is still an estimate, and the variance shows on tasks that need sharp, confident retrieval",
      "Enough random features to be accurate erodes the speed advantage",
      "In practice it never displaced exact attention at scale",
    ],
    chooseWhen:
      "When you need linear cost and want a principled bound on what you gave up. Historically it matters most as the answer to 'how close can cheap get?' — and the answer shaped the field's expectations.",
    demo: null,
    diagram: null,
  },

  {
    id: "delta-rule",
    name: "The delta rule, fast-weight programmers",
    date: "2021-02-22",
    source: {
      label: "Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers — arXiv:2102.11174",
      url: "https://arxiv.org/abs/2102.11174",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "Linear attention's state is written with S = S + v k-transpose: pure accumulation. If a key already maps to an answer and the answer should change, adding the new answer does not replace the old one — reading that key afterwards returns the sum of both. A memory that can only add cannot correct, and over a long sequence the interference compounds.",
    mechanism:
      "Read before writing. Query the state with the incoming key to see what it currently returns, subtract that from what it should return, and write only the difference: S = S + (v_new - S k) k-transpose. This is the delta rule from the 1960s, and it turns an accumulator into an addressable memory that can revise an entry in place.",
    buys: [
      "The state can update an old association instead of piling on top of it",
      "Capacity is used far better: writes stop fighting each other as the sequence grows",
      "Still fixed-size, still linear in sequence length",
    ],
    givesUp: [
      "The write now depends on the current state, which is sequential — the parallel scan that made linear attention fast is harder to keep",
      "More arithmetic per token than a plain accumulate",
    ],
    chooseWhen:
      "Any fixed-state model whose sequences are long enough for a key to be written more than once — which is all of them. The parallelisation problem it created took until 2024 to solve properly.",
    demo: "linear",
    diagram: null,
  },

  {
    id: "rope",
    name: "RoPE, rotary position embedding",
    date: "2021-04-20",
    source: {
      label: "Su et al., RoFormer — arXiv:2104.09864",
      url: "https://arxiv.org/abs/2104.09864",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "By 2021 there were two positional options and both were unsatisfying. A learned absolute table has a last row — the hard wall. Relative position tables put a per-pair term into the score and slow the kernel down. Neither gives distance for free.",
    mechanism:
      "Rotate instead of add. Take the query and key dimensions in pairs, treat each pair as an arrow on a plane, and rotate the arrow at position i by i times a small angle — different pairs rotating at different rates so distance is represented at several scales. Because a dot product depends only on the angle between two arrows, both absolute rotations cancel and what survives in the score is the difference i - j: the distance. Nothing is added to the input, nothing is stored per pair, and since the rotation is computed from a function it is defined at any position.",
    buys: [
      "Relative distance drops out of the standard dot product with no extra term and no table",
      "No length wall: the function evaluates anywhere, which is what makes context extension conceivable at all",
      "Applies to queries and keys only, so it composes with everything downstream",
    ],
    givesUp: [
      "Defined at a position is not the same as trained at it — far past the training length the rotations combine in ways the model never learned to read",
      "The rate schedule is a hyperparameter, and later methods spend their whole existence adjusting it",
    ],
    chooseWhen:
      "The default for a decoder-only language model, and it has been since roughly 2022. Everything in this timeline about extending context assumes it.",
    demo: "rope",
    diagram: null,
  },

  {
    id: "alibi",
    name: "ALiBi, attention with linear biases",
    date: "2021-08-27",
    source: {
      label: "Press et al., Train Short, Test Long — arXiv:2108.12409",
      url: "https://arxiv.org/abs/2108.12409",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "RoPE is defined beyond the training length but does not reliably work there. The question ALiBi asked was blunter: what is the least positional machinery that still extrapolates?",
    mechanism:
      "Do not encode position at all. Subtract a penalty proportional to the distance between query and key directly from the score, with a different fixed slope per head, so some heads look far and some look only nearby. No embeddings, no rotations, no learned parameters — one subtraction before softmax.",
    buys: [
      "Trains at a short length and evaluates at a much longer one with graceful degradation, which was the headline result",
      "Costs essentially nothing to compute and nothing to store",
      "Per-head slopes give a spread of receptive fields for free",
    ],
    givesUp: [
      "The bias is monotone: distant tokens are always penalised, so a genuinely important thing far back is fought by the mechanism itself",
      "No notion of position beyond distance — no periodicity, no structure",
      "Weaker than RoPE on tasks needing precise long-range retrieval, which is why frontier models mostly went the other way",
    ],
    chooseWhen:
      "When extrapolation beyond the training length matters more than sharp long-range recall, or when you want the simplest thing that could possibly work as a baseline.",
    demo: null,
    diagram: null,
  },

  {
    id: "flashattention",
    name: "FlashAttention",
    date: "2022-05-27",
    source: {
      label: "Dao et al., FlashAttention — arXiv:2205.14135",
      url: "https://arxiv.org/abs/2205.14135",
      kind: "paper",
    },
    verified: true,
    thread: "systems",
    problem:
      "Every mechanism above changes the mathematics to dodge the T^2 cost. FlashAttention asked whether the mathematics was the problem at all. On real hardware the T x T score matrix is written to high-bandwidth memory and read back for the softmax, and that round trip — not the multiply — is what the clock is waiting on.",
    mechanism:
      "Never materialise the score matrix. Walk the keys and values in tiles that fit in on-chip SRAM, and accumulate the softmax with a running maximum and running denominator so each tile can be folded in without seeing the others. The output is bit-comparable to standard attention; only the memory traffic changes. The backward pass recomputes tiles instead of storing them.",
    buys: [
      "Several times faster and dramatically less memory, with no approximation — the same numbers, computed better",
      "Memory becomes linear in sequence length, so context lengths that previously OOMed simply run",
      "Made exact attention competitive again and quietly undercut the case for approximating it",
    ],
    givesUp: [
      "Nothing mathematically. The cost is engineering: hand-written kernels per hardware generation and per variant",
      "Anything that does not fit the tiling pattern — an unusual mask, a per-pair bias — either gets no fast kernel or a slower one",
      "That constraint is real enough to shape architecture. Session 8 records V4 capping its sparse budget at 256 because of backward-kernel contention",
    ],
    chooseWhen:
      "Always, if a kernel exists for your setup. Its deeper lesson is that some attention bills are paid to the memory hierarchy, not to the asymptotics — and you should know which one you are actually being charged.",
    demo: null,
    diagram: null,
  },

  {
    id: "gqa",
    name: "Grouped-query attention",
    date: "2023-05-22",
    source: {
      label: "Ainslie et al., GQA — arXiv:2305.13245",
      url: "https://arxiv.org/abs/2305.13245",
      kind: "paper",
    },
    verified: true,
    thread: "cache",
    problem:
      "MQA cut the cache by collapsing to one key-value head and paid for it in quality. Full multi-head keeps quality and pays in cache. Four years of serving experience said the truth was somewhere in between, and nobody had gone looking.",
    mechanism:
      "Share, but not all the way. Split the query heads into groups and give each group its own key-value head — eight query heads over two key-value heads, say. Cache scales with the number of groups, and an existing multi-head checkpoint can be converted by mean-pooling the heads within each group and briefly fine-tuning.",
    buys: [
      "Most of MQA's cache saving with most of multi-head's quality — the trade curve turns out to be very forgiving in the middle",
      "Existing checkpoints convert cheaply instead of being retrained",
      "One integer to tune, and the effect on memory is exactly proportional",
    ],
    givesUp: [
      "It lowers the slope of the line; it does not stop the line rising. Cache still grows linearly with context",
      "At a million tokens a smaller constant is not a solution",
    ],
    chooseWhen:
      "Effectively the default for every open-weight decoder since 2023. The lesson is explicit that it is a baseline, not an answer: for V5's target it reduces the constant and leaves the growth.",
    demo: "cache",
    diagram: null,
  },

  {
    id: "position-interpolation",
    name: "Position interpolation",
    date: "2023-06-27",
    source: {
      label: "Chen et al., Extending Context Window of LLMs via Position Interpolation — arXiv:2306.15595",
      url: "https://arxiv.org/abs/2306.15595",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "RoPE can be evaluated at any position, and models trained at 2K or 4K nonetheless fell apart when asked for more. Extrapolating the rotation puts the model in an angular regime it never saw in training, and the attention scores there are not merely unfamiliar — they blow up.",
    mechanism:
      "Do not extrapolate, interpolate. To serve a context n times longer, divide every position index by n before computing the rotation, so position 8,000 is presented as position 2,000 and every angle stays inside the range training covered. A short fine-tune, reported at around a thousand steps, lets the model adjust to the denser spacing.",
    buys: [
      "Large extensions from a very small fine-tune, with a bound on why it should work at all",
      "Keeps every rotation inside the trained regime instead of walking off the end of it",
      "The first method to make RoPE extension routine rather than hopeful",
    ],
    givesUp: [
      "Neighbouring tokens are squeezed together: at 8x, adjacent positions are an eighth of their trained angular distance apart, and fine local distinctions blur",
      "It scales every frequency equally, which is exactly the flaw the next two entries were written to fix",
      "Requires the fine-tune; it is not an inference-time switch",
    ],
    chooseWhen:
      "Superseded in practice by NTK-aware scaling and YaRN, but it is the entry that established the shape of the whole extend-don't-retrain road — and the baseline they are both measured against.",
    demo: null,
    diagram: null,
  },

  {
    id: "ntk-aware",
    name: "NTK-aware base scaling",
    date: "2023-06-30",
    source: {
      label: "bloc97, community post on r/LocalLLaMA — no paper; refined later into YaRN",
      url: "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/",
      kind: "post",
    },
    verified: false,
    thread: "position",
    problem:
      "Position interpolation worked, but it squeezed every frequency alike. Neighbouring tokens ended up an nth of their trained angular distance apart, and fine detail at short range — which is most of language — got worse in exchange for reach the model rarely used.",
    mechanism:
      "Do not scale all frequencies equally. RoPE's dimension pairs rotate at different rates; the fast ones carry local detail, the slow ones carry long-range placement. Increasing the rotary base stretches the slow dimensions to cover the longer context while leaving the fast ones nearly untouched, so long range is extended without blurring what is nearby.",
    buys: [
      "Meaningful context extension with little or no fine-tuning — the reason it spread through the open-weight community in weeks",
      "Preserves short-range resolution, which naive interpolation destroys",
    ],
    givesUp: [
      "A heuristic, published as a forum post, with no derivation or guarantees at the time",
      "There is a ceiling: push the factor far enough and quality collapses anyway",
      "Extending a position function is not the same as the model having learned to use the new range",
    ],
    chooseWhen:
      "When you have a trained model and need more context this week without a training run. Its provenance is the point: this is a community post that changed practice before the theory caught up — hence no verified paper date on this card.",
    demo: null,
    diagram: null,
  },

  {
    id: "yarn",
    name: "YaRN",
    date: "2023-08-31",
    source: {
      label: "Peng et al., YaRN — arXiv:2309.00071",
      url: "https://arxiv.org/abs/2309.00071",
      kind: "paper",
    },
    verified: true,
    thread: "position",
    problem:
      "NTK-aware scaling worked but nobody could say why, or when it would stop working. Interpolation was principled but damaged local detail. What was missing was a method that treated the two regimes separately and could be reasoned about.",
    mechanism:
      "Sort the rotary dimensions by wavelength. Dimensions whose period is shorter than the training context are already fully observed and are left alone; dimensions whose period exceeds it are interpolated; the band in between is blended. A temperature adjustment on the attention logits compensates for the entropy change the stretch introduces. A short fine-tune on long samples finishes the job.",
    buys: [
      "Large extensions — commonly 16x to 32x — with a fraction of the tokens full long-context training would need",
      "Explains and subsumes the earlier heuristics rather than competing with them",
      "Well characterised enough to be a standard config flag in serving stacks",
    ],
    givesUp: [
      "Still an extension: quality past the trained range is earned, not free, and it degrades with the factor",
      "Needs the fine-tune to reach its published numbers — it is not purely an inference switch",
      "More knobs to set than 'scale the base'",
    ],
    chooseWhen:
      "The standard route when you are on Road 1 — train at an affordable length, then stretch. It is what DroPE is doing the same job as in the V4 record, and the honest comparison for V5's extend-or-build-native question.",
    demo: null,
    diagram: null,
  },

  {
    id: "attention-sinks",
    name: "Attention sinks, StreamingLLM",
    date: "2023-09-29",
    source: {
      label: "Xiao et al., Efficient Streaming Language Models with Attention Sinks — arXiv:2309.17453",
      url: "https://arxiv.org/abs/2309.17453",
      kind: "paper",
    },
    verified: true,
    thread: "cache",
    problem:
      "The obvious way to bound a growing cache during endless generation is to drop the oldest entries. It fails immediately and spectacularly: perplexity explodes the moment the very first tokens fall out of the window, and nobody could explain why those particular tokens mattered.",
    mechanism:
      "The explanation is softmax. Every row must sum to one, so when a query has nothing it genuinely wants, the weight has to go somewhere — and models learn to dump it on the first few positions, which every query can see. Those tokens are not carrying meaning, they are a drain. Evict them and the excess weight redistributes onto real tokens, corrupting the output. The fix is to pin the first handful of tokens permanently and slide the window over everything else.",
    buys: [
      "Genuinely unbounded generation at constant memory, with stable quality, from four extra cached tokens",
      "Needs no retraining — it is a serving-time change",
      "Explains a real failure mode rather than papering over it",
    ],
    givesUp: [
      "The middle of the conversation is still gone. This bounds memory, it does not extend understanding",
      "A fact stated outside the window and the sinks is unrecoverable",
    ],
    chooseWhen:
      "Long-running chat and streaming where the recent past is what matters and you need a hard memory ceiling. Do not confuse it with long context: it is a way to forget safely.",
    demo: null,
    diagram: "sinks",
  },

  {
    id: "mamba",
    name: "Selective state space",
    date: "2023-12-01",
    source: {
      label: "Gu and Dao, Mamba — arXiv:2312.00752",
      url: "https://arxiv.org/abs/2312.00752",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "Linear attention and the earlier state-space models were fast and had fixed-size state, and both were bad at the one thing language models must do: notice that this particular token matters and hold onto it. Their dynamics were the same for every input, so they could not choose what to remember.",
    mechanism:
      "Make the state-space parameters functions of the current token. That input-dependence is what lets the model decide, per token, whether to write to the state or let it decay — but it also destroys the convolution that made these models fast. Mamba pays that back with a hardware-aware parallel scan that keeps the state in fast memory and never materialises it across the sequence.",
    buys: [
      "Fixed-size state and linear time, with selectivity that finally makes it competitive with attention on language",
      "O(1) memory per generated token — no cache at all",
      "A working demonstration that recurrence can be trained at scale on modern hardware",
    ],
    givesUp: [
      "Exact recall of arbitrary earlier tokens. A compressed state cannot reproduce something it chose not to keep, and copy-heavy tasks show it",
      "Depends on a custom kernel; the naive implementation is slow",
      "Pure state-space stacks have generally lost to hybrids in practice",
    ],
    chooseWhen:
      "As the fixed-state half of a hybrid, which is how it is nearly always used. Its real contribution to this timeline is proving that state models deserved a second look — which is what DeltaNet and V4's DDDGDDDG schedule are built on.",
    demo: null,
    diagram: null,
  },

  {
    id: "mla",
    name: "Multi-head latent attention",
    date: "2024-05-07",
    source: {
      label: "DeepSeek-AI, DeepSeek-V2 — arXiv:2405.04434",
      url: "https://arxiv.org/abs/2405.04434",
      kind: "paper",
    },
    verified: true,
    thread: "cache",
    problem:
      "GQA reduces the cache by making heads share. That is a blunt instrument: it throws away head diversity to save bytes, and the saving is capped by how much sharing quality will tolerate. The question nobody had asked was whether the keys and values need storing in that form at all.",
    mechanism:
      "Store a compressed latent instead. Project keys and values down to a small shared latent vector, cache only that, and reconstruct per-head keys and values on the fly with up-projections that can be folded into the surrounding weight matrices, so the reconstruction is close to free. RoPE does not survive the folding, so a few dimensions are carried separately just to hold position — the detail behind the lesson's note that DeepSeek applies rotation to only part of the head.",
    buys: [
      "Cache far smaller than GQA — the reported figure is a large multiple, not a percentage",
      "Head diversity is kept: every head still gets its own key and value after reconstruction",
      "Reported to match or beat multi-head quality, so the saving is not obviously paid for in accuracy",
    ],
    givesUp: [
      "Real architectural complexity: an extra projection pair, a split positional path, and weight folding that has to be got right",
      "The latent width is a capacity ceiling chosen before training",
      "Cache still grows linearly with context — a smaller constant, same shape",
    ],
    chooseWhen:
      "When serving memory is the binding constraint and you are training from scratch, so you can afford the architectural surgery. It is the strongest current answer to 'make the per-token cache smaller' — and the lesson still classes it with GQA as not solving growth.",
    demo: "cache",
    diagram: "mla",
  },

  {
    id: "parallel-deltanet",
    name: "Parallelizable DeltaNet",
    date: "2024-06-10",
    source: {
      label: "Yang et al., Parallelizing Linear Transformers with the Delta Rule — arXiv:2406.06484",
      url: "https://arxiv.org/abs/2406.06484",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "The delta rule fixed linear attention's memory but broke its training speed. Each write needs the state produced by the previous write, so the sequence has to be walked one token at a time — exactly the sequential dependency the Transformer was invented to escape. For three years that made it a good idea nobody could afford to train.",
    mechanism:
      "Rewrite the recurrence as a matrix product. Each delta update is a rank-one change to the state, and a chunk of consecutive rank-one updates can be reassociated into a form computed with dense matrix multiplies over the chunk — the WY representation from classical numerical linear algebra. Sequential between chunks, fully parallel within them, and dense matmul is what accelerators are built for.",
    buys: [
      "The delta rule at Transformer-like training throughput — orders of magnitude faster than the token-by-token form",
      "Made a correcting fixed-state layer practical at billion-parameter scale for the first time",
      "Keeps O(1) generation with no cache",
    ],
    givesUp: [
      "Chunking is another hyperparameter, and the gain depends on the kernel being written well",
      "Still a fixed-size state: parallel training does not make a compressed memory exact",
    ],
    chooseWhen:
      "Whenever you want a delta-rule layer at all — there is no reason to use the sequential form now. This is the work that put the D in V4's DDDGDDDG.",
    demo: null,
    diagram: null,
  },

  {
    id: "gated-deltanet",
    name: "Gated DeltaNet",
    date: "2024-12-09",
    source: {
      label: "Yang et al., Gated Delta Networks — arXiv:2412.06464",
      url: "https://arxiv.org/abs/2412.06464",
      kind: "paper",
    },
    verified: true,
    thread: "state",
    problem:
      "The delta rule can correct an association it is told about. It has no way to let anything go. Over a very long sequence the state fills with associations that were written once and never revisited, and stale content crowds out what is current — precise editing without any forgetting.",
    mechanism:
      "Add a data-dependent decay alongside the delta write, so each step both scales the existing state down by a learned gate and applies the correction. Gating alone forgets uniformly; the delta rule alone edits precisely but never releases. Together the model learns when to hold, when to revise, and when to clear — and the chunked parallel form survives the addition.",
    buys: [
      "Precise editing and controlled forgetting in one state, which is what a bounded memory actually needs",
      "Clear wins over plain DeltaNet and plain gated linear attention on long-context and recall benchmarks",
      "Still fixed-state, still parallelisable, still no KV cache",
    ],
    givesUp: [
      "Forgetting is irreversible — a badly timed gate discards something that mattered, and there is no cache to fall back on",
      "More machinery per layer, and the gate is another thing to tune",
    ],
    chooseWhen:
      "As the fixed-state layer in a hybrid stack. It is the current best answer to 'what should the D layers be', and the reason the lesson describes modern variants as learning how strongly to write and when old information should fade.",
    demo: "linear",
    diagram: null,
  },

  {
    id: "nsa",
    name: "Natively trainable sparse attention",
    date: "2025-02-16",
    source: {
      label: "Yuan et al., Native Sparse Attention — arXiv:2502.11089",
      url: "https://arxiv.org/abs/2502.11089",
      kind: "paper",
    },
    verified: true,
    thread: "compute",
    problem:
      "Sparse attention had a persistent gap between paper and practice. Most schemes were applied at inference to a model trained densely, so the model never learned to work with them; and the honest catch in naive top-k is that finding the best k keys means scoring all T of them first, which is the cost you were trying to avoid.",
    mechanism:
      "Three branches, trained from the start. Compress runs of tokens into block summaries for the coarse view; select scores those cheap summaries and reads exact tokens from only the top-scoring blocks; a sliding window keeps local detail intact. Their outputs are combined by a learned gate. The block structure is chosen so the whole thing maps onto efficient kernels, and because sparsity is present during training the model learns to rely on it.",
    buys: [
      "Sparsity that is trained rather than retrofitted, reported at or above full-attention quality",
      "Real end-to-end speedups on long sequences, forward and backward, because the pattern was co-designed with the kernel",
      "The compression branch reduces what must be stored; the selection branch reduces what must be read",
    ],
    givesUp: [
      "Considerable complexity: three branches, a gate, block-aligned kernels",
      "Block granularity is a floor on precision — a single crucial token inside an unselected block is still missed",
      "Committed at training time, so it cannot be bolted onto an existing dense model",
    ],
    chooseWhen:
      "Road 2 — building an architecture that can be trained long in the first place. This is the direct ancestor of the compressed-plus-top-k scheme Session 8 describes, and the counterweight to the linear-state family in V5's open question.",
    demo: "compress",
    diagram: "nsa",
  },

  {
    id: "dsa",
    name: "DeepSeek sparse attention",
    date: "2025-09-29",
    source: {
      label: "DeepSeek-V3.2-Exp release — model card and tech report, no arXiv v1 to cite",
      url: "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp",
      kind: "release",
    },
    verified: false,
    thread: "compute",
    problem:
      "NSA proved trained sparsity works, but it has to be designed in from the beginning. That leaves the practical question of what to do with a strong dense model you already own and cannot afford to retrain from scratch.",
    mechanism:
      "Separate ranking from reading. A small low-rank indexer scores which earlier positions are worth attending to, and exact attention runs only over the top-scoring ones — so the expensive operation sees a short list while the cheap one sees everything. Introduced by continuing training from an existing dense checkpoint rather than starting over, and combined with the latent cache from MLA.",
    buys: [
      "Sub-quadratic long-context attention reached by adapting an existing model instead of retraining it",
      "The indexer is cheap enough that selection stops being the bottleneck — the catch in naive top-k, answered",
      "Reported quality parity with the dense model it was adapted from, at markedly lower long-context serving cost",
    ],
    givesUp: [
      "The indexer is an approximation: a genuinely relevant position it ranks low is simply not read",
      "Another trained component that can be wrong, and it must be trained to agree with the attention it gates",
      "This card is sourced from a release, not a peer-reviewed paper with an arXiv v1 date — treat the numbers accordingly",
    ],
    chooseWhen:
      "When you have a dense model and need long context without a from-scratch run. Together with NSA it is the compression-and-sparsity family V5 must weigh against linear state.",
    demo: "topk",
    diagram: null,
  },

  {
    // No public source, so no date exists to verify. The value below is a sort key that puts
    // this entry last, and the card says so in as many words. It is never presented as a date.
    id: "drope",
    name: "DroPE",
    date: "2026-01-01",
    source: {
      label: "LightningLM V4 cookbook, via the ERA V5 Session 8 lesson — no public source",
      url: "",
      kind: "course",
    },
    verified: false,
    thread: "position",
    problem:
      "V4 trained at 8K and needed to serve far beyond it. RoPE guarantees only that a rotation can be computed at position 256K, not that the model has learned to read it — and the lesson is emphatic that those are different claims.",
    mechanism:
      "What the record establishes: a positional recalibration step named DroPE, applied before annealing — that is, while the model still had training steps and learning rate left to adapt — taking a model trained at 8K to a reported 256K, an extension of 32x. What the record does not establish: the algorithm, or which rotary dimensions it touches. Anything more specific than the previous sentence is a hypothesis until it is checked against the reference implementation, so this card does not offer one.",
    buys: [
      "A reported 32x extension without training at the target length, which is the cheap road (256K / 8K = 32)",
      "Applied before annealing, so the change was trained into the model rather than switched on at inference",
    ],
    givesUp: [
      "Verifiability. There is no public paper, no date and no algorithm — this entry sits at the end of the timeline by convention, not by an established date",
      "An extension factor is evidence for one model and one procedure. It does not promise that 320x or another architecture works",
    ],
    chooseWhen:
      "Not a recommendation — a record. It is on the timeline because V5's first open question is extend-or-build-native, and this is the only in-house data point on the extend side. Compare it against YaRN, which has a paper.",
    demo: null,
    diagram: null,
  },
];

export const byDate = () => [...mechanisms].sort((a, b) => a.date.localeCompare(b.date));
