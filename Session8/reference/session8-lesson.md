# Session 8: Modern Attention Variants

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms8msrhwu8ss0yxe9zg/lesson
Captured: 2026-08-15

Widgets (17) saved under `widgets/`, inline scripts collapsed into `widgets-scripts-dump.txt`.
Videos embedded on the page: https://youtu.be/sMgWxXFAvHM and https://youtu.be/mxg43ZJ86Fg

## 1. What this session is

At the end of Session 7, we had reached this point:

    token id
       ↓
    fixed byte codec
       ↓
    code
       ↓
    trainable projection
       ↓
    [B, T, D]

So every token has now become a vector of width D.

The obvious next question is: what consumes those vectors?

That is what this session is about.

Attention is the first place where tokens start interacting with one another. Until now, token 17 could become a beautiful 4,096-dimensional vector and still know absolutely nothing about token 16 or token 18.

Attention gives each token a way to look at the other tokens in its context and bring back the information that matters.

We will build the ordinary version first. Only after that will we ask what becomes difficult when the context gets long:

- comparing every token with every other token becomes expensive,
- remembering the useful parts of every earlier token takes memory.

Those are two different problems. The rest of the session is a tour of the different mechanisms people have proposed to reduce one or both of them.

## 2. Attention

We have used the word attention since Session 1, but we have deliberately not built it yet.

Now we will.

Assume one token enters the attention layer as a vector:

    x

The layer makes three different projections of that same vector:

                 ┌── Wq ──> query
    x ───────────┼── Wk ──> key
                 └── Wv ──> value

Why three? Because looking for information and giving information are different jobs.

A useful mental model is:

    query = what am I looking for?
    key   = what kind of information do I contain?
    value = what information should I give you if you choose me?

Suppose the sequence is:

    The cat sat on the mat

When the model is processing `sat`, its query may strongly match the key for `cat`, because knowing who sat matters. If that match is strong, more of `cat`'s value flows into the representation for `sat`.

That is attention in one sentence:

Each token asks a question, every other token advertises what it contains, and the best matches contribute the most information.

The standard scaled dot-product attention equation says exactly the same thing:

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Read it from the inside out:

- `QKᵀ` compares every query with every key and produces the score matrix.
- Dividing by `√d_k` keeps those scores from growing too large as the query/key width increases. Here `d_k` is the number of components in each query and key for one head.
- `softmax` is applied across each query's row of scores, turning that row into positive weights that add to one.
- Multiplying by `V` uses those weights to combine the value vectors.

*Widget: the whole layer on one screen, computed live. A small transformer runs inside the page: 32 dimensions, 4 heads, 2 blocks. Follow one word from the left. Its embedding splits into a query, a key and a value, the ribbons show where each one goes, the dot grid in the middle is how strongly every word attends to every earlier word, and the bars on the right are the next word. Step through the heads and the blocks, and pull the temperature to watch the prediction sharpen and spread.*

Mechanically, standard scaled dot-product attention is only four steps.

1. **Score.** Compare the current query with every key using a dot product. One query-key pair gives one number.
2. **Scale.** Divide the score by `sqrt(d_k)`. As the query/key dimension grows, raw dot products tend to grow too; scaling keeps the numbers in a useful range for softmax.
3. **Softmax.** Convert the scores into positive weights that add to one.
4. **Weighted sum.** Multiply every value by its weight and add them together. The result is one new vector for the current token.

So for one token:

    query
      │
      ├── compare with key 0 ──> score 0 ──┐
      ├── compare with key 1 ──> score 1 ──┤
      ├── compare with key 2 ──> score 2 ──┤─> softmax ─> weights
      └── compare with key 3 ──> score 3 ──┘
                                           │
    values ────────────────────────────────┘
                     ↓
               weighted sum
                     ↓
               output vector

For a decoder model there is one extra rule: the causal mask.

With that mask included, the equation becomes:

    CausalAttention(Q, K, V) = softmax(QKᵀ / √d_k + M) V

`M` is 0 where attention is allowed and `-∞` for a future token. It is added before softmax, so every forbidden position receives weight zero.

If we are generating token 4, token 4 is allowed to read tokens 0, 1, 2 and 3. It is not allowed to read token 5, because token 5 is in the future.

During training, however, we want to process the whole sequence in parallel. So we do not physically stop the computation at every token. Instead, we set every forbidden future score to `-infinity` before softmax:

    allowed score = normal number
    future score  = -infinity

    softmax(-infinity) = 0

The future therefore receives exactly zero attention.

This is what lets us train on the whole sequence at once while preserving left-to-right generation.

At this point, keep one picture in mind:

    tokens
      ↓
    queries, keys, values
      ↓
    which earlier tokens matter?
      ↓
    weighted information comes back

Everything else in this session changes one part of that picture. Some methods change how many tokens we compare. Some change what we store. Some change how position is represented. We will name those changes only after the standard mechanism is clear.

Two facts from standard attention will matter for the rest of the session.

**Fact 1:** every token is compared with every other token. Six tokens give 6 x 6 = 36 scores. Six hundred tokens give 600 x 600 = 360,000. Ten thousand tokens give one hundred million scores. This exact all-to-all comparison is both the strength and the cost of attention.

**Fact 2:** attention itself does not know token order. If token 0 and token 4 contain the same word and enter with the same vector, their query, key and value projections are also the same. Something else has to tell the model that one came earlier and one came later. We will fix that in Section 8.

*Widget: six tokens, four numbers each, with every value computed. Go through the five steps: Q/K/V projection, the full 36-score matrix, scaling and softmax, the causal mask, and finally the weighted sum. On the mask tab, turn the mask off. You should immediately see attention weight move onto tokens that have not happened yet. That is the exact problem the causal mask solves.*

## 3. The two bills attention sends

The 6 x 6 score matrix above looks harmless because six is small. The first problem appears when the context grows: every new token has more other tokens to compare with.

    T = 1,000   →       1,000,000 query-key scores
    T = 10,000  →     100,000,000 scores
    T = 100,000 →  10,000,000,000 scores

That is the first bill: compute. The pairwise attention work grows approximately as T². Double the context and this part becomes roughly four times larger.

There is a second bill, but it belongs to a different moment: generation.

When a model writes a reply one token at a time, it has already computed the key and value for the earlier tokens. It would be wasteful to throw those away and calculate them again every time the next token is generated. So it keeps them:

    after "The"          → K₁, V₁
    after "The cat"      → K₁, V₁, K₂, V₂
    after "The cat sat"  → K₁, V₁, K₂, V₂, K₃, V₃

This growing list of saved keys and values is the **KV cache**. It is simply the model's saved attention history for one active conversation.

The two bills therefore grow differently:

| BILL | WHAT GROWS | SIMPLE QUESTION |
| --- | --- | --- |
| Compute | The number of token-to-token comparisons, roughly T² | Can we afford to calculate all those comparisons? |
| KV-cache memory | One saved key and value for every earlier token, roughly T | Can we afford to keep all that conversation history? |

The model weights are a useful comparison. They are mostly fixed: load them once and many users can share them. A KV cache is not shared:

    user 1 conversation → its own cache
    user 2 conversation → its own cache
    user 3 conversation → its own cache

So a longer context creates two separate engineering questions:

Does attention become too expensive to compute, or does the saved conversation history become too large to store?

Do not try to calculate the full memory bill yet. First make the difference visible. Section 10 will return to the cache formula, precision, concurrency and head sharing. The small widget below introduces only the two growth patterns.

*Widget: a first picture of the two bills. Move the context slider and watch two simple curves: attention compute grows quadratically, while the KV cache grows linearly. The widget first shows how the cache accumulates one key-value pair per earlier token, then compares the two growth patterns. The detailed cache calculator comes later, after the cache has become familiar.*

## 4. What happens if you remove the softmax

Everything from here through Section 7 builds directly on the standard attention we just constructed.

Start with one strange question:

What if we remove the softmax?

For the moment, forget production architectures. We are doing a small arithmetic experiment.

### First compute attention without softmax

Use ordinary numbers before we return to vectors and matrices.

Suppose we have one query and three old key-value pairs:

    query q = 2

    key k₁ = 0.5   value v₁ = 10
    key k₂ = 1.0   value v₂ = 20
    key k₃ = 1.5   value v₃ = 30

Compare the query with each key:

    q × k₁ = 2 × 0.5 = 1
    q × k₂ = 2 × 1.0 = 2
    q × k₃ = 2 × 1.5 = 3

Those three numbers are the attention scores. Without softmax, use them directly on the values:

    output = 1×10 + 2×20 + 3×30
           = 10 + 40 + 90
           = 140

This is the direct route:

    query arrives
       ↓
    visit every old key
       ↓
    weight every old value
       ↓
    output = 140

### Now change only the order

Write the same calculation without doing the multiplications yet:

    (q×k₁)×v₁ + (q×k₂)×v₂ + (q×k₃)×v₃

Every term contains the same query `q`, so we can factor it out:

    q × (k₁v₁ + k₂v₂ + k₃v₃)

The part inside the brackets does not need the query. We can calculate it while the old tokens arrive:

    S = k₁v₁ + k₂v₂ + k₃v₃

    S = 0.5×10 + 1.0×20 + 1.5×30
      = 5 + 20 + 45
      = 70

Now the new query reads only `S`:

    output = q × S
           = 2 × 70
           = 140

Both routes return exactly the same answer:

    direct route    = 140
    regrouped route = 140

That precomputed `S` is the **running state**.

Instead of keeping a growing list:

    k₁,v₁
    k₂,v₂
    k₃,v₃
    ...

we continually update one object:

    S = S + key×value

With real vectors, `S` is a matrix rather than one number. The same distributive rule gives us:

    direct:    y = Σ (q · kⱼ) vⱼ

    regrouped: S = Σ vⱼ kⱼᵀ
               y = S q

If keys and values are four numbers wide, `S` remains a 4 × 4 matrix:

    after 10 tokens        → one 4×4 matrix
    after 1,000 tokens     → one 4×4 matrix
    after 1,000,000 tokens → one 4×4 matrix

More information is written into the same state. Its shape does not grow with the number of tokens.

### Now put softmax back

Our raw scores were:

    [1, 2, 3]

Softmax first exponentiates them:

    exp(1) ≈ 2.72
    exp(2) ≈ 7.39
    exp(3) ≈ 20.09

    total ≈ 30.20

Then it divides every number by that shared total:

    softmax weights ≈ [0.09, 0.24, 0.67]

The output is now:

    0.09×10 + 0.24×20 + 0.67×30 ≈ 25.8

The important part is not the new answer. It is the shared denominator:

    exp(score₁) + exp(score₂) + exp(score₃)

To calculate the weight of key 1, we need the scores of keys 2 and 3. If one score changes, all three softmax weights change.

That is what it means to say that softmax ties the scores together.

Without softmax, each term is independent, so we can factor out the query and pre-combine the old key-value pairs.

With softmax, the weight for key j is:

                    exp(q × kⱼ)
    weight j = ─────────────────────────
                Σ exp(q × every old key)

A different query creates different scores, a different denominator and different weights for every key. Exact softmax attention therefore needs the individual old keys to remain available when each new query arrives.

The whole distinction is:

    Softmax OFF
      → old key-value pairs can be folded into one fixed state

    Exact softmax ON
      → each new query must revisit the individual old keys

Removing softmax lets us summarise the past before the query arrives. Exact softmax needs the whole score list after the query arrives.

This opens the door to **linear attention**: replace a growing exact history with a fixed-size state and make sequence work grow linearly rather than quadratically.

### So what happened to softmax?

In the simple experiment above, softmax is genuinely gone. It is not applied later. That changes the mechanism, not just its implementation.

Softmax was doing several useful jobs:

- It made every weight positive. A token could receive more or less attention, but not an uncontrolled negative weight.
- It made the weights add to one. The output remained a weighted average, so its scale did not automatically grow just because the context contained more tokens.
- It created competition. Giving more weight to one key necessarily left less weight for the others.
- It gave every query a fresh distribution over the exact old keys. Different queries could select very different parts of the same history.

Removing softmax gives up those guarantees.

Raw dot-product scores can be negative or very large. Their sum is not normalised. As more tokens are added, the magnitude of the accumulated state can grow. And because the complete past is compressed into one fixed-size object, different memories can interfere with one another.

So the two mechanisms are not expected to produce the same answer:

| EXACT SOFTMAX ATTENTION | SIMPLE NO-SOFTMAX STATE |
| --- | --- |
| Keeps individual keys and values | Compresses them into one state |
| Produces positive weights that sum to one | Uses raw, unnormalised scores |
| Lets every query form a fresh distribution | Lets every query read the same accumulated state differently |
| More exact access to old tokens | Cheaper but lossy memory |
| Growing KV cache and expensive all-pairs work | Fixed-size state and linear sequence work |

### Then why remove softmax at all?

Because removing it unlocks the regrouping that makes fixed-state, linear-time attention possible.

We are deliberately exchanging exact token-by-token lookup for a compressed memory that can be much cheaper to train and serve.

That exchange creates the next problems in the session:

    Section 5  → naive state writes accumulate and interfere
    Section 6  → the delta rule learns to correct old state
    Section 7  → sparse attention keeps softmax but reads fewer keys
    Section 13 → hybrid schedules mix fixed-state and softmax-style layers

Practical linear-attention systems therefore rarely delete softmax and call the work finished. They add back some of its useful behaviour through feature maps, explicit normalisation, gates, decay and better write rules. Hybrid architectures also keep occasional softmax or sparse-attention layers for direct access to old tokens.

The conclusion of this section is not that softmax was bad:

Softmax gives excellent query-specific selection, but it prevents this simple fixed-state regrouping. Removing it buys efficiency and creates a new memory-quality problem that the following sections must solve.

*Widget: one claim, shown two ways. The left path visits every old key and value. The right path reads one pre-built state. With softmax off, both outputs match exactly. Turn softmax on and the fixed-state regrouping stops matching. Use Replay to watch where the paths separate.*

## 5. A new state can still carry the old contribution

The previous section compressed the past into one fixed-size state. Under the hood, that state is a matrix: a grid of numbers.

You do not need to calculate that grid here. For this section, think of it as a small scratchpad with two jobs:

    WRITE: connect a clue to an answer
    READ:  use the clue to recover the answer

Suppose the clue is key A.

The first write says:

    key A → old answer

Now a read with key A returns the old answer. So the state can act as a memory.

Later, the answer changes:

    key A → new answer

What should happen? The new answer should replace the old one.

The state variable `S` is overwritten, but look at how its new contents are calculated:

    new state = old state + new write

After the update there is only one matrix—the new state. However, the numerical contribution from the old association is still inside it because the calculation started with the old state.

For example, suppose key A currently returns 40 and should now return 55. An add-only write produces:

    new read for A = 40 + 55 = 95   ← not 55

The memory is not holding two visible records. It is holding one newly computed matrix whose summed numbers still contain the effect of the old answer.

Three things are worth separating:

1. There is only one current matrix after each write.
2. Its fixed size is not the problem; the matrix did not need to grow.
3. The add-only write rule always carries the old contribution forward; it has no correction term that cancels what is no longer wanted.

A useful memory must do more than remember new information. It must also know how to correct old information.

The next section gives the state that missing ability: read the old answer first, then write only the correction.

*Widget: try the two writes in order. The first state makes key A return 40. The second write creates one new state, but that state was calculated as old state plus the complete new answer. Key A therefore returns 95 instead of 55. There are never two separate state matrices.*

## 6. The delta rule: write only what needs to change

The arithmetic is simple. We discuss it because a fixed-size attention state must be able to update its memory, not merely pile up more information.

Suppose key A currently returns 40, but the answer should now be 55.

Adding the complete new answer repeats Section 5's mistake:

    40 + 55 = 95   ← wrong

The delta rule first measures the gap:

    current answer:      40
    wanted answer:       55
    correction (delta):  55 - 40 = 15

    write the correction: 40 + 15 = 55

That is the whole rule:

Read what memory says, find the difference, and write only that difference.

Why does this matter for attention? Removing softmax gave us a compact state, but an add-only state could not revise an old association. The delta rule turns that state into an updateable memory. Modern variants can also learn how strongly to write or when old information should fade, but those are later refinements—not new ideas required here.

*Widget: follow one correction. Read 40, calculate the missing +15, and apply it to reach 55. The comparison at the bottom shows why adding the whole new answer produces 95 instead.*

## 7. The other lever: do not look at everything

Linear attention changed the kind of memory the model uses. Sparse attention takes a different route:

Keep normal softmax attention, but let each query use only a small number of keys.

Imagine that one query has twelve earlier tokens it could look at. Full attention uses all twelve. Top-k attention keeps only the k keys with the highest scores and drops the rest before forming the output.

For one query, the process is:

    compare the query with the candidate keys
       ↓
    keep the best k
       ↓
    apply softmax to those k scores and combine those k values

If k is much smaller than the context length T, the final attention calculation uses far fewer value vectors.

    full attention:  use T values
    top-k attention: use k values, where k << T

This can work when most of the useful attention is concentrated on a few tokens. But that is an empirical property, not a guarantee: some queries need evidence from many places, and dropping a useful key can change the answer.

There is also a catch hidden in the phrase "keep the best k":

How do we know which keys are best?

The naïve method first computes a score for every key, sorts those scores, and only then keeps the top k.

    score all T keys → discover top k → use only k values

That reduces the work after selection, but it does not remove the cost of scoring every candidate. If scoring all keys was the expensive part, naïve top-k has not solved the main problem.

Practical sparse-attention systems therefore need a cheaper proposal step: for example, a local window, a learned router, or a compressed index that suggests where to look. Exact attention then runs only on those candidates.

This creates the real trade-off:

    cheaper candidate search
         versus
    risk of failing to propose a useful key

So sparse attention is not simply "full attention with most entries deleted." Its success depends on finding a candidate-selection method that is both cheaper than full scoring and accurate enough for the task.

**Architecture note.** LightningLM V4 used sparse-attention G-layers alongside DeltaNet layers. Its maximum budget was reduced from 1024 to 256 because of backward-kernel contention in that particular hardware and software stack. That makes 256 an implementation constraint from that run, not a universal law of sparse attention. V4 also varied the budget by token; that is a later design choice, not required to understand the core top-k idea here.

*Widget: move k and follow the same query through three steps. Every key is scored, only the highest k survive, and only those values enter the final sum. The bottom bars make the catch visible: value work falls with k, while the naïve selection cost stays at all twelve scores.*

## 8. Position, and the part that is usually skipped

Section 2 left one problem unresolved.

The query–key dot product compares the content of two token vectors. By itself, that number does not say whether the tokens are next to each other or hundreds of positions apart.

The causal mask helps with one part of order: it prevents a token from looking into the future. But among the tokens it is allowed to see, the dot product still needs a clue about distance.

For example, imagine that the same word appears twice:

    position 2  → "bank"
    position 20 → "bank"

If both occurrences produced the same key vector, a content-only comparison would not know which occurrence was nearby. Attention needs position to become part of the query–key comparison.

### RoPE adds position by turning vectors

Rotary position embedding (RoPE) handles this with rotation.

Take two dimensions from a query or key and view them as one arrow on a 2D plane:

    (x0, x1) → one 2D arrow

Choose a small rotation angle θ for each step in the sequence. Then rotate the arrow according to the token's position:

    position 0 → rotate by 0 × θ
    position 1 → rotate by 1 × θ
    position 2 → rotate by 2 × θ
    ...

So a query at position i is rotated by iθ, while a key at position j is rotated by jθ.

The useful part appears when we take their dot product. A dot product depends on the angle between the two arrows. Both absolute rotations cancel, leaving only their difference:

    R(iθ)q · R(jθ)k
      = q · R((j − i)θ)k

The positional part of the score therefore depends on:

    i − j

That is the distance between the tokens.

Suppose a key is at position 2 and a query is at position 8. Their gap is 6. Now move both tokens ten places forward:

    positions 2 and 8   → gap 6
    positions 12 and 18 → gap 6

Both arrows rotate further, but they rotate together. The angle between them stays the same, so their positional relationship stays the same.

RoPE turns relative token distance into relative rotation inside the attention score.

This is more useful than attaching only an absolute label such as "I am token 8." The relationship "that token is six places behind me" still makes sense when the whole sequence moves.

### The implementation detail people often miss

A real attention head has more than two dimensions. RoPE groups the dimensions into 2D pairs and performs the same rotation in each pair. Different pairs can rotate at different rates, allowing the head to represent distance at more than one scale.

Not every dimension has to be rotated. Some implementations rotate only part of the head and leave the remaining dimensions unchanged. DeepSeek-V4, for example, applies RoPE to the last 64 dimensions. That is an implementation choice; it does not change the central mechanism above.

There is also an important boundary to the claim. Because RoPE computes rotations from a function, it has no finite lookup-table wall. The function can be evaluated at a position beyond the training length. But this does not prove that the model will behave well there. Section 9 deals with that separate long-context problem.

*Widget: a one-screen RoPE demonstration. Switch RoPE off to see that identical content vectors cannot reveal distance. Switch it on and change the distance to see the relative angle and score change. Then move both tokens together: their absolute positions change, but their distance, relative angle and score stay fixed.*

## 9. DroPE, and how eight thousand became two hundred and fifty-six thousand

RoPE solved one positional problem: it replaced a finite lookup table with a rotation that can be calculated at any token position.

But being calculable is not the same as working well.

Imagine a model trained only on sequences up to 8K tokens. We can still ask RoPE for the rotation at position 256K. The formula will return an answer. That tells us only this:

    the positional rule is defined at 256K

It does not tell us this:

    the model can use a 256K context reliably

The second claim is about the whole trained model. During training, every layer learned to operate with the positional patterns it encountered inside the 8K range. Far beyond that range, the rotations may repeat patterns or combine in ways the model never learned to interpret.

So long-context extension needs evidence. It is not guaranteed just because the positional formula keeps producing numbers.

This is where DroPE appears in the V4 record.

The reported training story is:

    trained context:  8K
    reported context: 256K
    extension:        32×

The cookbook also records:

    positional recalibration: DroPE, applied before annealing

"Before annealing" matters. It tells us that this was part of the training process. The positional behaviour was changed while the model still had training steps and learning rate available to adapt. DroPE should therefore not be presented as a switch that magically turns an 8K model into a 256K model at inference time.

There is also an important evidence boundary.

What the available record establishes: V4 trained at 8K, reportedly reached 256K, and used a step called DroPE before annealing.

What the available record does not establish: the exact DroPE algorithm or which rotary dimensions it changes. Any more detailed mechanism is a hypothesis until it is checked against the reference implementation.

That boundary is the main lesson of this section:

A position function can exist beyond training length. Model capability at that length must still be earned—and demonstrated.

The number to remember is:

    256K ÷ 8K = 32×

This result is why context extension is attractive. Training every step at the final context length is expensive. Recalibrating after shorter-context training can be much cheaper. But a reported extension factor is evidence for that specific model and procedure, not a promise that every model or every longer target will work.

*Widget: defined is not proven. Switch between plain RoPE at 256K and the reported V4 + DroPE result. Plain RoPE shows that the rotation can still be calculated; it does not establish model quality. The DroPE view adds only what the source record supports: an 8K → 256K result, a 32× extension, with positional recalibration applied before annealing. The widget deliberately does not simulate an unverified DroPE mechanism.*

## 10. The cache bill, done properly

Section 3 introduced the KV cache as the saved keys and values from earlier tokens. Now we can calculate exactly how much memory that saved history needs.

Start with one token in one layer.

The layer stores two vectors:

    one key + one value
        K       V

That is where the first `2` in the formula comes from.

Each vector contains `head_dim` numbers, and there is one pair for every KV head. So one token in one layer needs:

    2 × kv_heads × head_dim numbers

But every layer makes its own keys and values. Multiplying by the number of layers gives the cache added by one token across the whole model:

    2 × layers × kv_heads × head_dim numbers per token

Now let the conversation contain T tokens. Every token has left its own K/V record behind:

    2 × layers × kv_heads × head_dim × T numbers per user

Finally, each stored number occupies memory. With bf16, each number takes two bytes:

    cache bytes per user
      = 2 × layers × kv_heads × head_dim × T × bytes_per_number

The formula looks long, but every factor answers a simple question:

    2               key and value
    layers          each layer has its own cache
    kv_heads        each KV head stores its own vectors
    head_dim        numbers in each vector
    T               tokens saved for this conversation
    bytes_per_number storage used by each number

### One user's cache becomes every user's cache

The formula above is for one active conversation. A server normally handles several conversations at once, and their histories cannot be shared. If there are B active users with the same context length:

    total cache bytes = cache bytes per user × B

This is the batch multiplier often included at the end of the full formula:

    total cache bytes
      = 2 × layers × kv_heads × head_dim × T × batch × bytes_per_number

For the comparison example used in this lesson—48 layers, 8 KV heads, head dimension 128 and bf16—the raw cache at 32,768 tokens is:

    one user    ≈  6.44 GB
    eight users ≈ 51.54 GB

Double the context and both numbers double. Keep the context fixed and double the active users, and the total also doubles.

That is the operational meaning of calling KV cache a per-user serving cost. Model weights are loaded once and shared. Conversation history is private to each active sequence.

Context length sets the cache cost of one conversation. Concurrency multiplies that cost by the number of active conversations.

The calculation above counts only the raw K and V tensors. A real inference server also needs memory for model weights, activations, temporary attention workspaces and allocator headroom. So the formula explains the cache bill; it does not predict total accelerator memory by itself.

For later comparisons, this lesson uses 8 KV heads, bf16 and head dimension 128 as its GQA yardstick. The next section changes one factor—`kv_heads`—and shows exactly what grouped-query attention saves.

*Widget: a progressive cache formula. Use Next multiplier to see why each factor exists. Then change context length to alter one user's cache and active users to see that private cache copied across concurrent conversations. The architecture stays fixed so the two serving multipliers remain easy to see.*

## 11. Grouped-query attention, and why it is the baseline rather than the answer

Section 10 gave us the KV-cache formula. For this section, keep your eye on just two terms:

    cache size ∝ kv_heads × context length

Grouped-query attention (GQA) reduces the first term.

In ordinary multi-head attention, every query head can have its own key and value head. That means every token adds many K/V vectors to the cache.

GQA keeps the different query heads, but lets several of them share one K/V head:

    query heads 1, 2, 3, 4 ──> shared K/V head A
    query heads 5, 6, 7, 8 ──> shared K/V head B

The query heads can still ask different questions. We simply store fewer versions of the keys and values they search through.

For example, compare eight query heads in three arrangements:

    MHA: 8 query heads, 8 K/V heads
    GQA: 8 query heads, 2 K/V heads
    MQA: 8 query heads, 1 K/V head

In this example, GQA stores one quarter as many K/V heads as MHA. Holding everything else fixed, its KV cache is therefore four times smaller.

The one-K/V-head extreme is called multi-query attention (MQA). It saves even more cache, although more sharing can affect model quality. The widget below compares memory scaling only; it does not predict that quality trade-off.

So why is GQA a baseline rather than the final answer?

Because it changes how much we store per token, but it still stores something for every token.

    MHA cache: 8 × T
    GQA cache: 2 × T
    MQA cache: 1 × T

GQA lowers the slope of the line. It does not make the line stop growing. Double the context length and the GQA cache still doubles; at a million tokens, it still contains entries from a million positions.

GQA is a strong practical baseline because K/V sharing gives a large cache saving with a useful quality trade-off. But it does not solve long-context memory: the cache still grows linearly with context.

The next section asks what happens if we reduce the other term—how many token positions must be stored.

*Widget: a head-sharing visual. Switch between MHA, GQA and MQA to see eight query heads reuse fewer stored K/V heads. Then increase context length. Sharing makes the cache line less steep, but every line continues upward because every new token still adds to the cache.*

## 12. Compressing the sequence itself

Section 11 reduced the number of K/V heads stored for each token. But even with GQA, the cache still keeps an entry for every token position.

This section changes the other part of the problem:

What if several nearby token positions could share one stored entry?

Start with a sequence that stores one K/V entry per token:

    t1 → KV1
    t2 → KV2
    t3 → KV3
    t4 → KV4

Now combine those four positions into one block summary:

    t1, t2, t3, t4 → compressed block 1

If every block contains m tokens, the number of long-range entries falls from roughly T to T / m:

    stored positions before: T
    stored positions after:  T / m

This is sequence compression. GQA stores fewer heads for each position; sequence compression stores fewer positions.

DeepSeek-V4's Compressed Sparse Attention adds a second saving. After making the block summaries, it does not run expensive attention over every summary. It selects only the top-k summaries that look most relevant to the current query.

    all tokens
       ↓ compress nearby tokens
    fewer block summaries
       ↓ select top-k blocks
    fewer expensive attention reads

These two steps reduce different costs:

    compression → fewer entries stored
    top-k       → fewer summaries read by expensive attention

But selecting the best blocks creates a practical question: if we must run full attention over every block just to find the best ones, we have not saved much work.

DeepSeek uses a small low-rank indexer to rank the block summaries cheaply. The indexer does not produce the final attention output. It only chooses which blocks deserve the more expensive read.

The trade-off is important. Compression can lose token-level detail because one summary now speaks for several tokens. Approximate top-k selection can also miss a useful block. A real architecture must protect important recent detail and train the summaries and indexer well; the widget below demonstrates the storage-and-read scaling idea, not the model's quality.

The reported DeepSeek-V4 architecture also interleaves a heavily compressed dense form with the top-k sparse form. The shared idea is the same: old history does not have to keep one equally expensive representation for every original token.

Sequence compression reduces how much history is stored. Top-k selection reduces how much of that compressed history is read for one query.

*Widget: a two-stage sequence-compression visual. Increase the number of tokens represented by each block and watch the number of stored positions fall. Then change top-k and watch expensive attention read only that many summaries. The small note about the low-rank indexer explains how the ranking can be cheaper than the final attention calculation.*

## 13. Schedules across depth

So far, each attention mechanism may have sounded like a complete replacement for the others:

    standard attention
        OR
    linear attention
        OR
    sparse attention

But a deep model does not have to make that choice once for the whole network.

Different layers can use different memory systems.

A fixed-state layer compresses the past into a small running state. That makes it cheap to serve, because its memory does not keep growing with every earlier token. The tradeoff is that the old sequence is no longer available token by token.

A sparse-attention layer keeps keys and values from the sequence and lets a query read selected earlier tokens directly. That restores exact token access, but every such layer adds a KV cache that grows with context length.

This creates a useful division of labor:

    many fixed-state layers
      → process and combine information cheaply

    occasional sparse-attention layers
      → revisit selected earlier tokens directly

The order of those layers is called the depth schedule.

LightningLM V4 used this repeating eight-layer motif:

    D D D G D D D G

where:

    D = DeltaNet fixed-state layer
    G = sparse-attention layer

So every motif contains six fixed-state layers and two sparse-attention layers.

**Why not use G everywhere?** More G layers give the model more frequent opportunities to read exact earlier tokens, but they also create more per-sequence KV cache. In the V4 configuration, changing from one sparse-attention layer per eight layers to eight per eight makes the KV state 8.0× larger, while the estimated mixing compute rises only about 1.41×.

For this configuration, the main price of adding more G layers is therefore serving memory, not FLOPs.

**Why not use D everywhere?** A fixed recurrent state is cheap, but it is a compressed summary. If every layer uses only that state, the model never gets another direct look at the exact old token representations.

The schedule chooses a compromise:

    more D layers → lower serving-memory cost, more compression
    more G layers → more frequent exact access, larger KV cache

V4 kept DDDGDDDG from the 1.78B seed model to the 120B run. That is meaningful evidence that the mixture works across scale.

It is not evidence that six D layers and two G layers are the best possible ratio. The neighboring schedules were not cleanly ablated.

DDDGDDDG is a successful design choice, not a proven optimum.

The broader lesson is more important than this exact motif:

Not every layer needs the same memory system. The schedule itself is part of the architecture.

*Widget: compare three depth schedules. Each D block compresses history into a fixed-size state; each G block gives the model another opportunity to read selected earlier tokens directly. Switch among the schedules to see the tradeoff between frequent exact access and the KV cache carried by attention layers. The V4 preset is marked separately because it is reported evidence; the alternatives are illustrations, not experiments the paper ran.*

## 14. State that outlives the window

Everything so far has described memory inside the current sequence window.

Now suppose we split a very long document into chunks. Chunk 1 ends, its internal attention state is discarded, and chunk 2 begins. What information crosses that boundary?

Without an explicit mechanism, the answer may be: none of it.

V4 adds a Memory Stream for this boundary. The mechanism carries just one thing forward:

At the end of a chunk, take its final hidden state and use it as one summary vector for the next chunk.

    chunk 1 ──writes──> one memory vector ──nudges──> chunk 2

The vector has the same width as the model. It does not get longer when the document gets longer. After two chunks or two thousand chunks, the stream is still one model-width vector.

That is fixed-size, or O(1), cross-chunk state.

It is also a severe compression. One vector cannot preserve every sentence from every earlier chunk. It can only carry a useful summary signal.

### Two small rules make the stream practical

First, the vector is written with stop-gradient:

    memory = stop_gradient(final hidden state)

The next chunk can read that memory during the forward pass, but its training loss cannot send gradients backward through the boundary into all previous chunks. This avoids building one enormous training graph across the complete document.

Second, each token gets a small learned gate:

    updated token = current token + scale × gate × memory

The gate is between zero and one.

    gate near 0 → this token mostly ignores the old summary
    small gate  → the old summary gives this token a nudge
    large gate  → the old summary can begin to dominate the present

The gate is learned because different tokens need different amounts of old information. A question referring to the previous chunk may use the stream; an unrelated token may largely ignore it.

The reported V4 injection scale starts near 0.078 and reaches about 0.391 by the 2B stage. The average injected memory is only about 6% of the current embedding magnitude.

That is the right mental model:

The Memory Stream is a small nudge from the previous chunk, not a copy of the old chunk and not a replacement for the current token.

*Widget: try the boundary. Read chunk 1, cross the boundary, and ask the question in chunk 2. Turn the stream off to see that nothing survives. With the stream on, adjust the learned gate: a closed gate ignores the memory, while the default small gate creates roughly a 6% nudge. The red crossed arrow is the stop-gradient rule—information moves forward, but training does not backpropagate through the previous chunk.*

## 15. Long context is a system, not a number

We can now collect the whole session.

When somebody says:

    "our model supports 256K context"

that sounds like one specification.

It is not.

For that claim to be meaningful, several different things must work at the same time.

**1. Position must still make sense**

The positional mechanism must remain meaningful at the target length, either because the model was trained there natively or because a method such as DroPE, NTK-aware scaling or YaRN extended it successfully.

**2. The cache must fit**

A theoretical 1M-token model is not useful if one sequence consumes all accelerator memory.

This is where GQA, MQA and sequence compression matter.

**3. Compute must be affordable**

Even if the cache fits, full quadratic attention over the entire context may still be too expensive.

This is where sparse attention, top-k selection and linear attention matter.

**4. State outside the active window must be handled**

If the system chunks documents or discards internal state at boundaries, some mechanism may be needed to carry useful information forward. V4's Memory Stream is one answer.

**5. Training must expose the model to the behaviour we expect**

This distinction is critical:

    architecture is mathematically defined at 256K

is not the same as:

    model has learned to use 256K well

A positional function may extrapolate forever while the model's competence does not.

**6. Evaluation has to test understanding, not only retrieval**

A model can sometimes find one planted sentence in a huge context while failing to combine evidence, follow long dependencies or understand the document globally.

So a successful needle-in-a-haystack test is useful, but it is not a complete long-context evaluation.

Context length is not one number. It is the point where position, memory, compute, training and evaluation all still work together.

There is one more dimension that matters specifically for the model we are building.

Context windows are advertised in tokens.

Users do not think in tokens. They have documents, code, legal files, textbooks and conversations.

Session 3 showed why that difference matters.

Suppose the same meaning requires:

    English → 1x tokens
    Telugu  → ~3x tokens

Then a fixed 256K-token context holds much less Telugu text than English text.

Very roughly, if fertility were exactly 3x for the same semantic content:

    256K token window in English-equivalent content
       ↓
    ~1/3 as much comparable Telugu content

The accelerator does not care. It charges us per token either way.

The same nominal 256K context can represent very different amounts of human language depending on tokenizer fertility.

That connects Session 3, Session 7 and this session directly.

Tokenizer efficiency determines how quickly a language consumes the window. Embedding and architecture choices determine what the model costs. Attention determines what happens when that window becomes long.

For an India-first model, long context is therefore not just a premium feature. It also helps recover some of the effective document length lost when a language requires more tokens for the same content.

*Widget: a context-readiness board. Choose a target context and a model configuration. The board checks the six conditions above and reports which constraint fails first and by how much. Beside it, the same document is represented under different tokenizer fertility assumptions so that the cost appears as lost document capacity rather than an abstract ratio. A model's advertised context length is useful only when we know which of these constraints is actually binding.*

## 16. What V5 has to decide

We should not end this session by pretending V5's attention architecture has already been chosen.

V4 is finished.

Its configuration gives us evidence.

It does not automatically give us the V5 configuration.

The frontier moved after those choices were made, the hardware changed, the kernels changed, and some of our own V4 choices were never ablated.

So the useful question is not:

    What did V4 use?

It is:

    Which V4 decisions are now evidence,
    which are merely defaults,
    and what must be re-tested for V5?

This is also where the cohort contributes. The candidate techniques and ablations from these sessions are inputs into the V5 architecture review.

Some conclusions are already strong enough to carry forward.

### What the current evidence already settles

- A stored absolute position table is out. Session 7 already showed the hard length wall it creates.
- GQA alone is not enough for the target. It reduces KV-cache growth by a constant factor but does not remove linear growth with context.
- Some stronger long-context mechanism is required. Across the three reference architectures, every one uses at least one of linear state, sparsity or sequence compression. None uses none.

The unresolved questions are more interesting.

| OPEN QUESTION | NEEDS |
| --- | --- |
| Extend or build native. V4 stretched 32x with DroPE. DeepSeek built for a million natively. Extension is much cheaper but has a ceiling; native long training costs much more. | An extension-factor study at V5's target, plus a real compute and memory estimate for training natively at that length |
| The schedule and its ratio. DDDGDDDG worked, but the ratio was never varied systematically. | A schedule ablation at a scale large enough that the result is likely to transfer |
| Linear state versus sequence compression as the primary long-context lever. V4 and Qwen3.6 chose the first family; DeepSeek chose the second. | A matched-budget comparison at genuinely long context, not a short-context benchmark |
| The sparsity budget. V4's cap of 256 came partly from kernel constraints on the previous hardware/software stack. | Re-measurement on the current kernels and current hardware rather than inheritance |
| Whether the Memory Stream still earns its place once the normal context window is already very long. | An ablation at the actual target context length |
| Every head count and head dimension. Those numbers were derived around a width of 4,096, and V5 has not committed to that width. | Decide the model width first, then derive the attention geometry from it |

This is a useful state to be in.

We are not searching the entire architecture space from zero.

We know the broad design pressures:

- position must extrapolate or be trained long
- cache cannot remain huge
- compute cannot remain quadratic everywhere
- exact memory is still useful somewhere
- fixed state is cheap but lossy
- compression is cheap but destroys detail unless repaired

What remains is an experimental question about which mixture and which ratios give the best trade at V5's actual target.

We know the shape of the decision. We do not yet know the winning configuration. Getting to this shape is itself the part that took the field years; the remaining questions are narrow enough that well-designed proxy runs can actually settle them.

*Widget: a V5 attention decision board. Build a candidate from the choices in this session: layer schedule, position scheme, extension policy, sparse or compressed attention, head layout, cache precision, cross-chunk state and target context. The board uses the same formulas introduced earlier to report cache, compute, reach and training-cost estimates. Failed constraints are named numerically. The output is a candidate specification, which is what an architecture proposal should eventually become.*

## 17. Where this leaves you: two roads

Everything in this session can now be organised around two basic strategies.

### Road 1: train short, then stretch

Train at a context length you can afford.

Then modify or recalibrate the positional system so the model can operate much farther beyond that training length.

    affordable training length
       ↓
    positional extension / recalibration
       ↓
    much longer serving context

This is the cheaper road.

It is the road V4 took with DroPE:

    8K → 256K

The limitation is that an extension method has a practical ceiling. We cannot assume that because 32x worked, 320x or 3,200x will also work.

### Road 2: build long, then train long

Design the attention architecture so long sequences are affordable natively.

That means reducing the compute and memory costs enough that the model can actually train on long sequences rather than merely extrapolating its position function after training.

    long-context architecture
       ↓
    train on long sequences
       ↓
    native long-context competence

This is the expensive road.

DeepSeek's compression and sparsity approach is an example of this direction.

It does not have the same positional-extension ceiling because the model is being trained in the long regime itself, although it pays for that in training complexity and cost.

Neither road is automatically correct.

The choice depends on the target context, training budget, serving budget, expected workloads and how much quality is lost when stretching.

And several mechanisms from this session can serve either road. GQA reduces cache in both. Sparse attention can reduce long-context compute in both. Better positional schemes can help both.

Before the assignment, put an actual target on the problem:

    32K?
    256K?
    1M?
    10M?

Only then ask which road is realistic.

*Widget: the fork behind the whole session. One road trains at an affordable length and stretches the positional behaviour afterwards. The other changes the architecture enough to make long-sequence training practical in the first place. Pick a target context and trace both roads through training cost, KV cache, inference compute and failure mode. "Long context" is not one technique; it is a system design choice.*

## 18. The assignment

> This assignment is for the whole cohort. If it comes out well, I want to keep the best one and put it in front of next year's batch.
>
> Work with your AI agent and build a web app that explains every attention mechanism we covered today visually. Host it wherever you like: Netlify, Vercel or anything equivalent.
>
> Start with standard scaled dot-product attention.
>
> Do not start with GQA, RoPE or DeltaNet and assume the reader already understands what they modify.
>
> The app should first make this mechanism obvious:
>
>     Q × K
>       ↓
>     scores
>       ↓
>     scale
>       ↓
>     mask
>       ↓
>     softmax
>       ↓
>     weighted sum of V
>
> Everything else in the assignment should then be presented as a response to some limitation of what came before.
>
> **Put them in the order they were launched**
>
> This is the part I care about most.
>
> Do not arrange the techniques in the order I taught them today.
>
> Do not group them into a neat taxonomy first.
>
> Arrange them chronologically, by when the technique actually appeared.
>
> Why?
>
> Because I want the app to show the field thinking.
>
> Standard attention was not "bad" and then replaced by a "better" attention. It solved a major problem and created new costs. Later work attacked different parts of those costs.
>
> So the timeline should read more like:
>
>     here is what existed
>           ↓
>     here is the problem people hit
>           ↓
>     here is the new mechanism somebody proposed
>           ↓
>     here is what it fixed
>           ↓
>     here is the new trade-off it introduced
>
> If you do this properly, you should see the priorities of the field move over time: exact global attention, cheaper decoding memory, better position handling, longer contexts, recurrent state returning, sparsity returning, compression becoming more aggressive.
>
> A chronological story lets you see why an idea appeared. A list only tells you that it exists.
>
> **Pros and cons, honestly written**
>
> Every mechanism in this session is a trade.
>
> If your page contains a method with five advantages and no meaningful downside, go back and investigate it again.
>
> For each technique, answer three questions:
>
> - What does it buy?
> - What does it give up?
> - When would I actually choose it?
>
> The last question matters.
>
> A mechanism can be a very good choice for a 2K-context chatbot and a bad choice for a 1M-token agent. That does not make the mechanism good or bad in isolation. It means the architecture has a workload.
>
> **What to cover, at minimum**
>
> Standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi, MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the delta rule and Gated DeltaNet, MLA, sparse and top-k attention, DeepSeek's compressed sparse attention, and DroPE.
>
> Add anything important I missed. Finding a relevant mechanism that was not covered in class counts in your favour.
>
> But if you add one, it has to fit the same standard: original date, original motivation, mechanism, advantage, cost and where it belongs in the timeline.
>
> **What to submit**
>
> Make something you would genuinely send to a friend who asked:
>
>     "How does attention work now?"
>
> Interactive is useful if the interaction teaches something.
>
> Animation is useful if motion makes the mechanism easier to understand.
>
> Neither is required.
>
> A clear static explanation is better than a clever interaction that does not teach the mechanism.
>
> Your agent can generate much of the HTML, JavaScript and visual scaffolding.
>
> Your job is to make sure the story is technically correct.
>
> Submit:
>
> - the live link,
> - the GitHub repository,
> - and a README that lists the sources used for the chronology.
>
> The chronology needs sources because dates are exactly where an AI agent can sound completely confident while being wrong.
>
> One warning: check every launch date against the actual paper, release or primary source. Do not accept a date because an agent gave it confidently. I got a mechanism wrong in Session 7 by trusting a confident sentence and had to correct it at the beginning of today's class. If your agent makes the same kind of mistake, catch it. If I have made one here, catch that too.
>
> The purpose of the assignment is not to produce eighteen cards about attention.
>
> The purpose is to be able to look at the timeline and understand:
>
>     what problem existed
>           ↓
>     why somebody changed the mechanism
>           ↓
>     what became cheaper
>           ↓
>     what became worse
>           ↓
>     what the next paper then had to fix
>
> If your app makes that progression obvious, it has done the job.
