"""Packing policies: assemble tokenized shards (documents) into fixed-length bins.

`long_context` and `structure_preserving` are implemented as tagged variants
of the shared one-doc-per-bin packer (`_one_doc_per_bin`) since they are
structurally identical - only the *justification* differs (a bigger window
vs. never merging multi-turn conversations into one bin).
"""

DEFAULT_SEQ_LEN = 128
LONG_CONTEXT_SEQ_LEN = 256

LANE_POLICY = {
    "general": "concat_and_chop",
    "code": "best_fit",
    "indic": "best_fit",
    "agentic": "structure_preserving",
    "reasoning": "long_context",
}


def _doc_tokens(shard):
    tokens, roles = [], []
    for seg in shard["segments"]:
        tokens.extend(seg["token_ids"])
        roles.extend([seg["role"]] * len(seg["token_ids"]))
    return tokens, roles


def _bin(policy, docs):
    return {"policy": policy, "docs": docs}


def _one_doc_per_bin(shards, seq_len, tag):
    bins = []
    for shard in shards:
        tokens, roles = _doc_tokens(shard)
        tokens, roles = tokens[:seq_len], roles[:seq_len]
        bins.append(_bin(tag, [{"shard_id": shard["shard_id"], "tokens": tokens, "roles": roles}]))
    return bins


def pack_pad_only(shards, seq_len=DEFAULT_SEQ_LEN):
    return _one_doc_per_bin(shards, seq_len, "pad_only")


def pack_long_context(shards, seq_len=LONG_CONTEXT_SEQ_LEN):
    return _one_doc_per_bin(shards, seq_len, "long_context")


def pack_structure_preserving(shards, seq_len=DEFAULT_SEQ_LEN):
    return _one_doc_per_bin(shards, seq_len, "structure_preserving")


def pack_concat_and_chop(shards, seq_len=DEFAULT_SEQ_LEN, eos_id=0):
    stream = []
    for shard in shards:
        tokens, roles = _doc_tokens(shard)
        for tok, role in zip(tokens, roles):
            stream.append((shard["shard_id"], tok, role))
        stream.append((shard["shard_id"], eos_id, "eos"))

    bins = []
    for start in range(0, len(stream), seq_len):
        window = stream[start:start + seq_len]
        docs, current_shard, cur_tokens, cur_roles = [], None, [], []
        for shard_id, tok, role in window:
            if shard_id != current_shard:
                if current_shard is not None:
                    docs.append({"shard_id": current_shard, "tokens": cur_tokens, "roles": cur_roles})
                current_shard, cur_tokens, cur_roles = shard_id, [], []
            cur_tokens.append(tok)
            cur_roles.append(role)
        if current_shard is not None:
            docs.append({"shard_id": current_shard, "tokens": cur_tokens, "roles": cur_roles})
        bins.append(_bin("concat_and_chop", docs))
    return bins


def pack_greedy(shards, seq_len=DEFAULT_SEQ_LEN):
    bins = []  # list of [remaining_capacity, docs]
    for shard in shards:
        tokens, roles = _doc_tokens(shard)
        tokens, roles = tokens[:seq_len], roles[:seq_len]
        entry = {"shard_id": shard["shard_id"], "tokens": tokens, "roles": roles}
        for slot in bins:
            if slot[0] >= len(tokens):
                slot[1].append(entry)
                slot[0] -= len(tokens)
                break
        else:
            bins.append([seq_len - len(tokens), [entry]])
    return [_bin("greedy", docs) for _, docs in bins]


def pack_best_fit(shards, seq_len=DEFAULT_SEQ_LEN):
    sized = []
    for shard in shards:
        tokens, roles = _doc_tokens(shard)
        tokens, roles = tokens[:seq_len], roles[:seq_len]
        sized.append({"shard_id": shard["shard_id"], "tokens": tokens, "roles": roles})
    sized.sort(key=lambda e: -len(e["tokens"]))

    bins = []  # list of [remaining_capacity, docs]
    for entry in sized:
        best_idx, best_remaining = None, None
        for i, (remaining, _) in enumerate(bins):
            if remaining >= len(entry["tokens"]) and (best_remaining is None or remaining < best_remaining):
                best_idx, best_remaining = i, remaining
        if best_idx is None:
            bins.append([seq_len - len(entry["tokens"]), [entry]])
        else:
            bins[best_idx][1].append(entry)
            bins[best_idx][0] -= len(entry["tokens"])
    return [_bin("best_fit", docs) for _, docs in bins]


POLICY_FUNCS = {
    "pad_only": pack_pad_only,
    "long_context": pack_long_context,
    "structure_preserving": pack_structure_preserving,
    "concat_and_chop": pack_concat_and_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
}


def pack_lane(lane, shards, seq_len=DEFAULT_SEQ_LEN, eos_id=0):
    policy = LANE_POLICY[lane]
    if policy == "long_context":
        return pack_long_context(shards)
    if policy == "concat_and_chop":
        return pack_concat_and_chop(shards, seq_len, eos_id)
    return POLICY_FUNCS[policy](shards, seq_len)


def bin_capacity(b, seq_len=DEFAULT_SEQ_LEN):
    return LONG_CONTEXT_SEQ_LEN if b["policy"] == "long_context" else seq_len


def packing_utilization(bins, seq_len=DEFAULT_SEQ_LEN):
    total_capacity = sum(bin_capacity(b, seq_len) for b in bins)
    total_used = sum(len(d["tokens"]) for b in bins for d in b["docs"])
    return round(100.0 * total_used / total_capacity, 2) if total_capacity else 0.0
