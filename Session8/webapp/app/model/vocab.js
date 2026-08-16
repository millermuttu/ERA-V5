// A small fixed vocabulary. Small enough that the next-token bars name words the reader can read,
// large enough that a sentence about the thing the app is explaining tokenizes without holes.

export const VOCAB = [
  "<unk>", "<end>",
  // function words
  "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "it", "its",
  "is", "are", "was", "were", "be", "been", "of", "in", "on", "at", "to", "from", "for",
  "with", "without", "into", "under", "over", "before", "after", "while", "when", "where",
  "not", "no", "every", "each", "all", "some", "one", "two", "three", "more", "less", "most",
  "she", "he", "they", "we", "you", "i", "her", "his", "their", "our", "your", "my",
  // the sentence material
  "lighthouse", "keeper", "wrote", "code", "notebook", "hid", "third", "stair", "storm",
  "reached", "island", "dawn", "letter", "sea", "night", "morning", "window", "door", "key",
  "book", "page", "word", "line", "map", "path", "river", "bridge", "town", "clock", "hour",
  "cat", "sat", "mat", "dog", "ran", "fast", "slow", "walked", "looked", "found", "lost",
  "remembered", "forgot", "asked", "answered", "read", "wrote", "kept", "left", "gave",
  // the subject matter
  "model", "token", "tokens", "attention", "query", "key", "value", "score", "scores",
  "softmax", "mask", "position", "context", "memory", "cache", "state", "window", "sparse",
  "linear", "layer", "head", "heads", "block", "sequence", "length", "long", "short",
  "cheap", "expensive", "fast", "slower", "reads", "stores", "forgets", "keeps", "learns",
];

const INDEX = new Map(VOCAB.map((w, i) => [w, i]));
export const UNK = 0;
export const V = VOCAB.length;

/** Lowercase word split; anything outside the vocabulary becomes <unk> but keeps its surface form. */
export function tokenize(text, limit = 26) {
  const words = String(text)
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, limit);
  return words.map((w) => ({ word: w, id: INDEX.has(w) ? INDEX.get(w) : UNK, known: INDEX.has(w) }));
}

export const word = (id) => VOCAB[id] ?? "<unk>";

export const PRESETS = [
  "The lighthouse keeper wrote the code in a notebook and hid it under the third stair",
  "The cat sat on the mat and the dog ran past the door",
  "A long context is cheap to store only if the model forgets most of it",
  "She read every page and remembered the one line that answered the question",
];
