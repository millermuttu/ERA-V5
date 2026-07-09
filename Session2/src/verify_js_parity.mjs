// Loads the generated page's injected data (base_chars, merges, fixtures) and
// verifies that an INDEPENDENT reference reimplementation of the encoder
// (built here, in Node) reproduces the Python ids for every fixture — a
// cross-check of the fixtures and data injection, not of the page's live
// encoder. The page's own live encoder is separately covered by the in-page
// self-test badge on every load. Exit 0 = parity holds, 1 = mismatch.
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

function blob(id) {
  const re = new RegExp(
    `<script id="${id}" type="application/json">([\\s\\S]*?)</script>`);
  const m = html.match(re);
  if (!m) throw new Error(`missing blob ${id}`);
  return JSON.parse(m[1]);
}

const BASE_CHARS = blob('base-chars-data');
const MERGES = blob('merges-data');
const FIXTURES = blob('fixtures-data');

// Reconstruct the encoder independently from the injected raw data, mirroring
// bpe_tokenizer.py — this is the reference the page's own encoder must match.
const UNK = 0, UNK_STR = '�';
const charToId = new Map(BASE_CHARS.map((c, i) => [c, i + 1]));
const idToStr = [UNK_STR, ...BASE_CHARS];
const rank = new Map();
for (const [a, b] of MERGES) {
  rank.set(a + ',' + b, idToStr.length);
  idToStr.push(idToStr[a] + idToStr[b]);
}
function pretok(text) {
  const units = text.match(/\s*\S+/gu) || [];
  const consumed = units.reduce((n, u) => n + u.length, 0);
  if (consumed < text.length) units.push(text.slice(consumed));
  return units;
}
function encodeUnit(u) {
  let seq = [...u].map((ch) => charToId.get(ch) ?? UNK);
  while (seq.length > 1) {
    let best = Infinity, at = -1;
    for (let i = 0; i < seq.length - 1; i++) {
      const r = rank.get(seq[i] + ',' + seq[i + 1]);
      if (r !== undefined && r < best) { best = r; at = i; }
    }
    if (at < 0) break;
    seq.splice(at, 2, best);
  }
  return seq;
}
function encode(text) { return pretok(text).flatMap(encodeUnit); }

let fails = 0;
for (const { text, ids } of FIXTURES) {
  const got = encode(text);
  if (JSON.stringify(got) !== JSON.stringify(ids)) {
    fails++;
    console.error(`MISMATCH ${JSON.stringify(text)}\n  py ${ids}\n  js ${got}`);
  }
}
if (fails) { console.error(`${fails} fixture(s) failed`); process.exit(1); }
console.log(`JS parity OK: ${FIXTURES.length}/${FIXTURES.length} fixtures`);
