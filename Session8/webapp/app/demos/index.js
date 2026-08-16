import { mountAttention } from "./attention.js";
import { mountLinear } from "./linear.js";
import { mountRope } from "./rope.js";
import { mountCache } from "./cache.js";
import { mountTopk } from "./topk.js";
import { mountCompress } from "./compress.js";

export const DEMOS = {
  attention: mountAttention,
  linear: mountLinear,
  rope: mountRope,
  cache: mountCache,
  topk: mountTopk,
  compress: mountCompress,
};
