// One entry per concept, added as each is researched and built. A concept with no entry here
// renders as a placeholder, which is how the deck shows work in progress honestly.
import { transformerCard } from "./transformer.js";
import { sinusoidalCard } from "./sinusoidal.js";
import { relativePositionsCard } from "./relative-positions.js";
import { learnedAbsoluteCard } from "./learned-absolute.js";
import { transformerXLCard } from "./transformer-xl.js";
import { sparseTransformerCard } from "./sparse-transformer.js";

export const CARDS = {
  transformer: transformerCard,
  sinusoidal: sinusoidalCard,
  "relative-positions": relativePositionsCard,
  "learned-absolute": learnedAbsoluteCard,
  "transformer-xl": transformerXLCard,
  "sparse-transformer": sparseTransformerCard,
};
