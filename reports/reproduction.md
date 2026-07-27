# Oxford-IIIT Pets baseline reproduction

This is the implementation check that precedes both CUB studies. We reproduce
the frozen-encoder Oxford-IIIT Pets experiment from Gupta et al.,
*Canonicalizing Multimodal Contrastive Representation Learning*, for the two
model pairs used throughout this repository.

## Setup

- OpenAI CLIP ViT-B/32 → LAION-400M CLIP ViT-B/32.
- OpenAI CLIP ViT-L/14 → FLAVA.
- Encoders remain frozen.
- A centered orthogonal Procrustes map is fitted on all 3,680 paired Oxford
  `trainval` images.
- Evaluation uses 3,669 official Oxford test images and 37 class prompts.
- The image-fitted rotation is shared across image and text embeddings; image
  and text centering are computed separately.

The reference author repository was evaluated at commit
`3b446d853b6f8fb0f412b6d598d81f8014720e18` with the same split and released
model weights.

## Matched centered-map results

| Metric | OpenAI B/32 → LAION B/32 | OpenAI L/14 → FLAVA |
| --- | ---: | ---: |
| Paired image cosine, before → after | 0.0223 → **0.8883** | −0.0208 → **0.7942** |
| Paired text cosine, before → after | 0.2258 → **0.7725** | 0.0054 → **0.6112** |
| Image→image class retrieval, before → after | 3.24% → **99.43%** | 1.83% → **97.30%** |
| Text→text class retrieval, before → after | 0.00% → **100.00%** | 2.70% → **83.78%** |
| Native source zero-shot | 87.46% | 93.43% |
| Native target zero-shot | 85.53% | 68.85% |
| Aligned source image → native target text | 69.04% | 74.49% |

The aggregate values match the author-code logs at printed precision. This
establishes the frozen alignment pipeline before transferring the map to CUB.

## Artifacts

Exact outputs are stored under
[`artifacts/results/standalone/`](../artifacts/results/standalone/).
