# Oxford-IIIT Pet baseline reproduction

**Status: complete.** The independent implementation matches the aggregate
outputs of the authors' code for both selected frozen-encoder pairs.

## Scope

1. OpenAI CLIP ViT-B/32 → LAION-400M CLIP ViT-B/32.
2. OpenAI CLIP ViT-L/14 → FLAVA.

The author repository was evaluated at commit
`3b446d853b6f8fb0f412b6d598d81f8014720e18`; the independent implementation
then ran against the same Oxford split and model weights.

## Protocol

- Encoder weights remain frozen.
- All image and text embeddings are L2-normalized.
- The centered orthogonal Procrustes map is fitted on all 3,680 paired
  Oxford `trainval` images.
- Evaluation uses 3,669 official test images and 37 class prompts.
- Image means center aligned images; instance-text means center aligned text.
  The image-fitted rotation is shared across both modalities.
- Centered and rotation-only variants are retained.

The released seed loop does not resample anchors or otherwise change the
computation, so seeds 42–44 are identical by construction.

## Centered-map results

| Metric | OpenAI B/32 → LAION B/32 | OpenAI L/14 → FLAVA |
|---|---:|---:|
| Image cosine, before | 0.0223 | −0.0208 |
| Image cosine, after | **0.8883** | **0.7942** |
| Text cosine, before | 0.2258 | 0.0054 |
| Text cosine, after | **0.7725** | **0.6112** |
| Image→image retrieval, before | 3.24% | 1.83% |
| Image→image retrieval, after | **99.43%** | **97.30%** |
| Text→text retrieval, before | 0.00% | 2.70% |
| Text→text retrieval, after | **100.00%** | **83.78%** |
| Native source zero-shot | 87.46% | 93.43% |
| Native target zero-shot | 85.53% | 68.85% |
| Aligned source image + target text | 69.04% | 74.49% |
| Target image + aligned source text | 81.79% | 70.26% |
| Aligned source image + aligned source text | 80.78% | 90.19% |

These values match the corresponding author-code logs at the printed
precision. The exact JSON outputs are stored under
[`artifacts/results/standalone/`](../artifacts/results/standalone/).
