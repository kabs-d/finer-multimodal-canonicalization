# Attribute-text global retrieval

This report records the Phase II attribute-text retrieval probe. The experiment
uses attribute-only prompts and retrieves from all CUB test images where the
queried attribute is visible.

```text
a photo of a bird with {attribute phrase}.
```

No decoders are trained. Image embeddings are loaded from existing frozen CUB
caches. Text embeddings are computed for the 312 CUB attribute prompts.

## Attribute sets

- **CLIP-readable subset:** 95 metadata-filtered attributes with ordinary visual
  phrases. This excludes bird-annotation jargon such as `primary color`,
  `upper tail`, `under tail`, `nape`, tiny eye/leg colors, and obscure colors.
- **All 312 attributes:** the full CUB attribute vocabulary.

The readable subset is fixed in code as `CLIP_READABLE_ATTRIBUTE_INDICES`.

## Metrics

- **P@k:** fraction of the top-k retrieved visible candidates that are
  visible-positive for the queried attribute.
- **Ranking accuracy:** probability that the prompt scores a visible-positive
  image above a visible-negative image. Ties score 0.5.
- **Random P@k:** the visible-positive base rate for that attribute, averaged
  over attributes. Random ranking accuracy is 50%.

## CLIP-readable subset

Random/base-rate P@k is 19.19%. Random ranking accuracy is 50%.

| Pair | Condition | P@1 | P@5 | P@10 | Ranking |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native source | 37.89 | 43.37 | 41.58 | 62.82 |
| OpenAI B/32 → LAION B/32 | Native target | 47.37 | 41.68 | 38.32 | 63.71 |
| OpenAI B/32 → LAION B/32 | Unaligned source→target | 15.79 | 21.26 | 18.63 | 50.73 |
| OpenAI B/32 → LAION B/32 | Oxford-Q source→target | 35.79 | 34.53 | 34.95 | 61.79 |
| OpenAI B/32 → LAION B/32 | CUB-train-Q source→target | 41.05 | 40.00 | 38.63 | 62.91 |
| OpenAI L/14 → FLAVA | Native source | 36.84 | 37.05 | 37.89 | 62.63 |
| OpenAI L/14 → FLAVA | Native target | 30.53 | 34.74 | 33.58 | 60.06 |
| OpenAI L/14 → FLAVA | Unaligned source→target | 23.16 | 19.16 | 18.32 | 50.78 |
| OpenAI L/14 → FLAVA | Oxford-Q source→target | 27.37 | 29.89 | 30.21 | 57.22 |
| OpenAI L/14 → FLAVA | CUB-train-Q source→target | 38.95 | 33.89 | 33.47 | 58.72 |

## All 312 attributes

Random/base-rate P@k is 11.40%. Random ranking accuracy is 50%.

| Pair | Condition | P@1 | P@5 | P@10 | Ranking |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native source | 33.33 | 33.97 | 32.31 | 64.84 |
| OpenAI B/32 → LAION B/32 | Native target | 34.94 | 32.50 | 31.51 | 66.57 |
| OpenAI B/32 → LAION B/32 | Unaligned source→target | 8.33 | 11.41 | 10.80 | 49.32 |
| OpenAI B/32 → LAION B/32 | Oxford-Q source→target | 29.81 | 28.01 | 27.66 | 64.23 |
| OpenAI B/32 → LAION B/32 | CUB-train-Q source→target | 33.65 | 31.54 | 30.13 | 65.13 |
| OpenAI L/14 → FLAVA | Native source | 28.21 | 29.23 | 29.39 | 64.56 |
| OpenAI L/14 → FLAVA | Native target | 23.08 | 26.03 | 26.35 | 62.20 |
| OpenAI L/14 → FLAVA | Unaligned source→target | 11.86 | 11.79 | 11.31 | 50.86 |
| OpenAI L/14 → FLAVA | Oxford-Q source→target | 21.79 | 23.08 | 22.76 | 59.37 |
| OpenAI L/14 → FLAVA | CUB-train-Q source→target | 28.85 | 25.45 | 24.71 | 60.84 |

## Takeaway

Attribute-only text retrieval is well above random for native models. The
unaligned cross-space control collapses to random. Orthogonal alignment recovers
most of the retrieval signal, and CUB-train-Q is consistently stronger than
Oxford-Q.

Artifacts:

- `artifacts/results/attribute_text_retrieval/cub_openai_vitb32_to_laion_vitb32_linear_attribute_text_global_retrieval/`
- `artifacts/results/attribute_text_retrieval/cub_openai_vitl14_to_flava_linear_attribute_text_global_retrieval/`
- `docs/phase2/figures/global_attribute_p10.svg`
- `docs/phase2/figures/global_attribute_ranking.svg`
