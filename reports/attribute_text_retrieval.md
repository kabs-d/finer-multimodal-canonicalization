# Phase II technical record: attribute-text image retrieval

This report supplies the full numbers behind
[Attribute-Text Image Retrieval under Canonical Alignment](../docs/phase2/README.md).
It asks whether an attribute-only prompt—for example, `a photo of a bird with
yellow wing`—retrieves CUB test images that visibly possess that attribute
after the prompt is mapped across model spaces.

## Evaluation

For each CUB attribute, candidates are all official CUB test images where the
attribute is visible. The query's positives and negatives are therefore the
visible-positive and visible-negative images for that attribute. No image
encoder is rerun and no decoder is trained.

Conditions are native source, native target, unaligned source text → target
images, Oxford-Pets-$Q$ aligned source text → target images, and CUB-train-$Q$
aligned source text → target images. Image embeddings come from the existing
frozen CUB caches; prompts use `a photo of a bird with {attribute phrase}.`

P@k is the fraction of the top-$k$ retrieved images that are positive. Ranking
accuracy is the fraction of visible positive–negative image pairs for which
the positive gets a higher prompt score; ties receive half credit. Scores are
macro-averaged across attributes, so random ranking is 50%.

## CLIP-readable attribute subset

The pre-specified 95-attribute subset uses ordinary visual phrases such as
`white belly`, `black bill`, `striped wing`, and `long wings`. Random P@k is
the average visible-positive base rate: 19.19%.

| Pair | Condition | P@1 | P@5 | P@10 | Ranking |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native source | 37.89 | 43.37 | 41.58 | 62.82 |
|  | Native target | 47.37 | 41.68 | 38.32 | 63.71 |
|  | Unaligned source→target | 15.79 | 21.26 | 18.63 | 50.73 |
|  | Oxford-Pets $Q$ | 35.79 | 34.53 | 34.95 | 61.79 |
|  | CUB-train $Q$ | 41.05 | 40.00 | 38.63 | 62.91 |
| OpenAI L/14 → FLAVA | Native source | 36.84 | 37.05 | 37.89 | 62.63 |
|  | Native target | 30.53 | 34.74 | 33.58 | 60.06 |
|  | Unaligned source→target | 23.16 | 19.16 | 18.32 | 50.78 |
|  | Oxford-Pets $Q$ | 27.37 | 29.89 | 30.21 | 57.22 |
|  | CUB-train $Q$ | 38.95 | 33.89 | 33.47 | 58.72 |

## All 312 CUB attributes

This set retains the raw CUB vocabulary, including technical annotation terms
and sparse attributes. Its random P@k base rate is 11.40%.

| Pair | Condition | P@1 | P@5 | P@10 | Ranking |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native source | 33.33 | 33.97 | 32.31 | 64.84 |
|  | Native target | 34.94 | 32.50 | 31.51 | 66.57 |
|  | Unaligned source→target | 8.33 | 11.41 | 10.80 | 49.32 |
|  | Oxford-Pets $Q$ | 29.81 | 28.01 | 27.66 | 64.23 |
|  | CUB-train $Q$ | 33.65 | 31.54 | 30.13 | 65.13 |
| OpenAI L/14 → FLAVA | Native source | 28.21 | 29.23 | 29.39 | 64.56 |
|  | Native target | 23.08 | 26.03 | 26.35 | 62.20 |
|  | Unaligned source→target | 11.86 | 11.79 | 11.31 | 50.86 |
|  | Oxford-Pets $Q$ | 21.79 | 23.08 | 22.76 | 59.37 |
|  | CUB-train $Q$ | 28.85 | 25.45 | 24.71 | 60.84 |

## Interpretation and scope

For both pairs, unaligned source text retrieves at the random base rate and
ranks positives at chance. Both orthogonal maps restore substantial
attribute-level text-to-image behavior. CUB-train $Q$ is generally strongest;
the Oxford-Pets map remains clearly above unaligned despite never seeing CUB
images when fitted.

The readable subset is the most natural public demonstration. The full 312
set is retained as a tougher annotation-level stress test, not as evidence
that every raw CUB label is an equally natural text prompt.

Machine-readable outputs are under
`artifacts/results/attribute_text_retrieval/`.
