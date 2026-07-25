# Phase II: fine-grained retrieval under canonical alignment

This cache-only experiment asks whether aligned source embeddings can retrieve
target-space birds with matching fine-grained attributes inside the same
species. No encoder is rerun, no decoder is trained, and no model is
fine-tuned.

## Protocol

For every official CUB test image, candidates are restricted to other test
images from the same species. The query image itself is excluded. Retrieval is
by cosine similarity after L2 normalization.

Conditions:

- native target query → native target candidates;
- Oxford-Q aligned source query → native target candidates;
- CUB-train-Q aligned source query → native target candidates;
- unaligned source query → native target candidates;
- random same-species baseline.

Rare attributes are selected using official CUB train labels only: bottom
quartile by visible-positive prevalence among evaluable attributes. This gives
78 rare attributes out of 312 evaluable attributes, with maximum rare
prevalence 1.78%.

## Metrics

**Same-species attribute overlap@k**: for each query and top-k set, average
over retrieved candidates the fraction of the query's visible-positive
attributes that are also visible-positive in the candidate.

**Rare-attribute recall@k**: for each query with at least one visible-positive
rare attribute, report the fraction of its rare positives recovered by at
least one top-k candidate.

## OpenAI ViT-B/32 → LAION ViT-B/32

| Condition | Overlap@1 | Overlap@5 | Overlap@10 | Rare@1 | Rare@5 | Rare@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native target | **46.14%** | **45.55%** | **45.02%** | 10.53% | 29.34% | **40.85%** |
| Oxford-Q aligned source | 45.19% | 44.90% | 44.69% | **10.71%** | **30.14%** | 40.01% |
| CUB-train-Q aligned source | 45.74% | 45.31% | 44.92% | 10.45% | 29.97% | 40.57% |
| Unaligned source | 42.38% | 43.35% | 43.53% | 8.30% | 27.15% | 39.02% |
| Random same-species | 43.69% | 43.46% | 43.46% | 8.62% | 27.55% | 38.79% |

## OpenAI ViT-L/14 → FLAVA

| Condition | Overlap@1 | Overlap@5 | Overlap@10 | Rare@1 | Rare@5 | Rare@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native target | 45.90% | 45.22% | 44.80% | 9.23% | **29.69%** | **41.16%** |
| Oxford-Q aligned source | 45.49% | 44.78% | 44.44% | 8.76% | 28.03% | 40.56% |
| CUB-train-Q aligned source | **46.28%** | **45.37%** | **44.91%** | **9.69%** | 29.52% | 41.10% |
| Unaligned source | 43.04% | 43.12% | 43.31% | 7.85% | 26.33% | 39.81% |
| Random same-species | 43.69% | 43.46% | 43.46% | 8.62% | 27.55% | 38.79% |

## Takeaway

Same-species retrieval is intentionally difficult: random candidates already
share many species-typical attributes. The useful signal is the gain over this
baseline. Aligned source queries consistently recover target-space neighbors
with more matching attributes than the unaligned source condition, especially
for strict top-1 retrieval.

The CUB-train-Q control is strongest for overlap@1 and often approaches or
slightly exceeds native target retrieval. Oxford-Q remains competitive despite
being fitted on Oxford Pets rather than birds, which supports the Phase I
claim that the canonical map transfers fine-grained neighborhood structure,
not just decoder compatibility.

Figures are rendered in
`docs/frozen_encoder/figures/fine_retrieval_attribute_overlap.svg` and
`docs/frozen_encoder/figures/fine_retrieval_rare_recall.svg`.
