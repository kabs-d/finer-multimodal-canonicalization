# Phase II: Fine-Grained Retrieval Under Canonical Alignment

Phase II keeps the encoders frozen and removes the decoder. It asks whether
orthogonal alignment preserves fine-grained nearest-neighbor structure:

> Can an aligned source embedding retrieve target-space birds from the same
> species that share the query bird's visible attributes?

This is deliberately stricter than class retrieval. Candidate pools are
restricted to the same CUB species, and the query image itself is excluded.

## What is measured

- Same-species attribute overlap@k for `k = 1, 5, 10`.
- Rare-attribute recall@k, where rare attributes are selected from official
  CUB train labels only as the bottom quartile by visible-positive prevalence.
- Cross-space compatibility:
  - native target query → native target candidates;
  - Oxford-Q aligned source query → native target candidates;
  - CUB-train-Q aligned source query → native target candidates;
  - unaligned source query → native target candidates;
  - random same-species baseline.

The experiment is cache-only: no encoder reruns, decoder training, or
fine-tuning.

## Same-species attribute overlap

For each query bird, retrieve top-k birds from the same species. The score is
the fraction of the query's visible-positive attributes also visible-positive
in the retrieved birds.

The figure reports gain over a random same-species baseline, because random is
already strong when candidates are from the same bird species.

![Same-species attribute retrieval gain over random](../frozen_encoder/figures/fine_retrieval_attribute_overlap.svg)

| Pair | Condition | Overlap@1 | Overlap@5 | Overlap@10 |
| --- | --- | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native target | **46.14%** | **45.55%** | **45.02%** |
| OpenAI B/32 → LAION B/32 | Oxford-Q aligned source | 45.19% | 44.90% | 44.69% |
| OpenAI B/32 → LAION B/32 | CUB-train-Q aligned source | 45.74% | 45.31% | 44.92% |
| OpenAI B/32 → LAION B/32 | Unaligned source | 42.38% | 43.35% | 43.53% |
| OpenAI B/32 → LAION B/32 | Random same-species | 43.69% | 43.46% | 43.46% |
| OpenAI L/14 → FLAVA | Native target | 45.90% | 45.22% | 44.80% |
| OpenAI L/14 → FLAVA | Oxford-Q aligned source | 45.49% | 44.78% | 44.44% |
| OpenAI L/14 → FLAVA | CUB-train-Q aligned source | **46.28%** | **45.37%** | **44.91%** |
| OpenAI L/14 → FLAVA | Unaligned source | 43.04% | 43.12% | 43.31% |
| OpenAI L/14 → FLAVA | Random same-species | 43.69% | 43.46% | 43.46% |

## Rare-attribute recall

Rare attributes are defined from official CUB train labels only. We select the
bottom quartile by visible-positive prevalence: 78 rare attributes out of 312
evaluable attributes, with maximum rare prevalence 1.78%.

For each query with at least one visible-positive rare attribute, recall@k is
the fraction of those rare positives recovered by at least one retrieved bird.

![Rare-attribute retrieval gain over random](../frozen_encoder/figures/fine_retrieval_rare_recall.svg)

| Pair | Condition | Rare@1 | Rare@5 | Rare@10 |
| --- | --- | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | Native target | 10.53% | 29.34% | **40.85%** |
| OpenAI B/32 → LAION B/32 | Oxford-Q aligned source | **10.71%** | **30.14%** | 40.01% |
| OpenAI B/32 → LAION B/32 | CUB-train-Q aligned source | 10.45% | 29.97% | 40.57% |
| OpenAI B/32 → LAION B/32 | Unaligned source | 8.30% | 27.15% | 39.02% |
| OpenAI B/32 → LAION B/32 | Random same-species | 8.62% | 27.55% | 38.79% |
| OpenAI L/14 → FLAVA | Native target | 9.23% | **29.69%** | **41.16%** |
| OpenAI L/14 → FLAVA | Oxford-Q aligned source | 8.76% | 28.03% | 40.56% |
| OpenAI L/14 → FLAVA | CUB-train-Q aligned source | **9.69%** | 29.52% | 41.10% |
| OpenAI L/14 → FLAVA | Unaligned source | 7.85% | 26.33% | 39.81% |
| OpenAI L/14 → FLAVA | Random same-species | 8.62% | 27.55% | 38.79% |

## Interpretation

Same-species random retrieval is already strong because birds from the same
species share many canonical attributes. The important comparison is therefore
gain over random and over the unaligned source control. Aligned source queries
retrieve target-space neighbors with more matching fine-grained attributes,
especially at strict top-1.

The strongest clean result is not “retrieval is solved.” It is narrower and
better:

> Orthogonal alignment preserves enough fine-grained neighborhood structure
> that source embeddings can retrieve target-space birds with more matching
> attributes than unaligned or random same-species controls.

For artifact provenance, see
[the fine-grained retrieval report](../../reports/fine_grained_retrieval.md).

## Parked direction

Contrastive fine-tuning remains a possible later experiment, but it is not the
current Phase II. Fine-tuning would introduce new optimization and data-design
confounds; retrieval is the cleaner extension of the frozen canonicalization
result.
