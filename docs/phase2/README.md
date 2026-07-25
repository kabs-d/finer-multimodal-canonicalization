# Phase II: Fine-Grained Retrieval Under Canonical Alignment

Phase II keeps the encoders frozen and removes the decoder. It asks whether
orthogonal alignment preserves fine-grained nearest-neighbor structure:

> Can an aligned source embedding retrieve target-space birds from the same
> species that share the query bird's visible attributes?

This is deliberately stricter than class retrieval. Candidate pools are
restricted to the same CUB species, and the query image itself is excluded.

## Completed retrieval probes

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

## Main interpretation

Same-species random retrieval is already strong because birds from the same
species share many canonical attributes. The important comparison is therefore
gain over random and over the unaligned source control. Aligned source queries
retrieve target-space neighbors with more matching fine-grained attributes,
especially at strict top-1.

For numbers, see
[the fine-grained retrieval report](../../reports/fine_grained_retrieval.md).

## Parked direction

Contrastive fine-tuning remains a possible later experiment, but it is not the
current Phase II. Fine-tuning would introduce new optimization and data-design
confounds; retrieval is the cleaner extension of the frozen canonicalization
result.
