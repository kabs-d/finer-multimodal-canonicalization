# Finer Multimodal Canonicalization

Can an alignment between two multimodal models preserve the details that make
images visually different—not just their broad semantic meaning?

This repository studies that question using **CUB-200-2011**, a fine-grained
bird dataset with 200 species and 312 per-image attributes such as bill shape,
wing color, head color, and tail pattern. Each attribute can also be marked
“not visible,” which lets us evaluate only what the image actually shows.

We study two simple tests:

- **Attribute decoder transfer:** train a decoder in one model’s space to
  predict a bird’s visible attributes, then reuse it on another model after
  alignment.
- **Attribute-guided retrieval:** use a text prompt describing an attribute,
  such as “a bird with a yellow wing,” and retrieve matching CUB images.

The main alignment is learned from Oxford-IIIT Pets and transferred to CUB.
A separate CUB-trained alignment provides an in-domain comparison.

## What the results show

![Fine-grained attribute transfer](docs/frozen_encoder/figures/within_species_ranking.svg)

Aligned embeddings retain clear attribute-level information, including when
the map was learned on Oxford-Pets. Performance moves from approximately
chance-level ranking before alignment to around 60% after alignment.

![Attribute-guided retrieval](docs/phase2/figures/global_attribute_p10.svg)

For attribute-only text queries, unaligned cross-model retrieval is close to
the random attribute-frequency baseline. Alignment recovers much of the
native retrieval signal, while the CUB-trained map is consistently stronger
than the Oxford-Pets map.

These figures summarize the central result: canonical alignment transfers
useful fine-grained behavior, but not perfectly.

## Explore the experiments

- [Frozen encoder and decoder-transfer study](docs/frozen_encoder/README.md)
- [Attribute-guided retrieval study](docs/phase2/README.md)
- [Detailed results and protocols](reports/)

## Repository map

```text
docs/frozen_encoder/              Frozen encoder and decoder-transfer study
docs/phase2/                      Attribute-guided retrieval study
reports/                          Detailed results and protocols
artifacts/                        Compact machine-readable results
configs/                          Locked experiment configurations
src/canonical_study/              Alignment, evaluation, and decoder code
scripts/                          Reproducible launchers and figure rendering
tests/                            Alignment, metric, decoder, and regression tests
```

Downloaded datasets, model weights, large embedding caches, and per-image
prediction dumps are intentionally kept out of the exportable repository.

## Reproduce

```bash
./scripts/bootstrap.sh
.venv/bin/python -m unittest discover -s tests -v
./scripts/run_attribute_text_global_retrieval.sh
python3 scripts/render_phase1_figures.py
python3 scripts/render_attribute_text_retrieval_figures.py
```
