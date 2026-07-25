# Finer Multimodal Canonicalization

This repository studies a sharper version of the question raised by
Gupta et al., *Canonicalizing Multimodal Contrastive Representation Learning*:

> Once two frozen multimodal encoders are orthogonally aligned, do their
> representations merely look similar, or can they share fine-grained
> downstream readouts?

The current result is a compact Phase I extension on CUB-200-2011 bird
attributes. A decoder trained only in one model's native space can be reused on
another model's embeddings after alignment, and the transferred signal remains
above chance even inside the same bird species.

## Read the study

- [Frozen encoder canonical readouts](docs/frozen_encoder/README.md)  
  The completed experiment: cosine alignment, paper-style classification and
  retrieval diagnostics, linear attribute readout transfer, an in-domain
  CUB-fitted alignment control, a fixed MLP decoder-capacity control, and
  same-species fine-grained retrieval.

- [Phase II: fine-grained retrieval](docs/phase2/README.md)  
  The decoder-free extension: aligned source queries retrieve native target
  birds from the same species and are scored by shared visible attributes.

## Headline

On CUB test images, an Oxford-Pets-fitted orthogonal map transfers measurable
within-species attribute structure:

- OpenAI ViT-B/32 → LAION ViT-B/32: within-species attribute ranking rises
  from 50.32% unaligned to 59.69% aligned.
- OpenAI ViT-L/14 → FLAVA: within-species attribute ranking rises from 49.74%
  unaligned to 57.44% aligned.

Fitting the same kind of map on official CUB train images raises the aligned
linear decoder to 60.78% and 60.23%, respectively. The gain shows that part of
the Phase I gap is cross-dataset alignment error, but even the in-domain map
does not fully match native target readouts.

## Repository map

```text
docs/frozen_encoder/              Phase I narrative and figures
docs/phase2/                      Phase II fine-grained retrieval summary
reports/phase1_results.md         Detailed Phase I tables and protocol
reports/cub_train_q_control.md    In-domain CUB-train-fitted Q control
reports/fine_grained_retrieval.md Decoder-free same-species retrieval results
reports/reproduction.md           Oxford baseline reproduction notes
artifacts/results/phase1_summary.json
                                  Compact machine-readable headline results
artifacts/results/frozen_decoder/ Decoder summaries and manifests
artifacts/alignments/             Saved orthogonal maps and diagnostics
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
python3 scripts/render_phase1_figures.py
python3 scripts/render_fine_grained_retrieval_figures.py
```

The tmux launchers run the two configured model pairs concurrently on the
freest GPUs:

```bash
./scripts/launch_frozen_decoder_tmux.sh
./scripts/launch_mlp_decoder_tmux.sh
./scripts/launch_cub_train_q_control_tmux.sh
```

See the Phase I README for the exact leakage boundary and interpretation.

## Reference

```bibtex
@article{gupta2026canonicalizing,
  title={Canonicalizing Multimodal Contrastive Representation Learning},
  author={Gupta, Sharut and Kansal, Sanyam and Jegelka, Stefanie and
          Isola, Phillip and Garg, Vikas},
  journal={arXiv preprint arXiv:2602.17584},
  year={2026}
}
```
