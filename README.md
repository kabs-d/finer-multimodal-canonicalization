# Do canonical representations preserve fine-grained readouts?

This repository reproduces the frozen-encoder alignment experiments from
Gupta et al., *Canonicalizing Multimodal Contrastive Representation Learning*,
and tests a direct extension suggested by the paper:

> Can a decoder trained in one model's representation space be reused on a
> second model after orthogonal alignment?

The answer is **yes for measurable, image-specific bird attributes, but the
strongest transferable signal is already linearly accessible**.

## Phase I in one table

An orthogonal map \(Q_{\text{Oxford}}\) is fitted from 3,680 paired
Oxford-IIIT Pet training images. The rotation is then frozen and transferred
to CUB-200-2011; only model-specific CUB training means are recomputed, exactly
following the paper's cross-dataset centering protocol.

| Frozen model pair | CUB image retrieval, unaligned → aligned | Joint zero-shot species classification, native source → aligned | Within-species attribute ranking, unaligned → aligned → native target |
|---|---:|---:|---:|
| OpenAI ViT-B/32 → LAION ViT-B/32 | 0.47% → **90.42%** | 53.04% → **44.68%** | 50.32% → **59.69%** → 62.02% |
| OpenAI ViT-L/14 → FLAVA | 0.26% → **40.01%** | 62.19% → **54.00%** | 49.74% → **57.44%** → 60.86% |

The attribute test compares positive and negative birds **within the same
species**. Chance, an unaligned decoder, and a ground-truth-species lookup are
all approximately 50%. The aligned decoder remains 7–10 points above chance,
so the result cannot be explained by transferring species identity alone.

## What is new here?

The paper establishes strong pointwise alignment, class retrieval, and
zero-shot interoperability, and gives qualitative evidence for fine-grained
semantic preservation. It explicitly leaves lightweight decoder evaluation as
a next step.

This study turns that suggestion into a leakage-controlled quantitative test:

```text
Oxford trainval images ── fit Q ───────────────────────────────┐
                                                               │ frozen
CUB train images ─────── estimate means; train h_B on f_B(x)  │
                                                               ▼
CUB test images ──────── compare h_B(f_B(x)),
                                h_B(align_Q(f_A(x))),
                                h_B(f_A(x))
```

- Both contrastive encoders and \(Q_{\text{Oxford}}\) stay frozen.
- The 312-output decoder \(h_B\) is trained only on native target embeddings.
- Aligned-source embeddings are never used for training, early stopping,
  threshold selection, or architecture selection.
- The official 5,794-image CUB test split remains untouched.
- Results use 294 sufficiently supported attributes and five decoder seeds.

This asks for **functional compatibility**, not merely high cosine similarity:
does alignment make a decoder learned in model \(B\)'s coordinates work on
model \(A\)'s representations?

## Linear versus nonlinear reuse

A prespecified MLP
`Linear(d,512) → GELU → Dropout(0.1) → Linear(512,312)` is compared with the
linear decoder under the same training protocol.

| Decoder input | OpenAI → LAION | OpenAI → FLAVA |
|---|---:|---:|
| Linear, native target | 62.02% | 60.86% |
| MLP, native target | **62.30%** | **61.16%** |
| Linear, aligned source | **59.69%** | **57.44%** |
| MLP, aligned source | 59.49% | 57.29% |
| Linear, unaligned source | 50.32% | 49.74% |
| MLP, unaligned source | 50.21% | 49.53% |

The MLP has a small native advantage, showing that the comparison is not
vacuous. That advantage disappears after alignment. On thresholded per-bird
predictions, the aligned MLP recovers more true attributes but also
hallucinates more; it is a recall–hallucination tradeoff rather than a clean
improvement.

**Phase I conclusion.** An Oxford-fitted orthogonal orientation transfers
quantitatively detectable, within-species attribute structure to a new
dataset and task. A frozen target-space decoder can reuse that structure
without aligned examples. The additional nonlinear discrimination exposed by
this MLP, however, is not preserved better than the linear readout.

See the [complete Phase I results](reports/phase1_results.md) for the
classification table, per-bird attribute counts, metric definitions,
limitations, and artifact provenance.

## Reproduce

The implementation is independent and does not vendor the authors' source,
model weights, or datasets. The Oxford baseline matches the authors' released
code outputs for both configured pairs.

```bash
./scripts/bootstrap.sh
.venv/bin/python -m unittest discover -s tests -v
./scripts/launch_tmux.sh
```

For the CUB extension, first review and accept the dataset's image-use terms:

```bash
/tmp/canonical-study-venv/bin/python -m canonical_study prepare-cub \
  --data-root /tmp/canonical-study-cub \
  --accept-research-terms

CANONICAL_STUDY_PYTHON=/tmp/canonical-study-venv/bin/python \
CANONICAL_STUDY_CUB_DATA_ROOT=/tmp/canonical-study-cub \
CANONICAL_STUDY_MODEL_CACHE_ROOT=/tmp/canonical-study-model-cache \
CANONICAL_STUDY_CUB_EMBEDDING_ROOT=/tmp/canonical-study-cub-embeddings \
CANONICAL_STUDY_CUB_PREDICTION_ROOT=/tmp/canonical-study-cub-predictions \
  ./scripts/launch_frozen_decoder_tmux.sh
```

The MLP runner reuses the frozen CUB embeddings and fixed Oxford alignments:

```bash
CANONICAL_STUDY_PYTHON=/tmp/canonical-study-venv/bin/python \
CANONICAL_STUDY_CUB_EMBEDDING_ROOT=/tmp/canonical-study-cub-embeddings \
CANONICAL_STUDY_CUB_PREDICTION_ROOT=/tmp/canonical-study-cub-predictions \
  ./scripts/launch_mlp_decoder_tmux.sh
```

Both launchers select the two GPUs with the most free memory and run the two
model pairs concurrently.

## Repository map

```text
reports/phase1_results.md       main experiment, results, and interpretation
reports/reproduction.md         Oxford parity details
reports/frozen_decoder_protocol.md
                                locked CUB protocol and leakage boundary
artifacts/results/phase1_summary.json
                                compact machine-readable headline results
artifacts/results/frozen_decoder/
                                run manifests, summaries, and checkpoints
configs/                        locked baseline and decoder specifications
src/canonical_study/            alignment, evaluation, and decoder code
scripts/                        reproducible two-GPU tmux launchers
tests/                          alignment, metric, decoder, and regression tests
```

Downloaded data, model weights, embeddings, and per-image predictions are kept
outside the exportable repository.

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
