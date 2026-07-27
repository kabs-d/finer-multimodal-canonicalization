# Phase I technical record: frozen-encoder transfer

This report is the numerical record behind the
[Frozen-Encoder Transfer under Canonical Alignment](../docs/frozen_encoder/README.md)
study. The central question is whether an orthogonal map learned from paired
Oxford-IIIT Pets images remains useful for CUB-200-2011, where evaluation
depends on fine-grained bird attributes rather than broad semantic classes.

## Protocol and leakage boundary

We use two frozen pairs:

1. OpenAI CLIP ViT-B/32 → LAION CLIP ViT-B/32.
2. OpenAI CLIP ViT-L/14 → FLAVA.

$Q_{\text{Oxford}}$ is fitted once by centered Procrustes on 3,680 paired
Oxford `trainval` image embeddings. It is never refitted, selected, or tuned
using CUB. For cross-dataset evaluation, the affine means are recomputed from
the 5,994 official CUB training images, following the paper's protocol.

$Q_{\text{CUB-train}}$ is an in-domain control fitted only on those same 5,994
CUB training pairs; all results are evaluated on the disjoint 5,794-image CUB
test split. Both encoders remain frozen. Decoder training, early stopping, and
attribute thresholds use only the CUB training/validation partition.

## Paired geometry and exact image retrieval

Paired cosine compares the two models' embeddings of the same CUB image.
Exact image retrieval asks whether a source query retrieves its *same-photo*
target embedding among all 5,794 CUB test candidates.

| Pair | Image cosine: before / Oxford / CUB-train | Exact paired image top-5: unaligned / Oxford / CUB-train |
| --- | ---: | ---: |
| OpenAI B/32 → LAION B/32 | 0.030 / 0.803 / 0.905 | 0.0% / 95.8% / 99.8% |
| OpenAI L/14 → FLAVA | 0.011 / 0.681 / 0.835 | 0.1% / 52.4% / 95.9% |

Thus the Oxford-Pets map transfers substantial paired-image geometry to CUB
without ever seeing CUB pairs. The CUB-trained control improves it further,
especially for the more heterogeneous OpenAI→FLAVA pair.

## Strict cross-model species classification

For this test, a source image is mapped into the target coordinate system and
classified against the **native target model's** 200 CUB class-name prompts.
The native target bar is only a reference; source text is not transformed in
this strict condition.

| Pair | Native target | Oxford-Pets $Q$ | CUB-train $Q$ |
| --- | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | 56.4% | 18.5% | 47.7% |
| OpenAI L/14 → FLAVA | 40.5% | 8.6% | 41.1% |

The CUB-trained control is close to native-target classification. Oxford-Pets
$Q$ remains clearly weaker on this strict image-to-native-text test, which is
why strong paired image geometry should not be mistaken for complete
cross-modal compatibility.

## Fine-grained attribute transfer

Each decoder predicts CUB's 312 per-image visual attributes. Labels marked
not visible are masked; 294 attributes remain eligible. A result reports the
percentage of a bird's visible-positive attributes recovered by a frozen
decoder. The table uses the two-layer target-space MLP shown in the main README.

| Pair | Native target | Unaligned source | Oxford-Pets $Q$ | CUB-train $Q$ |
| --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION B/32 | 70.80% | 43.10% | 66.02% | 70.21% |
| OpenAI L/14 → FLAVA | 70.71% | 34.29% | 62.21% | 70.68% |

The source embeddings cannot simply be passed to the target decoder: the
unaligned control drops sharply. Oxford-Pets $Q$ restores most of the lost
attribute recovery, while the in-domain map nearly reaches native target
performance. Full bidirectional decoder and linear/MLP comparisons are in the
[bidirectional decoder-transfer appendix](bidirectional_decoder_transfer.md).

## Scope

This phase establishes transfer for two frozen encoder pairs, one fine-grained
dataset, and fixed decoder architectures. It does not claim that every
nonlinear decoder or fine-tuned encoder will retain the same compatibility.

Machine-readable run summaries and alignment artifacts are committed under
`artifacts/results/frozen_decoder/`, `artifacts/results/deep_mlp_probe/`, and
`artifacts/alignments/`.
