# Frozen CUB decoder-transfer protocol

## Question

Does the orthogonal map fitted from paired Oxford-IIIT Pet image embeddings
preserve enough fine-grained information that a decoder trained in the target
representation can be reused on aligned source representations?

This is deliberately stricter than showing high paired cosine or species
retrieval. The primary comparison uses one target-space decoder:

```text
native target:    h_B(f_B(x))
aligned source:   h_B(normalize((f_A(x) - mu_A,CUB) Q_Oxford + mu_B,CUB))
unaligned source: h_B(f_A(x))
```

The source-native control `h_A(f_A(x))` distinguishes an alignment failure from
an attribute that the source representation never made linearly decodable.

## Fixed quantities and leakage boundary

- `Q_Oxford` is fitted once using the 3,680 paired Oxford `trainval` images.
- `Q_Oxford` is never fitted, selected, or tuned on CUB.
- Both contrastive encoders remain in evaluation mode with zero trainable
  parameters.
- Only CUB official-training image means are recomputed for cross-dataset
  centering.
- Official CUB test images are never used for fitting means, decoder weights,
  early stopping, or hyperparameter selection.
- The raw-\(Q\) ablation is disabled for this phase.

The official CUB training split is divided 80/20 within every species using
split seed 2026. The internal validation partition is used only for decoder
early stopping. The official test split remains untouched.

## Attributes

CUB supplies 312 binary attribute labels per image and a certainty code per
image–attribute cell. Cells labeled `not visible` are masked. The remaining
certainty levels—`guessing`, `probably`, and `definitely`—are retained.
Masking a cell means that it contributes to neither loss nor metrics; it does
not remove the image.

The declared evaluation set includes attributes with at least 20 positive and
20 negative observations in the internal training partition and at least 5 of
each in official test. These support thresholds are fixed in the JSON configs.

## Decoder and metrics

Each decoder is a single affine layer with 312 independent logits. Training uses
masked class-balanced binary cross entropy, AdamW, learning rate `1e-3`, weight
decay `1e-4`, batch size 256, at most 200 epochs, and patience 20. Positive
weights are clipped to `[0.25, 20]`. Results use seeds 42–46.

Primary metric: macro mean average precision across eligible attributes.
Secondary metrics: micro average precision, macro AUROC, per-attribute AP,
aligned-minus-native transfer gap, and aligned/native retention. A paired
species-cluster bootstrap with 1,000 replicates gives a 95% interval for the
macro-mAP transfer gap.

Before decoder training, the run also reports the paper-style CUB alignment
block: image↔image and text↔text retrieval, zero-shot image↔text classification,
and paired cosine before/after alignment.

## Artifact contract

Compact, exportable artifacts stay under:

```text
artifacts/alignments/                 serialized Oxford Q and diagnostics
artifacts/results/frozen_decoder/     summaries, manifests, checkpoints, CSVs
artifacts/logs/frozen_decoder/        live job logs
```

Large/reconstructable CUB images, frozen embeddings, and per-image predictions
are external or gitignored. Every result manifest states that encoders were
frozen, CUB rotation refitting was false, CUB recentering was true, and the raw
rotation ablation was false.
