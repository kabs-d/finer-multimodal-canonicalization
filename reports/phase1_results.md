# Phase I: fine-grained decoder reuse after canonical alignment

## Research question

Gupta et al. show that a centered orthogonal map can make independently
trained multimodal contrastive encoders interoperable. Their evaluation
establishes pointwise correspondence, class-level retrieval, zero-shot
classification, and qualitative preservation of fine-grained semantics. The
paper identifies lightweight decoding after alignment as a natural next step.

We test the stronger operational claim:

> If a decoder is trained only in model \(B\)'s native coordinates, can it be
> reused on model \(A\)'s embeddings after applying an orthogonal map learned
> on a different dataset?

This separates geometric similarity from functional interchangeability.

## Experimental design

We study two frozen pairs:

1. OpenAI CLIP ViT-B/32 → LAION-400M CLIP ViT-B/32.
2. OpenAI CLIP ViT-L/14 → FLAVA.

For each pair, \(Q_{\text{Oxford}}\) is the centered orthogonal Procrustes
rotation fitted from all 3,680 paired Oxford-IIIT Pet `trainval` images. We
then transfer that rotation to CUB-200-2011 without refitting it. Following
the paper's cross-dataset protocol, source and target means are recomputed
using only the official CUB training split:

\[
\operatorname{align}(z)
=
\operatorname{normalize}\!\left(
  (z-\mu_{A,\mathrm{CUB}})Q_{\mathrm{Oxford}}
  + \mu_{B,\mathrm{CUB}}
\right).
\]

The official CUB split supplies 5,994 training and 5,794 test images across
200 bird species. The training split is divided 80/20 within species using
split seed 2026. Both encoders remain frozen throughout.

### Leakage boundary

- CUB never changes \(Q_{\text{Oxford}}\).
- Only official CUB training embeddings determine the new means.
- The target decoder is trained only on native target embeddings.
- Aligned and unaligned source embeddings are test conditions, never training
  or model-selection inputs.
- Per-attribute thresholds are selected from native-target validation
  predictions and reused unchanged in every test condition.
- Official CUB test images are excluded from fitting means, decoder weights,
  early stopping, and threshold selection.

## 1. Oxford baseline reproduction

Our independent implementation exactly matches the aggregate outputs produced
by the authors' code for both selected model pairs.

| Oxford metric | OpenAI B/32 → LAION B/32 | OpenAI L/14 → FLAVA |
|---|---:|---:|
| Paired image cosine, before → after | 0.022 → **0.888** | −0.021 → **0.794** |
| Image→image class retrieval, before → after | 3.24% → **99.43%** | 1.83% → **97.30%** |
| Text→text class retrieval, before → after | 0.00% → **100.00%** | 2.70% → **83.78%** |
| Aligned source image + target text | 69.04% | 74.49% |
| Aligned source image + aligned source text | 80.78% | 90.19% |

This establishes that the local implementation recovers the paper baseline
before introducing CUB or learned decoders.

## 2. Does \(Q_{\text{Oxford}}\) transfer to CUB classification?

Yes, although the strength depends on the task and model pair.

### Cross-model geometry on CUB

| Metric, before → after alignment | OpenAI B/32 → LAION B/32 | OpenAI L/14 → FLAVA |
|---|---:|---:|
| Paired image cosine | 0.030 → **0.803** | 0.011 → **0.681** |
| Image→image class retrieval | 0.47% → **90.42%** | 0.26% → **40.01%** |
| Text→text class retrieval | 0.50% → **62.00%** | 0.00% → **19.00%** |
| Paired class-text cosine | 0.142 → **0.702** | −0.015 → **0.498** |

The rotation is learned only on pet images; these gains are measured on a new
200-class bird distribution.

### Paper-style zero-shot CUB species accuracy

No classifier or attribute decoder is trained for this table. Images are
classified by cosine similarity to class-name text prompts, following the
paper's evaluation.

| Image and text coordinates | OpenAI B/32 → LAION B/32 | OpenAI L/14 → FLAVA |
|---|---:|---:|
| Native source image + native source text | 53.04% | 62.19% |
| Native target image + native target text | 56.39% | 40.52% |
| Aligned source image + target text | 18.50% | 8.60% |
| Target image + aligned source text | 31.81% | 13.62% |
| Aligned source image + aligned source text | **44.68%** | **54.00%** |

Jointly mapping source image and text embeddings retains much more
classification performance than mixing a mapped modality with the target's
native opposite modality. Thus \(Q_{\text{Oxford}}\) transfers strongly at
the intra-modal level, while cross-modal coordinate compatibility is only
partial on CUB.

## 3. Fine-grained decoder transfer

CUB provides 312 binary per-image attributes. Cells marked `not visible` are
excluded from both training and evaluation. After fixed support filtering,
294 attributes remain. The average test bird has 31.47 visible positive
attributes.

The linear decoder is one affine \(d\!\rightarrow\!312\) layer trained with
masked, class-balanced binary cross entropy. It has one output logit per
attribute.

The target-space decoder is evaluated on:

\[
h_B(f_B(x)), \qquad
h_B(\operatorname{align}(f_A(x))), \qquad
h_B(f_A(x)).
\]

The last condition is an intentionally unaligned control. A ground-truth
species lookup is also evaluated to expose how much apparent attribute
performance can arise from species identity.

### Interpretable per-bird counts

`Recovered` is a visible true attribute predicted present. `Missed` is a
visible true attribute predicted absent. `Hallucinated` is a visible absent
attribute predicted present. True negatives are intentionally omitted.
Values are means over test birds and seeds 42–46.

#### OpenAI ViT-B/32 → LAION ViT-B/32

| Decoder and input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Linear, native target | 21.71 | 9.77 | 25.20 |
| Linear, aligned source | 16.42 | 15.06 | **17.35** |
| Linear, unaligned source | 15.04 | 16.43 | 68.36 |
| MLP, native target | **21.78** | **9.69** | 23.96 |
| MLP, aligned source | 17.81 | 13.66 | 19.71 |
| MLP, unaligned source | 13.87 | 17.60 | 43.86 |
| Species only | 21.95 | 9.52 | 24.64 |

#### OpenAI ViT-L/14 → FLAVA

| Decoder and input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Linear, native target | 21.72 | 9.75 | 26.04 |
| Linear, aligned source | 15.02 | 16.45 | **16.26** |
| Linear, unaligned source | 12.85 | 18.62 | 57.14 |
| MLP, native target | **21.77** | **9.70** | 24.89 |
| MLP, aligned source | 17.07 | 14.40 | 20.77 |
| MLP, unaligned source | 11.69 | 19.78 | 57.04 |
| Species only | 21.95 | 9.52 | 24.64 |

Recovered counts alone are misleading: the unaligned conditions sometimes
recover many attributes by predicting far too many positives. Alignment
sharply reduces this pathological hallucination rate.

### Species-controlled fine-grained ranking

For each attribute–species group containing both positive and negative birds,
we ask whether a positive bird receives a higher decoder score than a negative
bird of the **same species**. Ties receive one half. We macro-average 29,497
valid attribute–species groups spanning all 294 attributes.

Chance is 50%. A species-only predictor is exactly 50% because species is held
constant inside every comparison.

| Decoder and input | OpenAI → LAION | OpenAI → FLAVA |
|---|---:|---:|
| Linear, native target | 62.02% | 60.86% |
| Linear, aligned source | **59.69%** | **57.44%** |
| Linear, unaligned source | 50.32% | 49.74% |
| MLP, native target | **62.30%** | **61.16%** |
| MLP, aligned source | 59.49% | 57.29% |
| MLP, unaligned source | 50.21% | 49.53% |
| Species only | 50.00% | 50.00% |

The aligned decoders stay well above chance and far above the unaligned
controls. Therefore \(Q_{\text{Oxford}}\) preserves image-specific attribute
variation beyond coarse bird taxonomy.

## 4. What does decoder capacity reveal?

The fixed MLP is
`Linear(d,512) → GELU → Dropout(0.1) → Linear(512,312)`. It is trained under
the same data split, loss, optimizer, stopping rule, thresholds, and five seeds
as the linear decoder.

Natively, the MLP improves within-species ranking by approximately 0.3 points
for both target models and hallucinates roughly one fewer attribute per bird.
After alignment:

- it recovers 1.40 more true attributes for LAION and 2.05 more for FLAVA;
- it also hallucinates 2.36 and 4.51 more attributes, respectively;
- its within-species ranking is 0.20 and 0.15 points below the linear decoder.

The nonlinear decoder is reusable, but its extra native discrimination does
not transfer. The evidence supports robust transfer of predominantly
linearly accessible fine-grained structure; it does not support a stronger
claim of improved nonlinear compatibility.

## Takeaways

1. **Cross-dataset orientation transfer is real.** A rotation fitted on Oxford
   Pets strongly aligns CUB image geometry without using CUB pairs to refit it.
2. **Alignment supports functional decoder reuse.** A decoder trained only in
   the target space remains useful on aligned source embeddings.
3. **The signal is finer than class identity.** Within-species ranking rules
   out a species-only explanation.
4. **More decoder capacity does not automatically transfer more structure.**
   The MLP's small native advantage disappears after alignment.
5. **Cross-modal transfer remains incomplete.** One-sided CUB zero-shot
   classification is substantially weaker than intra-modal alignment and
   jointly aligned classification.

## Scope and limitations

- This phase studies one transfer dataset, two model pairs, and one fixed MLP.
- The rotation is global across datasets, but the affine centering is
  dataset-specific, matching the paper's protocol.
- \(Q_{\text{Oxford}}\) uses all Oxford `trainval` images rather than a
  small-anchor regime.
- The MLP architecture was prespecified rather than tuned; the experiment
  tests transfer for this decoder, not every possible nonlinear readout.
- Both encoders remain frozen. Whether downstream encoder fine-tuning preserves
  this compatibility is the next experimental phase.

## Provenance

The compact machine-readable results are in
[`artifacts/results/phase1_summary.json`](../artifacts/results/phase1_summary.json).
Run-level manifests, summaries, checkpoints, and per-attribute results are
under [`artifacts/results/frozen_decoder/`](../artifacts/results/frozen_decoder/).
The locked protocol is documented in
[`reports/frozen_decoder_protocol.md`](frozen_decoder_protocol.md).
