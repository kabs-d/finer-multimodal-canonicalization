# CUB-train-fitted Q control

This control asks whether the Phase I decoder-transfer gap is mainly caused by
using a cross-dataset Oxford-IIIT Pet rotation. We refit the centered
orthogonal map on the official CUB training image pairs only, then evaluate on
the official CUB test split.

No encoder is fine-tuned. No decoder is retrained. The five existing linear
target-space decoder checkpoints are reused unchanged.

## Protocol

For each model pair, fit

$$
Q_{\mathrm{CUB-train}} =
\arg\min_{Q^\top Q=I}
\|(A_{\mathrm{train}}-\mu_A)Q-(B_{\mathrm{train}}-\mu_B)\|_F^2 .
$$

Then compare the new CUB-train-fitted map against the original Oxford-fitted
map using the same CUB test examples, masks, eligible 294 attributes, decoder
seeds, validation thresholds, and species-controlled metrics.

## Paper-style CUB diagnostics

| Pair | Q source | Image retrieval after Q | Joint zero-shot species |
| --- | --- | ---: | ---: |
| OpenAI B/32 -> LAION B/32 | Oxford trainval | 90.42% | 44.68% |
| OpenAI B/32 -> LAION B/32 | CUB train | **99.46%** | **44.77%** |
| OpenAI L/14 -> FLAVA | Oxford trainval | 40.01% | **54.00%** |
| OpenAI L/14 -> FLAVA | CUB train | **92.53%** | 53.78% |

The in-domain image rotation almost closes image-image retrieval, especially
for FLAVA. Zero-shot species classification barely changes, so the new Q is not
just producing a broad species-level win.

## Fine-grained attribute transfer

| Pair | Q source | Recovered | Missed | Hallucinated | Within-species ranking |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 -> LAION B/32 | Oxford trainval | 16.42 | 15.06 | **17.35** | 59.69% |
| OpenAI B/32 -> LAION B/32 | CUB train | **20.26** | **11.21** | 21.58 | **60.78%** |
| OpenAI L/14 -> FLAVA | Oxford trainval | 15.02 | 16.45 | **16.26** | 57.44% |
| OpenAI L/14 -> FLAVA | CUB train | **20.36** | **11.11** | 20.72 | **60.23%** |
| Species-only baseline | CUB train labels | 21.95 | 9.52 | 24.64 | 50.00% |

The CUB-fitted Q recovers roughly 4 to 5 more true visible attributes per bird
than the Oxford-fitted Q. This is a real in-domain alignment gain, not merely a
species lookup: within-species positive-vs-negative ranking also rises above
the Oxford-Q result for both pairs.

The tradeoff is hallucination. CUB-train-Q predicts more true attributes, but
also predicts more absent attributes. The best headline is therefore not
"attributes solved"; it is:

> In-domain orthogonal alignment substantially improves fine-grained readout
> transfer, while the species-controlled ranking gain shows that part of the
> recovered signal is genuinely within-species visual structure.

## Artifacts

- Results:
  `artifacts/results/frozen_decoder/*_cub_train_q_linear/`
- Fitted rotations:
  `artifacts/alignments/*_cub_train_q_linear.pt`
- Prediction files:
  `artifacts/predictions/cub/*_cub_train_q_linear/`
