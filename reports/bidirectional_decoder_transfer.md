# Bidirectional attribute-decoder transfer

This follow-up evaluates decoders in both directions of the same Oxford-IIIT
Pets-fitted canonical map. Source to target uses $Q$; target to source uses
$Q^\top$. No second transform is fitted.

The metric is the mean, over CUB test birds, of the percentage of that bird's
visible-positive ground-truth attributes recovered by the decoder. The table
uses five seeds and 294 eligible CUB attributes.

| Pair and decoder | Source native | Target native | Source decoder on aligned target | Target decoder on aligned source |
| --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION, linear | 70.11% | 69.66% | 64.34% | 52.87% |
| OpenAI B/32 → LAION, two-layer MLP | 70.82% | 70.80% | 67.59% | 66.02% |
| OpenAI L/14 → FLAVA, linear | 70.45% | 69.69% | 55.77% | 48.60% |
| OpenAI L/14 → FLAVA, two-layer MLP | 71.08% | 70.71% | 61.41% | 62.21% |

The native columns are close to 70% across models. The most visible change is
after alignment: the two-layer MLP recovers more attributes in both directions.
The asymmetry of the linear rows arises despite $Q^\top = Q^{-1}$, indicating
that the reversible map interacts differently with the two decoders' learned
decision boundaries.

## Protocol note

The linear numbers reuse the original five Phase I decoder checkpoints. The
two-layer MLP is an exploratory capacity probe using the same frozen embedding
caches, CUB split, seeds, and masked class-balanced BCE loss, but a different
validation stopping score and threshold implementation. Its improvement should
therefore be treated as preliminary until rerun under the fully locked Phase I
selection protocol.

Machine-readable summaries:

- `artifacts/results/linear_bidirectional_probe/`
- `artifacts/results/deep_mlp_probe/`
