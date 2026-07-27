# Bidirectional attribute-decoder transfer

This appendix asks whether frozen CUB attribute decoders transfer in both
directions of the same canonical map. Source → target applies $Q$; target →
source applies $Q^\top$. No reverse map is fitted.

The metric is the mean percentage of a test bird's visible-positive attributes
recovered by the decoder. Each result uses the 5,794 official CUB test images,
294 eligible attributes, and five random seeds.

## Decoder reuse in both directions

| Pair and decoder | Source native | Target native | Source decoder on aligned target | Target decoder on aligned source |
| --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION, linear | 70.11% | 69.66% | 64.34% | 52.87% |
| OpenAI B/32 → LAION, two-layer MLP | 70.82% | 70.80% | 67.59% | 66.02% |
| OpenAI L/14 → FLAVA, linear | 70.45% | 69.69% | 55.77% | 48.60% |
| OpenAI L/14 → FLAVA, two-layer MLP | 71.08% | 70.71% | 61.41% | 62.21% |

The two-layer MLP is fixed as:

```text
Linear(d, 512) → GELU → Dropout(0.1) → Linear(512, 256)
→ GELU → Dropout(0.1) → Linear(256, 312)
```

It is a fixed capacity follow-up, not an architecture search. The table shows
that the map remains useful in both directions, although the two decoder
decision boundaries do not behave symmetrically under $Q$ and $Q^\top$.

## Alignment controls for the two-layer MLP

| Pair and direction | Native | Unaligned | Oxford-Pets $Q$ | CUB-train $Q$ |
| --- | ---: | ---: | ---: | ---: |
| OpenAI B/32 → LAION, target → source | 70.82% | 42.19% | 67.59% | 69.68% |
| OpenAI B/32 → LAION, source → target | 70.80% | 43.10% | 66.02% | 70.21% |
| OpenAI L/14 → FLAVA, target → source | 71.08% | 43.24% | 61.41% | 67.88% |
| OpenAI L/14 → FLAVA, source → target | 70.71% | 34.29% | 62.21% | 70.68% |

The unaligned controls show that decoder reuse is not produced by merely
feeding arbitrary source coordinates into the target decoder. Oxford-Pets $Q$
recovers most of the loss, and the in-domain CUB-train control nearly reaches
native recovery in every displayed direction.

## Artifacts

- `artifacts/results/linear_bidirectional_probe/`
- `artifacts/results/deep_mlp_probe/`
