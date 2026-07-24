# Linear versus MLP attribute-decoder transfer

The MLP is fixed in advance as
`Linear(d,512) -> GELU -> Dropout(0.1) -> Linear(512,312)`. It is trained only
on native target embeddings using the same split, seeds, loss, optimizer, and
early-stopping protocol as the linear decoder. Neither decoder sees aligned
source embeddings during training or model selection.

All results use 5,794 official CUB test birds, 294 eligible attributes, and
seeds 42–46. An average bird has 31.47 visible positive attributes.

## Per-bird attribute recovery

### OpenAI ViT-B/32 to LAION ViT-B/32

| Decoder and input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Linear, native target | 21.71 | 9.77 | 25.20 |
| MLP, native target | **21.78** | **9.69** | **23.96** |
| Linear, aligned source | 16.42 | 15.06 | **17.35** |
| MLP, aligned source | **17.81** | **13.66** | 19.71 |
| Linear, unaligned source | **15.04** | **16.43** | 68.36 |
| MLP, unaligned source | 13.87 | 17.60 | **43.86** |
| Species only | 21.95 | 9.52 | 24.64 |

### OpenAI ViT-L/14 to FLAVA

| Decoder and input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Linear, native target | 21.72 | 9.75 | 26.04 |
| MLP, native target | **21.77** | **9.70** | **24.89** |
| Linear, aligned source | 15.02 | 16.45 | **16.26** |
| MLP, aligned source | **17.07** | **14.40** | 20.77 |
| Linear, unaligned source | **12.85** | **18.62** | 57.14 |
| MLP, unaligned source | 11.69 | 19.78 | **57.04** |
| Species only | 21.95 | 9.52 | 24.64 |

Natively, the MLP is slightly better: it recovers marginally more true
attributes and hallucinates roughly one fewer attribute per bird. On aligned
inputs, it recovers 1.40 additional attributes for LAION and 2.05 for FLAVA,
but also hallucinates 2.36 and 4.51 additional attributes, respectively. This
is a recall–hallucination tradeoff rather than clean dominance.

## Within-species fine-grained ranking

Chance and the ground-truth-species lookup baseline are both 50%.

| Decoder and input | OpenAI to LAION | OpenAI to FLAVA |
|---|---:|---:|
| Linear, native target | 62.02% | 60.86% |
| MLP, native target | **62.30%** | **61.16%** |
| Linear, aligned source | **59.69%** | **57.44%** |
| MLP, aligned source | 59.49% | 57.29% |
| Linear, unaligned source | 50.32% | 49.74% |
| MLP, unaligned source | 50.21% | 49.53% |
| Species only | 50.00% | 50.00% |

The MLP has a small native fine-grained advantage of about 0.3 percentage
points for both target models. That advantage disappears after alignment: the
aligned MLP is slightly below the aligned linear decoder for both pairs.

## Conclusion

The MLP is modestly more expressive on native target embeddings, and a frozen
target MLP remains reusable on aligned source embeddings far above chance.
However, the MLP's additional native within-species discrimination does not
transfer through the Oxford-fitted alignment. The evidence therefore supports
transfer of robust, largely linear fine-grained structure, but does not support
the stronger claim that the extra nonlinear structure exposed by this MLP is
preserved.
