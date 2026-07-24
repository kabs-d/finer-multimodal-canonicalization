# Interpretable CUB attribute-transfer results

This report uses the 5,794 official CUB test birds and 294 eligible attributes.
Labels marked `not visible` are excluded. An average bird has 31.47 visible
positive attributes among the eligible labels.

One decision threshold per attribute is selected using the native target
decoder on the internal validation partition. Those thresholds are frozen and
used unchanged for native-target, aligned-source, and unaligned-source test
predictions. Results below are means over decoder seeds 42–46.

## Per-bird attribute recovery

True-negative attributes are deliberately omitted. `Recovered` counts true
attributes predicted present, `missed` counts true attributes predicted absent,
and `hallucinated` counts absent attributes predicted present.

### OpenAI ViT-B/32 to LAION ViT-B/32

| Decoder input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Native target | 21.71 | 9.77 | 25.20 |
| Aligned source | 16.42 | 15.06 | 17.35 |
| Unaligned source | 15.04 | 16.43 | 68.36 |
| Species only | 21.95 | 9.52 | 24.64 |

### OpenAI ViT-L/14 to FLAVA

| Decoder input | Recovered | Missed | Hallucinated |
|---|---:|---:|---:|
| Native target | 21.72 | 9.75 | 26.04 |
| Aligned source | 15.02 | 16.45 | 16.26 |
| Unaligned source | 12.85 | 18.62 | 57.14 |
| Species only | 21.95 | 9.52 | 24.64 |

The species-only baseline uses ground-truth species and predicts each
attribute from its prevalence for that species in the decoder-training
partition. Its strong counts show that much of CUB's raw attribute
predictability can be explained by species identity. The unaligned decoder
also demonstrates why recovered counts must be accompanied by hallucinations:
it recovers some true attributes only by predicting far too many attributes.

## True fine-grained control

For every attribute–species group containing at least one visible positive and
one visible negative bird, this metric compares every positive–negative pair
within that species. It records how often the positive bird receives the higher
decoder score; ties receive one half. The final number is the equal-weight mean
over 29,497 valid attribute–species groups spanning all 294 attributes. Chance
is 50%.

| Decoder input | OpenAI to LAION | OpenAI to FLAVA |
|---|---:|---:|
| Native target | 62.02% | 60.86% |
| Aligned source | **59.69%** | **57.44%** |
| Unaligned source | 50.32% | 49.74% |
| Species only | 50.00% | 50.00% |

Because species is held constant within every comparison, neither species
identity nor a species lookup table can exceed chance. The aligned decoders
remain clearly above chance and substantially above unaligned inputs. Thus the
Oxford-fitted orientation with CUB centering preserves measurable
image-specific attribute variation beyond bird taxonomy, although it does not
retain all native fine-grained performance.

Machine-readable results are stored as `attribute_interpretability.json` in
each run directory under `artifacts/results/frozen_decoder/`.
