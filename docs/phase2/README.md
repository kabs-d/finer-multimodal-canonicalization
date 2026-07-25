# Phase II: Contrastive Fine-Tuning and Fine-Grained Retrieval

Phase II is planned, not yet claimed as a result.

The central question is whether the canonicalization observed for frozen
contrastive encoders survives after downstream contrastive fine-tuning.

## Working hypothesis

If both models are fine-tuned on the same narrow CUB-style contrastive
objective, coarse alignment may remain visible in cosine, retrieval, or
zero-shot species metrics. The sharper test is whether fine-grained
compatibility survives:

- Can a fixed or refit orthogonal map still support target-space attribute
  readout transfer?
- Does within-species attribute ranking degrade before coarse species
  classification does?
- Can fine-grained retrieval distinguish birds of the same species using
  attributes rather than only class identity?

## Planned design

1. Start from the frozen Phase I checkpoints and metrics.
2. Fine-tune each encoder pair with a contrastive image-text objective on the
   same train split.
3. Save multiple fine-tuning snapshots.
4. At each snapshot, measure:
   - paired cosine before and after alignment;
   - image-image and text-text retrieval;
   - paper-style zero-shot species classification;
   - target-space decoder transfer;
   - within-species attribute ranking;
   - fine-grained retrieval among same-species candidates.
5. Compare a fixed pre-fine-tuning $Q$ against a newly refit $Q_t$ at each
   snapshot.

The intended outcome is not merely “fine-tuning helps or hurts.” The
interesting result would be a separation: class-level compatibility remains
apparently healthy while fine-grained readout or retrieval compatibility
breaks earlier.

## Status

Phase II has not been run yet. The frozen-encoder Phase I results are the
completed baseline and should be treated as the reference point for any
fine-tuning experiment.
