#!/usr/bin/env python3
"""Train a two-hidden-layer CUB attribute decoder in both native spaces.

This is an exploratory capacity probe. It uses cached frozen embeddings only
and reports recovered visible-positive attributes as a percentage of the
ground-truth visible positives.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from canonical_study.alignment import OrthogonalAlignment
from canonical_study.decoder_experiment import (
    _attribute_eligibility,
    _positive_weights,
    _species_stratified_split,
)


class DeepAttributeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def predict(model, features, device):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(features.to(device))).cpu().numpy()


def thresholds(labels, mask, probabilities, eligible):
    y = labels.astype(bool)
    m = mask.astype(bool)
    p = probabilities
    out = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for j in np.flatnonzero(eligible):
        best = (-1.0, 0.5)
        for t in np.linspace(0.05, 0.95, 19):
            pred = p[:, j] >= t
            valid = m[:, j]
            tp = np.sum(pred[valid] & y[valid, j])
            fp = np.sum(pred[valid] & ~y[valid, j])
            fn = np.sum(~pred[valid] & y[valid, j])
            score = 2 * tp / max(2 * tp + fp + fn, 1)
            if score > best[0]:
                best = (score, float(t))
        out[j] = best[1]
    return out


def recovered_percent(labels, mask, probabilities, eligible, cutoffs):
    y = labels.astype(bool)
    m = mask.astype(bool)
    pred = probabilities >= cutoffs[None, :]
    positives = y & m & eligible[None, :]
    recovered = np.sum(pred & positives, axis=1)
    truth = np.sum(positives, axis=1)
    valid = truth > 0
    return float(np.mean(recovered[valid] / truth[valid]) * 100.0)


def train(features, labels, mask, train_idx, val_idx, eligible, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = DeepAttributeMLP(features.shape[1], labels.shape[1]).to(device)
    weight = _positive_weights(
        labels, mask, train_idx, 0.25, 20.0
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, 201):
        model.train(); order = train_idx.copy(); rng.shuffle(order)
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            logits = model(features[idx].to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, labels[idx].to(device), pos_weight=weight, reduction="none"
            )
            loss = (loss * mask[idx].to(device)).sum() / mask[idx].sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        vp = predict(model, features[val_idx], device)
        # Validation macro-F1 at 0.5 is used only as a stable exploratory stop rule.
        valid = mask[val_idx].numpy().astype(bool)
        truth = labels[val_idx].numpy().astype(bool)
        pred = vp >= 0.5
        f1s = []
        for j in np.flatnonzero(eligible):
            v = valid[:, j]; tp = np.sum(pred[v, j] & truth[v, j]); fp = np.sum(pred[v, j] & ~truth[v, j]); fn = np.sum(~pred[v, j] & truth[v, j])
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        score = float(np.mean(f1s)) if f1s else 0.0
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 20:
            break
    model.load_state_dict(best_state); model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "validation_macro_f1": best_score}


def run(pair, embedding_root, alignment_path, cub_alignment_path, output_root, device_name):
    device = torch.device(device_name)
    train_path = embedding_root / pair / "cub_train.pt"
    test_path = embedding_root / pair / "cub_test.pt"
    train_data = torch.load(train_path, map_location="cpu")
    test_data = torch.load(test_path, map_location="cpu")
    payload = torch.load(alignment_path, map_location="cpu")
    cub_payload = torch.load(cub_alignment_path, map_location="cpu") if cub_alignment_path else None
    source_mean = train_data["source"].mean(0, keepdim=True)
    target_mean = train_data["target"].mean(0, keepdim=True)
    alignment = OrthogonalAlignment(payload["rotation"], source_mean, target_mean)
    source_train = F.normalize(train_data["source"], dim=1)
    target_train = F.normalize(train_data["target"], dim=1)
    source_test = F.normalize(test_data["source"], dim=1)
    target_test = F.normalize(test_data["target"], dim=1)
    aligned_source_test = F.normalize(alignment.transform(source_test), dim=1)
    aligned_target_test = F.normalize((target_test - target_mean) @ payload["rotation"].T + source_mean, dim=1)
    if cub_payload:
        cub_alignment = OrthogonalAlignment(cub_payload["rotation"], source_mean, target_mean)
        cub_aligned_source_test = F.normalize(cub_alignment.transform(source_test), dim=1)
        cub_aligned_target_test = F.normalize((target_test - target_mean) @ cub_payload["rotation"].T + source_mean, dim=1)
    train_idx, val_idx = _species_stratified_split(train_data["labels"], validation_fraction=0.2, seed=2026)
    eligible = _attribute_eligibility(
        train_data["attributes"], train_data["attribute_mask"], train_idx,
        test_data["attributes"], test_data["attribute_mask"], min_train_positive=20,
        min_train_negative=20, min_test_positive=5, min_test_negative=5,
    )
    labels, mask = test_data["attributes"], test_data["attribute_mask"]
    rows = []
    for seed in [42, 43, 44, 45, 46]:
        source_model, source_training = train(source_train, train_data["attributes"], train_data["attribute_mask"], train_idx, val_idx, eligible, seed, device)
        target_model, target_training = train(target_train, train_data["attributes"], train_data["attribute_mask"], train_idx, val_idx, eligible, seed, device)
        source_val = predict(source_model, source_train[val_idx], device); target_val = predict(target_model, target_train[val_idx], device)
        source_cut = thresholds(train_data["attributes"][val_idx].numpy(), train_data["attribute_mask"][val_idx].numpy(), source_val, eligible)
        target_cut = thresholds(train_data["attributes"][val_idx].numpy(), train_data["attribute_mask"][val_idx].numpy(), target_val, eligible)
        sp = predict(source_model, source_test, device)
        tp = predict(target_model, target_test, device)
        rows.append({
            "seed": seed,
            "source_native_percent": recovered_percent(labels.numpy(), mask.numpy(), sp, eligible, source_cut),
            "target_native_percent": recovered_percent(labels.numpy(), mask.numpy(), tp, eligible, target_cut),
            "source_decoder_on_unaligned_target_percent": recovered_percent(labels.numpy(), mask.numpy(), predict(source_model, target_test, device), eligible, source_cut),
            "target_decoder_on_unaligned_source_percent": recovered_percent(labels.numpy(), mask.numpy(), predict(target_model, source_test, device), eligible, target_cut),
            "source_decoder_on_aligned_target_percent": recovered_percent(labels.numpy(), mask.numpy(), predict(source_model, aligned_target_test, device), eligible, source_cut),
            "target_decoder_on_aligned_source_percent": recovered_percent(labels.numpy(), mask.numpy(), predict(target_model, aligned_source_test, device), eligible, target_cut),
            "source_training": source_training,
            "target_training": target_training,
        })
        if cub_payload:
            rows[-1]["source_decoder_on_cub_aligned_target_percent"] = recovered_percent(labels.numpy(), mask.numpy(), predict(source_model, cub_aligned_target_test, device), eligible, source_cut)
            rows[-1]["target_decoder_on_cub_aligned_source_percent"] = recovered_percent(labels.numpy(), mask.numpy(), predict(target_model, cub_aligned_source_test, device), eligible, target_cut)
        print(f"{pair} seed {seed} complete")
    output = Path(output_root) / pair
    output.mkdir(parents=True, exist_ok=True)
    summary = {"architecture": "deep_mlp_512_256", "pair": pair, "eligible_attributes": int(eligible.sum()), "metric": "percent of visible-positive ground-truth attributes recovered", "seeds": rows}
    for key in ["source_native_percent", "target_native_percent", "source_decoder_on_unaligned_target_percent", "target_decoder_on_unaligned_source_percent", "source_decoder_on_aligned_target_percent", "target_decoder_on_aligned_source_percent"]:
        summary[key + "_mean"] = float(np.mean([r[key] for r in rows]))
    if cub_payload:
        for key in ["source_decoder_on_cub_aligned_target_percent", "target_decoder_on_cub_aligned_source_percent"]:
            summary[key + "_mean"] = float(np.mean([r[key] for r in rows]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--embedding-root", type=Path, default=Path("artifacts/embeddings/cub"))
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--cub-alignment", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/results/deep_mlp_probe"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.pair, args.embedding_root, args.alignment, args.cub_alignment, args.output_root, args.device)
