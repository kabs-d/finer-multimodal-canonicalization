#!/usr/bin/env python3
"""Evaluate existing linear CUB decoders in both canonical-alignment directions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from canonical_study.alignment import OrthogonalAlignment
from canonical_study.attribute_analysis import select_f1_thresholds
from canonical_study.decoder_experiment import _attribute_eligibility, _species_stratified_split
from canonical_study.decoders import build_decoder, decoder_probabilities


def recovered_percent(labels, mask, scores, thresholds, eligible):
    labels = np.asarray(labels, dtype=bool)
    observed_positive = labels & np.asarray(mask, dtype=bool) & eligible[None, :]
    predicted = scores >= thresholds[None, :]
    numerator = (predicted & observed_positive).sum(axis=1)
    denominator = observed_positive.sum(axis=1)
    return float(np.mean(numerator[denominator > 0] / denominator[denominator > 0]) * 100)


def run(pair, config_path, alignment_path, embedding_root, results_root, output_root, device_name):
    config = json.loads(Path(config_path).read_text())
    features = Path(embedding_root) / pair
    train = torch.load(features / "cub_train.pt", map_location="cpu")
    test = torch.load(features / "cub_test.pt", map_location="cpu")
    payload = torch.load(alignment_path, map_location="cpu")
    device = torch.device(device_name)

    source_train, target_train = F.normalize(train["source"], dim=1), F.normalize(train["target"], dim=1)
    source_test, target_test = F.normalize(test["source"], dim=1), F.normalize(test["target"], dim=1)
    source_mean, target_mean = train["source"].mean(0, keepdim=True), train["target"].mean(0, keepdim=True)
    alignment = OrthogonalAlignment(payload["rotation"], source_mean, target_mean)
    aligned_source = F.normalize(alignment.transform(source_test), dim=1)
    aligned_target = F.normalize((target_test - target_mean) @ payload["rotation"].T + source_mean, dim=1)
    train_indices, validation_indices = _species_stratified_split(train["labels"], validation_fraction=0.2, seed=2026)
    eligible = _attribute_eligibility(train["attributes"], train["attribute_mask"], train_indices, test["attributes"], test["attribute_mask"], min_train_positive=20, min_train_negative=20, min_test_positive=5, min_test_negative=5)
    labels, mask = test["attributes"].numpy(), test["attribute_mask"].numpy()
    val_labels, val_mask = train["attributes"][validation_indices].numpy(), train["attribute_mask"][validation_indices].numpy()
    checkpoint_dir = Path(results_root) / pair / "checkpoints"
    rows = []
    for seed in [42, 43, 44, 45, 46]:
        checkpoint = torch.load(checkpoint_dir / f"seed_{seed}.pt", map_location="cpu")
        source_decoder = build_decoder(config["decoder"], source_train.shape[1], train["attributes"].shape[1]).to(device)
        target_decoder = build_decoder(config["decoder"], target_train.shape[1], train["attributes"].shape[1]).to(device)
        source_decoder.load_state_dict(checkpoint["source_decoder"]); target_decoder.load_state_dict(checkpoint["target_decoder"])
        source_thresholds = select_f1_thresholds(val_labels, val_mask, decoder_probabilities(source_decoder, source_train[validation_indices], device).numpy(), eligible)
        target_thresholds = select_f1_thresholds(val_labels, val_mask, decoder_probabilities(target_decoder, target_train[validation_indices], device).numpy(), eligible)
        rows.append({
            "seed": seed,
            "source_native_percent": recovered_percent(labels, mask, decoder_probabilities(source_decoder, source_test, device).numpy(), source_thresholds, eligible),
            "target_native_percent": recovered_percent(labels, mask, decoder_probabilities(target_decoder, target_test, device).numpy(), target_thresholds, eligible),
            "source_decoder_on_aligned_target_percent": recovered_percent(labels, mask, decoder_probabilities(source_decoder, aligned_target, device).numpy(), source_thresholds, eligible),
            "target_decoder_on_aligned_source_percent": recovered_percent(labels, mask, decoder_probabilities(target_decoder, aligned_source, device).numpy(), target_thresholds, eligible),
        })
        print(f"{pair} seed {seed} complete", flush=True)
    summary = {"architecture": "linear", "pair": pair, "eligible_attributes": int(eligible.sum()), "metric": "mean per-bird percentage of visible-positive ground-truth attributes recovered", "seeds": rows}
    for metric in ["source_native_percent", "target_native_percent", "source_decoder_on_aligned_target_percent", "target_decoder_on_aligned_source_percent"]:
        summary[metric + "_mean"] = float(np.mean([row[metric] for row in rows]))
    destination = Path(output_root) / pair
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--embedding-root", default="artifacts/embeddings/cub")
    parser.add_argument("--results-root", default="artifacts/results/frozen_decoder")
    parser.add_argument("--output-root", default="artifacts/results/linear_bidirectional_probe")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.pair, args.config, args.alignment, args.embedding_root, args.results_root, args.output_root, args.device)
