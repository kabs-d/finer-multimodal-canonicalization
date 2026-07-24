"""Interpretable, species-controlled analysis of CUB attribute transfer."""

import json
from pathlib import Path

import numpy as np
import torch

from .decoder_experiment import (
    _attribute_eligibility,
    _species_stratified_split,
    read_decoder_config,
)
from .decoders import build_decoder, decoder_probabilities
from .metrics import l2_normalize


def select_f1_thresholds(
    labels: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Choose one threshold per attribute using validation F1."""
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    thresholds = np.full(labels.shape[1], np.inf, dtype=np.float64)
    for attribute_index in np.flatnonzero(eligible):
        observed = mask[:, attribute_index]
        observed_labels = labels[observed, attribute_index]
        observed_scores = scores[observed, attribute_index]
        positives = int(observed_labels.sum())
        if positives == 0:
            continue
        order = np.argsort(-observed_scores, kind="mergesort")
        sorted_labels = observed_labels[order]
        sorted_scores = observed_scores[order]
        threshold_ends = np.r_[
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
            sorted_scores.size - 1,
        ]
        true_positive = np.cumsum(sorted_labels)[threshold_ends]
        predicted_positive = threshold_ends + 1
        f1 = 2 * true_positive / (predicted_positive + positives)
        best = int(np.flatnonzero(f1 == f1.max())[0])
        thresholds[attribute_index] = sorted_scores[threshold_ends[best]]
    return thresholds


def per_bird_recovery(
    labels: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float]:
    """Average positive recovery errors per bird; true negatives are excluded."""
    labels = np.asarray(labels, dtype=bool)
    observed = np.asarray(mask, dtype=bool) & np.asarray(eligible, dtype=bool)[None, :]
    predictions = np.asarray(scores) >= np.asarray(thresholds)[None, :]
    correctly_recovered = (predictions & labels & observed).sum(axis=1)
    missed = ((~predictions) & labels & observed).sum(axis=1)
    hallucinated = (predictions & (~labels) & observed).sum(axis=1)
    return {
        "mean_present_attributes": float(
            (correctly_recovered + missed).mean()
        ),
        "mean_correctly_recovered": float(correctly_recovered.mean()),
        "mean_missed": float(missed.mean()),
        "mean_hallucinated": float(hallucinated.mean()),
    }


def within_species_ranking_accuracy(
    labels: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    species: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    """Macro accuracy over positive-negative pairs from the same species."""
    labels = np.asarray(labels, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    species = np.asarray(species, dtype=np.int64)
    group_accuracies: list[float] = []
    represented_attributes: set[int] = set()
    for attribute_index in np.flatnonzero(eligible):
        for species_index in np.unique(species):
            group = (
                (species == species_index)
                & mask[:, attribute_index]
            )
            group_labels = labels[group, attribute_index]
            if not group_labels.any() or group_labels.all():
                continue
            group_scores = scores[group, attribute_index]
            positive = group_scores[group_labels]
            negative = group_scores[~group_labels]
            comparisons = positive[:, None] - negative[None, :]
            accuracy = (
                np.count_nonzero(comparisons > 0)
                + 0.5 * np.count_nonzero(comparisons == 0)
            ) / comparisons.size
            group_accuracies.append(float(accuracy))
            represented_attributes.add(int(attribute_index))
    if not group_accuracies:
        raise ValueError("no within-species positive-negative groups are defined")
    values = np.asarray(group_accuracies, dtype=np.float64)
    return {
        "macro_pair_accuracy": float(values.mean()),
        "attribute_species_groups": int(values.size),
        "represented_attributes": len(represented_attributes),
    }


def _species_prevalence_scores(
    train_labels: np.ndarray,
    train_mask: np.ndarray,
    train_species: np.ndarray,
    fit_indices: np.ndarray,
    evaluation_species: np.ndarray,
) -> np.ndarray:
    """Predict attributes from ground-truth species and train prevalence only."""
    labels = train_labels[fit_indices]
    mask = train_mask[fit_indices]
    species = train_species[fit_indices]
    global_observed = mask.sum(axis=0)
    global_prevalence = np.divide(
        (labels * mask).sum(axis=0),
        global_observed,
        out=np.zeros(labels.shape[1], dtype=np.float64),
        where=global_observed > 0,
    )
    table: dict[int, np.ndarray] = {}
    for species_index in np.unique(train_species):
        selected = species == species_index
        observed = mask[selected].sum(axis=0)
        positives = (labels[selected] * mask[selected]).sum(axis=0)
        table[int(species_index)] = np.divide(
            positives,
            observed,
            out=global_prevalence.copy(),
            where=observed > 0,
        )
    return np.stack([table[int(index)] for index in evaluation_species])


def _aggregate_seed_results(seed_results: list[dict]) -> dict:
    conditions = seed_results[0]["conditions"].keys()
    output = {}
    count_metrics = [
        "mean_present_attributes",
        "mean_correctly_recovered",
        "mean_missed",
        "mean_hallucinated",
    ]
    for condition in conditions:
        output[condition] = {"per_bird_recovery": {}}
        for metric in count_metrics:
            values = np.asarray(
                [
                    result["conditions"][condition]["per_bird_recovery"][metric]
                    for result in seed_results
                ]
            )
            output[condition]["per_bird_recovery"][metric] = {
                "mean": float(values.mean()),
                "seed_std": float(values.std()),
            }
        values = np.asarray(
            [
                result["conditions"][condition]["within_species"][
                    "macro_pair_accuracy"
                ]
                for result in seed_results
            ]
        )
        first = seed_results[0]["conditions"][condition]["within_species"]
        output[condition]["within_species"] = {
            "macro_pair_accuracy": {
                "mean": float(values.mean()),
                "seed_std": float(values.std()),
            },
            "attribute_species_groups": first["attribute_species_groups"],
            "represented_attributes": first["represented_attributes"],
        }
    return output


def run_attribute_analysis(
    config_path: Path | str,
    embedding_root: Path | str,
    prediction_root: Path | str,
    output_root: Path | str,
) -> Path:
    config = read_decoder_config(config_path)
    run_id = config["run_id"]
    feature_run_id = config["feature_extraction"].get(
        "reuse_run_id", run_id
    )
    feature_dir = Path(embedding_root) / feature_run_id
    result_dir = Path(output_root) / run_id
    train_features = torch.load(
        feature_dir / "cub_train.pt", map_location="cpu"
    )
    first_prediction_path = (
        Path(prediction_root)
        / run_id
        / f"seed_{config['decoder']['seeds'][0]}.npz"
    )
    with np.load(first_prediction_path) as first_predictions:
        test_labels = first_predictions["labels"].astype(np.int64)
        test_mask = first_predictions["mask"].astype(bool)
        test_species = first_predictions["species"].astype(np.int64)

    decoder_config = config["decoder"]
    fit_indices, validation_indices = _species_stratified_split(
        train_features["labels"],
        validation_fraction=decoder_config["validation_fraction"],
        seed=decoder_config["split_seed"],
    )
    eligible = _attribute_eligibility(
        train_features["attributes"],
        train_features["attribute_mask"],
        fit_indices,
        torch.from_numpy(test_labels),
        torch.from_numpy(test_mask),
        min_train_positive=decoder_config["min_train_positive"],
        min_train_negative=decoder_config["min_train_negative"],
        min_test_positive=decoder_config["min_test_positive"],
        min_test_negative=decoder_config["min_test_negative"],
    )
    target_train = l2_normalize(train_features["target"]).float()
    train_labels = train_features["attributes"].numpy().astype(np.int64)
    train_mask = train_features["attribute_mask"].numpy().astype(bool)
    train_species = train_features["labels"].numpy().astype(np.int64)

    seed_results = []
    for seed in decoder_config["seeds"]:
        checkpoint = torch.load(
            result_dir / "checkpoints" / f"seed_{seed}.pt",
            map_location="cpu",
        )
        state = checkpoint["target_decoder"]
        model = build_decoder(
            config["decoder"],
            target_train.shape[1],
            train_features["attributes"].shape[1],
        )
        model.load_state_dict(state)
        validation_scores = decoder_probabilities(
            model,
            target_train[validation_indices],
            torch.device("cpu"),
        ).numpy()
        thresholds = select_f1_thresholds(
            train_labels[validation_indices],
            train_mask[validation_indices],
            validation_scores,
            eligible,
        )
        with np.load(
            Path(prediction_root) / run_id / f"seed_{seed}.npz"
        ) as predictions:
            conditions = {}
            for condition in [
                "native_target",
                "aligned_source",
                "unaligned_source",
            ]:
                scores = predictions[condition]
                conditions[condition] = {
                    "per_bird_recovery": per_bird_recovery(
                        test_labels,
                        test_mask,
                        scores,
                        thresholds,
                        eligible,
                    ),
                    "within_species": within_species_ranking_accuracy(
                        test_labels,
                        test_mask,
                        scores,
                        test_species,
                        eligible,
                    ),
                }
        seed_results.append({"seed": seed, "conditions": conditions})

    validation_species_scores = _species_prevalence_scores(
        train_labels,
        train_mask,
        train_species,
        fit_indices,
        train_species[validation_indices],
    )
    species_thresholds = select_f1_thresholds(
        train_labels[validation_indices],
        train_mask[validation_indices],
        validation_species_scores,
        eligible,
    )
    test_species_scores = _species_prevalence_scores(
        train_labels,
        train_mask,
        train_species,
        fit_indices,
        test_species,
    )
    species_only = {
        "uses_ground_truth_species": True,
        "per_bird_recovery": per_bird_recovery(
            test_labels,
            test_mask,
            test_species_scores,
            species_thresholds,
            eligible,
        ),
        "within_species": within_species_ranking_accuracy(
            test_labels,
            test_mask,
            test_species_scores,
            test_species,
            eligible,
        ),
    }
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "eligible_attributes": int(eligible.sum()),
        "test_birds": int(test_labels.shape[0]),
        "threshold_protocol": (
            "Per-attribute F1 thresholds selected on target-native validation "
            "scores and frozen for native, aligned, and unaligned test inputs."
        ),
        "per_bird_count_policy": (
            "Counts use visible eligible labels only and exclude true negatives."
        ),
        "within_species_policy": (
            "Equal-weight macro average over attribute-species groups containing "
            "at least one visible positive and negative; ties score 0.5."
        ),
        "aggregate": _aggregate_seed_results(seed_results),
        "species_only": species_only,
        "seeds": seed_results,
    }
    destination = result_dir / "attribute_interpretability.json"
    destination.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return destination
