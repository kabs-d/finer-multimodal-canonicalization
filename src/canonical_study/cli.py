"""Command-line entry point for reproducible baseline jobs."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .datasets import prepare_cub, prepare_oxford, validate_cub, validate_oxford
from .attribute_analysis import run_attribute_analysis
from .cub_train_q_control import run_cub_train_q_control
from .decoder_experiment import (
    materialize_oxford_alignment,
    read_decoder_config,
    run_frozen_decoder,
)
from .experiment import read_config, run_baseline
from .mlp_experiment import run_cached_mlp_decoder


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(destination: Path) -> dict:
    package_names = [
        "numpy",
        "open-clip-torch",
        "Pillow",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "timm",
        "tokenizers",
    ]
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in package_names},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        environment["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        environment["git_commit"] = output.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        environment["git_commit"] = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="canonical-study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--data-root", type=Path, required=True)

    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--data-root", type=Path, required=True)

    prepare_cub_parser = subparsers.add_parser("prepare-cub")
    prepare_cub_parser.add_argument("--data-root", type=Path, required=True)
    prepare_cub_parser.add_argument(
        "--accept-research-terms",
        action="store_true",
        help="confirm that the CUB image-use terms have been reviewed",
    )

    validate_cub_parser = subparsers.add_parser("validate-cub")
    validate_cub_parser.add_argument("--data-root", type=Path, required=True)

    inspect_config = subparsers.add_parser("validate-config")
    inspect_config.add_argument("--config", type=Path, required=True)

    inspect_decoder_config = subparsers.add_parser("validate-decoder-config")
    inspect_decoder_config.add_argument("--config", type=Path, required=True)

    alignment = subparsers.add_parser("materialize-oxford-alignment")
    alignment.add_argument("--config", type=Path, required=True)
    alignment.add_argument("--upstream-embedding-prefix", type=Path, required=True)
    alignment.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--embedding-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--model-cache-root", type=Path, required=True)
    run.add_argument("--upstream-embedding-prefix", type=Path)
    run.add_argument("--device", default="cuda")
    run.add_argument("--force", action="store_true")

    decoder = subparsers.add_parser("run-frozen-decoder")
    decoder.add_argument("--config", type=Path, required=True)
    decoder.add_argument("--alignment", type=Path, required=True)
    decoder.add_argument("--data-root", type=Path, required=True)
    decoder.add_argument("--embedding-root", type=Path, required=True)
    decoder.add_argument("--prediction-root", type=Path, required=True)
    decoder.add_argument("--output-root", type=Path, required=True)
    decoder.add_argument("--model-cache-root", type=Path, required=True)
    decoder.add_argument("--device", default="cuda")
    decoder.add_argument("--force", action="store_true")

    attributes = subparsers.add_parser("analyze-attributes")
    attributes.add_argument("--config", type=Path, required=True)
    attributes.add_argument("--embedding-root", type=Path, required=True)
    attributes.add_argument("--prediction-root", type=Path, required=True)
    attributes.add_argument("--output-root", type=Path, required=True)

    cub_train_q = subparsers.add_parser("cub-train-q-control")
    cub_train_q.add_argument("--config", type=Path, required=True)
    cub_train_q.add_argument("--data-root", type=Path, required=True)
    cub_train_q.add_argument("--embedding-root", type=Path, required=True)
    cub_train_q.add_argument("--prediction-root", type=Path, required=True)
    cub_train_q.add_argument("--output-root", type=Path, required=True)
    cub_train_q.add_argument("--alignment-root", type=Path, required=True)
    cub_train_q.add_argument("--model-cache-root", type=Path, required=True)
    cub_train_q.add_argument("--device", default="cuda")
    cub_train_q.add_argument("--force", action="store_true")

    mlp_decoder = subparsers.add_parser("run-cached-mlp-decoder")
    mlp_decoder.add_argument("--config", type=Path, required=True)
    mlp_decoder.add_argument("--alignment", type=Path, required=True)
    mlp_decoder.add_argument("--embedding-root", type=Path, required=True)
    mlp_decoder.add_argument("--prediction-root", type=Path, required=True)
    mlp_decoder.add_argument("--output-root", type=Path, required=True)
    mlp_decoder.add_argument("--device", default="cuda")
    mlp_decoder.add_argument("--force", action="store_true")

    environment = subparsers.add_parser("collect-env")
    environment.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = prepare_oxford(args.data_root)
    elif args.command == "validate-data":
        result = validate_oxford(args.data_root)
    elif args.command == "prepare-cub":
        result = prepare_cub(
            args.data_root,
            accepted_research_terms=args.accept_research_terms,
        )
    elif args.command == "validate-cub":
        result = validate_cub(args.data_root)
    elif args.command == "validate-config":
        result = read_config(args.config)
    elif args.command == "validate-decoder-config":
        result = read_decoder_config(args.config)
    elif args.command == "materialize-oxford-alignment":
        config = read_decoder_config(args.config)
        result = {
            "output": str(
                materialize_oxford_alignment(
                    args.upstream_embedding_prefix,
                    args.output,
                    source_model=config["source_model"],
                    target_model=config["target_model"],
                )
            )
        }
    elif args.command == "collect-env":
        result = collect_environment(args.output)
    elif args.command == "run":
        result = {
            "output": str(
                run_baseline(
                    args.config,
                    args.data_root,
                    args.embedding_root,
                    args.output_root,
                    args.model_cache_root,
                    args.device,
                    args.force,
                    args.upstream_embedding_prefix,
                )
            )
        }
    elif args.command == "run-frozen-decoder":
        result = {
            "output": str(
                run_frozen_decoder(
                    args.config,
                    args.alignment,
                    args.data_root,
                    args.embedding_root,
                    args.prediction_root,
                    args.output_root,
                    args.model_cache_root,
                    device_name=args.device,
                    force=args.force,
                )
            )
        }
    elif args.command == "analyze-attributes":
        result = {
            "output": str(
                run_attribute_analysis(
                    args.config,
                    args.embedding_root,
                    args.prediction_root,
                    args.output_root,
                )
            )
        }
    elif args.command == "cub-train-q-control":
        result = {
            "output": str(
                run_cub_train_q_control(
                    args.config,
                    args.data_root,
                    args.embedding_root,
                    args.prediction_root,
                    args.output_root,
                    args.alignment_root,
                    args.model_cache_root,
                    device_name=args.device,
                    force=args.force,
                )
            )
        }
    elif args.command == "run-cached-mlp-decoder":
        result = {
            "output": str(
                run_cached_mlp_decoder(
                    args.config,
                    args.alignment,
                    args.embedding_root,
                    args.prediction_root,
                    args.output_root,
                    device_name=args.device,
                    force=args.force,
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
