#!/usr/bin/env python
"""Pre-cache Hugging Face datasets referenced by an fev benchmark YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets as hf_datasets
import yaml


DEFAULT_BENCHMARK_YAML = Path(
    "chronos2/artifacts/chronos2_spectral_workflow/official_fev_benchmark_main.yaml"
)
DEFAULT_DATASET_REPO = "autogluon/fev_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cache the Hugging Face dataset configs referenced by an "
            "official fev benchmark YAML."
        )
    )
    parser.add_argument(
        "--benchmark-yaml",
        type=Path,
        default=DEFAULT_BENCHMARK_YAML,
        help=f"Path to the fev benchmark YAML (default: {DEFAULT_BENCHMARK_YAML})",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Only inspect the first N benchmark tasks. Defaults to all tasks.",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help=(
            "Only cache tasks whose dataset_path matches this repo "
            f"(default: {DEFAULT_DATASET_REPO})."
        ),
    )
    parser.add_argument(
        "--verbosity",
        choices=("warning", "info", "debug"),
        default="warning",
        help="datasets library log verbosity.",
    )
    return parser.parse_args()


def load_tasks(benchmark_yaml: Path) -> list[dict]:
    payload = yaml.safe_load(benchmark_yaml.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"Expected 'tasks' to be a list in {benchmark_yaml}")
    return tasks


def collect_dataset_configs(
    tasks: list[dict], dataset_repo: str, task_limit: int | None
) -> list[str]:
    if task_limit is not None:
        tasks = tasks[:task_limit]

    configs: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task.get("dataset_path") != dataset_repo:
            continue
        dataset_config = task.get("dataset_config")
        if not dataset_config or dataset_config in seen:
            continue
        seen.add(dataset_config)
        configs.append(dataset_config)
    return configs


def cache_dataset_configs(dataset_repo: str, configs: list[str]) -> tuple[int, list[str]]:
    cached = 0
    failures: list[str] = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] Caching {dataset_repo}:{config}")
        try:
            hf_datasets.load_dataset(
                path=dataset_repo,
                name=config,
                trust_remote_code=True,
            )
        except Exception as exc:  
            failures.append(config)
            print(f"  FAILED: {exc}")
        else:
            cached += 1
    return cached, failures


def main() -> int:
    args = parse_args()
    benchmark_yaml = args.benchmark_yaml.resolve()
    if not benchmark_yaml.exists():
        raise FileNotFoundError(f"Benchmark YAML not found: {benchmark_yaml}")

    getattr(hf_datasets.logging, f"set_verbosity_{args.verbosity}")()

    tasks = load_tasks(benchmark_yaml)
    configs = collect_dataset_configs(tasks, args.dataset_repo, args.task_limit)
    if not configs:
        print(
            "No matching dataset configs found in "
            f"{benchmark_yaml} for dataset repo {args.dataset_repo}."
        )
        return 0

    print(f"Benchmark YAML: {benchmark_yaml}")
    print(f"Tasks inspected: {min(len(tasks), args.task_limit) if args.task_limit else len(tasks)}")
    print(f"Unique dataset configs to cache: {len(configs)}")

    cached, failures = cache_dataset_configs(args.dataset_repo, configs)
    print(f"Cached {cached}/{len(configs)} dataset configs.")
    if failures:
        print(f"Failed configs ({len(failures)}): {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
