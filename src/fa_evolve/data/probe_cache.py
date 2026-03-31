"""Create and cache fixed probe sets for forgetting detection."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from fa_evolve.data.loader import VQASample, load_cluster_data


def create_probe_set(
    cluster_name: str,
    eval_split: str,
    probe_size: int,
    seed: int = 42,
) -> list[VQASample]:
    """Sample a fixed probe set from a cluster's eval split."""
    all_samples = load_cluster_data(cluster_name, eval_split)
    rng = random.Random(seed)
    if len(all_samples) <= probe_size:
        return all_samples
    return rng.sample(all_samples, probe_size)


def create_all_probe_sets(
    cluster_configs: dict,
    probe_size: int,
    seed: int = 42,
) -> dict[str, list[VQASample]]:
    """Create probe sets for all clusters.

    Args:
        cluster_configs: {cluster_name: {"split_eval": str, ...}}
        probe_size: Number of samples per cluster.
        seed: Random seed for reproducibility.

    Returns:
        {cluster_name: [VQASample, ...]}
    """
    probe_sets = {}
    for name, config in cluster_configs.items():
        probe_sets[name] = create_probe_set(
            cluster_name=name,
            eval_split=config["split_eval"],
            probe_size=probe_size,
            seed=seed,
        )
    return probe_sets


def save_probe_metadata(probe_sets: dict[str, list[VQASample]], cache_dir: str) -> None:
    """Save probe set metadata (questions + answers, not images) for reproducibility."""
    os.makedirs(cache_dir, exist_ok=True)
    for name, samples in probe_sets.items():
        metadata = [
            {"question": s.question, "answer": s.answer, "domain": s.domain, "metric": s.metric}
            for s in samples
        ]
        path = Path(cache_dir) / f"{name}_probes.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
