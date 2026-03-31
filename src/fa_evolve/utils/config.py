"""YAML config loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ClusterConfig:
    dataset_id: str
    split_eval: str
    split_train: str
    metric: str


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_dtype: str = "bfloat16"

    # Training loop
    num_rounds: int = 10
    samples_per_round: int = 5000

    # GRPO
    grpo_group_size: int = 8
    learning_rate: float = 1e-6
    kl_coeff: float = 0.01
    max_new_tokens: int = 512

    # Forgetting detection
    probe_size: int = 200

    # Curriculum
    curriculum_temperature: float = 1.0
    curriculum_floor: float = 0.05

    # Reward
    lambda_forgetting: float = 0.1

    # Ablation
    use_curriculum: bool = True
    use_reward_bonus: bool = True

    # Paths
    output_dir: str = "outputs"
    probe_cache_dir: str = "outputs/probe_cache"

    # Logging
    wandb_project: str = "fa-evolve"
    wandb_run_name: str | None = None
    log_to_wandb: bool = True

    # Clusters
    clusters: dict[str, ClusterConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        cluster_raw = raw.pop("clusters", {})
        clusters = {
            name: ClusterConfig(**cfg) for name, cfg in cluster_raw.items()
        }

        config = cls(**raw, clusters=clusters)
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / checkpointing."""
        d = {k: v for k, v in self.__dict__.items() if k != "clusters"}
        d["clusters"] = {
            name: {
                "dataset_id": c.dataset_id,
                "split_eval": c.split_eval,
                "split_train": c.split_train,
                "metric": c.metric,
            }
            for name, c in self.clusters.items()
        }
        return d
