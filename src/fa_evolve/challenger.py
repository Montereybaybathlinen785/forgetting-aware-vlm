"""Challenger: samples training tasks from dataset pool per curriculum distribution."""

from __future__ import annotations

import random

from fa_evolve.data.loader import VQASample, load_cluster_data


class Challenger:
    """Samples training problems according to the curriculum distribution."""

    def __init__(self, cluster_configs: dict, seed: int = 42):
        """
        Args:
            cluster_configs: {cluster_name: {"split_train": str, ...}}
            seed: Random seed.
        """
        self.cluster_configs = cluster_configs
        self.rng = random.Random(seed)
        self._cache: dict[str, list[VQASample]] = {}

    def _get_pool(self, cluster_name: str) -> list[VQASample]:
        """Lazily load and cache training data for a cluster."""
        if cluster_name not in self._cache:
            split = self.cluster_configs[cluster_name]["split_train"]
            self._cache[cluster_name] = load_cluster_data(cluster_name, split)
        return self._cache[cluster_name]

    def sample(
        self,
        curriculum_distribution: dict[str, float],
        n_samples: int,
    ) -> list[VQASample]:
        """Sample training data according to the curriculum distribution.

        Args:
            curriculum_distribution: {cluster_name: probability} summing to ~1.0.
            n_samples: Total number of samples to return.

        Returns:
            List of VQASamples drawn from clusters proportional to the distribution.
        """
        samples = []
        for cluster_name, prob in curriculum_distribution.items():
            n_cluster = round(prob * n_samples)
            if n_cluster == 0:
                continue
            pool = self._get_pool(cluster_name)
            if len(pool) == 0:
                continue
            drawn = self.rng.choices(pool, k=n_cluster)
            samples.extend(drawn)

        self.rng.shuffle(samples)
        return samples
