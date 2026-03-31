"""Forgetting-aware curriculum scheduling."""

from __future__ import annotations

import math


def compute_curriculum_distribution(
    forgetting_scores: dict[str, float],
    temperature: float = 1.0,
    floor: float = 0.05,
) -> dict[str, float]:
    """Compute sampling distribution biased toward forgotten skill clusters.

    Formula: p_i = floor + (1 - K*floor) * softmax(forgetting_scores / τ)

    Args:
        forgetting_scores: {cluster_name: forgetting_score} where score >= 0.
        temperature: Softmax temperature. Lower = more concentrated on forgotten clusters.
        floor: Minimum probability per cluster. Must satisfy K * floor < 1.

    Returns:
        {cluster_name: probability} summing to 1.0.
    """
    K = len(forgetting_scores)
    if K * floor >= 1.0:
        raise ValueError(
            f"K * floor must be < 1.0, got K={K}, floor={floor}, product={K * floor}"
        )

    clusters = list(forgetting_scores.keys())
    scaled = [forgetting_scores[c] / temperature for c in clusters]
    max_scaled = max(scaled)  # numerical stability
    exps = [math.exp(s - max_scaled) for s in scaled]
    exp_sum = sum(exps)
    softmax_probs = [e / exp_sum for e in exps]

    remaining = 1.0 - K * floor
    distribution = {}
    for i, c in enumerate(clusters):
        distribution[c] = floor + remaining * softmax_probs[i]

    return distribution
