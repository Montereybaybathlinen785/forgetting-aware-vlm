"""OpenRLHF-compatible reward function with forgetting-urgency bonus."""

from __future__ import annotations

import json
import re

import torch

from fa_evolve.evaluation import compute_reward as compute_accuracy_reward


def extract_answer(response: str) -> str:
    """Extract final answer from model response.

    Tries \\boxed{...} first, then "the answer is X" pattern, then last non-empty line.
    """
    # Try \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip()
    # Try "the answer is X" pattern (case-insensitive)
    match = re.search(r"(?:the\s+)?answer\s+is\s+(.+?)\.?\s*$", response, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fall back to last non-empty line
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]
    return ""


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    lambda_forgetting: float = 0.1,
    **kwargs,
) -> dict:
    """Compute rewards for a batch of model responses.

    Each label is a JSON string with keys:
        - answer: ground truth answer (str or list[str] for vqa_soft)
        - domain: skill cluster name
        - metric: metric name for evaluation
        - forgetting_urgency: normalized urgency score [0, 1]

    Returns:
        {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}
    """
    rewards = []
    accuracies = []

    for query, prompt, label_str in zip(queries, prompts, labels):
        response = query[len(prompt):]
        prediction = extract_answer(response)

        label = json.loads(label_str)
        gt_answer = label["answer"]
        metric = label["metric"]
        urgency = label.get("forgetting_urgency", 0.0)

        acc = compute_accuracy_reward(prediction, gt_answer, metric)
        accuracies.append(acc)

        total = acc + lambda_forgetting * urgency
        rewards.append(total)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    # scores is used by OpenRLHF for dynamic sample filtering (0-1 range).
    # rewards (unclamped) is used for advantage computation — the forgetting bonus is preserved.
    scores_tensor = torch.clamp(rewards_tensor, 0.0, 1.0)

    return {
        "rewards": rewards_tensor,
        "scores": scores_tensor,
        "extra_logs": {
            "accuracy": sum(accuracies) / max(len(accuracies), 1),
        },
    }
