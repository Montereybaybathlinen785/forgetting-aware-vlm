"""Answer verification for VQA benchmarks."""

from __future__ import annotations

import re
import string

from Levenshtein import distance as levenshtein_distance


def normalize_answer(answer: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = " ".join(answer.split())
    return answer


def parse_number(text: str) -> float | None:
    """Try to parse a number from text. Returns None if not a number."""
    text = text.strip().replace(",", "").rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _mathvista_match(pred: str, gt: str) -> float:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if pred_norm == gt_norm:
        return 1.0
    pred_num = parse_number(pred)
    gt_num = parse_number(gt)
    if pred_num is not None and gt_num is not None:
        denom = max(abs(gt_num), 1.0)
        if abs(pred_num - gt_num) / denom < 0.01:
            return 1.0
    return 0.0


def _anls(pred: str, gt: str) -> float:
    pred_norm = pred.lower().strip()
    gt_norm = gt.lower().strip()
    if not gt_norm and not pred_norm:
        return 1.0
    if not gt_norm or not pred_norm:
        return 0.0
    max_len = max(len(pred_norm), len(gt_norm))
    nld = levenshtein_distance(pred_norm, gt_norm) / max_len
    return 1.0 - nld if nld < 0.5 else 0.0


def _chartqa_match(pred: str, gt: str) -> float:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if pred_norm == gt_norm:
        return 1.0
    pred_num = parse_number(pred)
    gt_num = parse_number(gt)
    if pred_num is not None and gt_num is not None:
        denom = max(abs(gt_num), 1.0)
        if abs(pred_num - gt_num) / denom <= 0.05:
            return 1.0
    return 0.0


def _exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def _mc_match(pred: str, gt: str) -> float:
    return 1.0 if pred.strip().upper() == gt.strip().upper() else 0.0


def _vqa_soft_accuracy(pred: str, gt_answers: list[str]) -> float:
    pred_norm = normalize_answer(pred)
    count = sum(1 for a in gt_answers if normalize_answer(a) == pred_norm)
    return min(count / 3.0, 1.0)


_METRICS = {
    "mathvista": _mathvista_match,
    "anls": _anls,
    "chartqa": _chartqa_match,
    "exact_match": _exact_match,
    "mc_match": _mc_match,
    "vqa_soft": _vqa_soft_accuracy,
}


def compute_reward(prediction: str, ground_truth, metric_name: str) -> float:
    """Compute reward for a single prediction.

    Args:
        prediction: Model's predicted answer string.
        ground_truth: Ground truth answer (str, or list[str] for vqa_soft).
        metric_name: One of the registered metric names.

    Returns:
        Reward value in [0, 1].
    """
    if metric_name not in _METRICS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(_METRICS.keys())}")
    return _METRICS[metric_name](prediction, ground_truth)
