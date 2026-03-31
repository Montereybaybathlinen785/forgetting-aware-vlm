"""Forgetting detection via probe set evaluation."""

from __future__ import annotations


class ForgettingDetector:
    """Tracks per-cluster accuracy and computes forgetting scores.

    Forgetting score for cluster c:
        forgetting_score[c] = max(0, peak_acc[c] - current_acc[c])

    Forgetting urgency (normalized to [0, 1]):
        urgency[c] = forgetting_score[c] / (max(all_scores) + eps)
    """

    def __init__(self, cluster_names: list[str], eps: float = 1e-8):
        self.cluster_names = list(cluster_names)
        self.eps = eps
        self.peak_accuracies: dict[str, float] = {c: 0.0 for c in cluster_names}
        self.accuracy_history: dict[str, list[float]] = {c: [] for c in cluster_names}
        self._current_accuracies: dict[str, float] = {}

    def record_accuracies(self, accuracies: dict[str, float]) -> None:
        """Record evaluation results for one round and update peaks."""
        for c in self.cluster_names:
            acc = accuracies[c]
            self.accuracy_history[c].append(acc)
            self.peak_accuracies[c] = max(self.peak_accuracies[c], acc)
        self._current_accuracies = dict(accuracies)

    def compute_forgetting_scores(self) -> dict[str, float]:
        """Compute forgetting score per cluster: max(0, peak - current)."""
        return {
            c: max(0.0, self.peak_accuracies[c] - self._current_accuracies.get(c, 0.0))
            for c in self.cluster_names
        }

    def compute_forgetting_urgency(self) -> dict[str, float]:
        """Compute normalized forgetting urgency in [0, 1]."""
        scores = self.compute_forgetting_scores()
        max_score = max(scores.values())
        return {c: s / (max_score + self.eps) for c, s in scores.items()}

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "cluster_names": self.cluster_names,
            "peak_accuracies": self.peak_accuracies,
            "accuracy_history": self.accuracy_history,
            "current_accuracies": self._current_accuracies,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ForgettingDetector:
        """Restore from checkpoint."""
        detector = cls(cluster_names=state["cluster_names"])
        detector.peak_accuracies = state["peak_accuracies"]
        detector.accuracy_history = state["accuracy_history"]
        detector._current_accuracies = state["current_accuracies"]
        return detector
