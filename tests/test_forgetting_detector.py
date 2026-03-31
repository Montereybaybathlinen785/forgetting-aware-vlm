import pytest
from fa_evolve.forgetting_detector import ForgettingDetector


class TestForgettingScores:
    def test_no_forgetting_on_first_eval(self):
        detector = ForgettingDetector(cluster_names=["math", "docs"])
        detector.record_accuracies({"math": 0.7, "docs": 0.5})
        scores = detector.compute_forgetting_scores()
        assert scores["math"] == 0.0
        assert scores["docs"] == 0.0

    def test_forgetting_detected(self):
        detector = ForgettingDetector(cluster_names=["math", "docs"])
        detector.record_accuracies({"math": 0.8, "docs": 0.6})
        detector.record_accuracies({"math": 0.5, "docs": 0.7})
        scores = detector.compute_forgetting_scores()
        assert abs(scores["math"] - 0.3) < 1e-6
        assert scores["docs"] == 0.0

    def test_peak_tracking(self):
        detector = ForgettingDetector(cluster_names=["math"])
        detector.record_accuracies({"math": 0.5})
        detector.record_accuracies({"math": 0.8})
        detector.record_accuracies({"math": 0.6})
        scores = detector.compute_forgetting_scores()
        assert abs(scores["math"] - 0.2) < 1e-6


class TestForgettingUrgency:
    def test_normalization(self):
        detector = ForgettingDetector(cluster_names=["math", "docs", "charts"])
        detector.record_accuracies({"math": 0.9, "docs": 0.8, "charts": 0.7})
        detector.record_accuracies({"math": 0.5, "docs": 0.7, "charts": 0.7})
        urgency = detector.compute_forgetting_urgency()
        assert abs(urgency["math"] - 1.0) < 1e-6
        assert abs(urgency["docs"] - 0.25) < 1e-6
        assert abs(urgency["charts"] - 0.0) < 1e-2

    def test_no_forgetting_returns_zeros(self):
        detector = ForgettingDetector(cluster_names=["math", "docs"])
        detector.record_accuracies({"math": 0.5, "docs": 0.5})
        urgency = detector.compute_forgetting_urgency()
        assert urgency["math"] < 1e-6
        assert urgency["docs"] < 1e-6


class TestAccuracyHistory:
    def test_history_tracked(self):
        detector = ForgettingDetector(cluster_names=["math"])
        detector.record_accuracies({"math": 0.5})
        detector.record_accuracies({"math": 0.7})
        assert detector.accuracy_history["math"] == [0.5, 0.7]

    def test_state_serialization(self):
        detector = ForgettingDetector(cluster_names=["math", "docs"])
        detector.record_accuracies({"math": 0.8, "docs": 0.6})
        state = detector.state_dict()
        detector2 = ForgettingDetector.from_state_dict(state)
        assert detector2.peak_accuracies == detector.peak_accuracies
        assert detector2.accuracy_history == detector.accuracy_history
