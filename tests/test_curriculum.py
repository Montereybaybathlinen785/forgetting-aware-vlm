import pytest
from fa_evolve.curriculum import compute_curriculum_distribution


class TestCurriculumDistribution:
    def test_uniform_when_no_forgetting(self):
        """All forgetting scores 0 -> uniform distribution."""
        scores = {"math": 0.0, "docs": 0.0, "charts": 0.0}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.05)
        for p in dist.values():
            assert abs(p - 1 / 3) < 0.01

    def test_sums_to_one(self):
        scores = {"math": 0.3, "docs": 0.0, "charts": 0.1, "spatial": 0.0, "science": 0.2, "natural": 0.05}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.05)
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_floor_respected(self):
        """Even the cluster with 0 forgetting gets at least floor probability."""
        scores = {"math": 1.0, "docs": 0.0}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.1)
        assert dist["docs"] >= 0.1 - 1e-6

    def test_higher_forgetting_gets_higher_prob(self):
        scores = {"math": 0.5, "docs": 0.1}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.05)
        assert dist["math"] > dist["docs"]

    def test_low_temperature_concentrates(self):
        """Low tau should make distribution more peaked."""
        scores = {"math": 0.5, "docs": 0.1}
        dist_low = compute_curriculum_distribution(scores, temperature=0.1, floor=0.05)
        dist_high = compute_curriculum_distribution(scores, temperature=10.0, floor=0.05)
        assert dist_low["math"] - dist_low["docs"] > dist_high["math"] - dist_high["docs"]

    def test_invariant_k_floor_less_than_one(self):
        """Should raise if K * floor >= 1."""
        scores = {"a": 0.1, "b": 0.2}
        with pytest.raises(ValueError):
            compute_curriculum_distribution(scores, temperature=1.0, floor=0.6)

    def test_keys_preserved(self):
        scores = {"math": 0.3, "docs": 0.1}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.05)
        assert set(dist.keys()) == {"math", "docs"}

    def test_single_cluster(self):
        scores = {"math": 0.5}
        dist = compute_curriculum_distribution(scores, temperature=1.0, floor=0.05)
        assert abs(dist["math"] - 1.0) < 1e-6
