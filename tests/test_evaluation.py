import pytest
from fa_evolve.evaluation import normalize_answer, parse_number, compute_reward


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_strip_articles(self):
        assert normalize_answer("the cat") == "cat"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_strip_punctuation(self):
        assert normalize_answer("hello!") == "hello"
        assert normalize_answer("yes.") == "yes"

    def test_strip_whitespace(self):
        assert normalize_answer("  hello  world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("  The Answer is: YES! ") == "answer is yes"


class TestParseNumber:
    def test_integer(self):
        assert parse_number("42") == 42.0

    def test_float(self):
        assert parse_number("3.14") == 3.14

    def test_percentage(self):
        assert parse_number("75%") == 75.0

    def test_comma_separated(self):
        assert parse_number("1,234") == 1234.0

    def test_not_a_number(self):
        assert parse_number("hello") is None

    def test_negative(self):
        assert parse_number("-5.5") == -5.5


class TestMathVista:
    def test_exact_text_match(self):
        assert compute_reward("B", "B", "mathvista") == 1.0

    def test_numeric_match_within_tolerance(self):
        assert compute_reward("3.01", "3.0", "mathvista") == 1.0

    def test_numeric_mismatch(self):
        assert compute_reward("5.0", "3.0", "mathvista") == 0.0

    def test_text_mismatch(self):
        assert compute_reward("A", "B", "mathvista") == 0.0


class TestANLS:
    def test_exact_match(self):
        assert compute_reward("hello", "hello", "anls") == 1.0

    def test_close_match(self):
        reward = compute_reward("helo", "hello", "anls")
        assert 0.7 < reward < 0.9

    def test_distant_match(self):
        assert compute_reward("xyz", "hello", "anls") == 0.0

    def test_both_empty(self):
        assert compute_reward("", "", "anls") == 1.0

    def test_one_empty(self):
        assert compute_reward("", "hello", "anls") == 0.0


class TestChartQA:
    def test_exact_text_match(self):
        assert compute_reward("yes", "yes", "chartqa") == 1.0

    def test_numeric_within_5_percent(self):
        assert compute_reward("105", "100", "chartqa") == 1.0

    def test_numeric_outside_5_percent(self):
        assert compute_reward("110", "100", "chartqa") == 0.0


class TestExactMatch:
    def test_match(self):
        assert compute_reward("yes", "yes", "exact_match") == 1.0

    def test_case_insensitive(self):
        assert compute_reward("Yes", "yes", "exact_match") == 1.0

    def test_mismatch(self):
        assert compute_reward("no", "yes", "exact_match") == 0.0


class TestMCMatch:
    def test_correct(self):
        assert compute_reward("A", "A", "mc_match") == 1.0

    def test_incorrect(self):
        assert compute_reward("B", "A", "mc_match") == 0.0

    def test_case_insensitive(self):
        assert compute_reward("a", "A", "mc_match") == 1.0


class TestVQASoft:
    def test_unanimous(self):
        gt = ["cat"] * 10
        assert compute_reward("cat", gt, "vqa_soft") == 1.0

    def test_majority(self):
        gt = ["cat"] * 6 + ["dog"] * 4
        assert compute_reward("cat", gt, "vqa_soft") == 1.0

    def test_minority(self):
        gt = ["cat"] * 2 + ["dog"] * 8
        reward = compute_reward("cat", gt, "vqa_soft")
        assert abs(reward - 2 / 3) < 0.01

    def test_no_match(self):
        gt = ["dog"] * 10
        assert compute_reward("cat", gt, "vqa_soft") == 0.0


class TestUnknownMetric:
    def test_raises(self):
        with pytest.raises(ValueError):
            compute_reward("a", "b", "unknown_metric")
