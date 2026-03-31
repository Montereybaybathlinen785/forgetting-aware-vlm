import pytest
import json
import torch
from fa_evolve.reward import reward_func, extract_answer


class TestExtractAnswer:
    def test_boxed_format(self):
        response = "The answer is \\boxed{42}"
        assert extract_answer(response) == "42"

    def test_answer_is_pattern(self):
        response = "Let me think.\nStep 1: ...\nThe answer is B"
        assert extract_answer(response) == "B"

    def test_last_line_fallback(self):
        response = "Let me think.\nStep 1: ...\nB"
        assert extract_answer(response) == "B"

    def test_empty(self):
        assert extract_answer("") == ""


class TestRewardFunc:
    def test_correct_answer_no_forgetting(self):
        queries = ["Q: What is 2+2?\nA: 4"]
        prompts = ["Q: What is 2+2?\nA: "]
        labels = [json.dumps({"answer": "4", "domain": "math", "metric": "exact_match", "forgetting_urgency": 0.0})]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.1)
        assert result["rewards"][0].item() == 1.0

    def test_correct_answer_with_forgetting(self):
        queries = ["Q: What is 2+2?\nA: 4"]
        prompts = ["Q: What is 2+2?\nA: "]
        labels = [json.dumps({"answer": "4", "domain": "math", "metric": "exact_match", "forgetting_urgency": 0.5})]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.1)
        expected = 1.0 + 0.1 * 0.5
        assert abs(result["rewards"][0].item() - expected) < 1e-6

    def test_wrong_answer_with_forgetting(self):
        queries = ["Q: What is 2+2?\nA: 5"]
        prompts = ["Q: What is 2+2?\nA: "]
        labels = [json.dumps({"answer": "4", "domain": "math", "metric": "exact_match", "forgetting_urgency": 1.0})]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.1)
        expected = 0.0 + 0.1 * 1.0
        assert abs(result["rewards"][0].item() - expected) < 1e-6

    def test_batch(self):
        queries = ["Q1\nA: 4", "Q2\nA: wrong"]
        prompts = ["Q1\nA: ", "Q2\nA: "]
        labels = [
            json.dumps({"answer": "4", "domain": "math", "metric": "exact_match", "forgetting_urgency": 0.0}),
            json.dumps({"answer": "right", "domain": "docs", "metric": "exact_match", "forgetting_urgency": 0.0}),
        ]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.1)
        assert result["rewards"].shape == (2,)
        assert result["rewards"][0].item() == 1.0
        assert result["rewards"][1].item() == 0.0

    def test_returns_extra_logs(self):
        queries = ["Q\nA: 4"]
        prompts = ["Q\nA: "]
        labels = [json.dumps({"answer": "4", "domain": "math", "metric": "exact_match", "forgetting_urgency": 0.0})]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.1)
        assert "extra_logs" in result
        assert "accuracy" in result["extra_logs"]

    def test_vqa_soft_with_list_answer(self):
        queries = ["Q\nA: cat"]
        prompts = ["Q\nA: "]
        gt = ["cat"] * 6 + ["dog"] * 4
        labels = [json.dumps({"answer": gt, "domain": "natural", "metric": "vqa_soft", "forgetting_urgency": 0.0})]
        result = reward_func(queries, prompts, labels, lambda_forgetting=0.0)
        assert result["rewards"][0].item() == 1.0  # min(6/3, 1) = 1.0
