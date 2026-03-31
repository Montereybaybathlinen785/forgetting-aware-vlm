"""Prompt formatting for Qwen2.5-VL."""

from __future__ import annotations

import json

from fa_evolve.data.loader import VQASample


def format_prompt(sample: VQASample) -> str:
    """Format a VQA sample as a text prompt for Qwen2.5-VL.

    The image is handled separately by the model's processor.
    This returns the text portion of the prompt.
    """
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer the question about the image. "
        "Provide your final answer on the last line.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{sample.question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def format_training_sample(sample: VQASample, forgetting_urgency: float = 0.0) -> dict:
    """Format a VQA sample for OpenRLHF training JSONL.

    Returns a dict with keys: prompt, image, label (JSON string).
    Note: image must be set separately since PIL images need to be saved to disk.
    """
    label = json.dumps({
        "answer": sample.answer,  # str for most metrics, list[str] for vqa_soft
        "domain": sample.domain,
        "metric": sample.metric,
        "forgetting_urgency": forgetting_urgency,
    })

    return {
        "prompt": format_prompt(sample),
        "label": label,
    }
