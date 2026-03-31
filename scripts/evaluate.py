#!/usr/bin/env python3
"""Standalone evaluation of a VLM on probe sets."""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fa_evolve.data.probe_cache import create_all_probe_sets
from fa_evolve.evaluation import compute_reward
from fa_evolve.reward import extract_answer
from fa_evolve.utils.config import Config

logger = logging.getLogger("fa_evolve.eval")


def evaluate_model(model_path: str, probe_cache_dir: str, output_file: str, config_path: str = "configs/default.yaml"):
    """Evaluate model on all probe sets and write per-cluster accuracy."""
    config = Config.from_yaml(config_path)

    logger.info(f"Loading model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    probe_sets = create_all_probe_sets(
        cluster_configs={
            name: {"split_eval": c.split_eval}
            for name, c in config.clusters.items()
        },
        probe_size=config.probe_size,
    )

    accuracies = {}

    for cluster_name, samples in probe_sets.items():
        correct = 0
        total = 0
        metric = config.clusters[cluster_name].metric

        for sample in samples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample.image},
                        {"type": "text", "text": sample.question},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[sample.image],
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512)

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True)

            prediction = extract_answer(response)
            reward = compute_reward(prediction, sample.answer, metric)
            correct += reward
            total += 1

        accuracies[cluster_name] = correct / max(total, 1)
        logger.info(f"  {cluster_name}: {accuracies[cluster_name]:.3f} ({int(correct)}/{total})")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(accuracies, f, indent=2)

    logger.info(f"Results written to {output_file}")
    return accuracies


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--probe_cache_dir", type=str, default="outputs/probe_cache")
    parser.add_argument("--output_file", type=str, default="outputs/eval_results.json")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    evaluate_model(args.model, args.probe_cache_dir, args.output_file, args.config)


if __name__ == "__main__":
    main()
