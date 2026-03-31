#!/usr/bin/env python3
"""Entry point for the forgetting-aware VLM self-evolution pipeline."""

import argparse

from fa_evolve.orchestrator import Orchestrator
from fa_evolve.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Forgetting-Aware VLM Self-Evolution")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
        config.probe_cache_dir = f"{args.output_dir}/probe_cache"

    orchestrator = Orchestrator(config)
    orchestrator.run(resume=args.resume)


if __name__ == "__main__":
    main()
