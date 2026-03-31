"""Main orchestrator: iterative self-evolution training loop."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from fa_evolve.challenger import Challenger
from fa_evolve.curriculum import compute_curriculum_distribution
from fa_evolve.data.formatting import format_training_sample
from fa_evolve.data.probe_cache import create_all_probe_sets, save_probe_metadata
from fa_evolve.evaluation import compute_reward
from fa_evolve.forgetting_detector import ForgettingDetector
from fa_evolve.utils.config import Config
from fa_evolve.utils.logging_utils import (
    init_wandb,
    load_state,
    log_round_metrics,
    save_state,
    setup_logging,
)

logger = logging.getLogger("fa_evolve")


class Orchestrator:
    """Manages the iterative self-evolution training loop."""

    def __init__(self, config: Config):
        self.config = config
        self.cluster_names = list(config.clusters.keys())

        self.detector = ForgettingDetector(cluster_names=self.cluster_names)
        self.challenger = Challenger(
            cluster_configs={
                name: {"split_train": c.split_train}
                for name, c in config.clusters.items()
            }
        )

        self.current_round = 0
        self.solver_checkpoint = config.model_name

    def run(self, resume: bool = False) -> None:
        """Run the full training loop."""
        setup_logging()

        if resume:
            self._load_checkpoint()

        if self.config.log_to_wandb:
            init_wandb(
                self.config.wandb_project,
                self.config.wandb_run_name,
                self.config.to_dict(),
            )

        # Create probe sets (only once)
        logger.info("Creating probe sets...")
        probe_sets = create_all_probe_sets(
            cluster_configs={
                name: {"split_eval": c.split_eval}
                for name, c in self.config.clusters.items()
            },
            probe_size=self.config.probe_size,
        )
        save_probe_metadata(probe_sets, self.config.probe_cache_dir)

        for round_idx in range(self.current_round, self.config.num_rounds):
            logger.info(f"Starting round {round_idx}")
            self.current_round = round_idx

            # 1. Evaluate solver on probe sets
            accuracies = self._evaluate_solver(probe_sets)
            self.detector.record_accuracies(accuracies)

            # 2. Compute forgetting scores and curriculum
            forgetting_scores = self.detector.compute_forgetting_scores()
            forgetting_urgency = self.detector.compute_forgetting_urgency()

            if self.config.use_curriculum and round_idx > 0:
                curriculum_dist = compute_curriculum_distribution(
                    forgetting_scores,
                    temperature=self.config.curriculum_temperature,
                    floor=self.config.curriculum_floor,
                )
            else:
                K = len(self.cluster_names)
                curriculum_dist = {c: 1.0 / K for c in self.cluster_names}

            # 3. Log metrics
            log_round_metrics(
                round_idx, accuracies, forgetting_scores, curriculum_dist,
                log_to_wandb=self.config.log_to_wandb,
            )

            # 4. Challenger: sample training data
            training_samples = self.challenger.sample(
                curriculum_dist, self.config.samples_per_round
            )

            # 5. Prepare training data for OpenRLHF
            round_dir = Path(self.config.output_dir) / f"round_{round_idx}"
            round_dir.mkdir(parents=True, exist_ok=True)

            lambda_eff = self.config.lambda_forgetting if (self.config.use_reward_bonus and round_idx > 0) else 0.0

            self._write_training_data(
                training_samples, forgetting_urgency, round_dir
            )
            self._write_reward_config(lambda_eff, round_dir)

            # 6. Run GRPO training
            checkpoint_dir = round_dir / "checkpoint"
            self._run_grpo_training(round_dir, checkpoint_dir)

            # 7. Update solver checkpoint
            self.solver_checkpoint = str(checkpoint_dir)

            # 8. Save state
            self._save_checkpoint()

        logger.info("Training complete!")

    def _evaluate_solver(self, probe_sets: dict) -> dict[str, float]:
        """Evaluate solver on all probe sets. Returns per-cluster accuracy."""
        logger.info(f"Evaluating solver: {self.solver_checkpoint}")
        accuracies = {}

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate.py",
                    "--model", self.solver_checkpoint,
                    "--probe_cache_dir", self.config.probe_cache_dir,
                    "--output_file", str(Path(self.config.output_dir) / "eval_results.json"),
                ],
                capture_output=True, text=True, check=True,
            )
            with open(Path(self.config.output_dir) / "eval_results.json") as f:
                accuracies = json.load(f)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Evaluation failed: {e}. Using dummy accuracies.")
            accuracies = {c: 0.0 for c in self.cluster_names}

        return accuracies

    def _write_training_data(
        self,
        samples,
        forgetting_urgency: dict[str, float],
        round_dir: Path,
    ) -> None:
        """Write training samples to JSONL for OpenRLHF."""
        output_path = round_dir / "train.jsonl"
        with open(output_path, "w") as f:
            for i, sample in enumerate(samples):
                urgency = forgetting_urgency.get(sample.domain, 0.0)
                formatted = format_training_sample(sample, urgency)

                if sample.image is not None:
                    img_dir = round_dir / "images"
                    img_dir.mkdir(exist_ok=True)
                    img_path = img_dir / f"{i:06d}.png"
                    sample.image.save(str(img_path))
                    formatted["image"] = str(img_path)
                else:
                    formatted["image"] = ""

                f.write(json.dumps(formatted) + "\n")

        logger.info(f"Wrote {len(samples)} training samples to {output_path}")

    def _write_reward_config(self, lambda_eff: float, round_dir: Path) -> None:
        """Write the reward function config for this round."""
        config = {"lambda_forgetting": lambda_eff}
        with open(round_dir / "reward_config.json", "w") as f:
            json.dump(config, f)

    def _run_grpo_training(self, round_dir: Path, checkpoint_dir: Path) -> None:
        """Launch OpenRLHF GRPO training for one round."""
        script_path = Path("scripts/run_openrlhf.sh")
        if not script_path.exists():
            logger.warning("run_openrlhf.sh not found. Skipping GRPO training.")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return

        cmd = [
            "bash", str(script_path),
            "--pretrain", self.solver_checkpoint,
            "--dataset", str(round_dir / "train.jsonl"),
            "--reward_fn", "src/fa_evolve/reward.py",
            "--reward_config", str(round_dir / "reward_config.json"),
            "--save_path", str(checkpoint_dir),
            "--n_samples_per_prompt", str(self.config.grpo_group_size),
            "--learning_rate", str(self.config.learning_rate),
            "--kl_coeff", str(self.config.kl_coeff),
            "--max_new_tokens", str(self.config.max_new_tokens),
        ]

        logger.info(f"Running GRPO training: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"GRPO training failed: {e}")
            raise

    def _save_checkpoint(self) -> None:
        """Save orchestrator state."""
        state = {
            "current_round": self.current_round + 1,
            "solver_checkpoint": self.solver_checkpoint,
            "detector_state": self.detector.state_dict(),
        }
        save_state(state, self.config.output_dir)

    def _load_checkpoint(self) -> None:
        """Resume from saved state."""
        state = load_state(self.config.output_dir)
        if state is None:
            logger.info("No checkpoint found, starting fresh.")
            return
        self.current_round = state["current_round"]
        self.solver_checkpoint = state["solver_checkpoint"]
        self.detector = ForgettingDetector.from_state_dict(state["detector_state"])
        logger.info(f"Resumed from round {self.current_round}")
