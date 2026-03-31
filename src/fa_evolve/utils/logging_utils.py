"""Logging utilities for training metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("fa_evolve")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def init_wandb(project: str, run_name: str | None, config: dict) -> None:
    """Initialize W&B run."""
    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")


def log_round_metrics(
    round_idx: int,
    accuracies: dict[str, float],
    forgetting_scores: dict[str, float],
    curriculum_dist: dict[str, float],
    log_to_wandb: bool = True,
) -> None:
    """Log metrics for a training round."""
    logger.info(f"=== Round {round_idx} ===")
    for c in accuracies:
        logger.info(
            f"  {c}: acc={accuracies[c]:.3f} "
            f"forget={forgetting_scores.get(c, 0):.3f} "
            f"p={curriculum_dist.get(c, 0):.3f}"
        )

    if log_to_wandb:
        try:
            import wandb
            metrics = {"round": round_idx}
            for c in accuracies:
                metrics[f"accuracy/{c}"] = accuracies[c]
                metrics[f"forgetting/{c}"] = forgetting_scores.get(c, 0)
                metrics[f"curriculum/{c}"] = curriculum_dist.get(c, 0)
            metrics["accuracy/mean"] = sum(accuracies.values()) / len(accuracies)
            wandb.log(metrics, step=round_idx)
        except ImportError:
            pass


def save_state(state: dict, output_dir: str) -> None:
    """Save orchestrator state to JSON."""
    path = Path(output_dir) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(output_dir: str) -> dict | None:
    """Load orchestrator state from JSON, or None if not found."""
    path = Path(output_dir) / "state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
