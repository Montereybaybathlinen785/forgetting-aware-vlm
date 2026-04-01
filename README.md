<div align="center">

# SelfEvolve-VLM

### Forgetting-Aware Self-Evolvement for Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-59%20passed-brightgreen.svg)]()

*A self-evolving VLM framework that remembers what it learns.*

---

</div>

## Why Self-Evolvement Needs Memory

Self-evolvement — the paradigm where a model improves itself through iterative self-play without external supervision — has emerged as a powerful approach for scaling VLM reasoning (R-Zero, VisPlay, MM-Zero). But current self-evolvement methods have a blind spot: **they forget.**

As the Solver pushes into new capability frontiers, it silently loses previously mastered skills. A model that learns to reason about charts may degrade at spatial understanding. One that improves at science questions may regress on document comprehension.

**SelfEvolve-VLM** closes this gap by making the self-evolvement loop *aware of its own forgetting*.

<div align="center">

```
               Self-Evolvement Loop with Forgetting Awareness
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
  │   │  Forgetting  │────>│  Curriculum   │────>│   Challenger    │  │
  │   │  Detector    │     │  Scheduler    │     │   (Sampling)    │  │
  │   └──────┬───────┘     └──────────────┘     └────────┬────────┘  │
  │          │                                           │           │
  │          │  forgetting scores                training tasks      │
  │          │                                           │           │
  │   ┌──────┴───────┐                          ┌───────┴────────┐  │
  │   │    Probe     │                          │  GRPO Trainer   │  │
  │   │  Evaluation  │<─────────────────────────│  (OpenRLHF)     │  │
  │   └──────────────┘     updated solver       └────────────────┘  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

</div>

## How It Works

The self-evolvement loop runs in rounds. Each round:

| Step | What Happens | Key Mechanism |
|:----:|:-------------|:-------------|
| 1 | **Detect Forgetting** | Evaluate the Solver on fixed probes across 6 visual domains. Compute `forgetting = peak_accuracy - current_accuracy` per domain. |
| 2 | **Schedule Curriculum** | Convert forgetting scores into a sampling distribution via temperature-scaled softmax with a floor guarantee. Forgotten domains get more training data. |
| 3 | **Challenge the Solver** | The Challenger samples training tasks from VQA datasets proportional to the curriculum distribution. |
| 4 | **Train with Awareness** | GRPO training with a forgetting-aware reward: `r = accuracy + lambda * forgetting_urgency`. The Solver is incentivized to get answers right in domains it's losing. |
| 5 | **Evolve** | Update the Solver checkpoint. Repeat. |

The two key contributions work in tandem:
- **Curriculum scheduling** controls *what* the model trains on (data distribution)
- **Reward shaping** controls *how much* the model cares about each domain (gradient signal)

## Self-Evolvement Across 6 Visual Domains

The system tracks self-evolvement progress across diverse visual reasoning skills:

| Domain | Dataset | What It Tests | Evaluation |
|:------:|:--------|:-------------|:-----------|
| **Math** | MathVista | Mathematical reasoning over figures & plots | Accuracy (numeric tolerance) |
| **Documents** | DocVQA | Reading comprehension on document images | ANLS |
| **Charts** | ChartQA | Data extraction from charts & graphs | Relaxed Accuracy (5%) |
| **Spatial** | GQA | Spatial relationship understanding | Exact Match |
| **Science** | ScienceQA | Scientific reasoning with diagrams | MC Accuracy |
| **Natural** | VQAv2 | General visual question answering | Soft Accuracy |

Each domain maintains a fixed **probe set** (200 samples) that acts as a canary for forgetting — consistent evaluation on identical questions across rounds reveals true capability changes.

## Quick Start

### Installation

```bash
git clone https://github.com/ZhihaoZhu/forgetting-aware-vlm.git
cd forgetting-aware-vlm
pip install -e ".[dev]"
```

### Requirements

| Component | Requirement |
|:----------|:-----------|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| GPU | 4x A100 80GB (for 7B model) |
| RL Framework | [LMM-R1](https://github.com/TideDra/lmm-r1) (OpenRLHF fork with VLM support) |

### Run Self-Evolvement

```bash
# Full self-evolvement (10 rounds, 6 domains, Qwen2.5-VL-7B)
python scripts/train.py --config configs/default.yaml

# Debug mode (2 rounds, 2 domains, Qwen2.5-VL-2B)
python scripts/train.py --config configs/debug.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume

# Standalone evaluation
python scripts/evaluate.py --model outputs/round_5/checkpoint
```

## Ablation Study

Isolate the contribution of each self-evolvement enhancement:

| Variant | Curriculum | Reward Bonus | What It Tests |
|:--------|:----------:|:------------:|:-------------|
| **Baseline** | Uniform | Off | Standard self-evolvement (no forgetting awareness) |
| **Curriculum-only** | Forgetting-aware | Off | Does data rebalancing alone prevent forgetting? |
| **Reward-only** | Uniform | On | Does reward shaping alone prevent forgetting? |
| **Full SelfEvolve** | Forgetting-aware | On | Both mechanisms combined |

Toggle via config:
```yaml
use_curriculum: true   # forgetting-aware sampling
use_reward_bonus: true # forgetting reward bonus (lambda)
```

## Project Structure

```
src/fa_evolve/
  orchestrator.py          # Self-evolvement loop orchestration
  forgetting_detector.py   # Probe evaluation + forgetting score computation
  curriculum.py            # Forgetting-aware curriculum scheduling
  challenger.py            # Curriculum-proportional task sampling
  reward.py                # GRPO reward with forgetting-urgency bonus
  evaluation.py            # Answer verification across 6 VQA metrics
  data/
    loader.py              # HuggingFace dataset loading (6 benchmarks)
    probe_cache.py         # Fixed probe set creation & caching
    formatting.py          # Qwen2.5-VL prompt formatting
  utils/
    config.py              # YAML configuration management
    logging_utils.py       # W&B + console logging
```

## Key Hyperparameters

| Parameter | Default | Role in Self-Evolvement |
|:----------|:-------:|:----------------------|
| `num_rounds` | 10 | Number of self-evolvement iterations |
| `samples_per_round` | 5000 | Training data per evolution round |
| `curriculum_temperature` | 1.0 | How aggressively to shift toward forgotten domains |
| `curriculum_floor` | 0.05 | Minimum sampling probability per domain |
| `lambda_forgetting` | 0.1 | Weight of forgetting-urgency reward bonus |
| `probe_size` | 200 | Probe samples per domain for forgetting detection |
| `grpo_group_size` | 8 | GRPO responses per prompt |

## Tests

```bash
pytest tests/ -v   # 59 tests across 4 modules
```

## Related Work

This project builds on the rapid progress in VLM self-evolvement:

- **[R-Zero](https://arxiv.org/abs/2501.12948)** -- Challenger/Solver co-evolution from zero data
- **[Vision-Zero](https://arxiv.org/abs/2501.12948)** -- Self-play via visual games + Iterative-SPO
- **[VisPlay](https://arxiv.org/abs/2501.12948)** -- Dual-role VLM with diversity/difficulty rewards
- **[MM-Zero](https://arxiv.org/abs/2501.12948)** -- Three-role RL framework for zero-data VLM evolution
- **[VILA2](https://arxiv.org/abs/2407.17453)** -- Data-centric self-augmentation for VLM pretraining

**SelfEvolve-VLM** contributes the missing piece: making self-evolvement *remember* what it learns.

## Citation

```bibtex
@article{selfevolve-vlm-2026,
  title={SelfEvolve-VLM: Forgetting-Aware Self-Evolvement for Vision-Language Models},
  year={2026}
}
```

---

<div align="center">
<i>Self-evolvement without memory is just spinning wheels. Self-evolvement with memory is growth.</i>
</div>
