# Forgetting-Aware VLM Self-Evolution

Research prototype for **Adversarial Difficulty Scheduling with Forgetting-Aware Curriculum** -- a framework that augments VLM self-evolution (Challenger/Solver with GRPO) with explicit forgetting detection and adaptive curriculum scheduling.

## Key Idea

Existing VLM self-evolution methods (R-Zero, VisPlay, MM-Zero) treat each training iteration independently. They don't model which previously-learned visual skills the Solver might forget. This project adds:

1. **Forgetting Detector** -- periodically evaluates the Solver on fixed probe sets across 6 visual reasoning domains
2. **Curriculum Scheduler** -- biases training toward domains where forgetting is detected
3. **Reward Bonus** -- adds a forgetting-urgency term to the GRPO reward, incentivizing the Solver on degraded skills

## Architecture

```
For each round:
  1. Evaluate Solver on probe sets -> per-cluster accuracy
  2. Compute forgetting scores -> curriculum distribution
  3. Challenger samples training tasks per curriculum
  4. GRPO training with reward = accuracy + lambda * forgetting_urgency
  5. Update Solver checkpoint
```

## Skill Clusters

| Cluster | Dataset | Metric |
|---------|---------|--------|
| Math | MathVista | Accuracy |
| Documents | DocVQA | ANLS |
| Charts | ChartQA | Relaxed Accuracy |
| Spatial | GQA | Exact Match |
| Science | ScienceQA | MC Accuracy |
| Natural | VQAv2 | Soft Accuracy |

## Setup

```bash
pip install -e ".[dev]"
```

### Requirements
- Python 3.10+
- PyTorch 2.1+
- 4x A100 80GB (for 7B model with GRPO)
- [LMM-R1](https://github.com/TideDra/lmm-r1) (OpenRLHF fork with VLM support)

## Usage

### Full training
```bash
python scripts/train.py --config configs/default.yaml
```

### Debug run (small scale)
```bash
python scripts/train.py --config configs/debug.yaml
```

### Resume from checkpoint
```bash
python scripts/train.py --config configs/default.yaml --resume
```

### Standalone evaluation
```bash
python scripts/evaluate.py --model outputs/round_5/checkpoint --config configs/default.yaml
```

## Ablation Variants

Control via config flags:
- `use_curriculum: true/false` -- enable/disable forgetting-aware sampling
- `use_reward_bonus: true/false` -- enable/disable forgetting reward bonus (lambda)

| Variant | `use_curriculum` | `use_reward_bonus` |
|---------|-----------------|-------------------|
| Baseline | false | false |
| Curriculum-only | true | false |
| Reward-only | false | true |
| Full method | true | true |

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/fa_evolve/
  orchestrator.py          # Main iterative training loop
  forgetting_detector.py   # Probe evaluation + forgetting scores
  curriculum.py            # Forgetting-aware sampling distribution
  challenger.py            # Dataset sampling per curriculum
  reward.py                # OpenRLHF-compatible reward with forgetting bonus
  evaluation.py            # Answer verification (6 VQA metrics)
  data/
    loader.py              # HuggingFace dataset loading
    probe_cache.py         # Fixed probe set creation
    formatting.py          # Qwen2.5-VL prompt formatting
  utils/
    config.py              # YAML config loading
    logging_utils.py       # W&B + console logging
```

## Citation

```bibtex
@article{forgetting-aware-vlm-2026,
  title={Adversarial Difficulty Scheduling with Forgetting-Aware Curriculum for VLM Self-Evolution},
  year={2026}
}
```
