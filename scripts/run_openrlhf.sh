#!/bin/bash
# OpenRLHF GRPO training wrapper for one round of self-evolution.
# Uses LMM-R1 fork for Qwen2.5-VL support.
#
# Usage: bash scripts/run_openrlhf.sh --pretrain <path> --dataset <path> ...

set -euo pipefail

# Parse arguments
PRETRAIN=""
DATASET=""
REWARD_FN=""
REWARD_CONFIG=""
SAVE_PATH=""
N_SAMPLES=8
LR="1e-6"
KL_COEFF="0.01"
MAX_NEW_TOKENS=512

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrain) PRETRAIN="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --reward_fn) REWARD_FN="$2"; shift 2 ;;
        --reward_config) REWARD_CONFIG="$2"; shift 2 ;;
        --save_path) SAVE_PATH="$2"; shift 2 ;;
        --n_samples_per_prompt) N_SAMPLES="$2"; shift 2 ;;
        --learning_rate) LR="$2"; shift 2 ;;
        --kl_coeff) KL_COEFF="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== OpenRLHF GRPO Training ==="
echo "Model: $PRETRAIN"
echo "Dataset: $DATASET"
echo "Save to: $SAVE_PATH"

# Set REWARD_CONFIG as env var for the reward function to read
export FA_EVOLVE_REWARD_CONFIG="$REWARD_CONFIG"

python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN" \
    --save_path "$SAVE_PATH" \
    --save_hf_ckpt \
    --micro_train_batch_size 2 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 256 \
    --n_samples_per_prompt "$N_SAMPLES" \
    --max_samples 100000 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_max_len 2048 \
    --advantage_estimator group_norm \
    --remote_rm_url "$REWARD_FN" \
    --actor_learning_rate "$LR" \
    --init_kl_coeff "$KL_COEFF" \
    --num_episodes 1 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --dataset "$DATASET"

echo "=== Training complete ==="
