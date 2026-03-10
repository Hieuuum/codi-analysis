#!/usr/bin/env bash
# scripts/probe_latent_cot_hint.sh
#
# Replicates the decoded_latent.txt experiment:
#   - 2 latent steps
#   - CoT hint (truncated to n-1 equations) prepended to each question
#   - Full GSM8k-Aug test set
#
# Usage:
#   export CKPT_DIR=~/transfer/codi_gpt2
#   bash scripts/probe_latent_cot_hint.sh
#
# To run on a question subset, add: --question_indices "4,12,20,30"
# Or pass a file:                   --question_indices path/to/indices.txt

CKPT_DIR=${CKPT_DIR:-~/transfer/codi_gpt2}

python probe_latent_cot_hint.py \
    --data_name "zen-E/GSM8k-Aug" \
    --output_dir "outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 32 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 2 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR" \
    "$@"
# "$@" passes any extra CLI args through, e.g. --question_indices
