# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

This is a fork/working copy of **CODI** (Compressing Chain-of-Thought into Continuous Space via Self-Distillation, EMNLP 2025 — [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)), adapted for **latent-model positional analysis**. The repo trains small LMs (GPT-2, Llama-3.2-1B) to compress chain-of-thought reasoning into a small number of continuous "latent thought" vectors via self-distillation, and provides tooling to probe/interpret those latents (logit lens, bracket sweeps, position analysis).

Pretrained weights (not in repo): `zen-E/CODI-gpt2`, `zen-E/CODI-llama3.2-1b-Instruct` on HuggingFace.

## Environment

Python 3.12, CUDA required. Pinned versions in `requirements.txt`:
- `torch==2.7.1`, `transformers==4.52.4`, `peft==0.15.2`, `datasets==3.6.0`, `accelerate==1.7.0`, `safetensors==0.5.3`

Setup (per README):
```bash
conda create --name codi python=3.12
conda activate codi
pip install -r requirements.txt
```

A full conda env lock is in `environment.yml`.

## Repository Layout

- `src/model.py` — `CODI` model class + `ModelArguments`, `DataArguments`, `TrainingArguments` dataclasses. Wraps an HF causal LM with LoRA and an optional projection layer for latent thoughts. **Single source of truth for model architecture; other scripts import from here.**
- `train.py` — training entrypoint (self-distillation loss, teacher CE loss, distill loss with optional std-normalization).
- `test.py` — evaluation on GSM8K / SVAMP / GSM-Hard / MultiArith / Commonsense.
- `test_custom_questions.py` — evaluation on ad-hoc questions (see `example_questions.json`).
- `probe_latent_token.py` — injects partial CoT into prompts and probes latent-token hidden states; supports sweeping over CoT "bracket pairs" (e.g. `<<…>>`, `[…]`, `{{…}}`) to study positional/format sensitivity. Has module-level flags (`do_print`, `do_probe`, `log_wrong`, `sweep_all_brackets`, `cot_brackets`, `bracket_pairs`) near the top of the file.
- `analysis/logit_lens.py` — logit-lens decoding at the trained readout point (position `model_answer_position - 1`), plus tracking `P(target)` at `after_prompt_bot`, each latent step, and `decoder_readout`.
- `inspect_checkpoint.py` — utility for inspecting saved checkpoints.
- `scripts/` — shell wrappers for training/eval runs. Training scripts set `SAVE_DIR` and invoke `train.py`; test scripts invoke `test.py` / `probe_latent_token.py` with a `--ckpt_dir` (default `~/transfer/codi_gpt2`).
- `outputs/`, `logs/`, `results.json` — run outputs.
- `transfer/` — checkpoint staging directory referenced by scripts.
- `colab_experiment.ipynb` — exploratory notebook.

## Common Commands

All scripts assume the `codi` conda env is active and that a checkpoint exists at the path given by `--ckpt_dir` (commonly `~/transfer/codi_gpt2`).

Evaluation:
```bash
bash scripts/test_gpt2.sh          # GPT-2 on GSM8K
bash scripts/test_llama1b.sh       # Llama-3.2-1B on GSM8K
bash scripts/test_gpt2_commonsense.sh
bash scripts/test_custom_questions.sh
```
Change `--data_name` (`gsm8k`, `svamp`, `gsm-hard`, `multi-arith`, `commonsense`) inside the script to switch benchmarks.

Latent probing / interpretability:
```bash
bash scripts/probe_latent_token.sh
```
Outputs land in `outputs/`. `SAVE_DIR` must be exported before running.

Training:
```bash
bash scripts/train_gpt2_gsm8k-aug.sh
bash scripts/train_gpt2_gsm8k-aug-nl.sh
bash scripts/train_llama1b_gsm8k-aug.sh
bash scripts/train_llama1b_gsm8k-aug-nl.sh
bash scripts/train_gpt2_commonsense.sh
bash scripts/train_llama_commonsense.sh
```
Training scripts hard-code `SAVE_DIR` to `/scratch/...` paths — update for your filesystem before running.

## Key Model / Training Arguments

(See README for the canonical list.) Frequent flags:
- `--num_latent N` — latent thoughts used in training.
- `--inf_latent_iterations N` — latent thoughts used at inference.
- `--use_prj / --prj_dim / --prj_no_ln / --prj_dropout` — projection layer on the last-layer hidden state used as the latent.
- `--distill_loss_type {l1,l2,smoothl1}`, `--distill_loss_factor`, `--distill_loss_div_std`, `--ref_loss_factor` — loss composition.
- `--lora_r / --lora_alpha / --lora_init / --use_lora` — LoRA config.
- `--include_last_cot`, `--remove_eos`, `--max_token_num`, `--fix_attn_mask` — data/masking tweaks. `fix_attn_mask` defaults to False (known-bug compat flag).
- `--use_logit_lens`, `--logit_lens_example_idx` — enable logit-lens output in `test.py`.

## Working Notes

- Model definitions live only in `src/model.py`. When changing architecture, the dataclasses there drive CLI args across `train.py`, `test.py`, and `probe_latent_token.py` — keep them in sync.
- `probe_latent_token.py` has module-level globals (flags and `bracket_pairs` list) that are part of the experiment config, not CLI args — edit the file itself to sweep.
- Checkpoint paths in scripts point at `~/transfer/...` or `/scratch/...`; these are user-specific and need updating per machine.
- Recent commits (see `git log`) focus on CoT-format options (bracket variants) and refactoring so model/dataset load once per sweep — be mindful of that structure when adding experiments.
