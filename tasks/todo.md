# Experiment: Latent Steps vs CoT Hint Analysis

## Goal
Replicate the `decoded_latent.txt` run (2 latent steps + truncated CoT hint), then sweep
latent step counts on the 149 "original-only" questions to understand the contribution
of latent depth vs. explicit CoT hints.

## Tasks

### Phase 0: Setup
- [x] Map codebase (probe_latent_token.py, src/model.py, test.py)
- [x] Understand decoded_latent_comparison.txt (149 original-only, 146 modified-only, 398 both)
- [x] Confirm CoT format: `<<eq1>> <<eq2>> ...` space-separated blocks in `cot` field

### Phase 1: Replication Script
- [ ] Write `probe_latent_cot_hint.py` — mirrors probe_latent_token.py but:
        - Truncates CoT to n-1 `<<...>>` blocks and appends as hint to question
        - Adds `--question_indices` flag for targeted subset runs
        - Emits both .txt log (existing format) and .json summary
        - Defaults to inf_latent_iterations=2
- [ ] Write `scripts/probe_latent_cot_hint.sh`
- [ ] Write `colab_experiment.py` (# %% cell format for Colab upload)

### Phase 2: Run & Verify
- [ ] Run on full GSM8k-Aug test set with inf_latent_iterations=2, CoT hint
- [ ] Confirm ~544 correct (matching original decoded_latent.txt)
- [ ] Regenerate original_decoded_latent results (6 latents, no hint) using probe_latent_token.sh

### Phase 3: Latent Sweep on 149 Questions
- [ ] Run probe_latent_cot_hint.py on 149-question subset with latent_iterations in [2,4,6,8,10]
- [ ] Collect JSON summaries, plot accuracy curve

## Notes
- Model: GPT-2 + LoRA (r=128) + projection layer, bfloat16
- Checkpoint: ~/transfer/codi_gpt2
- Dataset: zen-E/GSM8k-Aug (test split, ~1319 questions)
- 149 target questions: from decoded_latent_comparison.txt "Only in original_decoded_latent.txt"
