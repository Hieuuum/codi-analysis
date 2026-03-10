# probe_latent_cot_hint.py
#
# Replication of the decoded_latent.txt experiment:
#   - Loads GSM8k-Aug test set
#   - Truncates each question's CoT to n-1 equations (removes last <<...>> block)
#   - Appends truncated CoT as a hint to the question input
#   - Runs CODI model with configurable inf_latent_iterations (default: 2)
#   - Supports --question_indices for targeted subset runs
#   - Emits: outputs/decoded_latent_cot_hint.txt   (same format as decoded_latent.txt)
#            outputs/decoded_latent_cot_hint.json   (structured summary for analysis)

import logging
import math
import re
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.nn import functional as F

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# ─── Config ──────────────────────────────────────────────────────────────────
do_print = True
probe_topk = 5
probe_idx = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─── CoT Truncation ───────────────────────────────────────────────────────────

def truncate_cot_to_penultimate(cot_str: str) -> str:
    """
    Given a CoT string like '<<16-3-4=9>> <<9*2=18>>',
    returns everything up to but NOT including the last <<...>> block.
    If there is only one equation (or none), returns empty string —
    meaning no hint is prepended (the question is passed as-is).
    """
    equations = re.findall(r'<<[^>]*>>', cot_str)
    if len(equations) <= 1:
        return ""
    # Find the start position of the last equation block
    last_eq = equations[-1]
    last_pos = cot_str.rfind(last_eq)
    return cot_str[:last_pos].strip()


def build_question_with_hint(question: str, cot_hint: str) -> str:
    """
    Appends the truncated CoT hint to the question string.
    The <<...>> calculator notation is preserved so it matches training distribution.
    """
    if not cot_hint:
        return question
    return f"{question} {cot_hint}"


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluation(model_args, data_args, training_args, question_indices: Optional[List[int]] = None):
    """
    Args:
        question_indices: If provided, only evaluate on these 0-indexed question indices.
                          If None, evaluate on the full test set.
    """
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, GPT-2 but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model = model.to('cuda')
    model.to(torch.bfloat16)

    # ── Load dataset ────────────────────────────────────────────────────────
    logging.warning("Downloading Data")
    if "zen-E/GSM8k-Aug" not in data_args.data_name:
        raise NotImplementedError("This script is designed for zen-E/GSM8k-Aug only.")

    dataset = load_dataset(data_args.data_name)
    test_set = dataset['test']

    logging.warning("Formatting inputs with CoT hints...")
    raw_questions, raw_answers, raw_cots = [], [], []

    for example in test_set:
        raw_questions.append(example["question"].strip().replace('  ', ' '))
        raw_answers.append(float(example["answer"].replace(",", "")))
        raw_cots.append(example["cot"])

    # ── Apply question_indices filter ────────────────────────────────────────
    if question_indices is not None:
        logging.warning(f"Filtering to {len(question_indices)} specified question indices.")
        # Store original indices for correct logging
        original_indices = question_indices
        raw_questions = [raw_questions[i] for i in question_indices]
        raw_answers   = [raw_answers[i] for i in question_indices]
        raw_cots      = [raw_cots[i] for i in question_indices]
    else:
        original_indices = list(range(len(raw_questions)))

    # ── Build questions with CoT hints ───────────────────────────────────────
    question, answer, procedures, hints = [], [], [], []
    for q, a, cot in zip(raw_questions, raw_answers, raw_cots):
        cot_hint = truncate_cot_to_penultimate(cot)
        q_with_hint = build_question_with_hint(q, cot_hint)
        question.append(q_with_hint)
        answer.append(a)
        procedures.append(cot)   # keep full CoT for logging (like existing format)
        hints.append(cot_hint)

    # ── Tokenise ─────────────────────────────────────────────────────────────
    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(f"Total examples: {len(question)} | batch size: {data_args.batch_size} | steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        batch_qs = question[i*data_args.batch_size:(i+1)*data_args.batch_size]
        batch = tokenizer(batch_qs, return_tensors="pt", padding="longest")

        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)

        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to('cuda'))

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []
    log = []
    json_results = []
    log_count = 0

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        top5_values_list, top5_indices_list = [], []

        with torch.no_grad():
            # Encode question + hint
            past_key_values = None
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"]
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Probe before projection
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
            top5_values_list.append(top5_values)
            top5_indices_list.append(top5_indices)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Latent iterations
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                top5_values_list.append(top5_values)
                top5_indices_list.append(top5_indices)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # EOT transition
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to('cuda')
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to('cuda')

            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
            output = eot_emb

            # Autoregressive decode
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]

            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k] = -float("inf")
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to('cuda')

            # ── Post-process batch ────────────────────────────────────────────
            top5_values_list = torch.cat(top5_values_list, dim=1)
            top5_indices_list = torch.cat(top5_indices_list, dim=1)

            if probe_idx is not None:
                top5_values_list = top5_values_list[:, probe_idx].unsqueeze(1)
                top5_indices_list = top5_indices_list[:, probe_idx].unsqueeze(1)

            for mini_step, pred_token in enumerate(pred_tokens):
                global_idx = original_indices[log_count]
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                pred_num = extract_answer_number(decoded_pred)
                gt = answer[log_count]
                is_correct = (int(gt) == int(pred_num)) if pred_num != float('inf') else False

                if do_print:
                    print(f"Question {global_idx} Starts...")
                    print(f"Q: {raw_questions[log_count]}")
                    print(f"Hint: {hints[log_count]}")
                    print(decoded_pred)
                    print(f"Question {global_idx} Ends")
                    print(f"Prediction={pred_num}; Groundtruth={gt} | Correct={is_correct}")
                    print("")

                ans_pred_list.append(pred_num)

                # Text log (matches decoded_latent.txt format)
                if is_correct:
                    log.append(f"Question{global_idx}...")
                    log.append(f"{raw_questions[log_count]}...")
                    log.append(f"CoT={procedures[log_count]}, Answer={gt}")
                    log.append(f"Hint={hints[log_count]}")
                    for jj in range(top5_indices_list.size(1)):
                        decoded_top5 = [tokenizer.decode(x) for x in top5_indices_list[mini_step, jj]]
                        log.append(f"decoded {jj}th latent (top5): {decoded_top5}")
                    log.append(f"Model Prediction: {tokenizer.decode(pred_token)}")
                    log.append("\n\n")

                # JSON summary (every question, correct or not)
                json_results.append({
                    "question_idx": global_idx,
                    "question": raw_questions[log_count],
                    "cot_hint": hints[log_count],
                    "full_cot": procedures[log_count],
                    "ground_truth": gt,
                    "prediction": pred_num,
                    "correct": is_correct,
                    "num_latent_steps": training_args.inf_latent_iterations,
                    "decoded_pred": decoded_pred,
                })

                log_count += 1

    accuracy = compute_accuracy(answer, ans_pred_list)

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)

    suffix = f"_latent{training_args.inf_latent_iterations}"
    if question_indices is not None:
        suffix += f"_subset{len(question_indices)}q"

    txt_path  = f"outputs/decoded_latent_cot_hint{suffix}.txt"
    json_path = f"outputs/decoded_latent_cot_hint{suffix}.json"

    with open(txt_path, "w") as f:
        f.write("\n".join(log))

    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "inf_latent_iterations": training_args.inf_latent_iterations,
                "num_questions": len(question),
                "question_indices": original_indices,
                "model": model_args.model_name_or_path,
                "ckpt_dir": model_args.ckpt_dir,
            },
            "accuracy": 100 * accuracy,
            "num_correct": sum(r["correct"] for r in json_results),
            "results": json_results,
        }, f, indent=2)

    print(f"\naccuracy: {100*accuracy:.2f}% | correct: {sum(r['correct'] for r in json_results)}/{len(question)}")
    print(f"average CoT length (tokens): {sum(len_cot)/len(len_cot):.1f}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")

    return 100 * accuracy


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1
    return acc / len(gold)


# ─── Entry Point ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentArguments:
    question_indices: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of 0-based question indices to evaluate, e.g. '4,12,20'. "
                "If omitted, the full test set is used. "
                "Can also be a path to a .txt file with one index per line."
            )
        }
    )


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExperimentArguments))
    model_args, data_args, training_args, exp_args = parser.parse_args_into_dataclasses()

    # Parse question_indices
    question_indices = None
    if exp_args.question_indices is not None:
        if os.path.isfile(exp_args.question_indices):
            with open(exp_args.question_indices) as f:
                question_indices = [int(line.strip()) for line in f if line.strip()]
        else:
            question_indices = [int(x.strip()) for x in exp_args.question_indices.split(",") if x.strip()]
        print(f"Running on {len(question_indices)} specified questions.")

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args, question_indices=question_indices)
        accu_list.append(accu)

    print(f"\nAverage accuracy over {training_args.inf_num_iterations} run(s): {sum(accu_list)/len(accu_list):.2f}%")
