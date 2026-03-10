#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json
from dataclasses import dataclass, field
from typing import Optional, List

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np
#from scipy.stats import mode

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True
probe_topk = 5
probe_idx = None
test_attention = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


@dataclass
class ExperimentArguments:
    question_indices: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated 0-based question indices to evaluate, e.g. '4,12,20'. "
                "Or a path to a .txt file with one index per line. "
                "If omitted, the full test set is used."
            )
        }
    )

def evaluation(model_args, data_args, training_args, question_indices: Optional[List[int]] = None):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
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
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
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
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "zen-E/GSM8k-Aug" in data_args.data_name:
        dataset = load_dataset(data_args.data_name)
        test_set = dataset['test']
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    all_questions = []
    all_answers = []
    all_procedures = []

    # get numerical answer
    for example in test_set:
        all_questions.append(f"{example[question_name].strip().replace('  ', ' ')}")
        all_answers.append(float(example[answer_name].replace(",", "")))
        all_procedures.append(example["cot"])

    # Apply question_indices filter
    if question_indices is not None:
        logging.warning(f"Filtering to {len(question_indices)} specified question indices.")
        original_indices = question_indices
        question   = [all_questions[i]   for i in question_indices]
        answer     = [all_answers[i]     for i in question_indices]
        procedures = [all_procedures[i]  for i in question_indices]
    else:
        original_indices = list(range(len(all_questions)))
        question   = all_questions
        answer     = all_answers
        procedures = all_procedures

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []

    gating_probs_sums = None
    len_cot = []
    model.eval()
    attn_to_latent_list = []
    top5_indices_list_decoded = []
    log_count = 0
    log = []
    json_results = []
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        top5_values_list, top5_indices_list = [], []
        with torch.no_grad():
            # encode the question
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)  
            top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
            top5_values_list.append(top5_values)
            top5_indices_list.append(top5_indices)
            
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Iterate the latent thoughts
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # decode the latent embeddings
                outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                # Probe the latent thought before the projection
                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                top5_values_list.append(top5_values)
                top5_indices_list.append(top5_indices)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
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
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # implement the sampling process
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

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

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences have finished
                if finished.all():
                    break

                #output = model.codi.get_base_model().transformer.wte(next_token_ids).unsqueeze(1).to(device)
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                # Extract the numbers in sentences 
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))
            
            top5_values_list = torch.cat(top5_values_list, dim=1)
            top5_indices_list = torch.cat(top5_indices_list, dim=1)

            if probe_idx is not None:
                top5_values_list = top5_values_list[:, probe_idx]
                top5_indices_list = top5_indices_list[:, probe_idx]
                top5_values_list = top5_values_list.unsqueeze(1)
                top5_indices_list = top5_indices_list.unsqueeze(1)

            # decode top5_indices_list
            for ii in range(len(top5_indices_list)): # batch
                global_idx = original_indices[log_count]
                pred_num = extract_answer_number(tokenizer.decode(pred_tokens[ii]))
                is_correct = int(answer[log_count]) == int(pred_num) if pred_num != float('inf') else False
                do_log = is_correct
                # snapshot before incrementing
                cur_question   = question[log_count]
                cur_procedure  = procedures[log_count]
                cur_answer     = answer[log_count]
                if do_log:
                    log.append(f"Question{global_idx}...")
                    log.append(f"{cur_question}...")
                    log.append(f"CoT={cur_procedure}, Answer={cur_answer}")
                log_count += 1
                top5_indices_list_decoded_tmp = []
                for jj in range(top5_indices_list.size(1)):
                    if do_log:
                        if test_attention:
                            log.append(f"decoded {jj}th latent's attended tokens (top5): {attn_to_lats[jj][ii]}")
                        log.append(f"decoded {jj}th latent (top5): {[tokenizer.decode(x) for x in top5_indices_list[ii, jj]]}")
                    for kk in range(top5_indices_list.size(2)):
                        top5_indices_list_decoded_tmp.append(tokenizer.decode(top5_indices_list[ii, jj, kk]))
                top5_indices_list_decoded.append(top5_indices_list_decoded_tmp)
                if do_log:
                    if test_attention:
                        log.append(f"decoded before answer token's attended tokens (top5): {attn_to_lats[-1][ii]}")
                    log.append(f"Model Prediction: {tokenizer.decode(pred_tokens[ii])}")
                    log.append("\n\n")

                # JSON record (every question)
                json_results.append({
                    "question_idx": global_idx,
                    "question": cur_question,
                    "full_cot": cur_procedure,
                    "ground_truth": cur_answer,
                    "prediction": pred_num,
                    "correct": is_correct,
                    "num_latent_steps": training_args.inf_latent_iterations,
                })
    accuracy = compute_accuracy(answer, ans_pred_list)

    os.makedirs("outputs", exist_ok=True)
    suffix = f"_latent{training_args.inf_latent_iterations}"
    if question_indices is not None:
        suffix += f"_subset{len(question_indices)}q"

    txt_path  = f"outputs/original_decoded_latent{suffix}.txt"
    json_path = f"outputs/original_decoded_latent{suffix}.json"

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

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")

    return 100*accuracy

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


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
    print(f"Average accuracy over {training_args.inf_num_iterations} run(s): {sum(accu_list)/len(accu_list):.2f}%")
