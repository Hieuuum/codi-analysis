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

"""
probe_latent_token.py

Evaluates a custom latent variable model (CODI) on GSM8k variants.
Injects partial mathematical Chain-of-Thought (CoT) into prompts and
probes the model's intermediate "latent thoughts" before generation.
"""

import logging
import math
import re
import os
import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from datasets import load_dataset
from safetensors.torch import load_file

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# ── Module-level inference settings ──────────────────────────────────────────
do_print = False
do_probe = True       # set False to skip latent token probing entirely (faster inference)
probe_topk = 20
probe_idx = None
test_attention = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_lora_config(model_args: ModelArguments) -> LoraConfig:
    """Build a LoraConfig by detecting the model architecture family."""
    task_type = TaskType.CAUSAL_LM
    name = model_args.model_name_or_path.lower()
    if any(n in name for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in name:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in name:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
    return LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
) -> tuple:
    """Construct CODI, load checkpoint weights, and set up the tokenizer."""
    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
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

    model = model.to('cuda').to(torch.bfloat16)
    return model, tokenizer


def prepare_dataset(
    data_args: DataArguments,
    training_args: TrainingArguments,
    model: CODI,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple:
    """Download and format the dataset, tokenize into batches ready for inference.

    Returns:
        (question_data, questions, answers, procedures)
        question_data: list of tokenized batches on cuda, each with input_ids / attention_mask / input_len.
    """
    logging.warning("Downloading Data")
    if "zen-E/GSM8k-Aug" not in data_args.data_name:
        raise NotImplementedError
    dataset = load_dataset(data_args.data_name)
    test_set = dataset['test']

    logging.warning("Formatting inputs...")
    questions, answers, procedures = [], [], []
    for example in test_set:
        raw_q = example["question"].strip().replace('  ', ' ')
        raw_cot = example["cot"]
        if not raw_cot or not raw_cot.strip():
            continue

        # Split the space-separated math annotators, e.g. '<<16-3-4=9>> <<9*2=18>>'
        thoughts = raw_cot.strip().split()
        first_n_minus_1 = " ".join(thoughts[:-1]) if len(thoughts) > 1 else ""
        final_question = f"{raw_q} {first_n_minus_1}" if first_n_minus_1 else raw_q

        questions.append(final_question)
        answers.append(float(example["answer"].replace(",", "")))
        procedures.append(raw_cot)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(questions) / data_args.batch_size)
    logging.warning(
        f"Total example: {len(questions)} | eval batch size: {data_args.batch_size} | "
        f"eval steps: {eval_step}"
    )

    question_data = []
    for i in range(eval_step):
        start = i * data_args.batch_size
        end = (i + 1) * data_args.batch_size if i < eval_step - 1 else len(questions)
        batch = tokenizer(questions[start:end], return_tensors="pt", padding="longest")

        # Append BoT token (and optionally EOS) to delimit the question from latent thinking
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor(
                [tokenizer.eos_token_id, model.bot_id], dtype=torch.long
            ).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch = batch.to(device)          # move tensors to device before adding non-tensor fields
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    return question_data, questions, answers, procedures


def _get_last_transformer_layer(model: CODI):
    """Return the final transformer layer module for forward hook registration.

    Handles LoRA wrapping (get_base_model) and GPT-2 / LLaMA / Pythia layouts.
    """
    name = model.model_name.lower()
    try:
        base = model.codi.get_base_model()  # unwrap PEFT/LoRA wrapper
    except AttributeError:
        base = model.codi

    if "gpt2" in name:
        return base.transformer.h[-1]
    elif "pythia" in name:
        return base.gpt_neox.layers[-1]
    else:  # llama, mistral, falcon, qwen, phi, ...
        return base.model.layers[-1]


def run_batch(
    batch: dict,
    model: CODI,
    tokenizer: transformers.PreTrainedTokenizer,
    training_args: TrainingArguments,
    gen_kwargs: dict,
    do_probe: bool,
    probe_topk: int,
    probe_idx,
) -> tuple:
    """Encode one batch, run latent iterations with probing, then generate answer tokens.

    Returns:
        (pred_tokens, topk_values, topk_indices)
        pred_tokens: list[list[int]], one token-id list per sequence in the batch.
        topk_values / topk_indices: (batch, n_probe_steps, probe_topk) tensors, or None if do_probe=False.
    """
    batch_size = batch["input_ids"].size(0)
    topk_values_list, topk_indices_list = [], []

    # Hook captures only the last layer's output hidden state so we can set
    # output_hidden_states=False and avoid materialising every layer's activations.
    _captured = {}
    def _capture_last_hidden(module, inp, out):
        # All transformer block types return (hidden_state, ...) as a tuple
        _captured['h'] = out[0] if isinstance(out, tuple) else out

    hook = _get_last_transformer_layer(model).register_forward_hook(_capture_last_hidden)
    try:
      with torch.no_grad():
        # Encode the question; the last hidden state becomes the first latent embedding
        outputs = model.codi(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=False,
            past_key_values=None,
            attention_mask=batch["attention_mask"],
        )
        past_key_values = outputs.past_key_values
        latent_embd = _captured['h'][:, -1, :].unsqueeze(1)

        # probe index 0 = after initial encoding; only compute if probing is enabled
        if do_probe and (probe_idx is None or probe_idx == 0):
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            topk_v, topk_i = torch.topk(probs, k=probe_topk, dim=2)
            topk_values_list.append(topk_v)
            topk_indices_list.append(topk_i)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Recurrent latent iterations: feed each hidden state back as the next input embedding
        for iter_idx in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=False,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = _captured['h'][:, -1, :].unsqueeze(1)

            # probe index iter_idx+1 = after this latent pass; skip if not the target
            if do_probe and (probe_idx is None or probe_idx == iter_idx + 1):
                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                topk_v, topk_i = torch.topk(probs, k=probe_topk, dim=2)
                topk_values_list.append(topk_v)
                topk_indices_list.append(topk_i)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        # Cache embedding lookup once — get_embd does string checks + try/except on every call
        embed_fn = model.get_embd(model.codi, model.model_name)

        # Inject EoT embedding to transition from latent space to token generation
        if training_args.remove_eos:
            eot_emb = embed_fn(
                torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
            ).unsqueeze(0)
        else:
            eot_emb = embed_fn(
                torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
            ).unsqueeze(0)
        eot_emb = eot_emb.expand(batch_size, -1, -1)
        output = eot_emb

        # Token-by-token autoregressive generation with per-sequence EOS tracking
        # Tokens are accumulated on GPU and transferred to CPU in one batch at the end
        finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
        all_token_ids = []
        for _ in range(gen_kwargs["max_new_tokens"]):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

            if training_args.greedy:
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
            else:
                logits /= gen_kwargs["temperature"]
                if gen_kwargs["top_k"] > 1:
                    top_k_vals, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                    logits[logits < top_k_vals[:, -1].unsqueeze(-1)] = -float("inf")
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

            all_token_ids.append(next_token_ids)
            finished = finished | (next_token_ids == tokenizer.eos_token_id)
            if finished.all():
                break
            output = embed_fn(next_token_ids).unsqueeze(1).to(device)

        # Single GPU→CPU transfer: move all steps at once, then trim each sequence at EOS
        token_matrix = torch.stack(all_token_ids, dim=1).cpu().tolist()  # [batch, steps]
        eos_id = tokenizer.eos_token_id
        pred_tokens = []
        for seq in token_matrix:
            tokens = []
            for tid in seq:
                tokens.append(tid)
                if tid == eos_id:
                    break
            pred_tokens.append(tokens)

    finally:
        hook.remove()

    if not do_probe:
        return pred_tokens, None, None

    topk_values = torch.cat(topk_values_list, dim=1)
    topk_indices = torch.cat(topk_indices_list, dim=1)

    if probe_idx is not None:
        # Only one probe was collected (at the target iteration), so it sits at index 0
        topk_values = topk_values[:, 0].unsqueeze(1)
        topk_indices = topk_indices[:, 0].unsqueeze(1)

    return pred_tokens, topk_values, topk_indices


def format_batch_logs(
    batch_offset: int,
    pred_tokens: list,
    topk_indices,           # torch.Tensor | None — None when do_probe=False
    questions: list,
    answers: list,
    procedures: list,
    tokenizer: transformers.PreTrainedTokenizer,
    log_count: int,
) -> tuple:
    """Decode predictions, extract numeric answers, and build log lines for correct examples.

    Returns:
        (ans_preds, log_lines, decoded_topk_flat)
        ans_preds: one float/int answer per sequence.
        log_lines: log strings for correctly predicted examples only.
        decoded_topk_flat: flat list of decoded topk token strings; empty when do_probe=False.
    """
    ans_preds = []
    log_lines = []
    decoded_topk_flat = []

    for ii, tokens in enumerate(pred_tokens):
        global_idx = log_count + ii
        decoded_pred = tokenizer.decode(tokens, skip_special_tokens=True)
        pred_answer = extract_answer_number(decoded_pred)
        ans_preds.append(pred_answer)

        if do_print:
            print(f"Question {batch_offset + ii} Starts...")
            print(f"Q: {questions[global_idx]}")
            print(decoded_pred)
            print(f"Question {batch_offset + ii} Ends")
            print(f"Prediction={pred_answer}; Groundtruth={answers[global_idx]}")
            print("")

        # do_log: reuse pred_answer already extracted above — avoids a second tokenizer.decode call
        do_log = int(answers[global_idx]) == int(pred_answer)
        if do_log:
            log_lines.append(f"Question{global_idx}...")
            log_lines.append(f"{questions[global_idx]}...")
            log_lines.append(f"CoT={procedures[global_idx]}, Answer={answers[global_idx]}")

        if topk_indices is not None:
            topk_indices_decoded_tmp = []
            for jj in range(topk_indices.size(1)):
                if do_log:
                    if test_attention:
                        pass  # placeholder: attn_to_lats not available in current flow
                    log_lines.append(
                        f"decoded {jj}th latent (topk): {[tokenizer.decode(x) for x in topk_indices[ii, jj]]}"
                    )
                for kk in range(topk_indices.size(2)):
                    topk_indices_decoded_tmp.append(tokenizer.decode(topk_indices[ii, jj, kk]))
            decoded_topk_flat.extend(topk_indices_decoded_tmp)

        if do_log:
            log_lines.append(f"Model Prediction: {tokenizer.decode(tokens)}")
            log_lines.append("\n\n")

    return ans_preds, log_lines, decoded_topk_flat


def evaluation(model_args, data_args, training_args):
    """Orchestrate CODI evaluation: load model, prepare data, run inference, log results."""
    if not model_args.lora_init:
        raise NotImplementedError

    lora_config = _build_lora_config(model_args)
    model, tokenizer = load_model_and_tokenizer(model_args, training_args, lora_config)
    question_data, questions, answers, procedures = prepare_dataset(
        data_args, training_args, model, tokenizer
    )

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    log = []
    log_count = 0
    len_cot = []

    #set_seed(42)
    for step, batch in enumerate(question_data):
        pred_tokens, topk_values, topk_indices = run_batch(
            batch, model, tokenizer, training_args, gen_kwargs, do_probe, probe_topk, probe_idx
        )
        for tokens in pred_tokens:
            len_cot.append(len(tokens))

        ans_preds, log_lines, _ = format_batch_logs(
            step * data_args.batch_size,
            pred_tokens, topk_indices,
            questions, answers, procedures,
            tokenizer, log_count,
        )
        ans_pred_list.extend(ans_preds)
        log.extend(log_lines)
        log_count += len(pred_tokens)

    accuracy = compute_accuracy(answers, ans_pred_list)
    correct_indices = [
        idx for idx, (p, g) in enumerate(zip(ans_pred_list, answers))
        if (g in p if isinstance(p, list) else p == g)
    ]

    summary = (
        f"====== SUMMARY ======\n"
        f"Total questions: {len(answers)}\n"
        f"Total correct: {len(correct_indices)}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Correct indices: {correct_indices}\n"
        f"=====================\n\n"
    )
    log.insert(0, summary)

    output_dir = training_args.output_dir if training_args.output_dir else "outputs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"decoded_latent_{training_args.inf_latent_iterations}_steps_cot_hint.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100 * accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot) / len(len_cot)}")
    return 100 * accuracy


def extract_answer_number(sentence: str) -> float:
    """Parses and extracts the final numerical floating-point answer from a generated text string."""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_accuracy(gold: list, pred: list) -> float:
    """Calculates the ratio of correctly predicted answers (exact matches) against the gold standard."""
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list) / len(accu_list)}")
