"""
Logit lens decoding for CoDi (your codebase version).

Key change vs your current script:
- We probe the *same readout point your training uses*:
  decode hidden states at position (model_answer_position - 1) during the DECODER forward
  (prompt+BoT -> latent rollout -> decoder tokens).

We still optionally track P(target) at:
- after_prompt_bot
- each latent_k token
- decoder_readout (the trained boundary)

Output JSON:
- positions: ["after_prompt_bot", "latent_0"... "latent_K-1", "decoder_readout"]
- heatmap: list[ position -> {layer_i: P(target_first_answer_token)} ]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from torch.nn import functional as F
from datasets import load_dataset
from safetensors.torch import load_file

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, TaskType


# -----------------------------
# Helpers: logit lens decoding
# -----------------------------

def _get_lm_head_and_ln_f(model: CODI, model_name: str):
    """
    Return (lm_head, ln_f or None).
    Paper-style: apply final LN before unembedding (ln_f exists for GPT-2; for LLaMA/Mistral usually None).
    """
    lm_head = model.codi.lm_head
    ln_f = model.get_ln_f(model.codi, model_name)
    return lm_head, ln_f


def _decode_residual_with_ln(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    ln_f: Optional[torch.nn.Module],
    vocab_size: int,
) -> torch.Tensor:
    """
    Decode residual with paper's definition: apply final LN then unembedding.
    hidden: (B, T, D) or (B, D); we decode the last position if (B, T, D).
    Returns probs: (B, vocab_size)
    """
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :]
    if ln_f is not None:
        hidden = ln_f(hidden)
    logits = lm_head(hidden)[..., :vocab_size]
    return F.softmax(logits.float(), dim=-1)


def _prob_of_token(probs: torch.Tensor, token_id: int) -> float:
    if probs.numel() == 0:
        return 0.0
    if token_id < 0 or token_id >= probs.size(-1):
        return 0.0
    return probs[0, token_id].item()


# -----------------------------
# Helpers: match your train.py
# -----------------------------

ANSWER_PROMPTS = ["The answer is:", "The next step result is:"]


def _answer_prompt_token_tensors(tokenizer: transformers.PreTrainedTokenizer) -> List[torch.Tensor]:
    # train.py: answer_prompts = [tokenizer.encode("The answer is:"), tokenizer.encode("The next step result is:")]
    ap = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in ANSWER_PROMPTS]
    # train.py removes BOS if present
    if tokenizer.bos_token_id is not None and len(ap[0]) > 0 and ap[0][0].item() == tokenizer.bos_token_id:
        ap = [x[1:] for x in ap]
    return ap


def _find_subsequence(tokens_1d: torch.Tensor, sub_1d: torch.Tensor) -> Optional[int]:
    """
    Returns the first index i such that tokens[i:i+len(sub)] == sub, else None.
    Matches the logic in your train.py get_answer_token_position (unfold + all).
    """
    if tokens_1d.dim() != 1 or sub_1d.dim() != 1:
        raise ValueError("Expected 1D tensors")
    sub_1d = sub_1d.to(tokens_1d.device)
    n, m = tokens_1d.numel(), sub_1d.numel()
    if m == 0 or n < m:
        return None
    # (n-m+1, m)
    windows = tokens_1d.unfold(0, m, 1)
    matches = (windows == sub_1d).all(dim=1).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def _build_decoder_ids_and_answer_pos(
    model: CODI,
    tokenizer: transformers.PreTrainedTokenizer,
    answer_value: Any,
    remove_eos: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int, str]:
    """
    Replicates train.py's decoder formatting enough to compute model_answer_position.

    train.py:
      answers list contains strings like "The answer is: 18"
      answers_id = tokenize(answers)
      answers_id += [eos] (always)
      then decoder_input_ids = [eot_id] + answers_id   (or [eot_id, eos] + answers_id if not remove_eos)

    We do the same, then:
      model_answer_position = index of token right after the answer prompt
      target_token_id = decoder_input_ids[model_answer_position]
    """
    # Build the natural language answer string (same style as train.py)
    try:
        ans_str = str(int(float(answer_value)))
    except Exception:
        ans_str = str(answer_value)

    answer_text = f"The answer is: {ans_str}"

    # Tokenize answer_text
    ids = tokenizer.encode(answer_text, add_special_tokens=False)
    # Always append eos (train.py does answers_id = answers_id + [eos])
    ids = ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])

    # Prepend eot (+ eos depending on remove_eos)
    if remove_eos:
        dec = [model.eot_id] + ids
    else:
        # train.py: [eot_id, eos] + answers_id
        eos = tokenizer.eos_token_id
        dec = [model.eot_id] + ([eos] if eos is not None else []) + ids

    decoder_input_ids = torch.tensor(dec, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Find model_answer_position (token index right after "The answer is:")
    ap = _answer_prompt_token_tensors(tokenizer)
    tok_1d = decoder_input_ids[0]
    start = _find_subsequence(tok_1d, ap[0])
    if start is None:
        # Fall back: if something changed in tokenizer spacing etc.
        raise RuntimeError("Could not find answer prompt tokens inside decoder_input_ids. Check tokenizer/use_fast/remove_eos.")
    model_answer_position = start + int(ap[0].numel())

    # Target is the *first answer token* (same thing you distill/train against via model_answer_position)
    target_token_id = int(decoder_input_ids[0, model_answer_position].item())
    target_label = tokenizer.decode([target_token_id])

    return decoder_input_ids, model_answer_position, target_token_id, target_label


# -----------------------------
# Main: run logit lens
# -----------------------------

POSITION_AFTER_PROMPT_BOT = "after_prompt_bot"
POSITION_DECODER_READOUT = "decoder_readout"


def run_logit_lens(
    model: CODI,
    tokenizer: transformers.PreTrainedTokenizer,
    question_data: List[Dict[str, torch.Tensor]],
    questions: List[str],
    procedures: List[str],
    answers: List[Any],
    training_args: TrainingArguments,
    example_idx: Optional[int] = 0,
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Track P(first answer token) across:
      - after_prompt_bot (last prompt token, i.e. BoT location)
      - latent_0..latent_{K-1} (each latent token)
      - decoder_readout (position = model_answer_position - 1 during decoder forward)

    This aligns with your train.py/model.py supervision point.
    """
    device = next(model.parameters()).device
    vocab_size = model.codi.config.vocab_size
    lm_head, ln_f = _get_lm_head_and_ln_f(model, model.model_name)
    lm_head.eval()
    if ln_f is not None:
        ln_f.eval()

    K = int(training_args.inf_latent_iterations)
    position_names = [POSITION_AFTER_PROMPT_BOT] + [f"latent_{i}" for i in range(K)] + [POSITION_DECODER_READOUT]

    all_results = []

    # Flatten question_data batches into (batch, row_index)
    idx_to_batch = []
    for batch in question_data:
        for i in range(batch["input_ids"].size(0)):
            idx_to_batch.append((batch, i))

    if example_idx is not None:
        indices_to_run = [example_idx]
    else:
        n = len(questions) if max_examples is None else min(len(questions), max_examples)
        indices_to_run = list(range(n))

    for idx in indices_to_run:
        if idx >= len(questions) or idx >= len(idx_to_batch):
            continue

        batch, mini = idx_to_batch[idx]
        single_batch = {
            "input_ids": batch["input_ids"][mini : mini + 1],
            "attention_mask": batch["attention_mask"][mini : mini + 1],
        }

        # Build decoder ids + target token (paper/your-train-style)
        decoder_input_ids, model_answer_pos, target_token_id, target_label = _build_decoder_ids_and_answer_pos(
            model=model,
            tokenizer=tokenizer,
            answer_value=answers[idx],
            remove_eos=bool(training_args.remove_eos),
            device=device,
        )

        result = {
            "example_idx": idx,
            "question": questions[idx],
            "procedure": procedures[idx],
            "answer": answers[idx],
            "positions": list(position_names),
            "target_token_id": target_token_id,
            "target_label": target_label,
            "model_answer_position": int(model_answer_pos),
            "heatmap": [],  # list length = len(position_names), each is {layer_i: prob}
        }

        with torch.no_grad():
            # ---- 1) Prompt+BoT forward (cache starts here) ----
            past_key_values = None
            out = model.codi(
                input_ids=single_batch["input_ids"],
                attention_mask=single_batch["attention_mask"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values

            # Decode at after_prompt_bot: last token of the prompt sequence (which ends with BoT token)
            # hidden_states[0]=embeds, hidden_states[1:]=blocks
            block_states = out.hidden_states[1:]
            row = {}
            for li, h in enumerate(block_states):
                probs = _decode_residual_with_ln(h[:, -1:, :], lm_head, ln_f, vocab_size)
                row[f"layer_{li}"] = _prob_of_token(probs, target_token_id)
            result["heatmap"].append(row)

            # Init latent embedding for iterative rollout
            latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)  # (1, 1, D)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # We'll extend the attention_mask during latent steps to be safe with left padding & caching
            base_mask = single_batch["attention_mask"]  # (1, Tprompt)

            # ---- 2) Latent rollout ----
            for lat in range(K):
                latent_mask = torch.cat(
                    [base_mask, torch.ones(1, 1 + lat, device=device, dtype=base_mask.dtype)],
                    dim=1,
                )

                out = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                    attention_mask=latent_mask,
                )
                past_key_values = out.past_key_values
                block_states = out.hidden_states[1:]

                # Decode at this latent token
                row = {}
                for li, h in enumerate(block_states):
                    probs = _decode_residual_with_ln(h[:, -1:, :], lm_head, ln_f, vocab_size)
                    row[f"layer_{li}"] = _prob_of_token(probs, target_token_id)
                result["heatmap"].append(row)

                # Next latent embedding
                latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # ---- 3) Decoder forward ----
            # Match your model.py: embed decoder_input_ids and feed with past_key_values
            emb_layer = model.get_embd(model.codi, model.model_name)
            decoder_embds = emb_layer(decoder_input_ids)  # (1, Tdec, D)

            # Build decoder attention mask (only strictly needed for left padding correctness)
            # Past length = len(prompt+BoT) + K latent tokens (each adds 1)
            # Provide a full mask length (past + current) if you want strictness like fix_attn_mask logic.
            decoder_mask = torch.ones((1, decoder_input_ids.size(1)), device=device, dtype=base_mask.dtype)

            full_mask = torch.cat(
                [base_mask, torch.ones((1, K), device=device, dtype=base_mask.dtype), decoder_mask],
                dim=1,
            )

            out = model.codi(
                inputs_embeds=decoder_embds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=full_mask,
            )
            block_states = out.hidden_states[1:]

            # Probe at decoder_readout = (model_answer_position - 1)
            readout_pos = int(model_answer_pos) - 1
            if readout_pos < 0 or readout_pos >= decoder_input_ids.size(1):
                raise RuntimeError(f"readout_pos={readout_pos} out of range for decoder length {decoder_input_ids.size(1)}")

            row = {}
            for li, h in enumerate(block_states):
                # take the hidden at readout_pos, decode it
                h_pos = h[:, readout_pos : readout_pos + 1, :]  # (1,1,D)
                probs = _decode_residual_with_ln(h_pos, lm_head, ln_f, vocab_size)
                row[f"layer_{li}"] = _prob_of_token(probs, target_token_id)
            result["heatmap"].append(row)

        all_results.append(result)

    out = {"positions": position_names, "results": all_results}

    # Summary when multiple examples
    if len(all_results) > 1:
        n_ex = len(all_results)
        position_names_out = all_results[0]["positions"]
        layer_keys = sorted(all_results[0]["heatmap"][0].keys(), key=lambda x: int(x.split("_")[1]))
        n_pos = len(position_names_out)
        n_layers = len(layer_keys)
        mean_heatmap = []
        for pos_i in range(n_pos):
            row = {}
            for layer_i, k in enumerate(layer_keys):
                row[k] = sum(r["heatmap"][pos_i].get(k, 0.0) for r in all_results) / n_ex
            mean_heatmap.append(row)
        per_position_mean = {}
        for pos_i, pname in enumerate(position_names_out):
            per_position_mean[pname] = sum(
                sum(r["heatmap"][pos_i].get(k, 0.0) for k in layer_keys) / n_layers
                for r in all_results
            ) / n_ex
        decoder_readout_mean = per_position_mean.get(POSITION_DECODER_READOUT, 0.0)
        out["summary"] = {
            "n_examples": n_ex,
            "mean_heatmap": mean_heatmap,
            "per_position_mean": per_position_mean,
            "decoder_readout_mean_p": decoder_readout_mean,
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    return out


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a short text summary when multiple examples were run."""
    summary = results.get("summary")
    if not summary:
        return
    n = summary["n_examples"]
    print(f"\n--- Logit lens summary (n={n} examples) ---")
    print(f"  decoder_readout mean P(target): {summary['decoder_readout_mean_p']:.4f}")
    per_pos = summary.get("per_position_mean", {})
    if per_pos:
        parts = [f"{k}: {v:.3f}" for k, v in per_pos.items()]
        print(f"  per-position mean P(target):  {', '.join(parts)}")
    print("---")


def plot_logit_lens_heatmap(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    use_mean: bool = True,
) -> Optional[Any]:
    """
    Heatmap: positions x layers, color = P(first answer token).
    If summary exists (multi-example run), plot mean heatmap unless use_mean=False.
    Otherwise uses first example.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    res_list = results.get("results", [])
    if not res_list:
        return None

    summary = results.get("summary")
    if summary and use_mean:
        positions = res_list[0]["positions"]
        heatmap_data = summary["mean_heatmap"]
        n_ex = summary["n_examples"]
        plot_title = f"Logit lens: P(first answer token) | mean over {n_ex} examples"
    else:
        r = res_list[0]
        positions = r["positions"]
        heatmap_data = r["heatmap"]
        plot_title = f"Logit lens: P(first answer token) | target: {r.get('target_label', '')}"

    n_pos = len(positions)
    layer_keys = sorted(heatmap_data[0].keys(), key=lambda x: int(x.split("_")[1]))
    n_layers = len(layer_keys)

    matrix = np.zeros((n_layers, n_pos))
    for pos_i, row in enumerate(heatmap_data):
        for layer_i, k in enumerate(layer_keys):
            matrix[layer_i, pos_i] = row.get(k, 0.0)

    fig, ax = plt.subplots(figsize=(max(7, n_pos * 0.75), max(4, n_layers * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(positions, rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Position")
    ax.set_ylabel("Layer")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(plot_title)

    plt.colorbar(im, ax=ax, label="P(target)")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# -----------------------------
# Loading model + GSM8k style data (as you had)
# -----------------------------

def load_model_and_data(
    ckpt_dir: str,
    model_name: str = "gpt2",
    data_name: str = "zen-E/GSM8k-Aug",
    batch_size: int = 1,
    inf_latent_iterations: int = 6,
    remove_eos: bool = True,
    use_prj: bool = True,
    prj_dim: int = 768,
):
    model_args = ModelArguments(model_name_or_path=model_name, lora_init=True, ckpt_dir=ckpt_dir)
    data_args = DataArguments(data_name=data_name, batch_size=batch_size)
    training_args = TrainingArguments(
        inf_latent_iterations=inf_latent_iterations,
        remove_eos=remove_eos,
        use_prj=use_prj,
        prj_dim=prj_dim,
        bf16=True,
    )

    task_type = TaskType.CAUSAL_LM
    target_modules = ["c_attn", "c_proj", "c_fc"] if "gpt2" in model_name else []
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = model.to("cuda").to(torch.bfloat16)
    model.eval()

    dataset = load_dataset(data_name)
    test_set = dataset["test"]

    questions, procedures, answers = [], [], []
    for ex in test_set:
        raw_q = ex["question"].strip().replace("  ", " ")
        raw_cot = ex.get("cot", "")
        if not raw_cot or not raw_cot.strip():
            continue
        parts = raw_cot.replace("<<", "").replace(">>", "").split()
        if len(parts) <= 1:
            continue
        truncated_cot = " ".join(parts[:-1])
        questions.append(f"{raw_q}")
        answers.append(float(ex["answer"].replace(",", "")))
        procedures.append(ex["cot"])

    # Tokenize prompt side and append BoT token exactly like your old code
    eval_steps = math.ceil(len(questions) / batch_size)
    question_data = []
    for i in range(eval_steps):
        start, end = i * batch_size, min((i + 1) * batch_size, len(questions))
        batch = tokenizer(
            questions[start:end],
            return_tensors="pt",
            padding="longest",
        )
        bot = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot)), dim=1)
        question_data.append({k: v.to("cuda") for k, v in batch.items()})

    return model, tokenizer, question_data, questions, procedures, answers, training_args


def main():
    parser = argparse.ArgumentParser(description="Logit lens analysis for CoDi (decoder-boundary faithful)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--data_name", type=str, default="zen-E/GSM8k-Aug")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inf_latent_iterations", type=int, default=6)
    parser.add_argument("--remove_eos", action="store_true", default=True)
    parser.add_argument("--use_prj", action="store_true", default=True)
    parser.add_argument("--no_prj", action="store_false", dest="use_prj")
    parser.add_argument("--example_idx", type=int, default=None, help="Single example index (if set, only this example; else use --max_examples)")
    parser.add_argument("--max_examples", type=int, default=1, help="Max examples to run when example_idx not set (default 1)")
    parser.add_argument("--output", type=str, default="outputs/logit_lens.json")
    parser.add_argument("--plot", action="store_true", help="Plot heatmap of P(target) over positions x layers")
    parser.add_argument("--plot_output", type=str, default=None, help="Path for heatmap image (default: output base + .png)")
    args = parser.parse_args()

    model, tokenizer, question_data, questions, procedures, answers, training_args = load_model_and_data(
        ckpt_dir=args.ckpt_dir,
        model_name=args.model_name,
        data_name=args.data_name,
        batch_size=args.batch_size,
        inf_latent_iterations=args.inf_latent_iterations,
        remove_eos=args.remove_eos,
        use_prj=args.use_prj,
    )

    example_idx = args.example_idx
    max_examples = args.max_examples
    if example_idx is not None:
        max_examples = None  # single example
    results = run_logit_lens(
        model=model,
        tokenizer=tokenizer,
        question_data=question_data,
        questions=questions,
        procedures=procedures,
        answers=answers,
        training_args=training_args,
        example_idx=example_idx,
        max_examples=max_examples,
        output_path=args.output,
    )
    n_run = len(results.get("results", []))
    print(f"Logit lens written to {args.output} ({n_run} examples)")

    _print_summary(results)

    if args.plot:
        plot_path = args.plot_output or (os.path.splitext(args.output)[0] + "_heatmap.png")
        fig = plot_logit_lens_heatmap(results, output_path=plot_path)
        if fig is None:
            print("Skipping heatmap (install matplotlib: pip install matplotlib)")
        else:
            print(f"Heatmap saved to {plot_path}")

    return results


if __name__ == "__main__":
    main()
