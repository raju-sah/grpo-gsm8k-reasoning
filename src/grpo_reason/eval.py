"""Greedy-decode evaluation harness on GSM8K fixed test subset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from grpo_reason.data import get_gsm8k_eval_subset
from grpo_reason.rewards import (
    correctness_reward,
    extract_answer,
    format_reward,
    int_reward,
    normalize_number,
)


def load_eval_model_and_tokenizer(
    model_name_or_path: str,
    base_model_name: str | None = None,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
):
    """Load model and tokenizer using Unsloth if available, otherwise transformers/peft."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except ImportError:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name or model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if base_model_name:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )

        if device == "cpu":
            model.to("cpu")
        model.eval()
        return model, tokenizer, "transformers"


def generate_greedy(
    model: Any,
    tokenizer: Any,
    backend: str,
    prompts_text: list[str],
    max_new_tokens: int = 512,
    lora_path: str | None = None,
) -> list[str]:
    """Generate greedy completions for a list of formatted prompt texts."""
    completions: list[str] = []

    # Check if vLLM / fast_generate is supported on the unsloth model
    if backend == "unsloth" and hasattr(model, "fast_generate"):
        try:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
            kwargs = {}
            if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
                try:
                    from vllm.lora.request import LoRARequest

                    kwargs["lora_request"] = LoRARequest("grpo_adapter", 1, lora_path)
                    print(f"Applying vLLM LoRARequest with adapter: {lora_path}")
                except Exception as le:
                    print(f"Notice: Could not initialize LoRARequest ({le})")

            outputs = model.fast_generate(prompts_text, sampling_params=sampling_params, **kwargs)
            for out in outputs:
                completions.append(out.outputs[0].text)
            return completions
        except (RuntimeError, ValueError, TypeError) as e:
            # Fall back to standard generate if fast_generate fails
            print(f"vLLM fast_generate failed ({e}), falling back to standard generate...")

    import torch

    device = next(model.parameters()).device
    for prompt in prompts_text:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return completions


def evaluate_model(
    model_name_or_path: str,
    base_model_name: str | None = None,
    subset_size: int = 300,
    seed: int = 42,
    max_new_tokens: int = 512,
    run_name: str = "baseline",
    steps: int = 0,
    output_dir: str | Path = "reports",
    prefill_think: bool = False,
) -> dict[str, Any]:
    """Evaluate a model on the fixed GSM8K test subset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GSM8K eval subset (n={subset_size}, seed={seed})...")
    dataset = get_gsm8k_eval_subset(subset_size=subset_size, seed=seed)

    print(f"Loading model '{model_name_or_path}'...")
    model, tokenizer, backend = load_eval_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        base_model_name=base_model_name,
    )

    print(f"Formatting {len(dataset)} prompts with chat template (prefill_think={prefill_think})...")
    formatted_prompts = [
        tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        + ("<think>\n" if prefill_think else "")
        for item in dataset
    ]

    is_lora = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    lora_path = model_name_or_path if is_lora else None

    print(f"Generating completions (greedy, max_new_tokens={max_new_tokens})...")
    completions = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        prompts_text=formatted_prompts,
        max_new_tokens=max_new_tokens,
        lora_path=lora_path,
    )

    if prefill_think:
        # Prepend the pre-filled <think>\n to reconstruct the full assistant output
        completions = ["<think>\n" + c for c in completions]

    # Format into completion structures expected by reward functions
    completion_dicts = [[{"role": "assistant", "content": c}] for c in completions]
    gold_answers = [item["answer"] for item in dataset]

    corr_rewards = correctness_reward(completion_dicts, answer=gold_answers)
    fmt_rewards = format_reward(completion_dicts)
    integer_rewards = int_reward(completion_dicts)

    n = len(dataset)
    acc = sum(r == 2.0 for r in corr_rewards) / n
    fmt_rate = sum(r > 0.0 for r in fmt_rewards) / n
    int_rate = sum(r > 0.0 for r in integer_rewards) / n

    # Token lengths
    token_lengths = [len(tokenizer.encode(c)) for c in completions]
    avg_tokens = sum(token_lengths) / n if n > 0 else 0.0

    metrics = {
        "run_name": run_name,
        "model": model_name_or_path,
        "steps": steps,
        "subset_size": n,
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 2),
        "format_compliance_pct": round(fmt_rate * 100, 2),
        "int_compliance_pct": round(int_rate * 100, 2),
        "avg_tokens": round(avg_tokens, 1),
    }

    # Save detailed outputs
    jsonl_path = output_dir / f"eval_{run_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item, c, cr, fr, ir, toks in zip(
            dataset, completions, corr_rewards, fmt_rewards, integer_rewards, token_lengths
        ):
            record = {
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_solution": item.get("gold_solution", ""),
                "completion": c,
                "extracted_answer": extract_answer(c),
                "normalized_answer": normalize_number(extract_answer(c)),
                "correctness_reward": cr,
                "format_reward": fr,
                "int_reward": ir,
                "token_length": toks,
            }
            f.write(json.dumps(record) + "\n")

    # Save samples file (first 5 + representative examples)
    samples_path = samples_dir / f"{run_name}_samples.md"
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"# Sample Completions: {run_name}\n\n")
        f.write(f"- Model: `{model_name_or_path}`\n")
        f.write(f"- Accuracy: {metrics['accuracy_pct']}%\n")
        f.write(f"- Format Compliance: {metrics['format_compliance_pct']}%\n\n")
        for i, (item, c, cr) in enumerate(zip(dataset, completions, corr_rewards)):
            if i >= 5:
                break
            status = "CORRECT" if cr == 2.0 else "INCORRECT"
            f.write(f"### Example {i + 1} [{status}]\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Gold Answer:** `{item['answer']}`\n\n")
            f.write(f"**Model Completion:**\n```\n{c}\n```\n\n---\n\n")

    # Append to results.md
    results_path = output_dir / "results.md"
    row = (
        f"| {run_name} | `{model_name_or_path}` | {steps} | "
        f"{metrics['accuracy_pct']}% | {metrics['format_compliance_pct']}% | {metrics['avg_tokens']} |\n"
    )
    if results_path.exists():
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(row)
    else:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("# Results\n\n")
            f.write("Eval set: GSM8K test, fixed subset (n=300, seed 42), greedy decoding.\n\n")
            f.write("| Run | Model | Steps | Accuracy | Format compliance | Avg tokens |\n")
            f.write("|---|---|---|---|---|---|\n")
            f.write(row)

    print("\n=== Eval Results ===")
    print(f"Accuracy: {metrics['accuracy_pct']}%")
    print(f"Format Compliance: {metrics['format_compliance_pct']}%")
    print(f"Int Compliance: {metrics['int_compliance_pct']}%")
    print(f"Avg Completion Tokens: {metrics['avg_tokens']}")
    print(f"Saved traces to: {jsonl_path}")
    print(f"Saved samples to: {samples_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K fixed test subset.")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path to checkpoint")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name if evaluating a LoRA adapter")
    parser.add_argument("--subset-size", type=int, default=300, help="Subset size (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection (default: 42)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--run-name", type=str, default="baseline", help="Run name identifier")
    parser.add_argument("--steps", type=int, default=0, help="Training step count (0 for baseline)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument(
        "--prefill-think",
        action="store_true",
        help="Pre-fill assistant turn with <think>\\n (standard DeepSeek-R1 evaluation method)",
    )

    args = parser.parse_args()
    evaluate_model(
        model_name_or_path=args.model,
        base_model_name=args.base_model,
        subset_size=args.subset_size,
        seed=args.seed,
        max_new_tokens=args.max_tokens,
        run_name=args.run_name,
        steps=args.steps,
        output_dir=args.output_dir,
        prefill_think=args.prefill_think,
    )


if __name__ == "__main__":
    main()
