"""GRPO training pipeline for reasoning using Unsloth + TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from grpo_reason.data import get_gsm8k_dataset
from grpo_reason.rewards import correctness_reward, format_reward, int_reward

REWARD_REGISTRY = {
    "correctness_reward": correctness_reward,
    "format_reward": format_reward,
    "int_reward": int_reward,
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = [
        d for d in path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def resolve_repo_id(repo_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user_info = api.whoami()
        user = user_info.get("name")
        if user:
            repo_name = repo_id.split("/")[-1]
            return f"{user}/{repo_name}"
    except Exception:
        pass
    return repo_id


def get_hub_callback(hub_cfg: dict | None):
    if not hub_cfg or not hub_cfg.get("push_to_hub", False):
        return None

    from transformers import TrainerCallback

    repo_id = resolve_repo_id(hub_cfg.get("repo_id", "grpo-checkpoints"))
    is_private = hub_cfg.get("private", True)

    class HubCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
            if not checkpoint_dir.exists():
                return
            print(f"\n[HubCallback] Pushing checkpoint-{step} to HF Hub: {repo_id}...")
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                api.create_repo(repo_id=repo_id, private=is_private, exist_ok=True)
                api.upload_folder(
                    folder_path=str(checkpoint_dir),
                    repo_id=repo_id,
                    commit_message=f"Upload LoRA checkpoint step {step}",
                )
                print(f"[HubCallback] Successfully pushed checkpoint-{step} to {repo_id}")
            except Exception as e:
                print(f"[HubCallback] Warning: Failed to push checkpoint to HF Hub: {e}")

    return HubCheckpointCallback()


def train_grpo(
    config_path: str | Path,
    resume: bool = False,
    checkpoint_path: str | None = None,
):
    cfg = load_config(config_path)
    print(f"Loaded config from {config_path}:")
    print(yaml.dump(cfg, default_flow_style=False))

    model_name = cfg["model_name"]
    output_dir = Path(cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_cfg = cfg.get("grpo", {})
    max_prompt_length = grpo_cfg.get("max_prompt_length", 256)
    max_completion_length = grpo_cfg.get("max_completion_length", 512)
    max_seq_length = max_prompt_length + max_completion_length

    # 1. Initialize Model & Tokenizer with Unsloth 4-bit
    from unsloth import FastLanguageModel

    print(f"Loading base model {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=cfg.get("load_in_4bit", True),
        fast_inference=True,
        gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
    )

    # 2. Attach LoRA adapters
    lora_cfg = cfg.get("lora", {})
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("rank", 16),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Load dataset
    print("Loading GSM8K training dataset...")
    train_dataset = get_gsm8k_dataset(split="train")

    # 4. Resolve reward functions
    reward_names = cfg.get(
        "rewards", ["correctness_reward", "format_reward", "int_reward"]
    )
    reward_funcs = [REWARD_REGISTRY[name] for name in reward_names]
    print(f"Active reward functions: {reward_names}")

    # 5. Build GRPOConfig
    from trl import GRPOConfig, GRPOTrainer

    report_to = grpo_cfg.get("report_to", "wandb")
    if os.environ.get("WANDB_DISABLED") == "true" or (
        report_to == "wandb" and not os.environ.get("WANDB_API_KEY")
    ):
        print(
            "[Notice] WANDB_API_KEY not set or wandb disabled. Falling back to report_to='none'."
        )
        report_to = "none"
        os.environ["WANDB_DISABLED"] = "true"

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(grpo_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type="cosine",
        logging_steps=grpo_cfg.get("logging_steps", 1),
        max_steps=grpo_cfg.get("max_steps", 50),
        save_steps=grpo_cfg.get("save_steps", 25),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 1),
        num_generations=grpo_cfg.get("num_generations", 8),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        beta=float(grpo_cfg.get("beta", 0.0)),
        fp16=grpo_cfg.get("fp16", True),
        bf16=False,
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
        report_to=report_to,
        save_strategy="steps",
        save_total_limit=3,
    )

    callbacks = []
    hub_cb = get_hub_callback(cfg.get("hub"))
    if hub_cb:
        callbacks.append(hub_cb)

    # 6. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    # 7. Check for resumption
    resume_from = checkpoint_path
    if resume and not resume_from:
        resume_from = find_latest_checkpoint(output_dir)

    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
    else:
        print("Starting training from step 0...")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final LoRA adapter
    final_output = output_dir / "final_adapter"
    print(f"Saving final adapter to {final_output}...")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    # Final hub upload if configured
    hub_cfg = cfg.get("hub")
    if hub_cfg and hub_cfg.get("push_to_hub", False):
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            repo_id = resolve_repo_id(hub_cfg.get("repo_id", "grpo-checkpoints"))
            print(f"Pushing final adapter to HF Hub: {repo_id}...")
            api.upload_folder(
                folder_path=str(final_output),
                repo_id=repo_id,
                commit_message="Upload final trained LoRA adapter",
            )
            print("Successfully uploaded final adapter to HF Hub.")
        except Exception as e:
            print(f"Warning: Failed to upload final adapter: {e}")

    print("GRPO training completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for GSM8K reasoning.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if found")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to resume from")

    args = parser.parse_args()
    train_grpo(
        config_path=args.config,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
