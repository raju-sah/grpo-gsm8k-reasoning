"""Kaggle entrypoint (eval).

Evaluates baseline (or checkpointed) models on the fixed 300-question GSM8K subset on Kaggle T4.
Secrets needed: HF_TOKEN, WANDB_API_KEY (configured in Kaggle Notebook -> Add-ons -> Secrets).
"""

import os
import subprocess
import sys


def setup_environment():
    print("=== Kaggle Eval Environment Setup ===")
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        try:
            hf_token = user_secrets.get_secret("HF_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print("HF_TOKEN successfully loaded from Kaggle Secrets.")
        except Exception as e:
            print(f"Note: HF_TOKEN not found in Kaggle Secrets ({e}).")

        try:
            wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = wandb_key
            print("WANDB_API_KEY successfully loaded from Kaggle Secrets.")
        except Exception as e:
            print(f"Note: WANDB_API_KEY not found in Kaggle Secrets ({e}).")
    except ImportError:
        print("Not running in Kaggle environment (kaggle_secrets not available).")

    # Install unsloth & vllm for fast T4 inference if not installed
    print("Ensuring dependencies are installed...")
    cmds = [
        "pip install --upgrade --no-cache-dir pip",
        "pip install --no-deps bitsandbytes",
        "pip install --no-deps trl",
        "pip install git+https://github.com/unslothai/unsloth.git",
        "pip install vllm datasets pyyaml",
    ]
    for cmd in cmds:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=False)

    # Clone or ensure grpo_reason is on sys.path
    repo_url = "https://github.com/raju-sah/grpo-gsm8k-reasoning.git"
    working_dir = "/kaggle/working"
    repo_dir = os.path.join(working_dir, "grpo-gsm8k-reasoning")

    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url}...")
        subprocess.run(f"git clone {repo_url} {repo_dir}", shell=True, check=True)

    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(repo_dir)


def main():
    setup_environment()

    from grpo_reason.eval import evaluate_model

    print("\n==========================================")
    print("Running Baseline Eval: Qwen2.5-0.5B-Instruct")
    print("==========================================")
    evaluate_model(
        model_name_or_path="unsloth/Qwen2.5-0.5B-Instruct",
        subset_size=300,
        seed=42,
        max_new_tokens=512,
        run_name="baseline-0.5b",
        steps=0,
        output_dir="/kaggle/working/reports",
    )

    print("\n==========================================")
    print("Running Baseline Eval: Qwen2.5-1.5B-Instruct")
    print("==========================================")
    evaluate_model(
        model_name_or_path="unsloth/Qwen2.5-1.5B-Instruct",
        subset_size=300,
        seed=42,
        max_new_tokens=512,
        run_name="baseline-1.5b",
        steps=0,
        output_dir="/kaggle/working/reports",
    )

    print("\nEvaluation complete! Output reports stored in /kaggle/working/reports.")


if __name__ == "__main__":
    main()
