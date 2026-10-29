"""Kaggle entrypoint (train).

Runs GRPO training on Kaggle T4 GPU using Unsloth 4-bit + vLLM + TRL GRPOTrainer.
Secrets needed: HF_TOKEN, WANDB_API_KEY (configured in Kaggle Notebook -> Add-ons -> Secrets).
"""

import os
import subprocess
import sys


def setup_environment():
    print("=== Kaggle Training Environment Setup ===")
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

    # Install unsloth & vllm for fast T4 training
    print("Ensuring dependencies are installed...")
    cmds = [
        "pip install --upgrade --no-cache-dir pip",
        "pip install --no-deps bitsandbytes",
        "pip install --no-deps trl",
        "pip install unsloth_zoo",
        "pip install git+https://github.com/unslothai/unsloth.git",
        "pip install vllm vllm-bnb-plugin datasets pyyaml wandb huggingface_hub",
    ]
    for cmd in cmds:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=False)

    # Login to services
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import login

            login(token=os.environ["HF_TOKEN"])
            print("Logged into Hugging Face Hub.")
        except Exception as e:
            print(f"HF login warning: {e}")

    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.login(key=os.environ["WANDB_API_KEY"])
            print("Logged into Weights & Biases.")
        except Exception as e:
            print(f"W&B login warning: {e}")

    # Clone or ensure grpo_reason is on sys.path
    repo_url = "https://github.com/raju-sah/grpo-gsm8k-reasoning.git"
    working_dir = "/kaggle/working"
    repo_dir = os.path.join(working_dir, "grpo-gsm8k-reasoning")

    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url}...")
        subprocess.run(f"git clone {repo_url} {repo_dir}", shell=True, check=True)
    else:
        print(f"Pulling latest changes in {repo_dir}...")
        subprocess.run("git pull origin main", shell=True, cwd=repo_dir, check=False)

    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(repo_dir)


def main():
    setup_environment()

    from grpo_reason.train import train_grpo

    # Default to debug run (0.5B, 50 steps); switch config for full 1.5B run
    config_file = os.environ.get("GRPO_CONFIG", "configs/grpo_debug_0.5b.yaml")
    resume = os.environ.get("RESUME_CHECKPOINT", "true").lower() in ("true", "1")

    print("\n==========================================")
    print(f"Starting GRPO Training with {config_file}")
    print(f"Auto-resume: {resume}")
    print("==========================================")

    train_grpo(
        config_path=config_file,
        resume=resume,
    )

    print("\nTraining run finished! Check outputs in /kaggle/working/outputs/.")


if __name__ == "__main__":
    main()
