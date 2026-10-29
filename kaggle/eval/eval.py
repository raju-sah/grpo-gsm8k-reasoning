"""Kaggle entrypoint (eval).

Evaluates baseline and GRPO-trained models on the fixed 300-question GSM8K subset on Kaggle T4.
Supports assistant prompt pre-filling (<think>\n) to evaluate strict XML reasoning compliance.
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
        "pip install unsloth_zoo",
        "pip install git+https://github.com/unslothai/unsloth.git",
        "pip install vllm vllm-bnb-plugin datasets pyyaml",
        "pip uninstall -y torchaudio",
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
    else:
        print(f"Pulling latest changes in {repo_dir}...")
        subprocess.run(f"git -C {repo_dir} pull", shell=True, check=False)

    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.chdir(repo_dir)


def main():
    setup_environment()

    repo_dir = "/kaggle/working/grpo-gsm8k-reasoning"
    src_dir = os.path.join(repo_dir, "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}"

    adapter_path = os.environ.get("ADAPTER_PATH")
    if not adapter_path:
        candidates = [
            "/kaggle/input/grpo-gsm8k-train/outputs/full_1.5b/final_adapter",
            "/kaggle/input/grpo-gsm8k-train/outputs/full_1.5b/checkpoint-500",
            "/kaggle/input/grpo-gsm8k-train/final_adapter",
        ]
        for c in candidates:
            if os.path.exists(c):
                adapter_path = c
                break

    # Recursive search fallback if not in standard candidates
    if not adapter_path and os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files:
                if "final_adapter" in root or "500" in root:
                    adapter_path = root
                    break
                elif not adapter_path:
                    adapter_path = root

    # Run separate subprocesses for each evaluation to isolate vLLM C++ CUDA contexts
    # 1. Baseline 1.5B Primed
    print("\n==========================================")
    print("Running Eval 1/2: Baseline Qwen2.5-1.5B (Primed with <think>\\n)")
    print("==========================================")
    cmd_base = [
        sys.executable,
        "-m",
        "grpo_reason.eval",
        "--model",
        "unsloth/Qwen2.5-1.5B-Instruct",
        "--prefill-think",
        "--run-name",
        "baseline-1.5b-primed",
        "--steps",
        "0",
        "--output-dir",
        "/kaggle/working/reports",
    ]
    subprocess.run(cmd_base, env=env, check=True)

    # 2. GRPO 1.5B Primed
    if adapter_path:
        print("\n==========================================")
        print(f"Running Eval 2/2: GRPO Qwen2.5-1.5B (Primed with <think>\\n, {adapter_path})")
        print("==========================================")
        cmd_grpo = [
            sys.executable,
            "-m",
            "grpo_reason.eval",
            "--model",
            adapter_path,
            "--prefill-think",
            "--run-name",
            "grpo-1.5b-primed",
            "--steps",
            "500",
            "--output-dir",
            "/kaggle/working/reports",
        ]
        subprocess.run(cmd_grpo, env=env, check=True)
    else:
        print("[Warning] Could not find trained adapter path in /kaggle/input/!")
        print(
            "Available in /kaggle/input/:",
            os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "None",
        )

    print("\nEvaluation complete! Output reports stored in /kaggle/working/reports.")


if __name__ == "__main__":
    main()
