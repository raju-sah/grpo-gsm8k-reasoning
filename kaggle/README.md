# Kaggle kernels (T4 only — vLLM needs compute capability ≥ 7.0)

```bash
kaggle kernels push -p kaggle/eval            # baseline / final eval
kaggle kernels push -p kaggle/train           # GRPO training
kaggle kernels status buggeddev/grpo-gsm8k-train
kaggle kernels output buggeddev/grpo-gsm8k-train -p kaggle/train/out
```

- `machine_shape: NvidiaTeslaT4` is set in each `kernel-metadata.json` (or pass `--accelerator NvidiaTeslaT4`).
- Attach secrets `HF_TOKEN`, `WANDB_API_KEY` once in the web UI (Add-ons → Secrets).
- 12 h session limit: checkpoints every 50 steps → push to HF Hub → resume next session.
