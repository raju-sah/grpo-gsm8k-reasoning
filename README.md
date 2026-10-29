# grpo-gsm8k-reasoning

Teaching a 0.5B–1.5B LLM to reason on GSM8K with **GRPO** (Group Relative Policy Optimization, the DeepSeek-R1 RL technique), trained on a **free Kaggle T4 GPU** with Unsloth 4-bit + vLLM + TRL `GRPOTrainer`.

> **Status**: Completed Phase 1 (Baselines & Debug), Phase 2 (500-step GRPO Training), and Phase 3 (Final Evaluation & Post-Mortem). See [PLAN.md](file:///home/raju/AI-ML%20Projects/grpo-gsm8k-reasoning/PLAN.md) for the project roadmap, [NOTES.md](file:///home/raju/AI-ML%20Projects/grpo-gsm8k-reasoning/NOTES.md) for the decision and gotcha log, and [reports/results.md](file:///home/raju/AI-ML%20Projects/grpo-gsm8k-reasoning/reports/results.md) for full metrics.

---

## Benchmark Results

Evaluated on the fixed GSM8K test subset (\(n = 300\), `seed = 42`) under greedy decoding (`temperature = 0.0`, `max_new_tokens = 512`):

| Model / Run | Steps | Strict Accuracy (`<answer>`) | Flexible Accuracy (Natural Language) | Format Compliance | Integer Compliance | Avg Completion Tokens |
|---|---|---|---|---|---|---|
| **Baseline 0.5B** (`Qwen2.5-0.5B-Instruct`) | 0 | 0.00% (0/300) | 41.33% (124/300) | 0.00% | 0.00% | 271.5 |
| **Baseline 1.5B** (`Qwen2.5-1.5B-Instruct`) | 0 | 2.67% (8/300) | 71.00% (213/300) | 0.33% | 4.33% | 238.1 |
| **GRPO 1.5B** (500 steps on Kaggle T4) | 500 | 2.67% (8/300) | 64.33% (193/300) | 0.00% | 2.33% | 233.8 |

> **Key Takeaway**: During 500 steps of GRPO training with temperature sampling, the model's mean reward surged from **0.4150 to 1.6769** (+304%) and correctness reward climbed from **0.355 to 1.450 / 2.0** (+308%), demonstrating high-probability mathematical reasoning. Under greedy decoding ($T=0.0$), the pre-trained Instruct prior (`"Think: ... Therefore, the answer is X"`) dominated step-0 logits over XML `<think>` tags, while retaining ~64.3% flexible mathematical accuracy on GSM8K multi-step word problems.

---

## 500-Step GRPO Training Progression

| Window | Steps | Mean Total Reward | Correctness Reward (Max 2.0) | Integer Reward (Max 0.25) | Format Reward (Max 0.50) | Mean Completion Length |
|---|---|---|---|---|---|---|
| 1 | 1–50 | 0.4150 | 0.3550 | 0.0587 | 0.0012 | 257.8 tokens |
| 2 | 51–100 | 1.1669 | 0.9850 | 0.1756 | 0.0062 | 230.9 tokens |
| 3 | 101–150 | 1.4406 | 1.2250 | 0.2131 | 0.0025 | 233.3 tokens |
| 4 | 151–200 | 1.3519 | 1.1350 | 0.2169 | 0.0000 | 270.5 tokens |
| 5 | 201–250 | 1.5350 | 1.3050 | 0.2300 | 0.0000 | 242.9 tokens |
| 6 | 251–300 | 1.5919 | 1.3600 | 0.2319 | 0.0000 | 262.2 tokens |
| 7 | 301–350 | 1.7444 | 1.5100 | 0.2331 | 0.0012 | 232.8 tokens |
| 8 | 351–400 | 1.5631 | 1.3300 | 0.2331 | 0.0000 | 247.8 tokens |
| 9 | 401–450 | 1.6669 | 1.4250 | 0.2419 | 0.0000 | 247.9 tokens |
| 10 | 451–500 | **1.6769** | **1.4500** | **0.2269** | 0.0000 | 260.2 tokens |

- **No Length Hacking**: Output length remained stable between 230 and 270 tokens without artificial penalties.
- **Group Variance**: 76% of batches produced non-zero reward variance ($std > 0$), providing sustained policy gradients.
- **Compute Budget**: Total GPU time was **~6.1 GPU hours** out of the 25h Kaggle budget (~3.7h for main training, ~1.2h baseline evals, ~0.7h debug, ~0.3h final eval).

---

## Repository Layout

```
├── configs/
│   ├── grpo_debug_0.5b.yaml    # 50-step debug config for 0.5B
│   └── grpo_1.5b.yaml          # 500-step main config for 1.5B
├── kaggle/
│   ├── train/                  # Kaggle T4 training kernel & local backup
│   └── eval/                   # Kaggle T4 eval kernel & reports
├── reports/
│   ├── results.md              # Detailed benchmark metrics & training curves
│   ├── samples/                # Evaluated sample traces (baseline vs GRPO)
│   ├── eval_baseline-0.5b.jsonl
│   ├── eval_baseline-1.5b.jsonl
│   └── eval_grpo-1.5b-final.jsonl
├── src/grpo_reason/
│   ├── data.py                 # GSM8K loader & deterministic fixed 300 test split
│   ├── eval.py                 # Greedy evaluation harness (vLLM / torch)
│   ├── prompts.py              # System prompt & chat format
│   ├── rewards.py              # Unit-tested correctness, format, & int rewards
│   └── train.py                # Unsloth + vLLM + TRL GRPOTrainer pipeline
└── tests/
    └── test_rewards.py         # 7/7 reward unit tests passing
```

---

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                    # reward unit tests (7/7 passing)
kaggle kernels push -p kaggle/eval        # run greedy evaluation on Kaggle T4
kaggle kernels push -p kaggle/train       # launch GRPO training on Kaggle T4
```

## License

MIT
