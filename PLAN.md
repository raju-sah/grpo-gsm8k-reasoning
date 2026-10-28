# PLAN — Teach a Small Model to Reason with RL (GRPO)

> Project 2 of 3 (flagship) · Difficulty ⭐⭐⭐⭐ · GPU: heavy (10–25 h Kaggle T4) · Weeks 3–4

## What it is

**GRPO — Group Relative Policy Optimization** (DeepSeekMath; used for DeepSeek-R1 / R1-Zero).
Outputs are **not** fed back as SFT training text. Instead:

1. For each question the model samples **G = 8** completions.
2. A reward function scores each (correct = 2.0, wrong = 0, + small format bonus).
3. Advantage is relative to the group:

   $$A_i = \frac{r_i - \operatorname{mean}(r_1..r_G)}{\operatorname{std}(r_1..r_G)}$$

4. Policy-gradient update: tokens of above-average answers become more likely, below-average
   less likely. Optional KL penalty (`beta`) keeps the policy near the reference model.

The group mean is the baseline → **no critic/value model** (unlike PPO) → fits on small GPUs.

> [!NOTE]
> If all 8 answers get the same reward, std = 0 and the question teaches nothing. Target a
> dataset the model solves **~20–80%** of the time.

## Requirements

| Item | Choice |
|---|---|
| Model | debug: `unsloth/Qwen2.5-0.5B-Instruct` → real: `unsloth/Qwen2.5-1.5B-Instruct` |
| Dataset | GSM8K (7.5k train / fixed 300-question test subset, seed 42) |
| Libraries | Unsloth (LoRA + 4-bit + vLLM fast generation) + TRL `GRPOTrainer` + `huggingface_hub` |
| GPU | Kaggle **T4** (vLLM needs compute capability ≥ 7.0; P100 is 6.0). **fp16**, not bf16. Single GPU |
| Tracking | Weights & Biases (`WANDB_API_KEY` as Kaggle secret) |
| Hub | Hugging Face Hub (`HF_TOKEN` as Kaggle secret) for checkpoint push/resume |

## Feasibility on Kaggle

- Bottleneck = generation (8 × ≤512 tokens per question).
- Qwen 1.5B + LoRA on T4 ≈ 30–90 s/step → 300–500 steps ≈ **4–10 h**.
- With baseline/final evals + failed runs: **~10–25 GPU h total**.
- 12 h session limit → **checkpoint every 50 steps** to `/kaggle/working`, push to HF Hub,
  resume from checkpoint in the next session.

**Realistic expectations:** big gain in format compliance; a few to ~10+ points GSM8K accuracy.
No R1-level "aha moments" at 1.5B.

## Steps

### Phase 1 — Baseline & debug (Week 3, ~6–10 GPU h)
- [ ] 1. **Baseline eval** (`src/grpo_reason/eval.py`): untouched model on GSM8K test (or fixed
      300-question subset, seed 42), greedy decoding, same prompt as training. Record in
      `reports/results.md`. **This number is the whole story of the project.**
- [ ] 2. **Format** (`prompts.py`): system prompt → `<think> ... </think><answer> 42 </answer>`.
- [ ] 3. **Reward functions** (`rewards.py`, unit-tested):
  - `correctness_reward`: 2.0 if extracted answer == gold, else 0
  - `format_reward`: 0.5 if tags present & well-formed
  - optional `int_reward` (answer is a number), length penalty against rambling
- [ ] 4. **Debug run on 0.5B** (`configs/grpo_debug_0.5b.yaml`, ~50 steps): pipeline works,
      reward std > 0, checkpoints saved and resumable.

### Phase 2 — Real run (Week 4, ~12–20 GPU h)
- [ ] 5. **Configure GRPO** (`configs/grpo_1.5b.yaml`):
  ```python
  GRPOConfig(
      num_generations=8, max_completion_length=512,
      learning_rate=5e-6, beta=0.0,  # or 0.04 for KL penalty
      per_device_train_batch_size=8, gradient_accumulation_steps=1,
      max_steps=500, save_steps=50, fp16=True, use_vllm=True,
      logging_steps=1, report_to="wandb",
  )
  ```
  LoRA rank 16–64, 4-bit load via Unsloth.
- [ ] 6. **Train & watch**: mean reward ↑, reward std > 0, completion length, KL.
      Flat reward at 0 → task too hard → easier model/dataset or short SFT warm-up (~200 formatted examples).
- [ ] 7. **Final eval**: same prompt, same test set, greedy. Compare to baseline.
- [ ] 8. **Publish**: LoRA → HF Hub; report with reward curves, before/after table, example traces.

> [!WARNING]
> **Reward hacking is common** — tags with garbage inside, repeated answers. Always read sample
> completions (log 4 per N steps to `reports/samples/`), not just the reward curve.

## Decisions (resolved via /grill-me)

- [x] **Model family & size**: Qwen2.5 (0.5B for debug, 1.5B for main run) — supported on T4 fp16 + Unsloth 4-bit, sits in the 30–70% GSM8K reward-variance sweet spot.
- [x] **Base vs Instruct**: Instruct (`Qwen2.5-0.5B-Instruct` / `Qwen2.5-1.5B-Instruct`) — chat template compatibility prevents early zero-variance format failure within 10–25h GPU budget.
- [x] **Eval subset**: Fixed 300-question subset (`seed=42`) across baseline, debug, and final evals (~5 min runtime on T4, saves GPU quota, ~2.8% SE).
- [x] **Reward shaping & anti-hacking**: Weights: Correctness 2.0, Format 0.5, Int 0.25. Strict single-tag pair check (exactly 1 `<think>`, 1 `</think>`, 1 `<answer>`, 1 `</answer>`) prevents tag repetition. No artificial length penalty to preserve reasoning emergence up to 512 tokens.
- [x] **KL penalty (`beta`)**: `0.0` — eliminates reference model forward passes, saves T4 VRAM and ~25–35% compute on verifiable rule-based math; LoRA acts as implicit regularizer.
- [x] **Checkpoint & resume strategy**: Push LoRA checkpoints (~50 MB) directly to private HF Hub repo every 50 steps (`rajucode/grpo-gsm8k-checkpoints`), keeping local copies in `/kaggle/working`. Kaggle secrets: `HF_TOKEN`, `WANDB_API_KEY`.

## Definition of done

- [ ] Baseline vs GRPO accuracy table (+ format-compliance rate) on identical eval set
- [ ] W&B reward/length/KL curves embedded in report
- [ ] ≥ 5 before/after reasoning traces, incl. one reward-hacking example if observed
- [ ] LoRA on HF Hub with model card
