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
| Model | debug: `Qwen2.5-0.5B-Instruct` (or `Qwen3-0.6B`) → real: `Qwen2.5-1.5B-Instruct` (or `Qwen3-1.7B`, `Gemma-3-1B`) |
| Dataset | GSM8K (7.5k train / 1.3k test). Later: MATH, Countdown |
| Libraries | Unsloth (LoRA + 4-bit + vLLM fast generation) + TRL `GRPOTrainer` |
| GPU | Kaggle **T4** (vLLM needs compute capability ≥ 7.0; P100 is 6.0). **fp16**, not bf16. Single GPU |
| Tracking | Weights & Biases (`WANDB_API_KEY` as Kaggle secret) |

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

## Open decisions (for /grill-me when we get here)

- [ ] Qwen2.5 vs Qwen3 (Qwen3 has native thinking mode — may already be near-ceiling on GSM8K)
- [ ] Instruct vs base model (base shows bigger relative gains)
- [ ] `beta` = 0 vs 0.04
- [ ] Eval subset: full 1,319 vs fixed 300
- [ ] Resume strategy across sessions: HF Hub checkpoints vs Kaggle dataset versions

## Definition of done

- [ ] Baseline vs GRPO accuracy table (+ format-compliance rate) on identical eval set
- [ ] W&B reward/length/KL curves embedded in report
- [ ] ≥ 5 before/after reasoning traces, incl. one reward-hacking example if observed
- [ ] LoRA on HF Hub with model card
