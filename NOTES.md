# NOTES — GRPO Reasoning

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | LoRA checkpoint push to private HF Hub every 50 steps via `HF_TOKEN` + `WANDB_API_KEY` | Adapter checkpoints are ~50 MB, fast to push; guarantees resumption across Kaggle 12h timeouts without dataset API quotas or latency. |
| 2026-10-28 | KL penalty `beta = 0.0` | Eliminates ref model forward passes, saving T4 VRAM and ~25–35% compute; rule-based math verifier resists drift and LoRA implicitly regularizes. |
| 2026-10-28 | Reward shaping: 2.0 Correctness, 0.5 Format, 0.25 Int + strict single-tag pair check, no length penalty | Dense format/int rewards bootstrap early gradient; single-tag count rule stops tag repetition hacks; avoids punishing reasoning length up to 512 tokens. |
| 2026-10-28 | GSM8K eval on fixed 300-question subset (seed 42) | Fast (~5 min greedy eval on T4 vs ~25 min), saves precious GPU quota, ~2.8% standard error is plenty to verify 5–15 point reasoning gains. |
| 2026-10-28 | Instruct variants (`Qwen2.5-0.5B-Instruct` / `Qwen2.5-1.5B-Instruct`) | Chat template compliance prevents early format failure and flat-reward zero-variance collapse within tight 10–25h budget. |
| 2026-10-28 | Model family: Qwen2.5 (0.5B debug, 1.5B full run) | Battle-tested with Unsloth 4-bit + vLLM on T4 fp16; baseline accuracy sits right in 30–70% sweet spot for GRPO group reward variance ($std > 0$). |
| 2026-10-28 | Repo scaffolded | — |

## GPU budget tracker (Kaggle ~30 h/week)

| Date | Run | Model | Steps | GPU h | Outcome |
|---|---|---|---|---|---|
| 2026-10-28 | Baseline Eval | Qwen2.5-0.5B / 1.5B | 0 | ~1.2 h | 0.5B: 0.0% strict acc (41.33% flex), 0.0% format (271 tok). 1.5B: 2.67% strict acc (71.00% flex), 0.33% format (238 tok). Both attempt free-form reasoning without `<think>`/`<answer>` tags. |
| 2026-10-29 | Debug Train | Qwen2.5-0.5B | 50 | ~0.7 h | 50/50 steps completed (runtime 16.8 min). Checkpoints saved at step 25 and 50. 6% non-zero reward variance groups due to 0.5B low baseline capacity. End-to-end Unsloth 4-bit + vLLM + TRL GRPOTrainer pipeline fully verified on T4 without OOM. |
| 2026-10-29 | Main Train | Qwen2.5-1.5B | 500 | ~3.9 h | 500/500 steps completed (runtime ~3.7 h). Checkpoints saved every 50 steps up to checkpoint-500 + final_adapter. Mean reward increased from 0.4150 (steps 1–50) to 1.6769 (steps 451–500), correctness reward surged 4x from 0.355 to 1.450, with healthy non-zero group reward variance across 76% of batches. Stable completion length ~250 tokens without length hacking. |
| 2026-10-29 | Final Eval | Qwen2.5-1.5B (GRPO 500) | 500 | ~0.3 h | 300 test problems on Kaggle T4 (greedy decoding, runtime 1.4 min via vLLM). Strict XML accuracy: 2.67%, Flexible accuracy: 64.33% (193/300), format compliance: 0.0%, average tokens: 233.8. Total GPU budget utilized: ~6.1 GPU hours out of 25h budget. |

## Reference notes

- **GSM8K gold answer** is after `#### ` in the `answer` field; strip commas (`1,000` → `1000`).
- **Unsloth + vLLM** on T4: use `fast_inference=True`, `load_in_4bit=True`, fp16. Keep
  `gpu_memory_utilization` ≈ 0.5–0.6 so training has room alongside vLLM.
- **Zero-variance groups** (all 8 same reward) give zero gradient — track the fraction of
  such groups; if > ~70%, data is too easy or too hard.
- **Checkpoint resume:** `trainer.train(resume_from_checkpoint=path)`; copy checkpoint dirs
  out of `/kaggle/working` (HF Hub or Kaggle dataset) before the 12 h limit.
- Papers: DeepSeekMath (GRPO, arXiv 2402.03300), DeepSeek-R1 (arXiv 2501.12948).

## Gotchas & Key Learnings

- **Sampling Exploration vs. Greedy Argmax Discrepancy:**
  During GRPO training with temperature sampling ($T \approx 0.9$), the policy explores candidate completions, discovering `<answer>...</answer>` tags and capturing the +2.0 correctness reward (training correctness reward surged from 0.3550 to 1.4500). However, during evaluation with greedy decoding ($T = 0.0$), the base model's strong pre-trained Instruct prior dominates the initial token distribution at position 0 (`"Think: "` rather than `"<think>"`). Once this autoregressive branch begins, the model produces natural language conclusions (`"Therefore, the answer is 109."`) instead of explicit XML tags.
- **Format Reward Regex Rigidity:**
  The `format_reward` regex required full string matching `^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$`. Because completions frequently included natural conversational prefixes or omitted the exact enclosing `<think>`, format reward was 0 in 491 of 500 steps. In future iterations, awarding soft partial credit for individual tags (`+0.25` for `<answer>` existence) provides a continuous gradient rather than an all-or-nothing step function.
- **Assistant Prompt Pre-filling (Cold-Start):**
  In DeepSeek-R1 and similar reasoning RL frameworks, the chat template or generation prompt is typically pre-filled with `<think>\n` at the start of the assistant turn. This forces greedy decoding along the thinking trajectory and ensures proper tag closure.
- **Unsloth vLLM bitsandbytes plugin:** `vLLM >= 0.28` moved bitsandbytes out of tree. Using `load_in_4bit=True` with `fast_inference=True` requires `pip install vllm-bnb-plugin` and `unsloth_zoo`.
- **Kaggle script secrets:** In script mode (`kernel_type: script`), `kaggle_secrets.UserSecretsClient()` requires internet enabled and secrets properly configured; handle connection errors gracefully. When `WANDB_API_KEY` is not attached, `wandb.init()` throws `UsageError`; resolved by setting `WANDB_DISABLED="true"` and falling back to `report_to="none"`.
- **PyTorch & TorchAudio CUDA mismatch in Kaggle environment:** When `vllm` installs/updates PyTorch (`2.13.0+cu130`), the preinstalled Kaggle `torchaudio` package (`cu128`) remains in site-packages. During vLLM model registry inspection, `transformers.audio_utils` imports `torchaudio`, triggering a `RuntimeError` due to mismatched CUDA runtimes. Fix: Run `pip uninstall -y torchaudio` during Kaggle environment setup. GSM8K reasoning is purely text-based, so torchaudio is not required.

- _(fill in as you hit them)_
