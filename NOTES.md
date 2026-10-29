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
| 2026-10-28 | Baseline Eval | Qwen2.5-0.5B / 1.5B | 0 | ~1.2 h | 0.5B: 0.0% acc, 0.0% format (271 tok). 1.5B: 2.67% acc, 0.33% format (238 tok). Both attempt free-form reasoning without <think>/<answer> tags. |

## Reference notes

- **GSM8K gold answer** is after `#### ` in the `answer` field; strip commas (`1,000` → `1000`).
- **Unsloth + vLLM** on T4: use `fast_inference=True`, `load_in_4bit=True`, fp16. Keep
  `gpu_memory_utilization` ≈ 0.5–0.6 so training has room alongside vLLM.
- **Zero-variance groups** (all 8 same reward) give zero gradient — track the fraction of
  such groups; if > ~70%, data is too easy or too hard.
- **Checkpoint resume:** `trainer.train(resume_from_checkpoint=path)`; copy checkpoint dirs
  out of `/kaggle/working` (HF Hub or Kaggle dataset) before the 12 h limit.
- Papers: DeepSeekMath (GRPO, arXiv 2402.03300), DeepSeek-R1 (arXiv 2501.12948).

## Gotchas

- **Unsloth vLLM bitsandbytes plugin:** `vLLM >= 0.28` moved bitsandbytes out of tree. Using `load_in_4bit=True` with `fast_inference=True` requires `pip install vllm-bnb-plugin` and `unsloth_zoo`.
- **Kaggle script secrets:** In script mode (`kernel_type: script`), `kaggle_secrets.UserSecretsClient()` requires internet enabled and secrets properly configured; handle connection errors gracefully.

- _(fill in as you hit them)_
