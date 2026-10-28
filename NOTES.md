# NOTES — GRPO Reasoning

Running log of decisions, findings, and gotchas. Newest first.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | Repo scaffolded | — |

## GPU budget tracker (Kaggle ~30 h/week)

| Date | Run | Model | Steps | GPU h | Outcome |
|---|---|---|---|---|---|
| | | | | | |

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

- _(fill in as you hit them)_
