# grpo-gsm8k-reasoning

Teaching a 0.5B–1.5B LLM to reason on GSM8K with **GRPO** (the DeepSeek-R1 RL technique),
trained on a **free Kaggle T4** with Unsloth + TRL.

> Status: 🚧 scaffolded — see [PLAN.md](PLAN.md) for the roadmap and [NOTES.md](NOTES.md) for the decision log.

## Results

| Model | GSM8K acc (greedy) | Format compliance | Avg. length |
|---|---|---|---|
| Baseline | _TBD_ | | |
| + GRPO | _TBD_ | | |

## Layout

```
src/grpo_reason/   prompts, rewards (unit-tested), data loading, eval, train
configs/           debug (0.5B) and full (1.5B) GRPO configs
kaggle/train/      Kaggle kernel (T4) for training
kaggle/eval/       Kaggle kernel for baseline / final evals
reports/           results.md, sample completions, curves
tests/
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                    # reward functions
kaggle kernels push -p kaggle/eval        # baseline eval on Kaggle T4
```

## License

MIT
