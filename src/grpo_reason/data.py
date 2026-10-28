"""GSM8K dataset loading, formatting, and splitting for GRPO and evaluation."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from grpo_reason.prompts import build_messages
from grpo_reason.rewards import extract_gsm8k_gold


def format_gsm8k_example(example: dict) -> dict:
    """Format a raw GSM8K example into prompt messages and extracted gold answer."""
    return {
        "prompt": build_messages(example["question"]),
        "answer": extract_gsm8k_gold(example["answer"]),
        "question": example["question"],
        "gold_solution": example["answer"],
    }


def load_raw_gsm8k(split: str = "train") -> Dataset:
    """Load raw GSM8K dataset split ('train' or 'test')."""
    return load_dataset("openai/gsm8k", "main", split=split)


def get_gsm8k_dataset(
    split: str = "train",
    subset_size: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load, format, and optionally subset GSM8K."""
    ds = load_raw_gsm8k(split=split)
    if subset_size is not None and subset_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(subset_size))
    return ds.map(format_gsm8k_example)


def get_gsm8k_eval_subset(subset_size: int = 300, seed: int = 42) -> Dataset:
    """Get the standard deterministic GSM8K evaluation subset."""
    return get_gsm8k_dataset(split="test", subset_size=subset_size, seed=seed)
