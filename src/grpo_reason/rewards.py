"""Reward functions for GRPO on GSM8K.

Signatures follow TRL's GRPOTrainer convention: ``fn(completions, **kwargs) -> list[float]``,
where ``completions`` are chat-format lists ``[{"role": "assistant", "content": str}]`` and
dataset columns (e.g. ``answer``) arrive as kwargs.
"""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


def normalize_number(s: str | None) -> str | None:
    """Pull the last number out of a string; strip commas, $ and trailing .0."""
    if s is None:
        return None
    nums = _NUM_RE.findall(s.replace(",", "").replace("$", ""))
    if not nums:
        return None
    n = nums[-1]
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def extract_gsm8k_gold(answer_field: str) -> str | None:
    """GSM8K gold answers follow ``#### `` at the end of the solution."""
    return normalize_number(answer_field.split("####")[-1])


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    preds = [normalize_number(extract_answer(_text(c))) for c in completions]
    golds = [normalize_number(a) for a in answer]
    return [2.0 if p is not None and p == g else 0.0 for p, g in zip(preds, golds)]


def format_reward(completions, **kwargs) -> list[float]:
    return [0.5 if _FORMAT_RE.match(_text(c)) else 0.0 for c in completions]


def int_reward(completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        a = extract_answer(_text(c))
        out.append(0.25 if a is not None and re.fullmatch(r"-?\d+", a.replace(",", "")) else 0.0)
    return out
