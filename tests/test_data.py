from grpo_reason.data import (
    format_gsm8k_example,
    get_gsm8k_eval_subset,
)
from grpo_reason.prompts import SYSTEM_PROMPT


def test_format_gsm8k_example():
    raw = {
        "question": "What is 2 + 2?",
        "answer": "2 + 2 = 4\n#### 4",
    }
    formatted = format_gsm8k_example(raw)

    assert formatted["question"] == "What is 2 + 2?"
    assert formatted["answer"] == "4"
    assert formatted["gold_solution"] == raw["answer"]
    assert formatted["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2 + 2?"},
    ]


def test_eval_subset_deterministic():
    # Use cached or small dataset
    s1 = get_gsm8k_eval_subset(subset_size=5, seed=42)
    s2 = get_gsm8k_eval_subset(subset_size=5, seed=42)
    s3 = get_gsm8k_eval_subset(subset_size=5, seed=99)

    assert len(s1) == 5
    assert [x["question"] for x in s1] == [x["question"] for x in s2]
    # Different seed should yield different sample
    assert [x["question"] for x in s1] != [x["question"] for x in s3]
