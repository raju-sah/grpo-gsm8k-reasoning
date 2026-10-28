from grpo_reason.rewards import (
    correctness_reward,
    extract_gsm8k_gold,
    format_reward,
    int_reward,
    normalize_number,
)


def c(text):
    return [{"role": "assistant", "content": text}]


GOOD = "<think>\n5 + 7 = 12\n</think>\n<answer>\n12\n</answer>"
BAD_FMT = "The answer is 12."


def test_gold_extraction():
    assert extract_gsm8k_gold("blah blah\n#### 1,234") == "1234"


def test_normalize():
    assert normalize_number("$18.00") == "18"
    assert normalize_number("no number") is None


def test_correctness():
    assert correctness_reward([c(GOOD), c(BAD_FMT)], answer=["12", "12"]) == [2.0, 0.0]


def test_format_and_int():
    assert format_reward([c(GOOD), c(BAD_FMT)]) == [0.5, 0.0]
    assert int_reward([c(GOOD)]) == [0.25]
