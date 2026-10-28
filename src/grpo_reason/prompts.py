"""Prompt format shared by training AND evaluation (must be identical)."""

SYSTEM_PROMPT = """Respond in the following format:
<think>
...step-by-step reasoning...
</think>
<answer>
...final numeric answer only...
</answer>"""


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
