---
language:
- en
license: mit
base_model:
- Qwen/Qwen2.5-1.5B-Instruct
- unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit
tags:
- grpo
- r1
- deepseek-r1
- reasoning
- gsm8k
- math
- reinforcement-learning
- unsloth
- trl
pipeline_tag: text-generation
datasets:
- openai/gsm8k
metrics:
- accuracy
model-index:
- name: qwen2.5-1.5b-grpo-gsm8k
  results:
  - task:
      type: text-generation
      name: Mathematical Reasoning
    dataset:
      name: GSM8K (Fixed 300 test subset, seed 42)
      type: openai/gsm8k
    metrics:
    - name: Strict Accuracy
      type: accuracy
      value: 36.33
    - name: Format Compliance
      type: accuracy
      value: 99.33
---

# Qwen2.5-1.5B-GRPO-GSM8K

A 1.5B parameter mathematical reasoning model trained with **Group Relative Policy Optimization (GRPO)** on GSM8K using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl) on a single **free Kaggle Tesla T4 GPU**.

- **GitHub Repository**: [raju-sah/grpo-gsm8k-reasoning](https://github.com/raju-sah/grpo-gsm8k-reasoning)
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Training Method**: GRPO (DeepSeek-R1 style rule-based RL with verifiable math rewards)
- **Adapter Type**: LoRA (Rank 32, Alpha 32, all linear layers)
- **License**: MIT

---

## Benchmark Results (GSM8K Test Subset, n=300)

Evaluated under greedy decoding (`temperature = 0.0`, `max_new_tokens = 512`) on a fixed 300-question test subset (`seed = 42`):

| Model / Run | Prompt Mode | Steps | Strict Accuracy (`<answer>`) | Format Compliance | Integer Compliance | Avg Tokens |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1.5B** (`Qwen2.5-1.5B-Instruct`) | Unprimed | 0 | 2.67% (8/300) | 0.33% | 4.33% | 238.1 |
| **GRPO 1.5B** (500 steps) | Unprimed | 500 | 2.67% (8/300) | 0.00% | 2.33% | 233.8 |
| **Baseline 1.5B** (`Qwen2.5-1.5B-Instruct`) | Primed (`<think>\n`) | 0 | 27.00% (81/300) | **99.33%** | 42.33% | 127.3 |
| **GRPO 1.5B** (500 steps) | Primed (`<think>\n`) | 500 | **36.33% (109/300)** | **99.33%** | **83.00%** | 149.9 |

### Key Improvements:
- **+9.33% Absolute Gain over Baseline** (+34.5% relative jump) on strict reasoning accuracy.
- **Integer Compliance Doubled**: 42.33% $\to$ **83.00%**, demonstrating clean numeric solution isolation.
- **Over 99% Format Compliance**: Reliably structures output into `<think>...</think><answer>...</answer>`.
- **Zero Length Hacking**: Output length remained compact and steady (~150 tokens) without artificial length penalties.

---

## 500-Step Training Reward Progression

Trained across 500 steps with $G = 8$ completions per group, learning rate $5 \times 10^{-6}$ (cosine decay), and KL penalty $\beta = 0.0$:

| Window | Steps | Mean Total Reward | Correctness Reward (Max 2.0) | Integer Reward (Max 0.25) | Format Reward (Max 0.50) | Mean Completion Length |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1–50 | 0.4150 | 0.3550 | 0.0587 | 0.0012 | 257.8 tokens |
| 2 | 51–100 | 1.1669 | 0.9850 | 0.1756 | 0.0062 | 230.9 tokens |
| 3 | 101–150 | 1.4406 | 1.2250 | 0.2131 | 0.0025 | 233.3 tokens |
| 4 | 151–200 | 1.3519 | 1.1350 | 0.2169 | 0.0000 | 270.5 tokens |
| 5 | 201–250 | 1.5350 | 1.3050 | 0.2300 | 0.0000 | 242.9 tokens |
| 6 | 251–300 | 1.5919 | 1.3600 | 0.2319 | 0.0000 | 262.2 tokens |
| 7 | 301–350 | 1.7444 | 1.5100 | 0.2331 | 0.0012 | 232.8 tokens |
| 8 | 351–400 | 1.5631 | 1.3300 | 0.2331 | 0.0000 | 247.8 tokens |
| 9 | 401–450 | 1.6669 | 1.4250 | 0.2419 | 0.0000 | 247.9 tokens |
| 10 | 451–500 | **1.6769** | **1.4500** | **0.2269** | 0.0000 | 260.2 tokens |

---

## Quickstart & Inference

### Using Transformers + PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "raju-ai/qwen2.5-1.5b-grpo-gsm8k"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

SYSTEM_PROMPT = """Respond in the following format:
<think>
...step-by-step reasoning...
</think>
<answer>
...final numeric answer only...
</answer>"""

question = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]

# Standard DeepSeek-R1 prompt priming: pre-fill assistant turn with <think>\n
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

output_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("<think>\n" + output_text)
```

---

## Example Model Output

```xml
<think>
To solve this problem, we need to determine the current ages of Darrell and Allen based on the given ratio and total age. Let's denote Darrell's age as \(7x\) and Allen's age as \(11x\), where \(x\) is a common multiplier. The sum of their ages is given as 162.

First, we set up the equation:
\[7x + 11x = 162\]

Combine the terms:
\[18x = 162\]

Solve for \(x\):
\[x = \frac{162}{18} = 9\]

Now, we can find the current ages of Darrell and Allen:
\[ \text{Darrell's age} = 7x = 7 \times 9 = 63 \]
\[ \text{Allen's age} = 11x = 11 \times 9 = 99 \]

To find Allen's age 10 years from now, we add 10 to his current age:
\[ \text{Allen's age 10 years from now} = 99 + 10 = 109 \]

Therefore, Allen's age 10 years from now is 109.
</think>
<answer>
109
</answer>
```

---

## Citation

If you use this model or code in your research or project:

```bibtex
@misc{qwen2.5-1.5b-grpo-gsm8k,
  author = {Raju Sah},
  title = {Qwen2.5-1.5B-GRPO-GSM8K: Mathematical Reasoning with GRPO RL},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/raju-ai/qwen2.5-1.5b-grpo-gsm8k}}
}
```
