import spaces  # noqa: I001 (must be first for ZeroGPU CUDA interception)
import re
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "raju-ai/qwen2.5-1.5b-grpo-gsm8k"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

print("Loading GRPO LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.to("cuda")
model.eval()
print("Model loaded successfully on ZeroGPU!")

SYSTEM_PROMPT = """Respond in the following format:
<think>
...step-by-step reasoning...
</think>
<answer>
...final numeric answer only...
</answer>"""


@spaces.GPU(duration=30)
def solve_math(question: str):
    if not question.strip():
        return "Please enter a valid math problem.", "", ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + "<think>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = "<think>\n" + output_text

    # Extract think section
    think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
    else:
        # Fallback if unclosed
        reasoning = full_text.replace("<think>", "").strip()

    # Extract answer section
    ans_match = re.search(r"<answer>(.*?)</answer>", full_text, re.DOTALL)
    if ans_match:
        answer = ans_match.group(1).strip()
    else:
        # Fallback regex for final number
        nums = re.findall(r"-?\d+(?:\.\d+)?", full_text.replace(",", "").replace("$", ""))
        answer = nums[-1] if nums else "N/A"

    return reasoning, answer, full_text


EXAMPLES = [
    [
        "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."
    ],
    [
        "Indras has 6 letters in her name. Her sister's name has 4 more letters than half of the letters in Indras' name. How many letters are in Indras and her sister's names?"
    ],
    [
        "Bethany can run 10 laps on the track in one hour. Trey can run 4 more laps than Bethany. Shaelyn can run half as many laps as Trey. Quinn can run 2 fewer laps than Shaelyn. How many more laps can Bethany run compared to Quinn?"
    ],
    [
        "James buys 5 packs of baseball cards with 12 cards in each pack. He gives 15 cards to his brother and 8 cards to his friend. How many baseball cards does James have left?"
    ],
]

with gr.Blocks(title="Qwen2.5-1.5B GRPO GSM8K Reasoning Demo") as demo:
    gr.Markdown(
        """
        # 🧮 Qwen2.5-1.5B-GRPO-GSM8K Reasoning Demo
        ### Group Relative Policy Optimization (GRPO) on GSM8K Math Word Problems
        *Trained with [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) on a single free Kaggle Tesla T4 GPU.*
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_box = gr.Textbox(
                label="Math Word Problem",
                placeholder="Enter a grade-school arithmetic or algebra problem...",
                lines=4,
            )
            submit_btn = gr.Button("🧠 Solve with GRPO Reasoning", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=input_box)

        with gr.Column(scale=5):
            answer_box = gr.Textbox(
                label="🎯 Final Numeric Answer",
                placeholder="The extracted numeric answer will appear here...",
                lines=1,
            )
            reasoning_box = gr.Markdown(
                label="💭 Step-by-Step Chain of Thought (<think>)"
            )
            with gr.Accordion("Raw XML Completion", open=False):
                raw_box = gr.Code(language="markdown")

    submit_btn.click(
        fn=solve_math,
        inputs=input_box,
        outputs=[reasoning_box, answer_box, raw_box],
    )

    gr.Markdown(
        """
        ---
        ### 📊 Benchmark Performance (GSM8K Test Set, n=300)
        | Metric | Baseline 1.5B | GRPO 1.5B (Trained) | Improvement |
        |---|:---:|:---:|:---:|
        | **Strict Accuracy** | 27.00% (81/300) | **36.33% (109/300)** | **+9.33% (+34.5% relative)** |
        | **Format Compliance** | 99.33% | **99.33%** | Complete tag compliance |
        | **Integer Compliance** | 42.33% | **83.00%** | **Nearly 2x increase** |

        - **Model on Hugging Face**: [`raju-ai/qwen2.5-1.5b-grpo-gsm8k`](https://huggingface.co/raju-ai/qwen2.5-1.5b-grpo-gsm8k)
        - **GitHub Source**: [`raju-sah/grpo-gsm8k-reasoning`](https://github.com/raju-sah/grpo-gsm8k-reasoning)
        """
    )

if __name__ == "__main__":
    demo.launch()
