"""Did RL training damage the model's ordinary behaviour? ("drift")

We trained the model to answer tool names when it sees the tool-selection
system prompt. A side effect could be that it starts saying "search" or
"calculator" EVERYWHERE, even in normal chat. This script asks ordinary
questions WITHOUT the tool instructions and prints each model's reply side by
side, so you can see whether training leaked into general behaviour.

Run from the project root, listing any trained adapters to compare:
    python -m src.check_drift checkpoints/grpo-lora checkpoints/grpo-t1.3-kl
"""

import argparse

import torch
from peft import PeftModel

from src.llm_policy import load_model_and_tokenizer

# Ordinary chat requests. Some deliberately look like tool tasks (a fact, a
# calculation) because those are where a drifted model is most likely to slip.
CHAT_PROMPTS = [
    "What is the capital of France?",
    "What is 12 times 12?",
    "Give me one synonym for 'happy'.",
    "Write one short sentence about the ocean.",
]

MAX_REPLY_TOKENS = 30  # enough to see the style of the reply


@torch.no_grad()
def reply(model, tokenizer, text):
    """Greedy reply to `text` as a plain chat message (no tool system prompt)."""
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_REPLY_TOKENS,
        do_sample=False, temperature=None, top_k=None, top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return " ".join(answer.split())  # put multi-line replies on one line


def main():
    """Print each chat prompt answered by the original model and by every adapter."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapters", nargs="+", help="folders with trained LoRA weights")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    # Load every adapter onto the same base model, each under its own name.
    # Only one is active at a time, and disable_adapter() gives the original.
    # peft forbids "." and "/" in adapter names, so use "adapter0", "adapter1", ...
    names = [f"adapter{i}" for i in range(len(args.adapters))]
    model = PeftModel.from_pretrained(model, args.adapters[0], adapter_name=names[0])
    for name, path in zip(names[1:], args.adapters[1:]):
        model.load_adapter(path, adapter_name=name)
    model.eval()

    for text in CHAT_PROMPTS:
        print(f"\nQ: {text}")
        with model.disable_adapter():
            print(f"  {'original model':<28} {reply(model, tokenizer, text)!r}")
        for name, path in zip(names, args.adapters):
            model.set_adapter(name)
            print(f"  {path:<28} {reply(model, tokenizer, text)!r}")


if __name__ == "__main__":
    main()
