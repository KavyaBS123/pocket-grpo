"""The LLM acting as an RL *policy*: given a task (state), it produces tools (the action).

In RL terms, a policy is a rule for choosing actions. In the tabular world the
policy will be "pick the tool with the highest Q-value". Here the policy is the
language model itself: its next-token probabilities define how likely it is to
answer "search", "calculator", or "database" for a given task.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.tasks import BASIC_SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Two tool names plus a comma are about 4-5 tokens. A small limit keeps
# generation fast; anything much longer would be rejected by parse_tools() anyway.
MAX_NEW_TOKENS = 12


def pick_device():
    """Use the Apple GPU (MPS) when available, otherwise fall back to the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model_and_tokenizer(model_name=MODEL_NAME, device=None, dtype=torch.float32):
    """Download (first run only, cached afterwards) and load the model and tokenizer.

    dtype controls the precision of the model weights:
      float32 -> 4 bytes per weight (~2 GB): most accurate, used for evaluation.
      float16 -> 2 bytes per weight (~1 GB): used for training on 8 GB Macs.
    """
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pad on the left so that, in a batch, every prompt *ends* at the same
    # position and generation starts right after it for all rows.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, task_text, system_prompt=BASIC_SYSTEM_PROMPT):
    """Wrap a task in the model's chat format (system message + user message).

    add_generation_prompt=True appends the "assistant is now speaking" marker,
    so the model's next tokens are its answer.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()  # No gradients needed just to generate text; saves memory.
def generate_replies(model, tokenizer, task_texts, sample=False, num_samples=1, batch_size=16,
                     system_prompt=BASIC_SYSTEM_PROMPT):
    """Generate replies for each task. Returns one list of `num_samples` strings per task.

    sample=False -> greedy decoding: always take the most likely next token.
                    Deterministic; this is how the model behaves when deployed.
    sample=True  -> draw tokens at random from the model's probabilities.
                    This is what RL trains on, because learning needs the model
                    to sometimes try answers other than its current favourite.
    """
    if not sample:
        num_samples = 1

    if sample:
        # temperature=1, top_k=0, top_p=1 means "sample from the model's true
        # probabilities". Qwen's defaults would otherwise sharpen and truncate
        # them, hiding how uncertain the model really is.
        gen_kwargs = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
    else:
        gen_kwargs = dict(do_sample=False, temperature=None, top_k=None, top_p=None)

    all_replies = []
    for start in range(0, len(task_texts), batch_size):
        batch = task_texts[start:start + batch_size]
        prompts = [build_prompt(tokenizer, text, system_prompt) for text in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )

        # `outputs` contains prompt + answer; keep only the newly generated tokens.
        prompt_length = inputs["input_ids"].shape[1]
        texts = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)

        # generate() returns samples grouped per prompt: [p0s0, p0s1, ..., p1s0, ...].
        for i in range(len(batch)):
            all_replies.append(texts[i * num_samples:(i + 1) * num_samples])

    return all_replies
