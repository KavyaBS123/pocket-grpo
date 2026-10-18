"""Measure how well the LLM picks tools, before (and later, after) RL training.

Run from the project root:
    python -m src.evaluate_llm                 # all 60 tasks
    python -m src.evaluate_llm --split test    # only the 18 held-out tasks
    python -m src.evaluate_llm --adapter checkpoints/grpo-lora   # the RL-trained model
"""

import argparse
from collections import defaultdict

import torch

from src.environment import CORRECT_TOOL
from src.llm_policy import generate_replies, load_model_and_tokenizer
from src.llm_reward import llm_reward, parse_tool
from src.tasks import ALL_TASKS, TEST_TASKS, TRAIN_TASKS

SPLITS = {"all": ALL_TASKS, "train": TRAIN_TASKS, "test": TEST_TASKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--samples", type=int, default=4, help="sampled replies per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", help="folder with trained LoRA weights (omit for the untrained model)")
    args = parser.parse_args()

    # Fixing the seed makes the *sampled* replies reproducible between runs.
    torch.manual_seed(args.seed)

    tasks = SPLITS[args.split]
    texts = [text for text, _ in tasks]
    states = [state for _, state in tasks]

    print("Loading model (the first run downloads ~1 GB)...")
    model, tokenizer = load_model_and_tokenizer()
    if args.adapter:
        # Imported here so the plain baseline run doesn't depend on peft.
        from peft import PeftModel

        # Attach the trained LoRA knobs on top of the frozen base model.
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded LoRA adapter from {args.adapter}")
    print(f"Model on {model.device}. Evaluating {len(tasks)} tasks ({args.split} split).\n")

    # --- 1. Greedy: the model's single "best guess" per task. ---------------
    greedy = [replies[0] for replies in generate_replies(model, tokenizer, texts, sample=False)]

    per_state = defaultdict(lambda: [0, 0])  # state -> [correct, total]
    invalid = 0
    mistakes = []
    for text, state, reply in zip(texts, states, greedy):
        is_correct = llm_reward(state, reply) > 0
        per_state[state][0] += is_correct
        per_state[state][1] += 1
        invalid += parse_tool(reply) is None
        if not is_correct:
            mistakes.append((text, CORRECT_TOOL[state], reply))

    total_correct = sum(c for c, _ in per_state.values())
    print(f"GREEDY accuracy: {total_correct}/{len(tasks)} = {total_correct / len(tasks):.0%}")
    print(f"  unparseable replies (format errors): {invalid}")
    for state, (c, n) in per_state.items():
        print(f"  {state:<16} -> {CORRECT_TOOL[state]:<10} {c}/{n}")

    if mistakes:
        print("\nGreedy mistakes (task | expected | model said):")
        for text, expected, reply in mistakes:
            print(f"  {text[:58]:<58} | {expected:<10} | {reply!r}")

    # --- 2. Sampled: what RL actually trains on. ----------------------------
    sampled = generate_replies(model, tokenizer, texts, sample=True, num_samples=args.samples)
    rewards = [[llm_reward(state, r) for r in replies] for state, replies in zip(states, sampled)]
    n_replies = len(tasks) * args.samples
    n_correct = sum(r > 0 for group in rewards for r in group)
    # A group has a learning signal only if its rewards are not all identical.
    n_mixed = sum(len(set(group)) > 1 for group in rewards)

    print(f"\nSAMPLED accuracy ({args.samples} per task): {n_correct}/{n_replies} = {n_correct / n_replies:.0%}")
    print(f"  tasks with mixed rewards (GRPO learning signal): {n_mixed}/{len(tasks)}")
    print("\nA few sampled groups:")
    for text, replies, group in list(zip(texts, sampled, rewards))[::10]:
        print(f"  {text[:50]:<50} -> {replies} rewards={group}")


if __name__ == "__main__":
    main()
