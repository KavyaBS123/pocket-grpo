"""Measure how well the LLM picks tools, before (and later, after) RL training.

Run from the project root:
    python -m src.evaluate_llm                              # basic task set, all tasks
    python -m src.evaluate_llm --tasks hard --split test    # hard set, held-out tasks only
    python -m src.evaluate_llm --adapter checkpoints/grpo-lora   # an RL-trained model
"""

import argparse
from collections import defaultdict

import torch

from src.llm_policy import generate_replies, load_model_and_tokenizer
from src.llm_reward import WRONG_ORDER_REWARD, parse_tools, sequence_reward
from src.tasks import TASK_SETS


def main():
    """Run the two evaluations and print a report.

    1. Greedy: one deterministic answer per task — what a user would get today.
    2. Sampled: several random answers per task — the distribution GRPO learns
       from, and whether it contains a learning signal (mixed rewards).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", choices=TASK_SETS, default="basic", help="which task set to use")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--samples", type=int, default=4, help="sampled replies per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", help="folder with trained LoRA weights (omit for the untrained model)")
    args = parser.parse_args()

    # Fixing the seed makes the *sampled* replies reproducible between runs.
    torch.manual_seed(args.seed)

    task_set = TASK_SETS[args.tasks]
    if args.split == "all":
        tasks = task_set["train"] + task_set["test"]
    else:
        tasks = task_set[args.split]
    texts = [task.text for task in tasks]
    system_prompt = task_set["system_prompt"]

    print("Loading model (the first run downloads ~1 GB)...")
    model, tokenizer = load_model_and_tokenizer()
    if args.adapter:
        # Imported here so the plain baseline run doesn't depend on peft.
        from peft import PeftModel

        # Attach the trained LoRA knobs on top of the frozen base model.
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded LoRA adapter from {args.adapter}")
    print(f"Model on {model.device}. Evaluating {len(tasks)} tasks ({args.tasks} set, {args.split} split).\n")

    # --- 1. Greedy: the model's single "best guess" per task. ---------------
    greedy = [r[0] for r in generate_replies(model, tokenizer, texts, sample=False, system_prompt=system_prompt)]

    per_category = defaultdict(lambda: [0, 0])  # category -> [correct, total]
    expected_by_category = {}
    invalid = wrong_order = 0
    mistakes = []
    for task, reply in zip(tasks, greedy):
        reward = sequence_reward(task.tools, reply)
        is_correct = reward > 0
        per_category[task.category][0] += is_correct
        per_category[task.category][1] += 1
        expected_by_category[task.category] = ", ".join(task.tools)
        invalid += parse_tools(reply) is None
        wrong_order += reward == WRONG_ORDER_REWARD
        if not is_correct:
            mistakes.append((task.text, expected_by_category[task.category], reply))

    total_correct = sum(c for c, _ in per_category.values())
    print(f"GREEDY accuracy: {total_correct}/{len(tasks)} = {total_correct / len(tasks):.0%}")
    print(f"  unparseable replies (format errors): {invalid}")
    print(f"  right tools, wrong order: {wrong_order}")
    for category, (c, n) in per_category.items():
        print(f"  {category:<21} -> {expected_by_category[category]:<20} {c}/{n}")

    if mistakes:
        print("\nGreedy mistakes (task | expected | model said):")
        for text, expected, reply in mistakes:
            print(f"  {text[:55]:<55} | {expected:<20} | {reply!r}")

    # --- 2. Sampled: what RL actually trains on. ----------------------------
    sampled = generate_replies(
        model, tokenizer, texts, sample=True, num_samples=args.samples, system_prompt=system_prompt
    )
    rewards = [[sequence_reward(task.tools, r) for r in replies] for task, replies in zip(tasks, sampled)]
    n_replies = len(tasks) * args.samples
    n_correct = sum(r > 0 for group in rewards for r in group)
    # A group has a learning signal only if its rewards are not all identical.
    n_mixed = sum(len(set(group)) > 1 for group in rewards)

    print(f"\nSAMPLED accuracy ({args.samples} per task): {n_correct}/{n_replies} = {n_correct / n_replies:.0%}")
    print(f"  tasks with mixed rewards (GRPO learning signal): {n_mixed}/{len(tasks)}")
    print("\nA few sampled groups:")
    for task, replies, group in list(zip(tasks, sampled, rewards))[::10]:
        print(f"  {task.text[:50]:<50} -> {replies} rewards={group}")


if __name__ == "__main__":
    main()
