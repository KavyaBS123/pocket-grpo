"""Train the LLM to pick the right tool with a simplified GRPO loop.

Run from the project root:
    python -m src.train_grpo

GRPO in four lines, for each practice question:
    1. Sample a GROUP of answers from the model (e.g. 8).
    2. Score each answer with the environment's reward (+1 / -1).
    3. Advantage = how much better each answer did than the group average.
    4. Make above-average answers more likely, below-average ones less likely.

Optional KL penalty (--kl-coef > 0): an extra cost for drifting away from the
original model, so the model learns the task without forgetting how it used to
behave everywhere else. With --kl-coef 0 (the default) there is no such cost.

Simplification compared with full GRPO (on purpose, to keep the idea visible):
    - No clipping. We take exactly one update per fresh batch of samples, so the
      "how much did the policy change since sampling" ratio that clipping limits
      is always 1, and clipping would have no effect anyway.
"""

import argparse
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from src.llm_policy import MAX_NEW_TOKENS, build_prompt, generate_replies, load_model_and_tokenizer
from src.llm_reward import llm_reward
from src.tasks import TEST_TASKS, TRAIN_TASKS


def parse_args():
    p = argparse.ArgumentParser(description="Simplified GRPO training for tool selection.")
    p.add_argument("--epochs", type=int, default=4, help="passes over the 42 practice questions")
    p.add_argument("--prompts-per-step", type=int, default=6, help="questions per optimizer update")
    p.add_argument("--group-size", type=int, default=8, help="sampled answers per question")
    p.add_argument("--micro-batch", type=int, default=4, help="answers per learning pass (lower = less memory)")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate: how big each nudge is")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling randomness; higher = more exploration")
    p.add_argument("--kl-coef", type=float, default=0.0, help="strength of the pull back toward the original model (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/grpo-lora", help="where to save the trained LoRA weights")
    return p.parse_args()


def add_lora(model):
    """Freeze the base model and attach small trainable LoRA matrices.

    r=16 is the "rank": how many knobs each LoRA layer gets. Higher = more
    capacity but more memory. We only attach LoRA to the attention projections,
    which is a common, memory-friendly choice.
    """
    config = LoraConfig(
        r=16,
        lora_alpha=32,  # scaling factor; alpha = 2 * r is a common default
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    # Gradient checkpointing: during the learning pass, don't keep every
    # intermediate result in memory; recompute them when gradients need them.
    # Slower, but it is what lets training fit next to other apps in 8 GB.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def sample_group(model, tokenizer, task_text, group_size, temperature):
    """Sample `group_size` answers for one question.

    Returns the token ids we need for the learning step, plus the decoded text
    we need for the reward. We keep the token ids (instead of re-tokenizing the
    text later) so that we compute probabilities for exactly the tokens the
    model actually produced.
    """
    # eval mode = "just generating": gradient checkpointing switches off and
    # generation can reuse its cache, which is much faster.
    model.eval()
    prompt = build_prompt(tokenizer, task_text)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_k=0,     # 0 and 1.0 disable Qwen's default truncation, so we sample
        top_p=1.0,   # from the model's real probabilities (honest exploration)
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion_ids = out[:, prompt_len:]  # shape: [group_size, num_new_tokens]

    # Mark which completion tokens are "real": everything up to and including
    # the first end-of-answer token. Tokens after it are just padding added
    # because other answers in the group were longer.
    eos_ids = torch.tensor(model.generation_config.eos_token_id, device=out.device).flatten()
    is_eos = torch.isin(completion_ids, eos_ids)
    num_tokens = completion_ids.shape[1]
    first_eos = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), num_tokens)
    positions = torch.arange(num_tokens, device=out.device).unsqueeze(0)
    completion_mask = (positions <= first_eos.unsqueeze(1)).long()

    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return inputs["input_ids"], completion_ids, completion_mask, texts


def token_log_probs(model, prompt_ids, completion_ids, completion_mask):
    """log P(token) for every generated token of every answer. Shape: [answers, tokens].

    The model gives a probability for each next token. The probability of a
    whole answer is the product of its token probabilities, so its log-prob is
    the SUM of these token log-probs (summed by the caller, after masking out
    padding). That sum is the quantity GRPO pushes up or down.
    """
    # train mode turns gradient checkpointing on. Qwen and our LoRA have no
    # dropout, so this does not change the math, only the memory use.
    model.train()
    group_size = completion_ids.shape[0]
    full_ids = torch.cat([prompt_ids.expand(group_size, -1), completion_ids], dim=1)
    attention_mask = torch.cat([torch.ones_like(prompt_ids).expand(group_size, -1), completion_mask], dim=1)

    num_tokens = completion_ids.shape[1]
    # logits_to_keep: only compute vocabulary scores for the last positions we
    # need. The vocabulary has ~152k entries, so scoring every prompt position
    # would waste a lot of memory. The score at position t predicts token t+1,
    # so to score `num_tokens` answer tokens we need `num_tokens + 1` positions
    # and drop the very last one.
    logits = model(input_ids=full_ids, attention_mask=attention_mask, logits_to_keep=num_tokens + 1).logits
    logits = logits[:, :-1, :]

    # Convert to float32 before log_softmax: with a float16 model, doing this
    # over ~152k vocabulary entries in half precision would lose accuracy.
    token_log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Pick out the log-prob of the token that was actually generated.
    return token_log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)


def kl_per_token(log_probs, ref_log_probs):
    """How far the trained model has drifted from the original, per token.

    Uses the estimator from the GRPO paper:  exp(d) - d - 1,  d = ref - current.
    It is always >= 0, and exactly 0 when both models give the token the same
    probability. The further apart they are, the bigger the penalty.
    """
    d = ref_log_probs - log_probs
    return torch.exp(d) - d - 1


def group_advantages(rewards):
    """Advantage_i = (reward_i - group mean) / (group std + small number).

    Positive -> this answer beat its siblings -> make it more likely.
    Negative -> it did worse -> make it less likely.
    If every answer got the same reward, all advantages are 0: no signal.
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)


@torch.no_grad()
def exam_accuracy(model, tokenizer):
    """Greedy accuracy on the 18 held-out questions the model never trains on."""
    model.eval()
    texts = [text for text, _ in TEST_TASKS]
    replies = generate_replies(model, tokenizer, texts, sample=False)
    correct = sum(llm_reward(state, r[0]) > 0 for (_, state), r in zip(TEST_TASKS, replies))
    return correct, len(TEST_TASKS)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading model...")
    # The frozen base model in float16 halves its memory (~1 GB). The LoRA
    # weights that we actually train are still kept in float32 by peft.
    model, tokenizer = load_model_and_tokenizer(dtype=torch.float16)
    model = add_lora(model)

    # Only the LoRA weights require gradients, so only they get updated.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    correct, total = exam_accuracy(model, tokenizer)
    print(f"Exam before training: {correct}/{total}\n")

    step = 0
    for epoch in range(1, args.epochs + 1):
        # Shuffle so the model doesn't see the questions in the same order every epoch.
        tasks = TRAIN_TASKS[:]
        random.shuffle(tasks)

        for start in range(0, len(tasks), args.prompts_per_step):
            batch = tasks[start:start + args.prompts_per_step]
            step += 1
            t0 = time.time()
            optimizer.zero_grad()

            batch_rewards = []
            groups_with_signal = 0
            total_loss = 0.0
            kl_values = []  # drift of each processed answer, for logging

            for task_text, state in batch:
                # 1. Sample a group of answers.
                prompt_ids, completion_ids, completion_mask, texts = sample_group(
                    model, tokenizer, task_text, args.group_size, args.temperature
                )
                # 2. Score each answer with the environment's reward.
                rewards = [llm_reward(state, text) for text in texts]
                batch_rewards += rewards

                has_signal = len(set(rewards)) > 1
                groups_with_signal += has_signal
                # All answers equally good/bad -> advantages are all 0. Without a
                # KL penalty such a group contributes nothing, so we skip the
                # expensive learning pass. WITH a KL penalty we still process it:
                # the KL term can pull the model back even when rewards are equal.
                if not has_signal and args.kl_coef == 0:
                    continue

                # 3. Advantages: how much better than the group average.
                advantages = group_advantages(rewards).to(model.device)

                # 4. Loss per answer = -advantage * log P(answer) + kl_coef * KL(answer).
                #    The first part pushes good answers up and bad ones down; the
                #    second part charges a cost for drifting from the original model.
                #    We process the group in small chunks to save memory. Dividing
                #    each chunk's sum by (group_size * questions in batch) makes the
                #    total exactly equal to the average over all answers in the batch.
                for c in range(0, args.group_size, args.micro_batch):
                    chunk = slice(c, c + args.micro_batch)
                    ids, mask = completion_ids[chunk], completion_mask[chunk]

                    log_probs = token_log_probs(model, prompt_ids, ids, mask)
                    # The original model's opinion of the same tokens. Disabling
                    # the LoRA adapter temporarily turns our model back into the
                    # untouched base model, so no second copy is needed in memory.
                    with torch.no_grad(), model.disable_adapter():
                        ref_log_probs = token_log_probs(model, prompt_ids, ids, mask)

                    answer_log_prob = (log_probs * mask).sum(dim=1)
                    answer_kl = (kl_per_token(log_probs, ref_log_probs) * mask).sum(dim=1)
                    kl_values += answer_kl.tolist()

                    per_answer_loss = -advantages[chunk] * answer_log_prob + args.kl_coef * answer_kl
                    loss = per_answer_loss.sum() / (args.group_size * len(batch))
                    # backward() ADDS these gradients to the running total; we
                    # update once after the whole batch (gradient accumulation).
                    loss.backward()
                    total_loss += loss.item()

            if groups_with_signal > 0 or args.kl_coef > 0:
                # Cap the size of the update so one unusual batch can't wreck the model.
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()

            # Hand cached GPU memory back to macOS so other apps (and the next
            # step) have room; otherwise PyTorch keeps it reserved.
            if model.device.type == "mps":
                torch.mps.empty_cache()
                memory = f" | GPU mem {torch.mps.driver_allocated_memory() / 1e9:.1f} GB"
            else:
                memory = ""

            n = len(batch_rewards)
            accuracy = sum(r > 0 for r in batch_rewards) / n
            # Average drift per answer (only over answers that got a learning pass).
            kl = f"{sum(kl_values) / len(kl_values):.3f}" if kl_values else "  -  "
            print(
                f"epoch {epoch} step {step:>3} | sampled accuracy {accuracy:5.0%} | "
                f"mean reward {sum(batch_rewards) / n:+.2f} | "
                f"groups with signal {groups_with_signal}/{len(batch)} | "
                f"KL {kl} | loss {total_loss:+.4f} | {time.time() - t0:.1f}s{memory}"
            )

        correct, total = exam_accuracy(model, tokenizer)
        print(f"--- end of epoch {epoch}: exam {correct}/{total} ---\n")

    # Saves only the small LoRA weights (a few MB), not the whole model.
    model.save_pretrained(args.out)
    print(f"Saved LoRA weights to {args.out}")


if __name__ == "__main__":
    main()
