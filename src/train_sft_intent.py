"""Supervised fine-tuning: teach Qwen to emit intent-gate JSON on Sarah prompts.

Run from project root:
    python -m src.train_sft_intent
    python -m src.train_sft_intent --jsonl data/sarah/labeled.jsonl --epochs 6

Each example is (full intent-gate prompt as user message) -> (gold JSON as assistant).
Loss is computed only on the assistant tokens, same chat layout as evaluate_sarah_intent.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from src.evaluate_sarah_intent import generate_intent
from src.llm_policy import MODEL_NAME, load_model_and_tokenizer, pick_device
from src.sarah.intent_dataset import build_sft_turns, turn_to_messages
from src.sarah.parse_intent import parse_intent_json
from src.sarah.score_intent import score_intent
from src.sarah.synthetic_data import SPLITS


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT for Sarah tenant intent JSON.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1, help="examples per optimizer step (keep 1 on 8 GB Macs)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--jsonl",
        action="append",
        default=[],
        metavar="PATH",
        help="labeled export JSONL (repeat for multiple files); merges with synthetic train",
    )
    p.add_argument("--no-synthetic", action="store_true", help="train only on --jsonl labels")
    p.add_argument(
        "--no-exam",
        action="store_true",
        help="skip held-out synthetic exam each epoch (much faster; long gate prompts)",
    )
    p.add_argument("--limit", type=int, default=0, help="max training turns (0 = all); use 2-3 for a quick smoke")
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="disable gradient checkpointing (faster steps if you have free RAM; close other apps)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="auto = MPS on Apple Silicon else CPU; use cpu if another train job may be on the GPU",
    )
    p.add_argument("--out", default="checkpoints/sft-intent-lora")
    p.add_argument(
        "--resume",
        default="",
        help="continue training an existing LoRA folder (same base model)",
    )
    p.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="epoch number for logging when resuming (default 1)",
    )
    return p.parse_args()


def release_mps_memory(model) -> None:
    """Return cached MPS memory after a step; helps avoid slow OOM on 8 GB Macs."""
    gc.collect()
    if getattr(model, "device", None) and model.device.type == "mps":
        torch.mps.empty_cache()


def add_lora(model, *, use_gradient_checkpointing: bool = True):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _chat_token_ids(tokenizer, messages, *, add_generation_prompt: bool) -> torch.Tensor:
    """apply_chat_template returns a tensor or a BatchEncoding depending on version."""
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def encode_sft_example(tokenizer, user_text: str, assistant_text: str, device):
    """Token ids and labels with -100 on prompt tokens (standard SFT masking)."""
    user_msgs = [{"role": "user", "content": user_text}]
    full_msgs = user_msgs + [{"role": "assistant", "content": assistant_text}]

    prompt_ids = _chat_token_ids(tokenizer, user_msgs, add_generation_prompt=True)
    full_ids = _chat_token_ids(tokenizer, full_msgs, add_generation_prompt=False)
    prompt_len = prompt_ids.shape[1]
    labels = full_ids.clone()
    labels[0, :prompt_len] = -100
    return full_ids.to(device), labels.to(device)


def pad_batch(tokenizer, batch_ids, batch_labels, device):
    """Left-pad sequences to the same length (tokenizer.padding_side is left)."""
    max_len = max(t.shape[1] for t in batch_ids)
    pad_id = tokenizer.pad_token_id
    input_rows, label_rows, attn = [], [], []
    for ids, lab in zip(batch_ids, batch_labels):
        pad_len = max_len - ids.shape[1]
        if pad_len:
            pad = torch.full((1, pad_len), pad_id, dtype=ids.dtype, device=device)
            pad_lab = torch.full((1, pad_len), -100, dtype=lab.dtype, device=device)
            ids = torch.cat([pad, ids], dim=1)
            lab = torch.cat([pad_lab, lab], dim=1)
        input_rows.append(ids)
        label_rows.append(lab)
        attn.append((ids != pad_id).long())
    return (
        torch.cat(input_rows, dim=0),
        torch.cat(label_rows, dim=0),
        torch.cat(attn, dim=0),
    )


@torch.no_grad()
def exam_on_split(model, tokenizer, turns):
    """Same metrics as evaluate_sarah_intent on a list of SarahTurn."""
    if not turns:
        return None
    n_parse = n_enums = n_route = n_full = 0
    for turn in turns:
        prompt = turn_to_messages(turn)[0]
        raw = generate_intent(model, tokenizer, prompt)
        parsed = parse_intent_json(raw, fallback_question=turn.user_message)
        sc = score_intent(parsed, turn.gold)
        n_parse += sc.parse_ok
        n_enums += sc.enums_valid
        n_route += sc.routing_exact
        n_full += sc.full_exact
    t = len(turns)
    return {
        "turns": t,
        "parse_ok": n_parse,
        "enums_valid": n_enums,
        "routing_exact": n_route,
        "full_exact": n_full,
    }


def format_exam(stats) -> str:
    if stats is None:
        return "(no exam turns)"
    t = stats["turns"]
    return (
        f"json_ok {stats['parse_ok']}/{t} | enums_valid {stats['enums_valid']}/{t} | "
        f"routing {stats['routing_exact']}/{t} | full {stats['full_exact']}/{t}"
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_turns = build_sft_turns(
        use_synthetic_train=not args.no_synthetic,
        jsonl_paths=args.jsonl,
    )
    if not train_turns:
        raise SystemExit("No labeled training turns. Use synthetic (default) and/or --jsonl with intent_gate_output.")
    if args.limit > 0:
        train_turns = train_turns[: args.limit]

    if args.device == "auto":
        device = pick_device()
    else:
        device = torch.device(args.device)
    print(
        f"Config: device={device} epochs={args.epochs} limit={args.limit or 'all'} "
        f"jsonl={args.jsonl or '(none)'} no_exam={args.no_exam} "
        f"grad_checkpoint={not args.no_checkpoint}",
        flush=True,
    )
    print(f"Loading {MODEL_NAME} (float16 for training)...")
    model, tokenizer = load_model_and_tokenizer(dtype=torch.float16, device=device)
    resume_path = args.resume.strip()
    if resume_path:
        print(f"Resuming LoRA from {resume_path}", flush=True)
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        if not args.no_checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.print_trainable_parameters()
    else:
        model = add_lora(model, use_gradient_checkpointing=not args.no_checkpoint)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    exam_turns = SPLITS["test"]
    print(f"Training examples: {len(train_turns)} | held-out synthetic exam: {len(exam_turns)}")
    if not args.no_exam:
        print(f"Exam before: {format_exam(exam_on_split(model, tokenizer, exam_turns))}")
    print(flush=True)

    first_epoch = max(1, args.start_epoch)
    last_epoch = first_epoch + args.epochs - 1
    for epoch in range(first_epoch, last_epoch + 1):
        order = train_turns[:]
        random.shuffle(order)
        epoch_loss = 0.0
        steps = 0
        t0 = time.time()

        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            ids_list, lab_list = [], []
            for turn in batch:
                user_text, target = turn_to_messages(turn)
                ids, lab = encode_sft_example(tokenizer, user_text, target, device)
                ids_list.append(ids)
                lab_list.append(lab)

            input_ids, labels, attention_mask = pad_batch(tokenizer, ids_list, lab_list, device)
            model.train()
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss_value = float(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                release_mps_memory(model)
                partial = Path(args.out)
                if partial.exists():
                    print(f"\nOOM at epoch {epoch} step {steps + 1}. Last weights on disk: {partial}", flush=True)
                raise SystemExit(
                    f"MPS ran out of memory: {exc}\n"
                    "Close other apps, use --no-exam, train one epoch at a time "
                    f"(weights save after each epoch to {args.out}), or --device cpu."
                ) from exc

            del out, loss, input_ids, labels, attention_mask, ids_list, lab_list
            epoch_loss += loss_value
            steps += 1
            print(
                f"  step {steps} | {batch[0].turn_id} | loss {loss_value:.4f}",
                flush=True,
            )
            release_mps_memory(model)

        exam_bit = ""
        if not args.no_exam:
            exam_bit = f" | exam {format_exam(exam_on_split(model, tokenizer, exam_turns))}"
            release_mps_memory(model)
        print(
            f"epoch {epoch}/{last_epoch} | steps {steps} | mean loss {epoch_loss / max(steps, 1):.4f} | "
            f"{time.time() - t0:.1f}s{exam_bit}",
            flush=True,
        )
        model.save_pretrained(args.out)
        print(f"  saved checkpoint -> {args.out}", flush=True)

    model.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}")
    print(f"Evaluate: python -m src.evaluate_sarah_intent --split test --adapter {args.out}")


if __name__ == "__main__":
    main()
