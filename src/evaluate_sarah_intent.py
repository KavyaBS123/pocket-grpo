"""Smoke-test Qwen on Sarah's intent-gate task (synthetic data, no backend export).

Run from project root:
    python -m src.evaluate_sarah_intent              # all synthetic turns
    python -m src.evaluate_sarah_intent --split test   # held-out only
    python -m src.evaluate_sarah_intent --limit 5    # quick check

Uses the real tenant_intent_gate prompt copied into prompts/sarah_tenant_intent_gate.md.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_policy import MODEL_NAME, pick_device
from src.sarah.intent_prompt import build_intent_classify_prompt
from src.sarah.parse_intent import parse_intent_json
from src.sarah.score_intent import score_intent
from src.sarah.synthetic_data import SPLITS

# Intent JSON can be longer than a single tool name.
MAX_NEW_TOKENS = 256


def load_model(adapter_path: str | None = None):
    """Load Qwen in float32; optionally merge a LoRA adapter from SFT or GRPO."""
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_intent(model, tokenizer, prompt: str) -> str:
    """Greedy JSON generation from the full intent-gate prompt.

    The whole gate prompt is sent as one user message (no system message),
    mirroring how tribe-chat sends it to Gemini; decoding is greedy because
    production uses temperature 0 for the intent call.
    """
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    """Classify every turn in the split and print per-turn results plus totals."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="max turns (0 = all in split)")
    parser.add_argument("--adapter", default="", help="path to LoRA checkpoint (e.g. checkpoints/sft-intent-lora)")
    args = parser.parse_args()

    turns = SPLITS[args.split]
    if args.limit > 0:
        turns = turns[: args.limit]

    adapter = args.adapter.strip() or None
    print(f"Loading {MODEL_NAME}" + (f" + adapter {adapter}" if adapter else "") + "...")
    model, tokenizer = load_model(adapter)
    print(f"Evaluating {len(turns)} synthetic turns ({args.split}).\n")

    n_parse = n_enums = n_route = n_full = 0
    for turn in turns:
        prompt = build_intent_classify_prompt(turn.user_message, turn.prior_turns)
        raw = generate_intent(model, tokenizer, prompt)
        parsed = parse_intent_json(raw, fallback_question=turn.user_message)
        sc = score_intent(parsed, turn.gold)
        n_parse += sc.parse_ok
        n_enums += sc.enums_valid
        n_route += sc.routing_exact
        n_full += sc.full_exact

        if not sc.parse_ok:
            status = "bad-json"
        elif not sc.enums_valid:
            # Typical untrained-model failure: it echoes the prompt's value
            # lists ("live_data | off_topic") instead of choosing one.
            status = "echo"
        elif sc.full_exact:
            status = "OK"
        elif sc.routing_exact:
            status = "route"
        else:
            status = "miss"
        print(f"[{status}] {turn.turn_id} ({turn.category})")
        print(f"  user: {turn.user_message[:70]}")
        print(f"  gold: intent={turn.gold.intent} resource={turn.gold.resource} type={turn.gold.request_type}")
        print(
            f"  pred: intent={parsed.intent} resource={parsed.resource} type={parsed.request_type} "
            f"parse_ok={parsed.parse_ok} enums_valid={parsed.enums_valid}"
        )
        if not sc.full_exact:
            print(f"  raw: {raw[:200]!r}...")
        print()

    t = len(turns)
    # json_ok = some JSON was found; enums_valid = the strict parse rate the
    # scorecard tracks (all enum fields valid, no defaults coerced).
    print(
        f"Summary: json_ok {n_parse}/{t} | enums_valid {n_enums}/{t} | "
        f"routing_exact {n_route}/{t} | full_exact {n_full}/{t}"
    )


if __name__ == "__main__":
    main()
