"""Training examples for Sarah intent SFT (synthetic + labeled JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from src.sarah.export_data import load_export_jsonl
from src.sarah.intent_prompt import build_intent_classify_prompt
from src.sarah.score_intent import IntentGold
from src.sarah.synthetic_data import SPLITS, SarahTurn


def gold_to_target_json(gold: IntentGold) -> str:
    """One-line JSON target matching tribe-chat intent_gate_output fields."""
    return json.dumps(
        {
            "intent": gold.intent,
            "resource": gold.resource,
            "request_type": gold.request_type,
            "rewritten_question": gold.rewritten_question,
            "chitchat_reply": gold.chitchat_reply,
            "confidence": gold.confidence,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def turn_to_messages(turn: SarahTurn) -> tuple[str, str]:
    """User prompt text and assistant JSON for one labeled turn."""
    if turn.gold is None:
        raise ValueError(f"turn {turn.turn_id} has no gold labels")
    prompt = build_intent_classify_prompt(turn.user_message, turn.prior_turns)
    target = gold_to_target_json(turn.gold)
    return prompt, target


def labeled_turns_from_jsonl(path: str | Path) -> list[SarahTurn]:
    """Load export JSONL and keep only rows with intent_gate_output."""
    turns, problems = load_export_jsonl(path)
    labeled = [t for t in turns if t.gold is not None]
    if problems:
        # Caller can log; we do not fail on skipped lines.
        pass
    return labeled


def build_sft_turns(
    *,
    use_synthetic_train: bool = True,
    jsonl_paths: list[str | Path] | None = None,
) -> list[SarahTurn]:
    """Merge synthetic train split with optional hand-labeled JSONL files."""
    out: list[SarahTurn] = []
    if use_synthetic_train:
        out.extend(SPLITS["train"])
    for path in jsonl_paths or []:
        out.extend(labeled_turns_from_jsonl(path))
    # Drop duplicates by turn_id while keeping first occurrence.
    seen: set[str] = set()
    unique: list[SarahTurn] = []
    for turn in out:
        if turn.turn_id in seen:
            continue
        seen.add(turn.turn_id)
        if turn.gold is None:
            continue
        unique.append(turn)
    return unique
