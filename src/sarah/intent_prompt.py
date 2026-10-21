"""Build the same style of prompt tribe-chat sends to the intent gate LLM.

In production, classify_tenant_intent() concatenates:
  1. tenant_intent_gate.md
  2. Recent conversation block
  3. Current user message
  4. "Return JSON only."

We mirror that here so Qwen sees the same task shape before real logs arrive.
"""

from pathlib import Path

# Copied from tribe-chat (see prompts/sarah_tenant_intent_gate.md header in repo).
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "sarah_tenant_intent_gate.md"


def load_intent_gate_prompt() -> str:
    """Read the copied production prompt from disk (single source of truth)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def format_history_block(history: list[dict[str, str]], *, empty_label: str = "(none)") -> str:
    """Match tribe-chat format_history_block: User:/Assistant: lines."""
    if not history:
        return empty_label
    lines = []
    for turn in history:
        role = str(turn.get("role", "user")).capitalize()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def build_intent_classify_prompt(
    user_message: str,
    prior_turns: list[dict[str, str]] | None = None,
) -> str:
    """Full text prompt for one intent-gate call (not chat-template wrapped)."""
    history = prior_turns or []
    return "\n\n".join(
        [
            load_intent_gate_prompt(),
            "## Recent conversation (up to 5 prior user/assistant turns)",
            format_history_block(history),
            f"## Current user message\n{user_message.strip()}",
            "Return JSON only.",
        ]
    )
