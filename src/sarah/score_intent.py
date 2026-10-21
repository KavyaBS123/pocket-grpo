"""Score a parsed intent against gold labels (for eval and future RL rewards)."""

from dataclasses import dataclass

from src.sarah.parse_intent import ParsedIntent


@dataclass
class IntentGold:
    """Expected intent-gate output for one synthetic or exported turn."""

    intent: str
    resource: str
    request_type: str
    rewritten_question: str = ""
    chitchat_reply: str = ""
    confidence: str = "high"
    # Substrings the rewrite must include (case-insensitive), e.g. "melbourne", "30000".
    rewrite_must_contain: tuple[str, ...] = ()


@dataclass
class IntentScore:
    """Breakdown of one turn's evaluation."""

    parse_ok: bool  # a JSON object was found and decoded
    enums_valid: bool  # all enum fields valid — the strict parse rate
    routing_exact: bool  # intent + resource + request_type all match gold
    rewrite_constraints_ok: bool
    full_exact: bool  # routing + rewritten_question exact match (when gold rewrite non-empty)


def _contains_all(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return all(n.lower() in lower for n in needles)


def score_intent(parsed: ParsedIntent, gold: IntentGold) -> IntentScore:
    """Compare model output to gold. Used offline; same logic can become RL reward later."""
    # A coerced field means the model did not really produce a usable answer,
    # so nothing built on top of it can count as correct.
    strict_ok = parsed.parse_ok and parsed.enums_valid
    routing_exact = (
        strict_ok
        and parsed.intent == gold.intent
        and parsed.resource == gold.resource
        and parsed.request_type == gold.request_type
    )
    rewrite_constraints_ok = _contains_all(parsed.rewritten_question, gold.rewrite_must_contain)
    full_exact = routing_exact and (
        not gold.rewritten_question.strip()
        or parsed.rewritten_question.strip() == gold.rewritten_question.strip()
    )
    if gold.rewrite_must_contain:
        full_exact = full_exact and rewrite_constraints_ok

    return IntentScore(
        parse_ok=parsed.parse_ok,
        enums_valid=parsed.enums_valid,
        routing_exact=routing_exact,
        rewrite_constraints_ok=rewrite_constraints_ok if gold.rewrite_must_contain else True,
        full_exact=full_exact,
    )


def scalar_reward(score: IntentScore) -> float:
    """Simple +1 / 0 / -1 style reward for future GRPO on intent JSON.

    A JSON object with invalid or coerced enum fields is a format failure (-1):
    template echoes must never earn reward, even if the coerced defaults
    happen to look plausible.
    """
    if not (score.parse_ok and score.enums_valid):
        return -1.0
    if score.full_exact:
        return 1.0
    if score.routing_exact and score.rewrite_constraints_ok:
        return 0.5
    if score.routing_exact:
        return 0.0
    return -1.0
