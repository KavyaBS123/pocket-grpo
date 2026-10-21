"""Parse intent-gate JSON from model text (same fields as tribe-chat TenantIntentResult)."""

import json
import re
from dataclasses import dataclass

# Allowed values mirror app/core/ai_query/schemas.py in tribe-chat.
VALID_INTENTS = frozenset({"greeting_chitchat", "general_tribe_info", "live_data", "off_topic"})
VALID_RESOURCES = frozenset(
    {
        "properties",
        "payments",
        "applications",
        "users",
        "maintenance_requests",
        "bookings",
        "amenities",
        "retail_stores",
        "tribe_hero",
        "run_club",
        "audit_logs",
        "general",
    }
)
VALID_REQUEST_TYPES = frozenset({"personal", "shared"})
VALID_CONFIDENCE = frozenset({"high", "low"})

# Greedy match from the first "{" to the last "}": enough because the prompt
# demands exactly one JSON object, and it survives stray prose or markdown
# fences around it (same trick as tribe-chat's _parse_intent_json).
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class ParsedIntent:
    """Structured intent gate output plus how well it parsed.

    The fields mirror the JSON keys the gate prompt asks for; `raw` keeps the
    model's original text for debugging when parsing fails.
    """

    parse_ok: bool  # a JSON object was found and decoded
    enums_valid: bool  # every enum field was present and valid (no default used)
    intent: str
    resource: str
    request_type: str
    rewritten_question: str
    chitchat_reply: str
    confidence: str
    raw: str


def parse_intent_json(raw: str, *, fallback_question: str = "") -> ParsedIntent:
    """Extract the first JSON object from `raw` and validate enum fields.

    Invalid or missing enum values fall back to safe defaults (so callers can
    always trust the fields), but `enums_valid` records that a fallback was
    used, so the strict parse rate stays measurable.
    """
    base = ParsedIntent(
        parse_ok=False,
        enums_valid=False,
        intent="live_data",
        resource="general",
        request_type="personal",
        rewritten_question=fallback_question.strip(),
        chitchat_reply="",
        confidence="low",
        raw=raw or "",
    )
    match = _JSON_OBJECT_RE.search(raw or "")
    if not match:
        return base
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return base
    if not isinstance(parsed, dict):
        return base

    intent = str(parsed.get("intent") or "").strip().lower()
    intent_ok = intent in VALID_INTENTS
    if not intent_ok:
        intent = "live_data"

    resource = str(parsed.get("resource") or "").strip().lower()
    resource_ok = resource in VALID_RESOURCES
    if not resource_ok:
        resource = "general"

    request_type = str(parsed.get("request_type") or "").strip().lower()
    request_type_ok = request_type in VALID_REQUEST_TYPES
    if not request_type_ok:
        request_type = "personal"

    rewritten = str(parsed.get("rewritten_question") or "").strip() or fallback_question.strip()
    chitchat = str(parsed.get("chitchat_reply") or "").strip()
    confidence = str(parsed.get("confidence") or "").strip().lower()
    confidence_ok = confidence in VALID_CONFIDENCE
    if not confidence_ok:
        confidence = "low"

    return ParsedIntent(
        parse_ok=True,
        # Strict validity: any coerced field (typical of untrained-model
        # template echoes) makes the whole reply a format failure.
        enums_valid=intent_ok and resource_ok and request_type_ok and confidence_ok,
        intent=intent,
        resource=resource,
        request_type=request_type,
        rewritten_question=rewritten,
        chitchat_reply=chitchat,
        confidence=confidence,
        raw=raw,
    )
