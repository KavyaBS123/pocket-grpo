"""Load and inspect tribe-chat conversation exports (backend handoff).

Two input formats are supported:

1. Export JSONL — one conversation turn per line, matching the schema shown in
   data/sarah/export_row_example.jsonl (conversation_id, turn_index, auth_mode,
   user_message, prior_turns, intent_gate_output, ...). intent_gate_output is
   Gemini's intent JSON and is present only if the export includes it.

2. DBeaver "export as HTML" dumps of the `ai_query_qa` table — the production
   Q&A log (id, user_id, role, conversation_id, question, answer, status,
   resource, request_type, created_at). It has the original question and final
   answer but NOT Gemini's raw gate output (intent, rewritten_question), so
   every turn imported this way is unlabeled; history is reconstructed from
   the earlier turns of the same conversation_id.

Rows are converted into the same SarahTurn objects the synthetic data uses,
so the evaluator (and later training code) work unchanged. CLI:

    python -m src.sarah.export_data data/sarah/export_row_example.jsonl
    python -m src.sarah.export_data data/sarah/ai_query_qa_sample.html
"""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

from src.sarah.score_intent import IntentGold
from src.sarah.synthetic_data import SarahTurn


def _gold_from_intent_output(payload: object) -> IntentGold | None:
    """Build IntentGold from the export's intent_gate_output; None if absent."""
    if not isinstance(payload, dict) or not payload:
        return None
    return IntentGold(
        intent=str(payload.get("intent") or ""),
        resource=str(payload.get("resource") or ""),
        request_type=str(payload.get("request_type") or ""),
        rewritten_question=str(payload.get("rewritten_question") or ""),
        chitchat_reply=str(payload.get("chitchat_reply") or ""),
        confidence=str(payload.get("confidence") or "high"),
    )


def parse_export_row(row: dict, line_no: int = 0) -> SarahTurn:
    """Convert one export row into a SarahTurn (gold=None when unlabeled)."""
    conversation_id = str(row.get("conversation_id") or f"line_{line_no}")
    turn_index = row.get("turn_index", line_no)
    payload = row.get("intent_gate_output")
    payload = payload if isinstance(payload, dict) else {}
    return SarahTurn(
        turn_id=f"{conversation_id}:{turn_index}",
        # Reporting bucket: the Gemini resource when labeled, else "unlabeled".
        category=str(payload.get("resource") or "unlabeled"),
        auth_mode=str(row.get("auth_mode") or "unknown"),
        prior_turns=list(row.get("prior_turns") or []),
        user_message=str(row.get("user_message") or "").strip(),
        gold=_gold_from_intent_output(payload),
    )


def load_export_lines(lines) -> tuple[list[SarahTurn], list[str]]:
    """Parse export lines; returns (turns, problem messages for skipped lines).

    Unusable lines are reported, not fatal: a real export is expected to have a
    few odd rows (empty prompts, truncated writes) that should not sink the run.
    """
    turns: list[SarahTurn] = []
    problems: list[str] = []
    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {line_no}: not valid JSON, skipped")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {line_no}: JSON is not an object, skipped")
            continue
        if not str(row.get("user_message") or "").strip():
            problems.append(f"line {line_no}: missing user_message, skipped")
            continue
        turns.append(parse_export_row(row, line_no))
    return turns, problems


def load_export_jsonl(path: str | Path) -> tuple[list[SarahTurn], list[str]]:
    """Read an export file from disk (see load_export_lines)."""
    text = Path(path).read_text(encoding="utf-8")
    return load_export_lines(text.splitlines())


def export_summary(turns: list[SarahTurn]) -> dict:
    """Counts used to judge an export before using it for eval/training."""
    labeled = sum(turn.gold is not None for turn in turns)
    follow_ups = sum(bool(turn.prior_turns) for turn in turns)
    conversations = {turn.turn_id.rsplit(":", 1)[0] for turn in turns}
    return {
        "turns": len(turns),
        "conversations": len(conversations),
        "labeled": labeled,
        "unlabeled": len(turns) - labeled,
        "follow_ups": follow_ups,
        "auth_modes": sorted({turn.auth_mode for turn in turns}),
    }


# ---------------------------------------------------------------------------
# ai_query_qa (DBeaver HTML) import
# ---------------------------------------------------------------------------

# Production trims what the intent gate sees: the last 5 user/assistant pairs,
# each content capped so one giant answer cannot fill the prompt
# (mirrors ai_query/conversation_history defaults in tribe-chat).
HISTORY_TURNS = 5
HISTORY_CONTENT_CAP = 600


class _DbeaverTableParser(HTMLParser):
    """Collect the table of a DBeaver 'export as HTML' file into rows of cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            # Collapse whitespace; &nbsp; (now \xa0) disappears entirely.
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_dbeaver_html(text: str) -> list[dict[str, str]]:
    """Parse a DBeaver HTML table dump into dicts keyed by the column header.

    DBeaver writes one row with the table name, one header row, then data;
    this reads the header names so any table works, not just ai_query_qa.
    """
    parser = _DbeaverTableParser()
    parser.feed(text)
    rows = parser.rows
    if len(rows) < 3:
        raise ValueError("HTML does not look like a DBeaver table export")
    header = rows[1]
    return [dict(zip(header, row)) for row in rows[2:] if len(row) == len(header)]


def _cap(text: str, limit: int = HISTORY_CONTENT_CAP) -> str:
    """Cap one history content so prompts stay small (same as production)."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def turns_from_ai_query_qa(
    rows: list[dict[str, str]], *, guest_user_id: str | None = None
) -> tuple[list[SarahTurn], list[str]]:
    """Group ai_query_qa rows into export turns with reconstructed history.

    Notes on what this data can and cannot be:
    - `question` is the ORIGINAL user message (good: that is the gate input).
    - History is rebuilt from earlier turns of the same conversation_id; in
      production the gate reads it from the Vanna store, so this is close but
      not byte-identical (see docs/SARAH_INTENT_PIPELINE.md).
    - `resource`/`request_type` are the POST-BOOST production decision, kept
      only as the reporting category; they are not Gemini's raw output.
    - There is no intent/rewritten_question anywhere in this table, so every
      turn is unlabeled (gold=None).
    """
    problems: list[str] = []
    by_conversation: dict[str, list[dict[str, str]]] = {}
    for row_no, row in enumerate(rows, 1):
        if not str(row.get("question") or "").strip():
            problems.append(f"row {row_no}: missing question, skipped")
            continue
        conversation_id = str(row.get("conversation_id") or "").strip() or f"row_{row_no}"
        by_conversation.setdefault(conversation_id, []).append(row)

    turns: list[SarahTurn] = []
    for conversation_id, conversation_rows in by_conversation.items():
        # created_at strings are "YYYY-MM-DD HH:MM:SS.mmm +offset" — sortable.
        conversation_rows = sorted(
            conversation_rows, key=lambda r: str(r.get("created_at") or "")
        )
        for index, row in enumerate(conversation_rows, start=1):
            prior_turns: list[dict[str, str]] = []
            for previous in conversation_rows[max(0, index - 1 - HISTORY_TURNS) : index - 1]:
                prior_turns.append(
                    {"role": "user", "content": _cap(str(previous.get("question") or ""))}
                )
                prior_turns.append(
                    {"role": "assistant", "content": _cap(str(previous.get("answer") or ""))}
                )
            user_id = str(row.get("user_id") or "")
            turns.append(
                SarahTurn(
                    turn_id=f"{conversation_id}:{index}",
                    category=str(row.get("resource") or "unlabeled"),
                    auth_mode=(
                        "guest"
                        if guest_user_id and user_id == guest_user_id
                        else "authenticated"
                    ),
                    prior_turns=prior_turns,
                    user_message=str(row.get("question") or "").strip(),
                    gold=None,
                )
            )
    return turns, problems


def load_ai_query_qa_html(
    path: str | Path, *, guest_user_id: str | None = None
) -> tuple[list[SarahTurn], list[str]]:
    """Read a DBeaver HTML dump of the ai_query_qa table."""
    text = Path(path).read_text(encoding="utf-8")
    return turns_from_ai_query_qa(parse_dbeaver_html(text), guest_user_id=guest_user_id)


def main():
    """Print a summary of an export file and flag anything unusable."""
    parser = argparse.ArgumentParser(description="Check a tribe-chat export before using it.")
    parser.add_argument(
        "path",
        help="export JSONL (export_row_example format) or a DBeaver HTML table dump",
    )
    parser.add_argument(
        "--guest-user-id",
        default="",
        help="user_id to treat as the guest account (HTML dumps only)",
    )
    args = parser.parse_args()

    try:
        if Path(args.path).suffix.lower() in (".html", ".htm"):
            turns, problems = load_ai_query_qa_html(
                args.path, guest_user_id=args.guest_user_id or None
            )
        else:
            turns, problems = load_export_jsonl(args.path)
    except ValueError as exc:
        raise SystemExit(f"Could not parse {args.path}: {exc}")
    summary = export_summary(turns)

    print(f"Checked {args.path}")
    for key, value in summary.items():
        print(f"  {key:<14} {value}")
    if problems:
        print(f"  problems       {len(problems)}")
        for problem in problems[:10]:
            print(f"    - {problem}")
    print()

    if summary["labeled"] == 0:
        print("No intent_gate_output labels in this file. Ways forward:")
        print("  1. Grep server logs for 'intent_llm_response' (raw Gemini gate output).")
        print("  2. Replay the turns through the production Gemini gate for imitation labels.")
        print("  3. Hand-label a 30+ turn exam set (scorecard requires follow-ups in it).")
    else:
        print("Labels present: ready for an agreement/scoring run once wired into the evaluator.")

    print("\nFirst turns:")
    for turn in turns[:3]:
        gold = (
            f"{turn.gold.intent}/{turn.gold.resource}/{turn.gold.request_type}"
            if turn.gold
            else "(unlabeled)"
        )
        print(f"  {turn.turn_id:<18} {turn.user_message[:60]!r} gold={gold}")


if __name__ == "__main__":
    main()
