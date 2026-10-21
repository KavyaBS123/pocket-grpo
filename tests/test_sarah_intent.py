"""Tests for Sarah intent parsing, scoring, and synthetic dataset."""

import unittest
from pathlib import Path

from src.sarah.export_data import (
    export_summary,
    load_export_jsonl,
    load_export_lines,
    parse_dbeaver_html,
    turns_from_ai_query_qa,
)
from src.sarah.intent_dataset import build_sft_turns, gold_to_target_json, turn_to_messages
from src.sarah.intent_prompt import build_intent_classify_prompt, load_intent_gate_prompt
from src.sarah.parse_intent import parse_intent_json
from src.sarah.score_intent import IntentGold, score_intent, scalar_reward
from src.sarah.synthetic_data import SPLITS, TEST_PER_CATEGORY


class TestPrompt(unittest.TestCase):
    """The rebuilt prompt must carry all four production parts."""

    def test_intent_gate_prompt_file_loads(self):
        # Smoke check that the copied production prompt is present and has the
        # two things the rest of the experiment depends on: JSON-only demand
        # and the live_data label.
        text = load_intent_gate_prompt()
        self.assertIn("Return **JSON only**", text)
        self.assertIn("live_data", text)

    def test_build_prompt_includes_history_and_user_message(self):
        prompt = build_intent_classify_prompt(
            "under 800",
            [{"role": "user", "content": "2BHK Merrylands"}],
        )
        self.assertIn("under 800", prompt)
        self.assertIn("2BHK Merrylands", prompt)
        self.assertIn("Return JSON only.", prompt)


class TestParseIntent(unittest.TestCase):
    """JSON extraction and enum validation, including noisy replies."""

    def test_parses_clean_json(self):
        raw = (
            '{"intent":"live_data","resource":"properties","request_type":"shared",'
            '"rewritten_question":"Show units","chitchat_reply":"","confidence":"high"}'
        )
        p = parse_intent_json(raw)
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.resource, "properties")

    def test_extracts_json_from_markdown_noise(self):
        # Models often wrap the JSON in prose or ``` fences; the parser must
        # still find and read it, or the whole gate call counts as failed.
        raw = 'Here you go:\n```json\n{"intent":"greeting_chitchat","resource":"general",'
        raw += '"request_type":"personal","rewritten_question":"","chitchat_reply":"Hi","confidence":"high"}\n```'
        p = parse_intent_json(raw)
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.intent, "greeting_chitchat")

    def test_invalid_json_fails_parse(self):
        # Garbage output must be flagged (parse_ok=False), not silently turned
        # into a default intent: production would reroute the user wrongly.
        p = parse_intent_json("I would use the calculator")
        self.assertFalse(p.parse_ok)
        self.assertFalse(p.enums_valid)

    def test_template_echo_parses_but_enums_invalid(self):
        # Seen with untrained Qwen: it copies the prompt's value list instead
        # of choosing one value. The JSON "parses", but no field is usable, so
        # the strict parse rate must count this as a failure.
        raw = (
            '{"intent":"greeting_chitchat | live_data","resource":"properties | users",'
            '"request_type":"personal | shared","rewritten_question":"x",'
            '"chitchat_reply":"","confidence":"high"}'
        )
        p = parse_intent_json(raw)
        self.assertTrue(p.parse_ok)
        self.assertFalse(p.enums_valid)


class TestScoreIntent(unittest.TestCase):
    """Scoring model output against gold labels (the future RL reward)."""

    def test_full_exact_gets_positive_reward(self):
        gold = IntentGold(
            intent="live_data",
            resource="users",
            request_type="personal",
            rewritten_question="Show my profile",
        )
        parsed = parse_intent_json(
            '{"intent":"live_data","resource":"users","request_type":"personal",'
            '"rewritten_question":"Show my profile","chitchat_reply":"","confidence":"high"}'
        )
        sc = score_intent(parsed, gold)
        self.assertTrue(sc.full_exact)
        self.assertEqual(scalar_reward(sc), 1.0)

    def test_must_contain_constraints(self):
        # Follow-up merge: the rewrite must keep the filters from earlier turns
        # (here "melbourne" and "30000"), else SQL searches the wrong thing.
        gold = IntentGold(
            intent="live_data",
            resource="properties",
            request_type="shared",
            rewritten_question="x",
            rewrite_must_contain=("melbourne", "30000"),
        )
        parsed = parse_intent_json(
            '{"intent":"live_data","resource":"properties","request_type":"shared",'
            '"rewritten_question":"Show 2 bed in Melbourne only","chitchat_reply":"","confidence":"high"}'
        )
        sc = score_intent(parsed, gold)
        self.assertFalse(sc.rewrite_constraints_ok)

    def test_enum_invalid_reply_gets_no_reward(self):
        # Even if the coerced defaults (live_data/general/personal) happen to
        # match a gold label, a reply with invalid enum values is a format
        # failure: -1, never a partial score.
        gold = IntentGold(intent="live_data", resource="general", request_type="personal")
        parsed = parse_intent_json('{"intent":"?","resource":"?","request_type":"?","confidence":"?"}')
        sc = score_intent(parsed, gold)
        self.assertFalse(sc.routing_exact)
        self.assertEqual(scalar_reward(sc), -1.0)


class TestExportData(unittest.TestCase):
    """The backend export format (see data/sarah/export_row_example.jsonl)."""

    EXPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "sarah" / "export_row_example.jsonl"

    def test_example_export_loads_without_problems(self):
        turns, problems = load_export_jsonl(self.EXPORT_PATH)
        self.assertEqual(problems, [])
        self.assertEqual(len(turns), 2)

    def test_rows_keep_history_and_gold(self):
        turns, _ = load_export_jsonl(self.EXPORT_PATH)
        first, follow_up = turns
        self.assertEqual(first.prior_turns, [])
        self.assertEqual(first.gold.intent, "live_data")
        self.assertEqual(first.gold.resource, "properties")
        self.assertEqual(len(follow_up.prior_turns), 2)  # the "under 30000" follow-up

    def test_summary_counts(self):
        turns, _ = load_export_jsonl(self.EXPORT_PATH)
        summary = export_summary(turns)
        self.assertEqual(summary["labeled"], 2)
        self.assertEqual(summary["follow_ups"], 1)

    def test_bad_lines_are_reported_not_fatal(self):
        lines = [
            "not json at all",
            '{"conversation_id":"c1","turn_index":1,"user_message":"Hi"}',
            '{"conversation_id":"c1","turn_index":2,"user_message":"   "}',
        ]
        turns, problems = load_export_lines(lines)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(problems), 2)
        self.assertIsNone(turns[0].gold)  # no intent_gate_output = unlabeled


class TestAiQueryQaImport(unittest.TestCase):
    """Importing a DBeaver HTML dump of the ai_query_qa table."""

    # Minimal copy of the DBeaver output shape: table-name row, header, data.
    HTML = """<table><tbody>
        <tr><th colspan="12"><pre><code>ai_query_qa</code></pre></th></tr>
        <tr><th>id</th><th>user_id</th><th>organisation_id</th><th>role</th>
        <th>conversation_id</th><th>question</th><th>answer</th><th>status</th>
        <th>resource</th><th>request_type</th><th>request_id</th><th>created_at</th></tr>
        <tr><td>1</td><td>u1</td><td>&nbsp;</td><td>tenant</td><td>c1</td>
        <td>2BHK in Merrylands</td><td>Here are units</td><td>success</td>
        <td>properties</td><td>shared</td><td>&nbsp;</td>
        <td>2026-08-11 00:00:01.000 +0530</td></tr>
        <tr class="odd"><td>2</td><td>u1</td><td>&nbsp;</td><td>tenant</td><td>c1</td>
        <td>under 800</td><td>Filtered list</td><td>success</td>
        <td>properties</td><td>shared</td><td>&nbsp;</td>
        <td>2026-08-11 00:01:02.000 +0530</td></tr>
        </tbody></table>"""

    def test_html_rows_parse_with_header_names(self):
        rows = parse_dbeaver_html(self.HTML)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "2BHK in Merrylands")
        self.assertEqual(rows[0]["organisation_id"], "")  # &nbsp; becomes empty

    def test_follow_up_gets_reconstructed_history(self):
        turns, problems = turns_from_ai_query_qa(parse_dbeaver_html(self.HTML))
        self.assertEqual(problems, [])
        first, follow_up = turns
        self.assertEqual(first.prior_turns, [])
        self.assertEqual([t["content"] for t in follow_up.prior_turns], ["2BHK in Merrylands", "Here are units"])
        self.assertEqual(follow_up.category, "properties")  # reporting bucket = resource
        self.assertIsNone(follow_up.gold)  # ai_query_qa has no Gemini output

    def test_guest_user_id_is_tagged(self):
        turns, _ = turns_from_ai_query_qa(parse_dbeaver_html(self.HTML), guest_user_id="u1")
        self.assertTrue(all(turn.auth_mode == "guest" for turn in turns))


class TestIntentDataset(unittest.TestCase):
    """SFT targets and merging synthetic + labeled JSONL."""

    def test_gold_to_target_json_is_compact(self):
        gold = IntentGold(
            intent="live_data",
            resource="properties",
            request_type="shared",
            rewritten_question="Show units",
        )
        text = gold_to_target_json(gold)
        self.assertIn('"intent":"live_data"', text)
        self.assertNotIn(" ", text.split(":")[0])  # no spaces after separators

    def test_turn_to_messages_includes_gate_prompt(self):
        turn = SPLITS["train"][0]
        user, assistant = turn_to_messages(turn)
        self.assertIn("Return JSON only.", user)
        self.assertIn('"intent"', assistant)

    def test_build_sft_includes_jsonl(self):
        path = Path(__file__).resolve().parents[1] / "data" / "sarah" / "labeled.jsonl"
        turns = build_sft_turns(use_synthetic_train=False, jsonl_paths=[path])
        self.assertGreaterEqual(len(turns), 2)


class TestSyntheticData(unittest.TestCase):
    """The hand-written dataset that stands in until the backend export arrives."""

    def test_train_test_disjoint(self):
        train_ids = {t.turn_id for t in SPLITS["train"]}
        test_ids = {t.turn_id for t in SPLITS["test"]}
        self.assertEqual(train_ids & test_ids, set())

    def test_follow_up_turn_exists(self):
        # Follow-ups ("under 800" after "2BHK in Merrylands") are the hardest
        # turns for the gate, so the dataset must contain at least one.
        ids = {t.turn_id for t in SPLITS["all"]}
        self.assertIn("prop_follow_1", ids)

    def test_every_turn_has_gold(self):
        for turn in SPLITS["all"]:
            self.assertTrue(turn.gold.intent)


if __name__ == "__main__":
    unittest.main()
