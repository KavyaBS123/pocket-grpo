"""Tests for the LLM task sets and the text-to-reward bridge.

These run without PyTorch or a model: they only check the plain-Python parts
that decide what the model is rewarded for.
"""

import unittest

from src.environment import ACTIONS, CORRECT_REWARD, CORRECT_TOOL, INCORRECT_REWARD, STATES, reward_for
from src.llm_reward import WRONG_ORDER_REWARD, parse_tools, sequence_reward
from src.tasks import BASIC_TEST_PER_STATE, TASK_SETS, TASKS_BY_STATE


class TestTaskSets(unittest.TestCase):
    """The task data itself: coverage, split sizes, and no train/test leakage."""

    def test_basic_tasks_cover_exactly_the_environment_states(self):
        self.assertEqual(set(TASKS_BY_STATE), set(STATES))

    def test_basic_split_sizes(self):
        # 6 states x 3 held-out tasks = 18 test tasks, 42 practice tasks.
        basic = TASK_SETS["basic"]
        self.assertEqual(len(basic["test"]), len(STATES) * BASIC_TEST_PER_STATE)
        self.assertEqual(len(basic["train"]), 42)

    def test_hard_set_contains_all_basic_tasks(self):
        # The hard set must still include one-tool questions, so the model also
        # learns when NOT to answer with two tools.
        for split in ("train", "test"):
            self.assertTrue(set(TASK_SETS["basic"][split]) <= set(TASK_SETS["hard"][split]))

    def test_train_and_test_do_not_overlap(self):
        # If a test sentence leaked into training, the exam score would be meaningless.
        for name, task_set in TASK_SETS.items():
            with self.subTest(task_set=name):
                train = {t.text for t in task_set["train"]}
                test = {t.text for t in task_set["test"]}
                self.assertEqual(train & test, set())

    def test_no_duplicate_sentences(self):
        # A duplicate would double-count one question and blur the train/test
        # boundary (a sentence could be in both sets through two tables).
        hard = TASK_SETS["hard"]
        texts = [t.text for t in hard["train"] + hard["test"]]
        self.assertEqual(len(texts), len(set(texts)))

    def test_every_task_uses_valid_tools(self):
        # Catches typos in the tool names written in the task tables.
        hard = TASK_SETS["hard"]
        for task in hard["train"] + hard["test"]:
            with self.subTest(task=task.text):
                self.assertTrue(1 <= len(task.tools) <= 2)
                self.assertTrue(all(tool in ACTIONS for tool in task.tools))


class TestParseTools(unittest.TestCase):
    """Text -> tool tuple (or None): the strict format contract the LLM must meet."""

    def test_single_tool(self):
        for tool in ACTIONS:
            self.assertEqual(parse_tools(tool), (tool,))

    def test_forgives_case_whitespace_and_punctuation(self):
        self.assertEqual(parse_tools("  Calculator.\n"), ("calculator",))
        self.assertEqual(parse_tools("`search`"), ("search",))
        self.assertEqual(parse_tools("Database, calculator."), ("database", "calculator"))
        self.assertEqual(parse_tools("search,calculator"), ("search", "calculator"))

    def test_rejects_sentences_and_unknown_tools(self):
        self.assertIsNone(parse_tools("I would use the calculator"))
        self.assertIsNone(parse_tools("search or database"))
        self.assertIsNone(parse_tools("search, hammer"))
        self.assertIsNone(parse_tools("hammer"))
        self.assertIsNone(parse_tools(""))
        self.assertIsNone(parse_tools("search,"))


class TestSequenceReward(unittest.TestCase):
    """The +1 / 0 / -1 reward that GRPO training optimizes."""

    def test_exact_sequence_is_rewarded(self):
        self.assertEqual(sequence_reward(("database", "calculator"), "database, calculator"), CORRECT_REWARD)

    def test_wrong_order_gets_partial_credit(self):
        self.assertEqual(sequence_reward(("database", "calculator"), "calculator, database"), WRONG_ORDER_REWARD)

    def test_missing_tool_is_penalized(self):
        self.assertEqual(sequence_reward(("database", "calculator"), "database"), INCORRECT_REWARD)

    def test_extra_tool_is_penalized(self):
        # Blocks the "list every tool for partial credit" cheat.
        self.assertEqual(sequence_reward(("database", "calculator"), "database, calculator, search"), INCORRECT_REWARD)
        self.assertEqual(sequence_reward(("search",), "search, calculator"), INCORRECT_REWARD)

    def test_unparseable_reply_is_penalized(self):
        self.assertEqual(sequence_reward(("calculator",), "Let me think about it..."), INCORRECT_REWARD)

    def test_single_tool_matches_original_environment_reward(self):
        # For one-tool tasks the new reward must be exactly the old +1 / -1 rule.
        for state in STATES:
            for action in ACTIONS:
                with self.subTest(state=state, action=action):
                    self.assertEqual(sequence_reward((CORRECT_TOOL[state],), action), reward_for(state, action))


if __name__ == "__main__":
    unittest.main()
