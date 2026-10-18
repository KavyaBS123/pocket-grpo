"""Tests for the LLM task dataset and the text-to-reward bridge.

These run without PyTorch or a model: they only check the plain-Python parts
that decide what the model is rewarded for.
"""

import unittest

from src.environment import CORRECT_REWARD, INCORRECT_REWARD, STATES
from src.llm_reward import llm_reward, parse_tool
from src.tasks import ALL_TASKS, TASKS_BY_STATE, TEST_PER_STATE, TEST_TASKS, TRAIN_TASKS


class TestTasks(unittest.TestCase):
    def test_tasks_cover_exactly_the_environment_states(self):
        # Every environment state needs tasks, and no task may use an unknown state.
        self.assertEqual(set(TASKS_BY_STATE), set(STATES))

    def test_split_sizes(self):
        # 6 states x 3 held-out tasks = 18 test tasks; the rest are for training.
        self.assertEqual(len(TEST_TASKS), len(STATES) * TEST_PER_STATE)
        self.assertEqual(len(TRAIN_TASKS) + len(TEST_TASKS), len(ALL_TASKS))

    def test_train_and_test_do_not_overlap(self):
        # If a test sentence leaked into training, the test score would be meaningless.
        train_texts = {text for text, _ in TRAIN_TASKS}
        test_texts = {text for text, _ in TEST_TASKS}
        self.assertEqual(train_texts & test_texts, set())

    def test_no_duplicate_sentences(self):
        texts = [text for text, _ in ALL_TASKS]
        self.assertEqual(len(texts), len(set(texts)))


class TestParseTool(unittest.TestCase):
    def test_accepts_plain_tool_names(self):
        for tool in ("search", "calculator", "database"):
            self.assertEqual(parse_tool(tool), tool)

    def test_forgives_case_whitespace_and_punctuation(self):
        self.assertEqual(parse_tool("  Calculator.\n"), "calculator")
        self.assertEqual(parse_tool("`search`"), "search")
        self.assertEqual(parse_tool('"DATABASE"'), "database")

    def test_rejects_sentences_and_unknown_tools(self):
        self.assertIsNone(parse_tool("I would use the calculator"))
        self.assertIsNone(parse_tool("search or database"))
        self.assertIsNone(parse_tool("hammer"))
        self.assertIsNone(parse_tool(""))


class TestLlmReward(unittest.TestCase):
    def test_correct_reply_is_rewarded(self):
        self.assertEqual(llm_reward("arithmetic", "calculator"), CORRECT_REWARD)

    def test_wrong_tool_is_penalized(self):
        self.assertEqual(llm_reward("arithmetic", "search"), INCORRECT_REWARD)

    def test_unparseable_reply_is_penalized_like_a_wrong_tool(self):
        self.assertEqual(llm_reward("arithmetic", "Let me think about it..."), INCORRECT_REWARD)


if __name__ == "__main__":
    unittest.main()
