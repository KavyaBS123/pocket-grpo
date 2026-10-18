import unittest

from src.environment import (
    ACTIONS,
    CORRECT_REWARD,
    CORRECT_TOOL,
    INCORRECT_REWARD,
    STATES,
    ToolSelectionEnv,
    reward_for,
)


class TestDefinitions(unittest.TestCase):
    def test_three_actions(self):
        self.assertEqual(ACTIONS, ("search", "calculator", "database"))

    def test_every_state_maps_to_a_valid_action(self):
        for state in STATES:
            self.assertIn(CORRECT_TOOL[state], ACTIONS)

    def test_every_action_is_correct_for_some_state(self):
        self.assertEqual(set(CORRECT_TOOL.values()), set(ACTIONS))


class TestRewardFunction(unittest.TestCase):
    def test_correct_tool_is_rewarded(self):
        for state in STATES:
            with self.subTest(state=state):
                self.assertEqual(reward_for(state, CORRECT_TOOL[state]), CORRECT_REWARD)

    def test_every_wrong_tool_is_penalized(self):
        for state in STATES:
            for action in ACTIONS:
                if action == CORRECT_TOOL[state]:
                    continue
                with self.subTest(state=state, action=action):
                    self.assertEqual(reward_for(state, action), INCORRECT_REWARD)

    def test_reward_signs(self):
        self.assertGreater(CORRECT_REWARD, 0)
        self.assertLess(INCORRECT_REWARD, 0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            reward_for("not_a_state", "search")
        with self.assertRaises(ValueError):
            reward_for("arithmetic", "hammer")


class TestEpisodeFlow(unittest.TestCase):
    def test_reset_returns_valid_state(self):
        env = ToolSelectionEnv(seed=0)
        for _ in range(50):
            self.assertIn(env.reset(), STATES)

    def test_step_returns_reward_done_and_info(self):
        env = ToolSelectionEnv()
        env.reset(state="arithmetic")
        reward, done, info = env.step("calculator")
        self.assertEqual(reward, CORRECT_REWARD)
        self.assertTrue(done)
        self.assertEqual(
            info,
            {"state": "arithmetic", "action": "calculator", "correct_action": "calculator"},
        )

    def test_wrong_tool_in_episode(self):
        env = ToolSelectionEnv()
        env.reset(state="customer_record")
        reward, done, _ = env.step("search")
        self.assertEqual(reward, INCORRECT_REWARD)
        self.assertTrue(done)

    def test_same_seed_gives_same_task_sequence(self):
        a = ToolSelectionEnv(seed=42)
        b = ToolSelectionEnv(seed=42)
        self.assertEqual([a.reset() for _ in range(20)], [b.reset() for _ in range(20)])

    def test_all_states_are_eventually_sampled(self):
        env = ToolSelectionEnv(seed=1)
        seen = {env.reset() for _ in range(200)}
        self.assertEqual(seen, set(STATES))


class TestValidation(unittest.TestCase):
    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            ToolSelectionEnv().step("search")

    def test_step_after_episode_ends_raises(self):
        env = ToolSelectionEnv()
        env.reset(state="lookup_fact")
        env.step("search")
        with self.assertRaises(RuntimeError):
            env.step("search")

    def test_invalid_action_raises_and_episode_stays_open(self):
        env = ToolSelectionEnv()
        env.reset(state="lookup_fact")
        with self.assertRaises(ValueError):
            env.step("hammer")
        reward, _, _ = env.step("search")
        self.assertEqual(reward, CORRECT_REWARD)

    def test_reset_with_invalid_state_raises(self):
        with self.assertRaises(ValueError):
            ToolSelectionEnv().reset(state="not_a_state")


if __name__ == "__main__":
    unittest.main()
