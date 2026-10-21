"""A tiny tool-selection environment for learning reinforcement learning.

Each episode is a single step:
    1. reset()  -> the environment presents a task (the state).
    2. step(a)  -> the agent picks a tool (the action) and receives a reward.

The agent never sees CORRECT_TOOL directly; it can only learn it from rewards.
"""

import random

ACTIONS = ("search", "calculator", "database")

CORRECT_TOOL = {
    "lookup_fact": "search",
    "current_events": "search",
    "arithmetic": "calculator",
    "unit_conversion": "calculator",
    "customer_record": "database",
    "order_history": "database",
}

STATES = tuple(CORRECT_TOOL)

CORRECT_REWARD = 1.0
INCORRECT_REWARD = -1.0


def validate_state(state):
    """Raise ValueError unless `state` is one of the six known task types."""
    if state not in CORRECT_TOOL:
        raise ValueError(f"Unknown state {state!r}. Valid states: {', '.join(STATES)}")


def validate_action(action):
    """Raise ValueError unless `action` is one of the three known tools."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}. Valid actions: {', '.join(ACTIONS)}")


def reward_for(state, action):
    """Reward for choosing `action` in `state`: +1 if correct, -1 otherwise."""
    validate_state(state)
    validate_action(action)
    return CORRECT_REWARD if CORRECT_TOOL[state] == action else INCORRECT_REWARD


class ToolSelectionEnv:
    """Single-step episodes: one task, one tool choice, one reward."""

    def __init__(self, seed=None):
        """seed makes the randomly drawn task sequence reproducible across runs."""
        self._rng = random.Random(seed)
        self._state = None
        self._done = True

    def reset(self, state=None):
        """Start a new episode and return its state.

        Pass `state` to choose the task explicitly; otherwise one is sampled uniformly.
        """
        if state is None:
            state = self._rng.choice(STATES)
        else:
            validate_state(state)
        self._state = state
        self._done = False
        return state

    def step(self, action):
        """Apply `action` to the current state. Returns (reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")
        validate_action(action)

        reward = reward_for(self._state, action)
        self._done = True
        info = {
            "state": self._state,
            "action": action,
            "correct_action": CORRECT_TOOL[self._state],
        }
        return reward, self._done, info
