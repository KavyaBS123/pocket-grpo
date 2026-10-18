"""Turn an LLM's raw text reply into a sequence of tools, then score it.

This is the bridge between the language model and our environment: the model
writes text, we parse it into tools, and compare them with the task's correct
tool sequence.

Reward rule:
    +1  exactly the right tools in the right order
     0  the right tools, but in the wrong order (partial credit)
    -1  anything else: wrong tool, missing tool, extra tool, or not a tool list

For one-tool tasks this is exactly the original +1 / -1 rule from environment.py.
"""

from src.environment import ACTIONS, CORRECT_REWARD, INCORRECT_REWARD

WRONG_ORDER_REWARD = 0.0

# Characters we forgive around each tool name, e.g. "Calculator." or "`search`".
_STRIP_CHARS = " \t\n.`'\""


def parse_tools(text):
    """Return a tuple of tool names if `text` is a comma-separated tool list, else None.

    "calculator"            -> ("calculator",)
    "Database, calculator." -> ("database", "calculator")
    "I would use search"    -> None   (a sentence, not a tool list)

    Deliberately strict: real tool-calling systems need a precise, machine-
    readable answer to execute anything, so RL must also teach the format.
    """
    parts = [part.strip(_STRIP_CHARS).lower() for part in text.split(",")]
    if all(part in ACTIONS for part in parts):
        return tuple(parts)
    return None


def sequence_reward(expected_tools, text):
    """Score the reply `text` against the correct tool sequence `expected_tools`."""
    tools = parse_tools(text)
    if tools is None:
        return INCORRECT_REWARD
    if tools == tuple(expected_tools):
        return CORRECT_REWARD
    # Same tools, each used the same number of times, but in a different order.
    # sorted() ignores order, so equal sorted lists mean "same tools, wrong order".
    # Extra or missing tools make the lists differ, so they still get -1;
    # otherwise the model could game the reward by always listing every tool.
    if sorted(tools) == sorted(expected_tools):
        return WRONG_ORDER_REWARD
    return INCORRECT_REWARD
