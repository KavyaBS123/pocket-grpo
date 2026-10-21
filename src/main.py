"""Play episodes with a random policy to see the environment in action.

Run from the project root:  python -m src.main
"""

import random

from src.environment import ACTIONS, ToolSelectionEnv


def run_random_policy(episodes=10, seed=0, verbose=True):
    """Play `episodes` episodes, choosing a tool uniformly at random.

    With three tools, guessing is right about 1/3 of the time. That number is
    the baseline the learning agent later has to beat.
    """
    env = ToolSelectionEnv(seed=seed)
    rng = random.Random(seed + 1)
    total_reward = 0.0
    correct = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = rng.choice(ACTIONS)
        reward, _, info = env.step(action)
        total_reward += reward
        correct += reward > 0
        if verbose:
            print(
                f"ep {episode:>3} | state={state:<16} action={action:<10} "
                f"reward={reward:+.0f} (correct: {info['correct_action']})"
            )

    print(f"\n{episodes} episodes: accuracy={correct / episodes:.0%}, total reward={total_reward:+.0f}")


if __name__ == "__main__":
    run_random_policy(episodes=10)
    print()
    run_random_policy(episodes=3000, verbose=False)
