"""Hand-written Sarah-like turns until the backend export arrives.

Each SarahTurn is one user message plus optional prior_turns and gold labels.
rewrite_must_contain checks that follow-up merges (beds, suburb, budget) survived.
"""

from collections import namedtuple

from src.sarah.score_intent import IntentGold

SarahTurn = namedtuple(
    "SarahTurn",
    "turn_id category auth_mode prior_turns user_message gold",
)

# Last TEST_PER_CATEGORY turns per category are held out (fixed split).
TEST_PER_CATEGORY = 1

_SYNTHETIC: list[tuple[str, list[SarahTurn]]] = [
    (
        "greeting",
        [
            SarahTurn(
                "greet_1",
                "greeting",
                "authenticated",
                [],
                "Hi Sarah!",
                IntentGold(
                    intent="greeting_chitchat",
                    resource="general",
                    request_type="personal",
                    chitchat_reply="Hi",
                    rewritten_question="",
                ),
            ),
            SarahTurn(
                "greet_2",
                "greeting",
                "guest",
                [],
                "Good morning, how are you?",
                IntentGold(
                    intent="greeting_chitchat",
                    resource="general",
                    request_type="shared",
                    chitchat_reply="Good morning",
                    rewritten_question="",
                ),
            ),
        ],
    ),
    (
        "tribe_info",
        [
            SarahTurn(
                "heroes_1",
                "tribe_info",
                "authenticated",
                [],
                "What is Tribe Heroes?",
                IntentGold(
                    intent="general_tribe_info",
                    resource="general",
                    request_type="shared",
                    chitchat_reply="Tribe Heroes",
                    rewritten_question="",
                ),
            ),
        ],
    ),
    (
        "properties_browse",
        [
            SarahTurn(
                "prop_1",
                "properties_browse",
                "guest",
                [],
                "Show me two-bedroom apartments in Melbourne.",
                IntentGold(
                    intent="live_data",
                    resource="properties",
                    request_type="shared",
                    rewritten_question=(
                        "Show vacant active units with 2 bedrooms in Melbourne including "
                        "project name, suburb, and weekly rent."
                    ),
                    rewrite_must_contain=("2", "melbourne", "bed"),
                ),
            ),
            SarahTurn(
                "prop_follow_1",
                "properties_browse",
                "guest",
                [
                    {"role": "user", "content": "Show me two-bedroom apartments in Melbourne."},
                    {"role": "assistant", "content": "Here are some available two-bedroom units in Melbourne."},
                ],
                "What about under ₹30,000 per month?",
                IntentGold(
                    intent="live_data",
                    resource="properties",
                    request_type="shared",
                    rewritten_question=(
                        "Show vacant 2 bedroom units in Melbourne under 30000 per month "
                        "including project name, suburb, and weekly rent."
                    ),
                    rewrite_must_contain=("2", "melbourne", "30000"),
                ),
            ),
            SarahTurn(
                "prop_2",
                "properties_browse",
                "authenticated",
                [],
                "2BHK under 900 in Merrylands",
                IntentGold(
                    intent="live_data",
                    resource="properties",
                    request_type="shared",
                    rewritten_question="Show vacant 2 bedroom units in Merrylands under 900 per week.",
                    rewrite_must_contain=("2", "merrylands", "900"),
                ),
            ),
            SarahTurn(
                "prop_follow_2",
                "properties_browse",
                "authenticated",
                [
                    {"role": "user", "content": "2BHK under 900 in Merrylands"},
                    {"role": "assistant", "content": "I found several units in Merrylands."},
                ],
                "under 800",
                IntentGold(
                    intent="live_data",
                    resource="properties",
                    request_type="shared",
                    rewritten_question="Show vacant 2 bedroom units in Merrylands under 800 per week.",
                    rewrite_must_contain=("2", "merrylands", "800"),
                ),
            ),
        ],
    ),
    (
        "profile",
        [
            SarahTurn(
                "profile_1",
                "profile",
                "authenticated",
                [],
                "Show me my profile details",
                IntentGold(
                    intent="live_data",
                    resource="users",
                    request_type="personal",
                    rewritten_question=(
                        "Show my profile for the current tenant including full name, email, phone, "
                        "account status, first and last name, date of birth, nationality, current city, "
                        "and current residential address."
                    ),
                    rewrite_must_contain=("profile",),
                ),
            ),
            SarahTurn(
                "profile_2",
                "profile",
                "authenticated",
                [],
                "What's my email?",
                IntentGold(
                    intent="live_data",
                    resource="users",
                    request_type="personal",
                    rewritten_question="What is my email address and phone number on my account?",
                    rewrite_must_contain=("email",),
                ),
            ),
        ],
    ),
    (
        "payments",
        [
            SarahTurn(
                "pay_1",
                "payments",
                "authenticated",
                [],
                "What do I owe on rent this month?",
                IntentGold(
                    intent="live_data",
                    resource="payments",
                    request_type="personal",
                    rewritten_question="What are my outstanding rent payments and recent payment history?",
                    rewrite_must_contain=("payment",),
                ),
            ),
        ],
    ),
    (
        "maintenance",
        [
            SarahTurn(
                "maint_1",
                "maintenance",
                "authenticated",
                [],
                "Show my maintenance requests",
                IntentGold(
                    intent="live_data",
                    resource="maintenance_requests",
                    request_type="personal",
                    rewritten_question="List my maintenance requests with status, description, and dates.",
                    rewrite_must_contain=("maintenance",),
                ),
            ),
        ],
    ),
    (
        "tricky",
        [
            SarahTurn(
                "tricky_pop",
                "tricky",
                "guest",
                [],
                "How many people live in Tokyo?",
                IntentGold(
                    intent="off_topic",
                    resource="general",
                    request_type="shared",
                    chitchat_reply="outside Tribe",
                    rewritten_question="",
                ),
            ),
            SarahTurn(
                "tricky_orders",
                "tricky",
                "authenticated",
                [],
                "How many orders did we receive last week?",
                IntentGold(
                    intent="live_data",
                    resource="retail_stores",
                    request_type="personal",
                    rewritten_question="Show my retail orders from the last week with status and amounts.",
                    rewrite_must_contain=("order",),
                ),
            ),
            SarahTurn(
                "tricky_calc",
                "tricky",
                "guest",
                [],
                "How many seconds are in a week?",
                IntentGold(
                    intent="live_data",
                    resource="general",
                    request_type="shared",
                    rewritten_question="Calculate how many seconds are in one week.",
                    rewrite_must_contain=("second", "week"),
                ),
            ),
        ],
    ),
    (
        "off_topic",
        [
            SarahTurn(
                "off_1",
                "off_topic",
                "guest",
                [],
                "Tell me a joke about penguins.",
                IntentGold(
                    intent="off_topic",
                    resource="general",
                    request_type="shared",
                    chitchat_reply="outside",
                    rewritten_question="",
                ),
            ),
        ],
    ),
]


def _split_category(turns: list[SarahTurn]) -> tuple[list[SarahTurn], list[SarahTurn]]:
    """Last TEST_PER_CATEGORY turns of a category go to the exam set."""
    if len(turns) <= TEST_PER_CATEGORY:
        return [], turns[:]
    return turns[:-TEST_PER_CATEGORY], turns[-TEST_PER_CATEGORY:]


def build_splits() -> dict[str, list[SarahTurn]]:
    """Assemble the fixed train/test/all lists used by the evaluator.

    Fixed (not random) so every experiment is measured on the same exam turns;
    tiny categories whose only turn is the held-out one simply have no train rows.
    """
    train, test = [], []
    for _cat, turns in _SYNTHETIC:
        tr, te = _split_category(turns)
        train.extend(tr)
        test.extend(te)
    return {"train": train, "test": test, "all": train + test}


SPLITS = build_splits()
