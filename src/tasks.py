"""Natural-language tasks for the LLM version of the tool-selection problem.

Each task is a Task(text, tools, category):
    text     - the sentence the LLM sees (the state)
    tools    - the correct tools, IN ORDER, as a tuple, e.g. ("database", "calculator")
    category - a label used only for reporting results per kind of task

Two task sets are available:
    "basic" - the original 60 one-tool questions.
    "hard"  - the basic questions PLUS two-tool tasks and "tricky" questions whose
              wording points toward the wrong tool.
"""

from collections import namedtuple

from src.environment import CORRECT_TOOL

Task = namedtuple("Task", "text tools category")

# ---------------------------------------------------------------------------
# System prompts: the instructions the LLM sees before every task. In RL terms
# they are part of the state: the same question with different instructions is
# a different situation.
# ---------------------------------------------------------------------------

_TOOL_LIST = (
    "Available tools:\n"
    "- search: retrieve information\n"
    "- calculator: perform calculations\n"
    "- database: retrieve structured records\n"
)

# Tool descriptions match the original project definitions on purpose. We do
# NOT give hints like "calculator also handles unit conversions"; that mapping
# is something the model should learn from reward.
BASIC_SYSTEM_PROMPT = (
    "You choose which tool should handle a user's task.\n"
    + _TOOL_LIST
    + "Reply with only the tool name and nothing else."
)

# The example uses placeholder names, not real tools, so it doesn't hint at
# any particular answer.
HARD_SYSTEM_PROMPT = (
    "You choose which tools should handle a user's task.\n"
    + _TOOL_LIST
    + "If the task needs one tool, reply with only that tool name.\n"
    "If it needs several tools, reply with the tool names in the order they "
    'should be used, separated by commas, like "first_tool, second_tool".\n'
    "Reply with nothing else."
)

# ---------------------------------------------------------------------------
# Basic one-tool tasks. Each state has 10 examples; the last BASIC_TEST_PER_STATE
# of each list form the held-out test set.
# ---------------------------------------------------------------------------

TASKS_BY_STATE = {
    "lookup_fact": [
        "Who wrote the novel Pride and Prejudice?",
        "What is the capital of Australia?",
        "How tall is Mount Kilimanjaro?",
        "When was the Eiffel Tower built?",
        "What language is spoken in Brazil?",
        "Who painted the Mona Lisa?",
        "Which planet has the most moons?",
        "Who invented the telephone?",
        "What does the abbreviation DNA stand for?",
        "Which country hosted the 2016 Summer Olympics?",
    ],
    "current_events": [
        "What is the weather forecast for Mumbai tomorrow?",
        "Who won yesterday's Champions League match?",
        "What are today's top technology headlines?",
        "What is the current price of Bitcoin?",
        "Is there a train strike in London this week?",
        "What movies are releasing in theaters this weekend?",
        "What did the central bank announce about interest rates today?",
        "Are there traffic jams on the highway to the airport right now?",
        "What topics are trending on social media today?",
        "Has the next iPhone been announced yet?",
    ],
    "arithmetic": [
        "What is 18% of 2,450?",
        "Calculate 347 multiplied by 29.",
        "What is the square root of 1,764?",
        "If a bill of 1,280 rupees is split among 5 people, how much does each pay?",
        "What is 2 to the power of 15?",
        "Add 4,567 and 8,912.",
        "What is a 15% tip on a 64 dollar meal?",
        "Compute (125 - 37) / 4.",
        "What is the average of 12, 45, and 78?",
        "How much is 7.5 times 640?",
    ],
    "unit_conversion": [
        "Convert 72 degrees Fahrenheit to Celsius.",
        "How many kilometers is 26.2 miles?",
        "Convert 5 feet 9 inches to centimeters.",
        "How many ounces are in 2.5 kilograms?",
        "Convert 3 hours 45 minutes into seconds.",
        "How many liters are in 12 gallons?",
        "Convert 150 pounds to kilograms.",
        "How many square meters is 1,200 square feet?",
        "Convert 90 kilometers per hour to meters per second.",
        "How many megabytes are in 3.5 gigabytes?",
    ],
    "customer_record": [
        "Look up the email address for customer ID 48213.",
        "What is the shipping address on file for account #5521?",
        "Show the phone number of customer Priya Sharma in our CRM.",
        "When did customer 10092 create their account?",
        "What membership tier is customer ID 7731 on?",
        "Retrieve the billing details for client account 3390.",
        "Which sales rep is assigned to customer Acme Corp in our records?",
        "Find the date of birth stored for user ID 2045.",
        "Is customer 66120 marked as active in our system?",
        "What is the loyalty points balance for customer 9087?",
    ],
    "order_history": [
        "List all orders placed by customer 48213 last month.",
        "What was the total of order #A-10457?",
        "Show the status of order number 77812.",
        "How many orders did account 5521 place in 2025?",
        "Which products were in order #30981?",
        "When was order 55102 shipped?",
        "Find all refunded orders for customer Rahul Verma.",
        "What was the most recent purchase by user ID 2045?",
        "Show every order over 500 dollars from client 3390.",
        "Has order #88213 been delivered yet?",
    ],
}
BASIC_TEST_PER_STATE = 3

# ---------------------------------------------------------------------------
# Hard tasks: category -> (correct tool sequence, example sentences).
# The last HARD_TEST_PER_CATEGORY sentences of each list are held out.
# ---------------------------------------------------------------------------

HARD_TASKS_BY_CATEGORY = {
    # Two tools: find information first, then do math with it.
    "search_then_calc": (("search", "calculator"), [
        "Look up the population of Japan and divide it by its 47 prefectures.",
        "Find the current price of gold per ounce and calculate the cost of 3.5 ounces.",
        "Look up the height of the Eiffel Tower in meters and convert it to feet.",
        "Find today's USD to INR exchange rate and convert 250 dollars to rupees.",
        "Look up the distance from Earth to the Moon and work out how long light takes to travel it.",
        "Find the current price of Bitcoin and calculate the value of 0.25 BTC.",
        "Look up the year the Taj Mahal was completed and compute how many years ago that was.",
        "Find the boiling point of ethanol in Celsius and convert it to Kelvin.",
    ]),
    # Two tools: fetch our own records first, then do math with them.
    "database_then_calc": (("database", "calculator"), [
        "Get the total of order #A-10457 from our records and add 18% tax.",
        "Retrieve customer 48213's last three order amounts and compute their average.",
        "Look up the loyalty points balance for customer 9087 and calculate its value at 0.25 rupees per point.",
        "Pull the quantity in order 30981 from our database and multiply it by the unit price of 45 dollars.",
        "Find how many orders account 5521 placed in 2025 and work out the monthly average.",
        "Retrieve the refund amounts for customer Rahul Verma and add them up.",
        "Get the shipping weight of order 55102 from our system and convert it from kilograms to pounds.",
        "Look up the credit limit for client account 3390 and calculate 15% of it.",
    ]),
    # Two tools: fetch a record first, then search the web about what it says.
    "database_then_search": (("database", "search"), [
        "Find the city on file for customer 48213 and look up today's weather there.",
        "Get the product name in order #30981 from our records and search for reviews of it online.",
        "Look up which courier shipped order 55102 and check the news for delays at that courier.",
        "Retrieve the company name for client account 3390 and find recent news about that company.",
        "Find the home country stored for customer Priya Sharma and look up its public holidays this month.",
        "Get the airline on booking record 7731 and search for that airline's baggage policy.",
        "Look up the supplier for product SKU 2045 in our database and find their official website.",
        "Retrieve the preferred language of customer 10092 and search for a common greeting in that language.",
    ]),
    # Tricky: numbers in the question, but the answer must be looked up.
    "tricky_search": (("search",), [
        "How many people live in Tokyo?",
        "What was the score in the last cricket World Cup final?",
        "How many calories are in a banana?",
        "What is the speed limit on German autobahns?",
        "How many moons does Saturn have according to the latest count?",
        "What is the current inflation rate in India?",
    ]),
    # Tricky: sounds like counting or math, but it is a query on our records.
    "tricky_database": (("database",), [
        "How many orders did we receive last week?",
        "Count how many customers in our CRM signed up in March.",
        "What is the account balance of customer 66120?",
        "What percentage discount was applied to order #88213?",
        "How many units of SKU 2045 are left in our inventory records?",
        "What was the order total for order #A-20931?",
    ]),
    # Tricky: sounds like a fact lookup, but it is pure math.
    "tricky_calculator": (("calculator",), [
        "What is the area of a circle with a radius of 7 cm?",
        "How many seconds are there in a week?",
        "If I save 2,000 rupees a month, how much will I have after 3 years?",
        "What is 30% off a price of 1,499?",
        "How many days are there from March 3 to July 19 in the same year?",
        "What is the compound interest on 10,000 at 8% per year for 2 years?",
    ]),
}
HARD_TEST_PER_CATEGORY = 2


def _split(sentences, tools, category, n_test):
    """Turn sentences into Tasks and split them: the last n_test go to the test set.

    The split is fixed (not random) so every run uses exactly the same test set,
    which makes before/after comparisons fair.
    """
    tasks = [Task(text, tools, category) for text in sentences]
    return tasks[:-n_test], tasks[-n_test:]


def _build():
    basic_train, basic_test, hard_train, hard_test = [], [], [], []
    for state, sentences in TASKS_BY_STATE.items():
        train, test = _split(sentences, (CORRECT_TOOL[state],), state, BASIC_TEST_PER_STATE)
        basic_train += train
        basic_test += test
    for category, (tools, sentences) in HARD_TASKS_BY_CATEGORY.items():
        train, test = _split(sentences, tools, category, HARD_TEST_PER_CATEGORY)
        hard_train += train
        hard_test += test
    return {
        "basic": {"train": basic_train, "test": basic_test, "system_prompt": BASIC_SYSTEM_PROMPT},
        # The hard set keeps the basic questions too, so the model must also
        # learn when NOT to answer with two tools.
        "hard": {
            "train": basic_train + hard_train,
            "test": basic_test + hard_test,
            "system_prompt": HARD_SYSTEM_PROMPT,
        },
    }


# TASK_SETS["hard"]["train"] -> list of Tasks, etc.
TASK_SETS = _build()
