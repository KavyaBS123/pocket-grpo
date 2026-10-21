# RL Tool Selection

A small learning project: an agent learns to pick the right tool
(`search`, `calculator`, `database`) for a task using tabular Q-learning.

Standard library only. Requires Python 3.11 (developed with 3.11.16).

## Setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv   # once
source .venv/bin/activate                    # every new terminal
python --version                             # should print Python 3.11.x
```

Leave the virtual environment with `deactivate`.

## Running tests

From the project root, with the virtual environment active:

```bash
python -m unittest discover -s tests -v
```

## Random-policy demo

```bash
python -m src.main
```

## LLM experiments

The LLM part needs PyTorch, Transformers, and PEFT (about 1-2 GB):

```bash
python -m pip install -r requirements.txt
python -m src.evaluate_llm                # baseline accuracy of Qwen2.5-0.5B-Instruct
python -m src.evaluate_llm --split test   # held-out tasks only
python -m src.train_grpo                  # RL training (~5 min on an M1; close other apps first)
python -m src.evaluate_llm --adapter checkpoints/grpo-lora   # evaluate the trained model

# Experiments: more exploration, and the KL penalty
python -m src.train_grpo --temperature 1.3 --out checkpoints/grpo-t1.3
python -m src.train_grpo --temperature 1.3 --kl-coef 0.04 --out checkpoints/grpo-t1.3-kl
python -m src.check_drift checkpoints/grpo-lora checkpoints/grpo-t1.3 checkpoints/grpo-t1.3-kl

# Hard task set: two-tool sequences and tricky questions
python -m src.evaluate_llm --tasks hard
python -m src.train_grpo --tasks hard --temperature 1.3 --epochs 6 --out checkpoints/grpo-hard
python -m src.evaluate_llm --tasks hard --adapter checkpoints/grpo-hard
```

Hard-set reward: +1 for the exact tool sequence, 0 for the right tools in the
wrong order, -1 for anything else (including extra tools, to block reward hacking).

## Sarah intent gate (offline, waiting on backend export)

Synthetic turns + smoke test using the copied `tenant_intent_gate` prompt:

```bash
python -m src.evaluate_sarah_intent --split test
python -m src.evaluate_sarah_intent --limit 3
python -m src.train_sft_intent --no-exam --limit 3 --epochs 2     # quick smoke (~minutes per step on M1)
python -m src.train_sft_intent --jsonl data/sarah/labeled.jsonl   # full run: synthetic train + labels
python -m src.evaluate_sarah_intent --split test --adapter checkpoints/sft-intent-lora
```

See `docs/SARAH_SCORECARD.md`, `docs/SARAH_INTENT_PIPELINE.md`, and
`data/sarah/export_row_example.jsonl` for backend handoff.

Training needs a few GB of free memory. On an 8 GB Mac with many apps open,
it slows to a crawl from swapping; close browsers and other heavy apps first.

The first run downloads the model (~1 GB) to `~/.cache/huggingface`.

## Layout

```
src/environment.py             states, actions, reward function, ToolSelectionEnv
src/main.py                    random-policy baseline demo
src/tasks.py                   task sets ("basic": 60 tasks, "hard": 102 tasks), train/test splits
src/llm_reward.py              parses the LLM's reply into a tool sequence and scores it
src/llm_policy.py              loads the LLM and generates replies
src/evaluate_llm.py            measures the LLM's tool-selection accuracy
src/train_grpo.py              simplified GRPO training loop with LoRA (optional KL penalty)
src/check_drift.py             compares ordinary chat replies before/after training
src/sarah/intent_prompt.py     rebuilds the tribe-chat intent prompt (gate doc + history + message)
src/sarah/parse_intent.py      JSON text -> ParsedIntent, same fields as tribe-chat TenantIntentResult
src/sarah/score_intent.py      scores parsed output against gold labels (becomes the RL reward)
src/sarah/synthetic_data.py    hand-written Sarah turns used until the backend export arrives
src/sarah/export_data.py       loads a backend export JSONL or DBeaver HTML dump + prints a summary
src/evaluate_sarah_intent.py   smoke-test Qwen on the Sarah intent-gate task
src/train_sft_intent.py        LoRA supervised fine-tuning on intent JSON (synthetic + JSONL)
src/sarah/intent_dataset.py    gold JSON targets and merge synthetic + labeled export rows
prompts/                       copied tribe-chat prompts (source of truth for the experiments)
data/sarah/                    export format example for the tribe-chat backend handoff
docs/                          pipeline notes and the Gemini-vs-Qwen scorecard
tests/                         unittest test suite
```
