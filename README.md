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
```

Training needs a few GB of free memory. On an 8 GB Mac with many apps open,
it slows to a crawl from swapping; close browsers and other heavy apps first.

The first run downloads the model (~1 GB) to `~/.cache/huggingface`.

## Layout

```
src/environment.py        states, actions, reward function, ToolSelectionEnv
src/main.py               random-policy baseline demo
src/tasks.py              60 natural-language tasks, split into train/test
src/llm_reward.py         parses the LLM's reply into a tool and scores it
src/llm_policy.py         loads the LLM and generates replies
src/evaluate_llm.py       measures the LLM's tool-selection accuracy
src/train_grpo.py         simplified GRPO training loop with LoRA (optional KL penalty)
src/check_drift.py        compares ordinary chat replies before/after training
tests/                    unittest test suite
```
