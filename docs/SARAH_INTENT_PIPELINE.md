# How tribe-chat intent gate maps to this repo

Verified against `/Users/kavya/Desktop/tribe-chat` (local clone).

## Production flow (intent only)

1. `QueryService.ask()` → `classify_tenant_intent()` in `tenant_intent_router.py`.
2. Prompt = `prompts/tenant_intent_gate.md` + **Recent conversation** + **Current user message** + `Return JSON only.`
3. History comes from Vanna Postgres store (`PostgresConversationStore`), trimmed to `AI_QUERY_INTENT_HISTORY_TURNS` (default 5 pairs).
4. Gemini (`GEMINI_INTENT_MODEL`, default `gemini-2.5-flash`) returns text; `_parse_intent_json()` extracts fields into `TenantIntentResult`.
5. Backend may **rewrite again** (`apply_property_*`, profile boost) before SQL — that is **not** part of the LLM output.

## What this repo mirrors

| tribe-chat | rleval |
|------------|--------|
| `tenant_intent_gate.md` | `prompts/sarah_tenant_intent_gate.md` (copy) |
| `format_history_block` + build in `classify_tenant_intent` | `src/sarah/intent_prompt.py` |
| `_parse_intent_json` | `src/sarah/parse_intent.py` |
| (no RL yet) | `src/sarah/score_intent.py` for eval / future reward |
| Real logs | `src/sarah/synthetic_data.py` until backend JSONL arrives |

## Persisted Q&A (partial)

`QueryService._persist_qa()` stores question, answer, status, resource, request_type — **not** full intent JSON. For experiments you still need log export or a new logging field.

## Run smoke test

```bash
source .venv/bin/activate
python -m src.evaluate_sarah_intent --split test
python -m src.evaluate_sarah_intent --limit 3   # quick
```

## SFT (synthetic + your labels)

Hand-label real turns in `data/sarah/labeled.jsonl` (same schema as `export_row_example.jsonl`).
Training uses the synthetic **train** split plus any JSONL rows that include `intent_gate_output`.

```bash
# One terminal only — kill stray `train_sft_intent` jobs or MPS can hang for 30+ minutes.
python -m src.train_sft_intent --no-exam --limit 2 --epochs 1 --out checkpoints/sft-intent-smoke
python -m src.train_sft_intent --no-exam --jsonl data/sarah/labeled.jsonl --epochs 4 --out checkpoints/sft-intent-lora
python -m src.evaluate_sarah_intent --split test --limit 3 --adapter checkpoints/sft-intent-smoke
```

Each step is ~2.2k tokens (full gate prompt). With gradient checkpointing, ~2 steps took ~1 minute on a free MPS device; `--no-checkpoint` often **OOM on 8 GB**.
