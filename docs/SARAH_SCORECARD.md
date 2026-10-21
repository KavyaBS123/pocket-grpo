# Sarah intent experiment — scorecard

Use this when comparing **Gemini (production)** vs **Qwen (offline)** on the intent gate only.

## Primary metrics (holdout set)

| Metric | Definition | Target (first milestone) |
|--------|------------|---------------------------|
| **Parse rate** | Output contains valid intent JSON with all enum fields valid (template echoes of the prompt's value lists do not count) | ≥ 95% |
| **Routing accuracy** | `intent` + `resource` + `request_type` match gold | Beat Gemini on same set |
| **Constraint recall** | Required substrings in `rewritten_question` (beds, suburb, budget) | ≥ 90% on follow-up turns |
| **Full match** | Routing + rewrite constraints (or exact rewrite if labelled) | Track trend; not required day one |

## Secondary (when export includes them)

| Metric | Definition |
|--------|------------|
| **Guest safety** | Personal `resource` + `request_type=personal` not predicted for guest on shared-only turns |
| **Agreement with Gemini** | % exact match to `intent_gate_output` in export (imitation baseline) |
| **Downstream proxy** | Same `answer_text` status when replaying intent only (optional, needs sandbox) |

## Suggested split

- **Train:** practice + SFT (if labels trusted).
- **Exam:** 30+ turns you label by hand (follow-ups mandatory).
- **Gemini mirror set:** random 20% of export to measure “matches production” without claiming correctness.

## When to add RL (GRPO)

Only after:

1. SFT Qwen reaches ≥ 80% parse rate on exam, and  
2. At least 30% of sampled groups have mixed rewards (same lesson as tool-selection).

## When *not* to switch production

- Parse rate &lt; Gemini on exam  
- Follow-up constraint recall &lt; Gemini  
- Any increase in guest → personal routing on labelled set  
