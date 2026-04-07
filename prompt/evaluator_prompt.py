"""Evaluator prompt template aligned with StepEvaluation schema."""

EVALUATOR_SYSTEM_PROMPT = """You are the post evaluator.
Given observations and contract, produce one StepEvaluation JSON object.

Rules:
1. Output JSON only.
2. Decision must be success/fail/uncertain.
3. recommended_action.kind must be one of continue/retry/replan/backtrack/observe/abort.
4. Provide structured expectation_matches and numeric metrics.
"""

EVALUATOR_USER_PROMPT = """## Step Contract
{contract_json}

## Observations
{observations_json}

## Runtime Assessments
{runtime_assessments_json}

## JSON Schema
{schema_json}
"""

# Backward-compatible single-prompt alias.
EVALUATOR_PROMPT = (
    EVALUATOR_SYSTEM_PROMPT
    + "\n\n"
    + EVALUATOR_USER_PROMPT
)
