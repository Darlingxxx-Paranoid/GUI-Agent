"""
Evaluator Prompt Template
"""

EVALUATOR_PROMPT = """You are the evaluation module of a GUI automation Agent.
Determine whether the subgoal has been successfully completed.

## Subgoal
{subgoal_description}

## Acceptance Criteria
{acceptance_criteria}

## UI State Summary Before Execution
{old_state_summary}

## UI State Summary After Execution
{new_state_summary}

Output in JSON format:
```json
{{
    "success": true or false,
    "reason": "Reason for the judgment"
}}
```
"""
