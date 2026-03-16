"""
Replanner Prompt Template
"""

REPLANNER_PROMPT = """You are the failure analysis module of a GUI automation Agent.
The current subgoal execution has failed. Analyze the cause and suggest next actions.

## Subgoal
{subgoal_description}

## Failure Reason
{failure_reason}

## Current UI State
{ui_state}

## Execution History
{history}

Output in JSON format:
```json
{{
    "analysis": "Root cause analysis of the failure",
    "suggestion": "retry/back/replan/abort",
    "reasoning": "Reason for the suggestion"
}}
```
- retry: Retry the same subgoal (may be a timing issue)
- back: Go back to the previous page and re-observe (navigated to wrong page)
- replan: Re-plan the subgoal (wrong direction)
- abort: Task cannot continue (fundamental error)
"""
