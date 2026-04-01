"""Oracle pre prompt template aligned with StepContract schema."""

ORACLE_PRE_PROMPT = """You are the pre-oracle module.
Build one StepContract JSON object strictly following the provided schema.

## Subgoal Summary
{goal_summary}

## Action Type
{action_type}

## Current UI
{ui_state}

## JSON Schema
{schema_json}

Rules:
1. Output JSON only.
2. Keep policies minimal: loop_guard, app_boundary, activity_boundary, visual_guard.
3. Expectations must use predicates and tier/polarity.
4. planning_hints can include helper hints but must not include decision logic.
"""
