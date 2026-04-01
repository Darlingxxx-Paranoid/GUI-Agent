"""Oracle pre prompt template aligned with StepContract schema."""

ORACLE_PRE_PROMPT = """You are the pre-oracle module.
Build one StepContract JSON object strictly following the provided schema.

## Final Task
{task_hint}

## Planning Intent
{plan_json}

## Current UI
{ui_state}

## JSON Schema
{schema_json}

Rules:
1. Output JSON only.
2. Keep policies minimal: loop_guard, app_boundary, activity_boundary, visual_guard.
3. Required success evidence must be encoded by expectations with tier="required".
4. visual_similarity_state can only be supporting evidence, never required.
5. Use app_boundary.boundary_mode = stay/switch/either and expected_packages to express boundary intent.
6. planning_hints can include diagnostics but must not include decision logic.
"""
