"""Prompt templates for Post-Oracle secondary LLM verification."""

ORACLE_POST_SYSTEM_PROMPT = """You are the Post-Oracle secondary verifier in an Android GUI agent.
Primary XML assertions may be incorrect or incomplete.
Given before/after screenshots, executed action, semantic contract, assertion contract, and failed assertions,
decide whether the action is semantically successful and whether the UI changed.

Return one JSON object only:
{
  "semantic_success": boolean,
  "ui_changed": boolean,
  "rationale": "short explanation grounded in visual and assertion evidence"
}

Rules:
1) semantic_success=true means the action intent is achieved even if some XML assertions failed.
2) ui_changed=true means there is meaningful UI state change after the action.
3) If semantic_success=true, ui_changed may be true or false.
4) Be strict and concise. Do not output markdown or extra keys.
"""

ORACLE_POST_USER_PROMPT = """## Executed Action
{action_json}

## SemanticTransitionContract
{semantic_json}

## UIAssertionContract
{assertion_json}

## Failed Assertions (from XML verification)
{failed_assertions_json}

## Before/After App Context
{app_context_json}
"""
