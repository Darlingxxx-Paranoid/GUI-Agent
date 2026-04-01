"""Replanner prompt template aligned with StepEvaluation."""

REPLANNER_PROMPT = """You are the replanner.
Given one failed/uncertain StepEvaluation, suggest next control action.

## Goal
{subgoal_description}

## Evaluation
{evaluation_json}

## Current UI
{ui_state}

Output JSON only:
{
  "suggestion": "retry|back|replan|abort",
  "reason": "short reason"
}
"""
