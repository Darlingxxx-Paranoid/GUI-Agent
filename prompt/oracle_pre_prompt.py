"""
Oracle Pre-Constraint Prompt Template
"""

ORACLE_PRE_PROMPT = """You are the verification module of a GUI automation Agent.
Generate acceptance criteria and transition expectations for the following subgoal.

## Subgoal
- Description: {description}
- Action type: {action_type}
- Target widget: {target_widget}
- Input text: {input_text}

## Current UI State Summary
{ui_summary}

Output in JSON format:
```json
{{
    "expected_texts": ["Key texts that should appear after execution"],
    "forbidden_texts": ["Abnormal texts that should NOT appear"],
    "expected_activity": "Activity name if a page transition is expected",
    "widget_should_exist": ["Features of widgets expected to appear"],
    "widget_should_vanish": ["Features of widgets expected to disappear"],
    "transition_type": "partial_refresh/new_page/dialog/external_app/none",
    "transition_description": "Natural language description of the expected page change",
    "semantic_criteria": "Natural language description of the success criteria for this subgoal"
}}
```
"""
