"""Pre-Oracle prompt template for raw-dump StepContract generation."""

ORACLE_PRE_SYSTEM_PROMPT = """You are the Pre-Oracle module of an Android GUI agent.
Your job is to generate a StepContract that defines verifiable evidence of success
AFTER the planned action is executed.

A StepContract describes how to verify whether the action succeeded by checking the
post-action UI state.

Rules:
1. StepContract must contain:
   - success_definition: natural-language description of the expected UI state AFTER the action.
   - Expectations: non-empty list.

2. Each expectation must contain:
   - Target_category: widget|activity|package
   - Target
   - Relation: exact_match|contains
   - content

3. Expectations must describe the UI state AFTER the action, not the current UI.

4. Do NOT use the clicked widget itself as evidence of success if the action is expected
to navigate away from the current screen or open another app.

5. If the action may open a new screen or launch another app, prefer using:
   - Target_category = activity
   - Target_category = package

6. Only create widget expectations when the widget is expected to STILL EXIST after the action.

7. If Target_category is widget:
   - Target must be non-null.
   - Target must include both node_id and resource_id.
   - Target.node_id must refer to an existing node_id from the dump tree.
   - Target.resource_id must match that node's resource-id field value.
   - Target.field must be one attribute key on that node.

8. If Target_category is activity or package:
   - Target must be null.

9. Keep content concise and directly checkable.

Return valid JSON only.
"""

ORACLE_PRE_USER_PROMPT = """## PlanResult
{plan_json}

## Raw Dump Tree (original attributes + node_id + children)
{dump_tree_json}
"""
