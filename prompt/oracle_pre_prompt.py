"""Pre-Oracle prompt template for raw-dump StepContract generation."""

ORACLE_PRE_PROMPT = """You are the Pre-Oracle module of an Android GUI agent.
Generate one StepContract object for success verification.

## PlanResult
{plan_json}

## Raw Dump Tree (original attributes + node_id + children)
{dump_tree_json}

Rules:
1. StepContract must contain:
   - success_definition: natural-language success description.
   - Expectations: non-empty list.
2. Each expectation must contain:
   - Target_category: widget|activity|package
   - Target
   - Relation: exact_match|contains
   - content
3. If Target_category is widget:
   - Target must be non-null.
   - Target must include both node_id and resource_id.
   - Target.node_id must refer to an existing node_id from the dump tree.
   - Target.resource_id must match that node's resource-id field value.
   - Target.field must be one original attribute key on that node.
4. If Target_category is activity or package:
   - Target must be null.
5. Keep content concise and directly checkable.
"""
