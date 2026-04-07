"""Action mapper prompt template aligned with ResolvedAction schema."""

ACTION_MAPPER_SYSTEM_PROMPT = """You are the action mapper.
Map one planning intent to one executable ResolvedAction JSON.

Rules:
1. Output JSON only.
2. Choose actionable coordinates from the widget list when possible.
3. Keep params execution-only (x/y/text/duration/key data), no evaluation semantics.
"""

ACTION_MAPPER_USER_PROMPT = """## Planning Intent
{intent_json}

## UI Widgets
{widget_list}

## JSON Schema
{schema_json}
"""

# Backward-compatible single-prompt alias.
ACTION_MAPPER_PROMPT = (
    ACTION_MAPPER_SYSTEM_PROMPT
    + "\n\n"
    + ACTION_MAPPER_USER_PROMPT
)
