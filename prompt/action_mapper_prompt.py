"""Action mapper prompt template aligned with ResolvedAction schema."""

ACTION_MAPPER_PROMPT = """You are the action mapper.
Map one planning intent to one executable ResolvedAction JSON.

## Planning Intent
{intent_json}

## UI Widgets
{widget_list}

## JSON Schema
{schema_json}

Rules:
1. Output JSON only.
2. Choose actionable coordinates from the widget list when possible.
3. Keep params execution-only (x/y/text/duration/key data), no evaluation semantics.
"""
