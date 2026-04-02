"""Prompt template for Planner module (task + screenshot)."""

PLANNER_SYSTEM_PROMPT = """You are the Plan module of an Android GUI agent.
Use task text, screenshot image, and UIED visible controls to produce one next-step plan.
"""

PLANNER_USER_PROMPT = """Task:
{task}

UIED Visible Controls (JSON list):
{uied_controls_json}

Rules:
1) If task is already complete in screenshot, set is_task_complete=true, action_type=wait, input_text=''.
2) If task is not complete, set is_task_complete=false and choose action_type from tap/input/swipe/back/enter/long_press/launch_app.
3) goal must describe one concrete next step only.
4) reasoning must be concise and grounded in the screenshot.
5) Use the UIED controls list as grounding evidence for visible targets.
6) You MUST output direct execution coordinates:
   - For tap/input/long_press/swipe: choose one target control from UIED list, output its id in target_control_id, and output action_x/action_y.
   - For back/enter/launch_app: set target_control_id=-1, action_x=-1, action_y=-1.
   - If is_task_complete=true: set target_control_id=-1, action_x=-1, action_y=-1.
7) Coordinates must be valid screen points and should align with the chosen control center or tappable area.
"""
