"""Pure-LLM planner: Task + Screenshot -> PlanResult."""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Perception.uied_controls import get_uied_visible_widgets_list
from prompt.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from utils.llm_client import LLMRequest

logger = logging.getLogger(__name__)

PlanActionType = Literal[
    "tap",
    "input",
    "swipe",
    "back",
    "enter",
    "long_press",
    "launch_app",
    "wait",
]


class PlanResult(BaseModel):
    """Structured output contract for the Plan module."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1, description="One-step goal description.")
    action_type: PlanActionType = Field(description="Next action type.")
    input_text: str = Field(default="", description="Input text when action_type is input.")
    target_control_id: int = Field(
        default=-1,
        description="Chosen UIED control id. Use -1 when no control target is needed.",
    )
    action_x: int = Field(
        default=-1,
        description="Direct execution X coordinate. Use -1 when not needed.",
    )
    action_y: int = Field(
        default=-1,
        description="Direct execution Y coordinate. Use -1 when not needed.",
    )
    is_task_complete: bool = Field(description="Whether the full task is already complete.")
    reasoning: str = Field(min_length=1, description="Planning reason based on current screenshot.")

    @model_validator(mode="after")
    def validate_completion_rules(self) -> "PlanResult":
        if self.is_task_complete:
            if self.action_type != "wait":
                logger.error(
                    "PlanResult校验失败: is_task_complete=true 但 action_type=%s (期望 wait)",
                    self.action_type,
                )
                raise ValueError("When is_task_complete=true, action_type must be 'wait'.")
            if self.input_text != "":
                logger.error(
                    "PlanResult校验失败: is_task_complete=true 但 input_text 非空"
                )
                raise ValueError("When is_task_complete=true, input_text must be empty.")
            if self.action_x != -1 or self.action_y != -1:
                logger.error(
                    "PlanResult校验失败: is_task_complete=true 但坐标非空(x=%s,y=%s)",
                    self.action_x,
                    self.action_y,
                )
                raise ValueError("When is_task_complete=true, action_x/action_y must be -1.")
            logger.info("PlanResult校验通过: 完成态(action_type=wait, input_text为空)")
            return self

        if self.action_type == "wait":
            logger.error(
                "PlanResult校验失败: is_task_complete=false 但 action_type=wait"
            )
            raise ValueError("When is_task_complete=false, action_type cannot be 'wait'.")

        point_actions = {"tap", "input", "long_press", "swipe"}
        if self.action_type in point_actions:
            if self.action_x < 0 or self.action_y < 0:
                logger.error(
                    "PlanResult校验失败: action_type=%s 但坐标非法(x=%s,y=%s)",
                    self.action_type,
                    self.action_x,
                    self.action_y,
                )
                raise ValueError(
                    "When action_type is tap/input/long_press/swipe, action_x/action_y must be >= 0."
                )
            if self.target_control_id < -1:
                logger.error(
                    "PlanResult校验失败: target_control_id 非法(%s)",
                    self.target_control_id,
                )
                raise ValueError("target_control_id must be >= -1.")

        non_point_actions = {"back", "enter", "launch_app"}
        if self.action_type in non_point_actions and (self.action_x != -1 or self.action_y != -1):
            logger.error(
                "PlanResult校验失败: action_type=%s 不应输出坐标(x=%s,y=%s)",
                self.action_type,
                self.action_x,
                self.action_y,
            )
            raise ValueError("When action_type is back/enter/launch_app, action_x/action_y must be -1.")

        logger.info("PlanResult校验通过: 非完成态(action_type=%s)", self.action_type)
        return self


class Planner:
    """Plan next step from only task and screenshot using structured LLM output."""

    def __init__(self, llm_client, cv_output_dir: str | None = None, cv_resize_height: int = 800):
        self.llm = llm_client
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_cv_dir = os.path.join(project_root, "data", "cv_output")
        self.cv_output_dir = str(cv_output_dir or default_cv_dir)
        self.cv_resize_height = int(cv_resize_height or 800)
        logger.info(
            "Planner 初始化完成 (pure LLM, task+screenshot+uied), cv_output_dir=%s",
            self.cv_output_dir,
        )

    def plan(self, task: str, screenshot: str) -> PlanResult:
        task_text = str(task or "").strip()
        if not task_text:
            raise ValueError("task 不能为空")

        screenshot_path = str(screenshot or "").strip()
        if not screenshot_path:
            raise ValueError("screenshot 不能为空")

        logger.info(
            "开始规划: task_len=%d, screenshot=%s",
            len(task_text),
            screenshot_path,
        )

        system_prompt = PLANNER_SYSTEM_PROMPT
        visible_widgets_list_json = "[]"
        try:
            visible_widgets_list = get_uied_visible_widgets_list(
                screenshot_path=screenshot_path,
                cv_output_dir=self.cv_output_dir,
                resize_height=self.cv_resize_height,
            )
            # Keep prompt size stable while preserving top controls for grounding.
            visible_widgets_list_json = json.dumps(visible_widgets_list[:160], ensure_ascii=False)
            logger.info(
                "Planner 注入 UIED Visible Widgets List: total=%d, used=%d",
                len(visible_widgets_list),
                min(len(visible_widgets_list), 160),
            )
        except Exception as exc:
            logger.warning("Planner 获取 UIED Visible Widgets List 失败，降级为空列表: %s", exc)

        user_prompt = PLANNER_USER_PROMPT.format(
            task=task_text,
            uied_visible_widgets_list_json=visible_widgets_list_json,
        )

        request = LLMRequest(
            system=system_prompt,
            user=user_prompt,
            images=[screenshot_path],
            response_format=PlanResult,
        )

        try:
            parsed = self.llm.chat(request)
            result = parsed if isinstance(parsed, PlanResult) else PlanResult.model_validate(parsed)
            logger.info(
                "规划完成: is_task_complete=%s, action_type=%s, goal=%s, point=(%s,%s), control_id=%s",
                result.is_task_complete,
                result.action_type,
                result.goal,
                result.action_x,
                result.action_y,
                result.target_control_id,
            )
            return result
        except Exception as exc:
            logger.error("规划失败: %s", exc)
            raise
