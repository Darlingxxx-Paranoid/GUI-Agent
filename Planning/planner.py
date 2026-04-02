"""Pure-LLM planner: Task + Screenshot -> PlanResult."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
            logger.info("PlanResult校验通过: 完成态(action_type=wait, input_text为空)")
            return self

        if self.action_type == "wait":
            logger.error(
                "PlanResult校验失败: is_task_complete=false 但 action_type=wait"
            )
            raise ValueError("When is_task_complete=false, action_type cannot be 'wait'.")
        logger.info("PlanResult校验通过: 非完成态(action_type=%s)", self.action_type)
        return self


class Planner:
    """Plan next step from only task and screenshot using structured LLM output."""

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("Planner 初始化完成 (pure LLM, task+screenshot)")

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

        system_prompt = (
            "You are the Plan module of an Android GUI agent. "
            "Use only the task text and the screenshot image to produce one next-step plan."
        )
        user_prompt = (
            "Task:\n"
            f"{task_text}\n\n"
            "Rules:\n"
            "1) Output must follow the provided response schema exactly.\n"
            "2) If the task is already complete in the screenshot, set is_task_complete=true, action_type=wait, and input_text=''.\n"
            "3) If the task is not complete, set is_task_complete=false and choose action_type from tap/input/swipe/back/enter/long_press/launch_app.\n"
            "4) goal must describe one concrete next step only.\n"
            "5) reasoning must be concise and grounded in the screenshot.\n"
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
                "规划完成: is_task_complete=%s, action_type=%s, goal=%s",
                result.is_task_complete,
                result.action_type,
                result.goal,
            )
            return result
        except Exception as exc:
            logger.error("规划失败: %s", exc)
            raise
