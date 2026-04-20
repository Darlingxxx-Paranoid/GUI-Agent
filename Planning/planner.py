"""Planner v2: semantic planning + target anchoring."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Perception.uied_controls import get_uied_visible_widgets_list
from prompt.planner_prompt import (
    PLANNER_ANCHOR_SYSTEM_PROMPT,
    PLANNER_ANCHOR_USER_PROMPT,
    PLANNER_DEVICE_CONTEXT_TEMPLATE,
    PLANNER_EXPERIENCE_CONTEXT_TEMPLATE,
    PLANNER_PROGRESS_CONTEXT_TEMPLATE,
    PLANNER_RUNTIME_EXCEPTION_CONTEXT_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
)
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

AnchorMethod = Literal[
    "none",
    "uied_text_match",
    "uied_class_match",
    "uied_keyword_overlap",
    "llm",
    "failed",
]

WIDGET_ACTION_TYPES: set[str] = {"tap", "input", "long_press", "swipe"}


class PlanResult(BaseModel):
    """Structured semantic output contract for Planner stage 1."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1, description="One-step semantic goal.")
    action_type: PlanActionType = Field(description="Next action type.")
    target_description: str = Field(
        default="",
        description="Human-readable widget target description for widget actions.",
    )
    input_description: str = Field(
        default="",
        description="Exact text to type when action_type is input.",
    )
    launch_package: str = Field(
        default="",
        description="Android package name when action_type is launch_app.",
    )
    launch_activity: str = Field(
        default="",
        description="Optional launch activity when action_type is launch_app.",
    )
    is_task_complete: bool = Field(description="Whether the full task is already complete.")
    reasoning: str = Field(min_length=1, description="Planning reason based on current screenshot.")

    @model_validator(mode="after")
    def validate_completion_rules(self) -> "PlanResult":
        action_type = str(self.action_type or "").strip().lower()

        if self.is_task_complete:
            if action_type != "wait":
                logger.error(
                    "PlanResult校验失败: is_task_complete=true 但 action_type=%s (期望 wait)",
                    self.action_type,
                )
                raise ValueError("When is_task_complete=true, action_type must be 'wait'.")
            if self.target_description != "":
                logger.error("PlanResult校验失败: is_task_complete=true 但 target_description 非空")
                raise ValueError("When is_task_complete=true, target_description must be empty.")
            if self.input_description != "":
                logger.error("PlanResult校验失败: is_task_complete=true 但 input_description 非空")
                raise ValueError("When is_task_complete=true, input_description must be empty.")
            if self.launch_package != "" or self.launch_activity != "":
                logger.error("PlanResult校验失败: is_task_complete=true 但 launch_package/activity 非空")
                raise ValueError(
                    "When is_task_complete=true, launch_package and launch_activity must be empty."
                )
            logger.info("PlanResult校验通过: 完成态(action_type=wait)")
            return self

        if action_type in WIDGET_ACTION_TYPES:
            if self.target_description == "":
                logger.error(
                    "PlanResult校验失败: action_type=%s 但 target_description 为空",
                    self.action_type,
                )
                raise ValueError(
                    "When action_type is tap/input/swipe/long_press, target_description must be non-empty."
                )
        elif self.target_description != "":
            logger.error(
                "PlanResult校验失败: action_type=%s 但 target_description 非空",
                self.action_type,
            )
            raise ValueError(
                "When action_type is not tap/input/swipe/long_press, target_description must be empty."
            )

        if action_type == "input":
            if self.input_description == "":
                logger.error("PlanResult校验失败: action_type=input 但 input_description 为空")
                raise ValueError("When action_type is input, input_description must be non-empty.")
        elif self.input_description != "":
            logger.error(
                "PlanResult校验失败: action_type=%s 但 input_description 非空",
                self.action_type,
            )
            raise ValueError("When action_type is not input, input_description must be empty.")

        if action_type == "launch_app":
            if self.launch_package == "":
                logger.warning(
                    "PlanResult提示: action_type=launch_app 但 launch_package 为空，将依赖任务解析兜底"
                )
        elif self.launch_package != "" or self.launch_activity != "":
            logger.error(
                "PlanResult校验失败: action_type=%s 但 launch_package/activity 非空",
                self.action_type,
            )
            raise ValueError(
                "When action_type is not launch_app, launch_package and launch_activity must be empty."
            )

        logger.info("PlanResult校验通过: 非完成态(action_type=%s)", self.action_type)
        return self


class AnchorResult(BaseModel):
    """Structured output of Planner stage 2 anchoring."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_widget_id: int = Field(
        default=-1,
        description="Chosen UIED widget id. Use -1 when no widget is selected.",
    )
    anchor_method: AnchorMethod = Field(description="Anchoring method used.")
    anchor_reason: str = Field(min_length=1, description="Why this widget was selected or failed.")

    @model_validator(mode="after")
    def validate_widget_id(self) -> "AnchorResult":
        if self.target_widget_id < -1:
            raise ValueError("target_widget_id must be >= -1.")
        return self


class _AnchorLLMOutput(BaseModel):
    """Internal structured output for LLM fallback in anchor stage."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_widget_id: int = Field(default=-1)
    anchor_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_widget_id(self) -> "_AnchorLLMOutput":
        if self.target_widget_id < -1:
            raise ValueError("target_widget_id must be >= -1.")
        return self


class Planner:
    """Plan next step from task+screenshot, then anchor target widget when needed."""

    def __init__(self, llm_client, cv_output_dir: str | None = None, cv_resize_height: int = 800):
        self.llm = llm_client
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_cv_dir = os.path.join(project_root, "data", "cv_output")
        self.cv_output_dir = str(cv_output_dir or default_cv_dir)
        self.cv_resize_height = int(cv_resize_height or 800)
        logger.info(
            "Planner 初始化完成 (v2: semantic plan + anchor), cv_output_dir=%s",
            self.cv_output_dir,
        )

    def plan(
        self,
        task: str,
        screenshot: str,
        runtime_exception_hint: str = "",
        progress_context: list[dict[str, Any]] | None = None,
        experience_context: list[dict[str, Any]] | None = None,
        current_package: str = "",
        task_target_app: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> PlanResult:
        task_text = str(task or "").strip()
        if not task_text:
            raise ValueError("task 不能为空")

        screenshot_path = str(screenshot or "").strip()
        if not screenshot_path:
            raise ValueError("screenshot 不能为空")

        logger.info(
            "开始规划(v2): task_len=%d, screenshot=%s",
            len(task_text),
            screenshot_path,
        )

        runtime_hint = str(runtime_exception_hint or "").strip()
        runtime_context = ""
        if runtime_hint:
            runtime_context = PLANNER_RUNTIME_EXCEPTION_CONTEXT_TEMPLATE.format(
                runtime_exception_hint=runtime_hint
            )
            logger.info("Planner 注入重规划提示: %s", runtime_hint)

        progress_items = self._normalize_progress_context(progress_context)
        progress_items_json = json.dumps(progress_items[:80], ensure_ascii=False)
        logger.info(
            "Planner 注入 progress_context: total=%d, used=%d",
            len(progress_items),
            min(len(progress_items), 80),
        )
        progress_context_block = PLANNER_PROGRESS_CONTEXT_TEMPLATE.format(
            progress_context_json=progress_items_json
        )

        experience_items = self._normalize_experience_context(experience_context)
        experience_items_json = json.dumps(experience_items[:3], ensure_ascii=False)
        logger.info(
            "Planner 注入 experience_context: total=%d, used=%d",
            len(experience_items),
            min(len(experience_items), 3),
        )
        experience_context_block = PLANNER_EXPERIENCE_CONTEXT_TEMPLATE.format(
            experience_context_json=experience_items_json
        )

        current_pkg = str(current_package or "").strip() or "(unknown)"
        target_app_payload = self._normalize_task_target_app(task_target_app)
        device_context_block = PLANNER_DEVICE_CONTEXT_TEMPLATE.format(
            current_package=current_pkg,
            task_target_app_json=json.dumps(target_app_payload, ensure_ascii=False),
        )
        logger.info(
            "Planner 注入 device/app context: current_package=%s, target_package=%s",
            current_pkg,
            target_app_payload.get("package", ""),
        )

        user_prompt = PLANNER_USER_PROMPT.format(
            task=task_text,
            runtime_exception_context=runtime_context,
            progress_context_block=progress_context_block,
            experience_context_block=experience_context_block,
            device_context_block=device_context_block,
        )

        request = LLMRequest(
            system=PLANNER_SYSTEM_PROMPT,
            user=user_prompt,
            images=[screenshot_path],
            response_format=PlanResult,
            audit_meta={
                "artifact_kind": "PlanResult",
                "step": step,
                "stage": "planner_plan",
            },
        )

        try:
            parsed = self.llm.chat(request)
            result = parsed if isinstance(parsed, PlanResult) else PlanResult.model_validate(parsed)
            logger.info(
                "规划完成(v2): is_task_complete=%s, action_type=%s, goal=%s, target_description=%s",
                result.is_task_complete,
                result.action_type,
                result.goal,
                result.target_description,
            )
            return result
        except Exception as exc:
            logger.error("规划失败(v2): %s", exc)
            raise

    def anchor(
        self,
        plan: PlanResult,
        screenshot: str,
        visible_widgets_list: list[dict[str, Any]] | None = None,
        step: int | None = None,
    ) -> AnchorResult:
        action_type = str(plan.action_type or "").strip().lower()
        if action_type not in WIDGET_ACTION_TYPES:
            return AnchorResult(
                target_widget_id=-1,
                anchor_method="none",
                anchor_reason=f"action_type={action_type} does not require widget anchor",
            )

        target_description = str(plan.target_description or "").strip()
        if not target_description:
            logger.warning("锚定失败: widget action 缺少 target_description")
            return AnchorResult(
                target_widget_id=-1,
                anchor_method="failed",
                anchor_reason="target_description is empty for widget action",
            )

        widgets = self._load_widgets_for_anchor(
            screenshot=screenshot,
            visible_widgets_list=visible_widgets_list,
        )
        if not widgets:
            return AnchorResult(
                target_widget_id=-1,
                anchor_method="failed",
                anchor_reason="visible widgets list is empty",
            )

        heuristic = self._anchor_with_heuristic(
            target_description=target_description,
            action_type=action_type,
            widgets=widgets,
        )
        if heuristic is not None:
            logger.info(
                "锚定命中(heuristic): widget_id=%s method=%s",
                heuristic.target_widget_id,
                heuristic.anchor_method,
            )
            return heuristic

        llm_result = self._anchor_with_llm(
            plan=plan,
            screenshot=screenshot,
            widgets=widgets,
            step=step,
        )
        logger.info(
            "锚定完成(llm fallback): widget_id=%s method=%s",
            llm_result.target_widget_id,
            llm_result.anchor_method,
        )
        return llm_result

    def _anchor_with_heuristic(
        self,
        target_description: str,
        action_type: str,
        widgets: list[dict[str, Any]],
    ) -> AnchorResult | None:
        target_norm = self._normalize_text(target_description)
        target_tokens = self._tokenize(target_norm)
        if not target_norm or not target_tokens:
            return None

        best: tuple[int, AnchorMethod, int, str] | None = None
        for widget in widgets:
            try:
                widget_id = int(widget.get("widget_id"))
            except Exception:
                continue

            widget_text = self._normalize_text(str(widget.get("text") or ""))
            widget_class = self._normalize_text(str(widget.get("class") or ""))
            combined = f"{widget_text} {widget_class}".strip()

            exact_text_hit = bool(widget_text and target_norm == widget_text)
            contains_text_hit = bool(widget_text and target_norm in widget_text)
            class_hit = bool(widget_class and target_norm in widget_class)

            token_hits = 0
            for token in target_tokens:
                if token in combined:
                    token_hits += 1

            score = 0
            method: AnchorMethod = "uied_keyword_overlap"
            if exact_text_hit:
                score += 220
                method = "uied_text_match"
            elif contains_text_hit:
                score += 160
                method = "uied_text_match"
            elif class_hit:
                score += 90
                method = "uied_class_match"

            score += token_hits * 18
            if action_type == "input" and any(key in widget_class for key in ["edit", "input", "text"]):
                score += 24

            if score <= 0:
                continue

            reason = (
                f"text='{str(widget.get('text') or '')[:64]}', "
                f"class='{str(widget.get('class') or '')[:64]}', token_hits={token_hits}, score={score}"
            )
            if best is None or score > best[2]:
                best = (widget_id, method, score, reason)

        if best is None:
            return None
        return AnchorResult(
            target_widget_id=int(best[0]),
            anchor_method=best[1],
            anchor_reason=best[3],
        )

    def _anchor_with_llm(
        self,
        plan: PlanResult,
        screenshot: str,
        widgets: list[dict[str, Any]],
        step: int | None,
    ) -> AnchorResult:
        widgets_json = json.dumps(widgets[:200], ensure_ascii=False)
        user_prompt = PLANNER_ANCHOR_USER_PROMPT.format(
            action_type=str(plan.action_type or ""),
            goal=str(plan.goal or ""),
            target_description=str(plan.target_description or ""),
            uied_visible_widgets_list_json=widgets_json,
        )

        request = LLMRequest(
            system=PLANNER_ANCHOR_SYSTEM_PROMPT,
            user=user_prompt,
            images=[str(screenshot or "").strip()],
            response_format=_AnchorLLMOutput,
            audit_meta={
                "artifact_kind": "AnchorResult",
                "step": step,
                "stage": "planner_anchor",
            },
        )

        widget_ids = {int(item.get("widget_id")) for item in widgets if self._safe_int(item.get("widget_id")) is not None}
        try:
            parsed = self.llm.chat(request)
            llm_output = (
                parsed if isinstance(parsed, _AnchorLLMOutput) else _AnchorLLMOutput.model_validate(parsed)
            )
            target_widget_id = int(llm_output.target_widget_id)
            if target_widget_id < 0:
                return AnchorResult(
                    target_widget_id=-1,
                    anchor_method="failed",
                    anchor_reason=f"llm rejected target: {llm_output.anchor_reason}",
                )
            if target_widget_id not in widget_ids:
                return AnchorResult(
                    target_widget_id=-1,
                    anchor_method="failed",
                    anchor_reason=(
                        f"llm returned out-of-list widget_id={target_widget_id}; reason={llm_output.anchor_reason}"
                    ),
                )
            return AnchorResult(
                target_widget_id=target_widget_id,
                anchor_method="llm",
                anchor_reason=llm_output.anchor_reason,
            )
        except Exception as exc:
            logger.warning("LLM 锚定失败: %s", exc)
            return AnchorResult(
                target_widget_id=-1,
                anchor_method="failed",
                anchor_reason=f"llm_anchor_error: {exc}",
            )

    def _load_widgets_for_anchor(
        self,
        screenshot: str,
        visible_widgets_list: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(visible_widgets_list, list):
            widgets = [item for item in visible_widgets_list if isinstance(item, dict)]
            logger.info("锚定阶段使用上游传入 widgets: count=%d", len(widgets))
            return widgets

        screenshot_path = str(screenshot or "").strip()
        if not screenshot_path:
            return []
        try:
            widgets = get_uied_visible_widgets_list(
                screenshot_path=screenshot_path,
                cv_output_dir=self.cv_output_dir,
                resize_height=self.cv_resize_height,
            )
            logger.info("锚定阶段提取 UIED Visible Widgets List: count=%d", len(widgets))
            return list(widgets or [])
        except Exception as exc:
            logger.warning("锚定阶段提取 UIED Visible Widgets List 失败: %s", exc)
            return []

    def _normalize_progress_context(
        self,
        progress_context: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for value in list(progress_context or []):
            if not isinstance(value, dict):
                continue
            goal = str(value.get("goal") or "").strip()
            action_type = str(value.get("action_type") or "").strip()
            target_description = str(value.get("target_description") or "").strip()
            input_description = str(
                value.get("input_description")
                or value.get("input_text")
                or ""
            )
            if not goal:
                continue
            items.append(
                {
                    "goal": goal,
                    "action_type": action_type,
                    "target_description": target_description,
                    "input_description": input_description,
                }
            )
        return items

    def _normalize_experience_context(
        self,
        experience_context: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for value in list(experience_context or []):
            if not isinstance(value, dict):
                continue
            goal = str(value.get("goal") or "").strip()
            action_type = str(value.get("action_type") or "").strip()
            if not (goal or action_type):
                continue
            items.append(
                {
                    "goal": goal,
                    "action_type": action_type,
                    "target_description": str(value.get("target_description") or "").strip(),
                    "input_description": str(
                        value.get("input_description")
                        or value.get("input_text")
                        or ""
                    ),
                    "failure_reason": str(value.get("failure_reason") or "").strip(),
                }
            )
        return items

    def _normalize_task_target_app(
        self,
        task_target_app: dict[str, Any] | None,
    ) -> dict[str, str]:
        if not isinstance(task_target_app, dict):
            return {}
        name = str(task_target_app.get("name") or "").strip()
        package = str(task_target_app.get("package") or "").strip()
        activity = str(task_target_app.get("activity") or "").strip()
        if not (name or package or activity):
            return {}
        return {
            "name": name,
            "package": package,
            "activity": activity,
        }

    def _normalize_text(self, text: str) -> str:
        token = str(text or "").strip().lower()
        token = re.sub(r"[\s_]+", " ", token)
        token = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", token)
        token = re.sub(r"\s+", " ", token).strip()
        return token

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in str(text or "").split():
            if len(token) >= 2:
                tokens.append(token)
        return tokens

    def _safe_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None
