"""Planner v2: semantic planning + target anchoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
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
    target_widget_bounds: list[int] = Field(
        default_factory=list,
        description="Selected widget bounds [x1,y1,x2,y2] when available.",
    )
    target_widget_center: list[int] = Field(
        default_factory=list,
        description="Selected widget center [x,y] when available.",
    )
    target_widget_class: str = Field(default="")
    target_widget_text: str = Field(default="")
    target_widget_resource_id: str = Field(default="")
    target_widget_content_desc: str = Field(default="")
    target_widget_hint: str = Field(default="")
    target_widget_source: str = Field(default="")

    @model_validator(mode="after")
    def validate_widget_id(self) -> "AnchorResult":
        if self.target_widget_id < -1:
            raise ValueError("target_widget_id must be >= -1.")
        if self.target_widget_bounds and len(self.target_widget_bounds) != 4:
            raise ValueError("target_widget_bounds must contain 4 integers when provided.")
        if self.target_widget_center and len(self.target_widget_center) != 2:
            raise ValueError("target_widget_center must contain 2 integers when provided.")
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


@dataclass(frozen=True)
class _HeuristicAnchorStats:
    result: AnchorResult
    top_score: float
    score_gap: float
    low_confidence: bool


class Planner:
    """Plan next step from task+screenshot, then anchor target widget when needed."""

    def __init__(self, llm_client, cv_output_dir: str | None = None, cv_resize_height: int = 800):
        self.llm = llm_client
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_cv_dir = os.path.join(project_root, "data", "cv_output")
        self.cv_output_dir = str(cv_output_dir or default_cv_dir)
        self.cv_resize_height = int(cv_resize_height or 800)
        self.anchor_confidence_min_score = 38.0
        self.anchor_confidence_min_gap = 8.0
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
        dump_tree: dict[str, Any] | None = None,
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

        widgets = self.prepare_widgets_for_anchor(
            screenshot=screenshot,
            visible_widgets_list=visible_widgets_list,
            dump_tree=dump_tree,
            action_type=action_type,
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
            if heuristic.low_confidence:
                logger.info(
                    "锚定低置信度(heuristic): widget_id=%s score=%.1f gap=%.1f，转LLM复核",
                    heuristic.result.target_widget_id,
                    heuristic.top_score,
                    heuristic.score_gap,
                )
                llm_result = self._anchor_with_llm(
                    plan=plan,
                    screenshot=screenshot,
                    widgets=widgets,
                    step=step,
                )
                if int(llm_result.target_widget_id) >= 0:
                    logger.info(
                        "锚定完成(llm override): widget_id=%s method=%s",
                        llm_result.target_widget_id,
                        llm_result.anchor_method,
                    )
                    return llm_result
                logger.info(
                    "LLM 复核未给出有效候选，回退 heuristic 结果: widget_id=%s",
                    heuristic.result.target_widget_id,
                )
                return heuristic.result
            logger.info(
                "锚定命中(heuristic): widget_id=%s method=%s score=%.1f gap=%.1f",
                heuristic.result.target_widget_id,
                heuristic.result.anchor_method,
                heuristic.top_score,
                heuristic.score_gap,
            )
            return heuristic.result

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

    def prepare_widgets_for_anchor(
        self,
        screenshot: str,
        visible_widgets_list: list[dict[str, Any]] | None = None,
        dump_tree: dict[str, Any] | None = None,
        action_type: str = "",
    ) -> list[dict[str, Any]]:
        widgets = self._load_widgets_for_anchor(
            screenshot=screenshot,
            visible_widgets_list=visible_widgets_list,
        )
        if not widgets and not isinstance(dump_tree, dict):
            return []

        normalized = self._normalize_widget_candidates(
            widgets=widgets,
            dump_tree=dump_tree,
            action_type=str(action_type or "").strip().lower(),
        )
        return normalized

    def _anchor_with_heuristic(
        self,
        target_description: str,
        action_type: str,
        widgets: list[dict[str, Any]],
    ) -> _HeuristicAnchorStats | None:
        target_norm = self._normalize_text(target_description)
        target_tokens = self._tokenize(target_norm)
        if not target_norm:
            return None

        position_target = self._infer_position_target(target_norm=target_norm)
        screen_w, screen_h = self._infer_screen_size_from_widgets(widgets=widgets)
        if not self._is_reliable_screen_size(screen_w=screen_w, screen_h=screen_h):
            position_target = None

        ranked: list[tuple[float, AnchorResult]] = []
        for widget in widgets:
            try:
                widget_id = int(widget.get("widget_id"))
            except Exception:
                continue

            (
                final_score,
                anchor_method,
                reason,
            ) = self._score_widget_candidate(
                widget=widget,
                action_type=action_type,
                target_norm=target_norm,
                target_tokens=target_tokens,
                screen_w=screen_w,
                screen_h=screen_h,
                position_target=position_target,
            )
            if final_score <= 0:
                continue

            valid, invalid_reason = self._validate_widget_for_target(
                widget=widget,
                action_type=action_type,
                target_norm=target_norm,
                screen_w=screen_w,
                screen_h=screen_h,
                position_target=position_target,
            )
            if not valid:
                continue

            reason_text = reason
            if invalid_reason:
                reason_text = f"{reason}, validation_note={invalid_reason}"
            ranked.append(
                (
                    float(final_score),
                    self._build_anchor_result(
                        widget_id=widget_id,
                        anchor_method=anchor_method,
                        anchor_reason=reason_text,
                        widget=widget,
                    ),
                )
            )

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0], reverse=True)
        top_score, top_result = ranked[0]
        second_score = ranked[1][0] if len(ranked) >= 2 else 0.0
        score_gap = float(top_score - second_score)
        low_confidence = (
            float(top_score) < float(self.anchor_confidence_min_score)
            or score_gap < float(self.anchor_confidence_min_gap)
        )
        top_result = top_result.model_copy(
            update={
                "anchor_reason": (
                    f"{top_result.anchor_reason}, "
                    f"top_score={top_score:.1f}, second_score={second_score:.1f}, "
                    f"gap={score_gap:.1f}, low_confidence={str(low_confidence).lower()}"
                )
            }
        )
        return _HeuristicAnchorStats(
            result=top_result,
            top_score=float(top_score),
            score_gap=float(score_gap),
            low_confidence=bool(low_confidence),
        )

    def _score_widget_candidate(
        self,
        widget: dict[str, Any],
        action_type: str,
        target_norm: str,
        target_tokens: list[str],
        screen_w: int,
        screen_h: int,
        position_target: tuple[float, float] | None,
    ) -> tuple[float, AnchorMethod, str]:
        widget_text_raw = str(widget.get("text") or "")
        widget_class_raw = str(widget.get("class") or "")
        widget_rid_raw = str(widget.get("resource_id") or "")
        widget_desc_raw = str(widget.get("content_desc") or "")
        widget_hint_raw = str(widget.get("hint") or "")
        widget_text = self._normalize_text(widget_text_raw)
        widget_class = self._normalize_text(widget_class_raw)
        widget_rid = self._normalize_text(widget_rid_raw)
        widget_desc = self._normalize_text(widget_desc_raw)
        widget_hint = self._normalize_text(widget_hint_raw)
        combined = " ".join(
            item for item in [widget_text, widget_class, widget_rid, widget_desc, widget_hint] if item
        ).strip()

        is_clickable = bool(widget.get("is_clickable"))
        is_focusable = bool(widget.get("is_focusable"))
        is_editable = bool(widget.get("is_editable"))
        source = str(widget.get("source") or "uied").strip().lower()

        expects_input_like = self._target_expects_input_like(
            target_norm=target_norm,
            action_type=action_type,
        )

        exact_text_hit = bool(widget_text and target_norm == widget_text)
        contains_text_hit = bool(widget_text and target_norm in widget_text)
        class_hit = bool(widget_class and target_norm in widget_class)
        desc_hit = bool(widget_desc and target_norm and (target_norm in widget_desc or widget_desc in target_norm))
        rid_hit = bool(widget_rid and target_norm and (target_norm in widget_rid or widget_rid in target_norm))
        hint_hit = bool(widget_hint and target_norm and (target_norm in widget_hint or widget_hint in target_norm))

        token_hits = 0
        for token in target_tokens:
            if token in combined:
                token_hits += 1

        text_score = 0.0
        method: AnchorMethod = "uied_keyword_overlap"
        if exact_text_hit:
            text_score += 120.0
            method = "uied_text_match"
        elif contains_text_hit:
            text_score += 72.0
            method = "uied_text_match"
        elif class_hit:
            text_score += 42.0
            method = "uied_class_match"
        if desc_hit:
            text_score += 52.0
        if rid_hit:
            text_score += 44.0
        if hint_hit:
            text_score += 50.0
        text_score += min(token_hits, 6) * 9.0

        action_score = 0.0
        if action_type in {"tap", "long_press", "swipe"}:
            if any(key in widget_class for key in ["compo", "button", "icon", "image", "combined"]):
                action_score += 14.0
            if "text" in widget_class:
                action_score -= 4.0
            if is_clickable or is_focusable:
                action_score += 18.0
            if is_editable and expects_input_like:
                action_score += 24.0
            if self._is_label_like(widget=widget) and expects_input_like:
                action_score -= 48.0
        elif action_type == "input":
            if any(key in widget_class for key in ["edit", "input", "field"]):
                action_score += 56.0
            elif "text" in widget_class:
                action_score += 4.0
            if any(key in combined for key in ["enter", "input", "search", "query", "field", "box", "type", "write"]):
                action_score += 12.0
            if is_editable:
                action_score += 52.0
            if is_focusable:
                action_score += 18.0
            if self._is_label_like(widget=widget):
                action_score -= 62.0

        geometry_score = 0.0
        width_ratio = 0.0
        height_ratio = 0.0
        area_ratio = 0.0
        bounds = self._extract_widget_bounds(widget=widget)
        if bounds is not None and screen_w > 0 and screen_h > 0:
            x1, y1, x2, y2 = bounds
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            width_ratio = float(width) / float(max(1, screen_w))
            height_ratio = float(height) / float(max(1, screen_h))
            area_ratio = float(width * height) / float(max(1, screen_w * screen_h))

            if area_ratio <= 0.20:
                geometry_score += 6.0
            else:
                geometry_score -= 10.0
            if width_ratio > 0.70 and height_ratio < 0.14:
                geometry_score -= 32.0
            if action_type in {"tap", "long_press"} and not widget_text and area_ratio < 0.08:
                geometry_score += 8.0
            if action_type == "input" and width_ratio > 0.22 and height_ratio > 0.04:
                geometry_score += 6.0
            if action_type == "input" and "text" in widget_class and not any(
                key in widget_class for key in ["edit", "input", "field"]
            ):
                geometry_score -= 8.0
            if area_ratio < 0.0003:
                geometry_score -= 42.0
            if min(width, height) <= 8:
                geometry_score -= 56.0

        text_len = len(widget_text_raw.strip())
        if text_len >= 40:
            geometry_score -= 22.0
        if text_len >= 90:
            geometry_score -= 30.0

        position_score = 0.0
        if position_target is not None and bounds is not None and screen_w > 0 and screen_h > 0:
            cx = float((bounds[0] + bounds[2]) / 2.0) / float(max(1, screen_w))
            cy = float((bounds[1] + bounds[3]) / 2.0) / float(max(1, screen_h))
            tx, ty = position_target
            distance = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            closeness = max(0.0, 1.0 - (distance / 1.2))
            position_score = closeness * 26.0
            if (not expects_input_like) and self._violates_position_hard_constraint(
                cx=cx,
                cy=cy,
                target_position=position_target,
            ):
                position_score -= 70.0

        source_score = 0.0
        if source == "dump":
            source_score += 8.0
            if not widget_text and (widget_desc or widget_rid or widget_hint):
                source_score += 12.0

        final_score = text_score + action_score + geometry_score + position_score + source_score
        reason = (
            f"text='{widget_text_raw[:64]}', class='{widget_class_raw[:48]}', "
            f"rid='{widget_rid_raw[:48]}', desc='{widget_desc_raw[:48]}', hint='{widget_hint_raw[:48]}', "
            f"token_hits={token_hits}, text_score={text_score:.1f}, action_score={action_score:.1f}, "
            f"geometry_score={geometry_score:.1f}, position_score={position_score:.1f}, "
            f"source_score={source_score:.1f}, editable={str(is_editable).lower()}, "
            f"clickable={str(is_clickable).lower()}, focusable={str(is_focusable).lower()}, "
            f"final_score={final_score:.1f}"
        )
        return float(final_score), method, reason

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
        target_norm = self._normalize_text(str(plan.target_description or ""))
        position_target = self._infer_position_target(target_norm=target_norm)
        screen_w, screen_h = self._infer_screen_size_from_widgets(widgets=widgets)
        if not self._is_reliable_screen_size(screen_w=screen_w, screen_h=screen_h):
            position_target = None
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
            widget = next(
                (
                    item
                    for item in widgets
                    if self._safe_int(item.get("widget_id")) is not None
                    and int(item.get("widget_id")) == target_widget_id
                ),
                None,
            )
            if widget is None:
                return AnchorResult(
                    target_widget_id=-1,
                    anchor_method="failed",
                    anchor_reason=f"llm selected widget_id={target_widget_id} but widget is missing",
                )

            valid, invalid_reason = self._validate_widget_for_target(
                widget=widget,
                action_type=str(plan.action_type or "").strip().lower(),
                target_norm=target_norm,
                screen_w=screen_w,
                screen_h=screen_h,
                position_target=position_target,
            )
            if not valid:
                return AnchorResult(
                    target_widget_id=-1,
                    anchor_method="failed",
                    anchor_reason=(
                        f"llm selected invalid candidate widget_id={target_widget_id}: "
                        f"{invalid_reason}; llm_reason={llm_output.anchor_reason}"
                    ),
                )

            return self._build_anchor_result(
                widget_id=target_widget_id,
                anchor_method="llm",
                anchor_reason=llm_output.anchor_reason,
                widget=widget,
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

    def _normalize_widget_candidates(
        self,
        widgets: list[dict[str, Any]],
        dump_tree: dict[str, Any] | None,
        action_type: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in list(widgets or []):
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            candidate["source"] = str(candidate.get("source") or "uied")
            candidate["resource_id"] = str(candidate.get("resource_id") or "")
            candidate["content_desc"] = str(candidate.get("content_desc") or "")
            candidate["hint"] = str(candidate.get("hint") or "")
            candidate["is_clickable"] = bool(candidate.get("is_clickable"))
            candidate["is_focusable"] = bool(candidate.get("is_focusable"))
            candidate["is_editable"] = bool(candidate.get("is_editable"))
            normalized.append(candidate)

        max_id = -1
        for item in normalized:
            wid = self._safe_int(item.get("widget_id"))
            if wid is not None:
                max_id = max(max_id, int(wid))

        dump_candidates = self._extract_dump_widget_candidates(
            dump_tree=dump_tree,
            start_widget_id=max_id + 1,
            action_type=action_type,
        )
        if dump_candidates:
            self._merge_dump_candidates_into_widgets(
                widgets=normalized,
                dump_candidates=dump_candidates,
            )

        screen_w, screen_h = self._infer_screen_size_from_widgets(widgets=normalized)
        if screen_w <= 0 or screen_h <= 0:
            screen_w = 1080
            screen_h = 1920

        filtered: list[dict[str, Any]] = []
        for item in normalized:
            valid, _ = self._validate_widget_basics(
                widget=item,
                action_type=action_type,
                screen_w=screen_w,
                screen_h=screen_h,
            )
            if valid:
                filtered.append(item)

        logger.info(
            "锚定候选归一化: raw=%d, merged=%d, filtered=%d",
            len(widgets or []),
            len(normalized),
            len(filtered),
        )
        return filtered

    def _extract_dump_widget_candidates(
        self,
        dump_tree: dict[str, Any] | None,
        start_widget_id: int,
        action_type: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(dump_tree, dict):
            return []

        current_id = max(0, int(start_widget_id))
        candidates: list[dict[str, Any]] = []

        def walk(node: Any) -> None:
            nonlocal current_id
            if not isinstance(node, dict):
                return

            for child in list(node.get("children") or []):
                walk(child)

            bounds = self._parse_dump_node_bounds(node=node)
            if bounds is None:
                return

            clickable = self._dump_attr_true(node=node, key="clickable")
            focusable = self._dump_attr_true(node=node, key="focusable")
            scrollable = self._dump_attr_true(node=node, key="scrollable")
            enabled = self._dump_attr_true(node=node, key="enabled", default=True)
            if not enabled:
                return

            class_name = str(node.get("class") or "")
            resource_id = str(node.get("resource-id") or "")
            text = str(node.get("text") or "")
            content_desc = str(node.get("content-desc") or "")
            hint = str(node.get("hint") or "")
            class_norm = self._normalize_text(class_name)
            editable = any(key in class_norm for key in ["edittext", "autocompletetextview", "textinput", "input"])

            has_signal = bool(
                clickable
                or focusable
                or editable
                or resource_id
                or content_desc
                or hint
            )
            if not has_signal:
                return
            if scrollable and action_type in {"tap", "long_press", "input"} and not editable:
                # Broad scroll containers are poor tap/input anchors in most cases.
                return

            x1, y1, x2, y2 = bounds
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]

            candidate = {
                "widget_id": int(current_id),
                "class": class_name,
                "text": text,
                "bounds": [int(x1), int(y1), int(x2), int(y2)],
                "center": center,
                "width": int(width),
                "height": int(height),
                "resource_id": resource_id,
                "content_desc": content_desc,
                "hint": hint,
                "is_clickable": bool(clickable),
                "is_focusable": bool(focusable),
                "is_editable": bool(editable),
                "source": "dump",
            }
            candidates.append(candidate)
            current_id += 1

        walk(dump_tree)
        return candidates

    def _merge_dump_candidates_into_widgets(
        self,
        widgets: list[dict[str, Any]],
        dump_candidates: list[dict[str, Any]],
    ) -> None:
        for dump_item in dump_candidates:
            dump_bounds = self._extract_widget_bounds(widget=dump_item)
            if dump_bounds is None:
                continue

            best_idx = -1
            best_iou = 0.0
            for idx, base_item in enumerate(widgets):
                base_bounds = self._extract_widget_bounds(widget=base_item)
                if base_bounds is None:
                    continue
                iou = self._bbox_iou(a=dump_bounds, b=base_bounds)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= 0.86:
                base = widgets[best_idx]
                if not str(base.get("resource_id") or "").strip():
                    base["resource_id"] = str(dump_item.get("resource_id") or "")
                if not str(base.get("content_desc") or "").strip():
                    base["content_desc"] = str(dump_item.get("content_desc") or "")
                if not str(base.get("hint") or "").strip():
                    base["hint"] = str(dump_item.get("hint") or "")
                if not str(base.get("text") or "").strip():
                    base["text"] = str(dump_item.get("text") or "")
                base["is_clickable"] = bool(base.get("is_clickable") or dump_item.get("is_clickable"))
                base["is_focusable"] = bool(base.get("is_focusable") or dump_item.get("is_focusable"))
                base["is_editable"] = bool(base.get("is_editable") or dump_item.get("is_editable"))
                continue

            widgets.append(dump_item)

    def _validate_widget_basics(
        self,
        widget: dict[str, Any],
        action_type: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[bool, str]:
        bounds = self._extract_widget_bounds(widget=widget)
        if bounds is None:
            return False, "invalid_bounds"
        x1, y1, x2, y2 = bounds
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = float(width * height)
        total = float(max(1, screen_w * screen_h))
        area_ratio = area / total

        has_text = bool(
            str(widget.get("text") or "").strip()
            or str(widget.get("content_desc") or "").strip()
            or str(widget.get("resource_id") or "").strip()
            or str(widget.get("hint") or "").strip()
        )
        interactive = bool(widget.get("is_clickable") or widget.get("is_focusable") or widget.get("is_editable"))

        if min(width, height) <= 6:
            return False, "too_thin"
        if area <= 360.0:
            return False, "too_small"
        if area_ratio > 0.40 and not has_text and not interactive:
            return False, "oversized_non_interactive"
        if action_type in {"tap", "long_press", "input"} and area_ratio > 0.92:
            return False, "fullscreen_candidate"

        return True, ""

    def _validate_widget_for_target(
        self,
        widget: dict[str, Any],
        action_type: str,
        target_norm: str,
        screen_w: int,
        screen_h: int,
        position_target: tuple[float, float] | None,
    ) -> tuple[bool, str]:
        valid, reason = self._validate_widget_basics(
            widget=widget,
            action_type=action_type,
            screen_w=screen_w,
            screen_h=screen_h,
        )
        if not valid:
            return False, reason

        bounds = self._extract_widget_bounds(widget=widget)
        if bounds is None:
            return False, "invalid_bounds"

        expects_input_like = self._target_expects_input_like(
            target_norm=target_norm,
            action_type=action_type,
        )
        if expects_input_like and self._is_label_like(widget=widget):
            return False, "label_like_for_input_target"
        if expects_input_like:
            class_norm = self._normalize_text(str(widget.get("class") or ""))
            signal_text = self._normalize_text(
                " ".join(
                    [
                        str(widget.get("text") or ""),
                        str(widget.get("resource_id") or ""),
                        str(widget.get("content_desc") or ""),
                        str(widget.get("hint") or ""),
                    ]
                )
            )
            has_input_class = any(key in class_norm for key in ["edit", "input", "field", "auto"])
            has_input_signal = any(
                key in signal_text
                for key in [
                    "field",
                    "recipient",
                    "subject",
                    "email",
                    "search",
                    "compose_to",
                    "address",
                    "输入",
                    "收件人",
                    "主题",
                    "邮箱",
                ]
            )
            if not (bool(widget.get("is_editable")) or has_input_class or has_input_signal):
                return False, "non_input_control_for_input_target"

        if (not expects_input_like) and position_target is not None and screen_w > 0 and screen_h > 0:
            cx = float((bounds[0] + bounds[2]) / 2.0) / float(max(1, screen_w))
            cy = float((bounds[1] + bounds[3]) / 2.0) / float(max(1, screen_h))
            if self._violates_position_hard_constraint(
                cx=cx,
                cy=cy,
                target_position=position_target,
            ):
                return False, "violates_position_constraint"

        return True, ""

    def _build_anchor_result(
        self,
        widget_id: int,
        anchor_method: AnchorMethod,
        anchor_reason: str,
        widget: dict[str, Any] | None,
    ) -> AnchorResult:
        bounds = self._extract_widget_bounds(widget=widget or {})
        center: list[int] = []
        if bounds is not None:
            center = [
                int((bounds[0] + bounds[2]) // 2),
                int((bounds[1] + bounds[3]) // 2),
            ]
        return AnchorResult(
            target_widget_id=int(widget_id),
            anchor_method=anchor_method,
            anchor_reason=str(anchor_reason or "").strip() or "anchor_selected",
            target_widget_bounds=list(bounds) if bounds is not None else [],
            target_widget_center=center,
            target_widget_class=str((widget or {}).get("class") or ""),
            target_widget_text=str((widget or {}).get("text") or ""),
            target_widget_resource_id=str((widget or {}).get("resource_id") or ""),
            target_widget_content_desc=str((widget or {}).get("content_desc") or ""),
            target_widget_hint=str((widget or {}).get("hint") or ""),
            target_widget_source=str((widget or {}).get("source") or ""),
        )

    def _target_expects_input_like(
        self,
        target_norm: str,
        action_type: str,
    ) -> bool:
        if action_type == "input":
            return True
        text = str(target_norm or "")
        keywords = [
            "field",
            "input",
            "recipient",
            "subject",
            "search",
            "to ",
            "to field",
            "email address",
            "textbox",
            "编辑",
            "输入",
            "收件人",
            "主题",
            "搜索",
            "to字段",
        ]
        return any(key in text for key in keywords)

    def _is_label_like(
        self,
        widget: dict[str, Any],
    ) -> bool:
        class_norm = self._normalize_text(str(widget.get("class") or ""))
        is_text_class = "text" in class_norm and not any(
            key in class_norm for key in ["edit", "input", "field", "auto"]
        )
        if not is_text_class:
            return False
        if bool(widget.get("is_clickable")) or bool(widget.get("is_focusable")) or bool(widget.get("is_editable")):
            return False
        return True

    def _violates_position_hard_constraint(
        self,
        cx: float,
        cy: float,
        target_position: tuple[float, float],
    ) -> bool:
        tx, ty = target_position
        if tx >= 0.80 and cx <= 0.55:
            return True
        if tx <= 0.20 and cx >= 0.45:
            return True
        if ty >= 0.80 and cy <= 0.55:
            return True
        if ty <= 0.20 and cy >= 0.45:
            return True
        return False

    def _parse_dump_node_bounds(
        self,
        node: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        raw = str(node.get("bounds") or "").strip()
        match = re.match(r"^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$", raw)
        if not match:
            return None
        try:
            x1 = int(match.group(1))
            y1 = int(match.group(2))
            x2 = int(match.group(3))
            y2 = int(match.group(4))
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _dump_attr_true(
        self,
        node: dict[str, Any],
        key: str,
        default: bool = False,
    ) -> bool:
        if not isinstance(node, dict):
            return bool(default)
        value = node.get(key)
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() == "true"

    def _bbox_iou(
        self,
        a: tuple[int, int, int, int] | None,
        b: tuple[int, int, int, int] | None,
    ) -> float:
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float((ax2 - ax1) * (ay2 - ay1))
        area_b = float((bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _extract_widget_bounds(
        self,
        widget: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        bounds = widget.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return None
        try:
            x1 = int(bounds[0])
            y1 = int(bounds[1])
            x2 = int(bounds[2])
            y2 = int(bounds[3])
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _infer_screen_size_from_widgets(
        self,
        widgets: list[dict[str, Any]],
    ) -> tuple[int, int]:
        max_x = 0
        max_y = 0
        for widget in widgets:
            bounds = self._extract_widget_bounds(widget=widget)
            if bounds is None:
                continue
            _, _, x2, y2 = bounds
            max_x = max(max_x, int(x2))
            max_y = max(max_y, int(y2))
        return max(max_x, 1), max(max_y, 1)

    def _is_reliable_screen_size(
        self,
        screen_w: int,
        screen_h: int,
    ) -> bool:
        return int(screen_w) >= 600 and int(screen_h) >= 1000

    def _infer_position_target(
        self,
        target_norm: str,
    ) -> tuple[float, float] | None:
        text = str(target_norm or "").strip().lower()
        if not text:
            return None

        right_hit = any(key in text for key in ["right", "右"])
        left_hit = any(key in text for key in ["left", "左"])
        center_x_hit = any(key in text for key in ["center", "middle", "中央", "中间"])
        top_hit = any(key in text for key in ["top", "upper", "上", "顶部"])
        bottom_hit = any(key in text for key in ["bottom", "lower", "下", "底部"])
        center_y_hit = any(key in text for key in ["center", "middle", "中央", "中间"])

        x_target: float | None = None
        y_target: float | None = None

        if right_hit and not left_hit:
            x_target = 0.88
        elif left_hit and not right_hit:
            x_target = 0.12
        elif center_x_hit:
            x_target = 0.50

        if top_hit and not bottom_hit:
            y_target = 0.12
        elif bottom_hit and not top_hit:
            y_target = 0.88
        elif center_y_hit:
            y_target = 0.50

        if x_target is None and y_target is None:
            return None
        if x_target is None:
            x_target = 0.50
        if y_target is None:
            y_target = 0.50
        return float(x_target), float(y_target)

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
