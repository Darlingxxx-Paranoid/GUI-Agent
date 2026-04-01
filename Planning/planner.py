"""Single-step planner that outputs V3.1 planning intent."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from Memory.memory_manager import MemoryManager
from Oracle.contracts import (
    GoalSpec,
    Selector,
    TargetRef,
    TargetResolved,
    ContractValidationError,
    dataclass_to_json_schema,
    parse_dataclass,
    parse_json_object,
    validate_action_type,
)
from Perception.context_builder import UIState, WidgetInfo
from prompt.planner_prompt import PLANNER_PROMPT
from utils.audit_recorder import AuditRecorder

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    goal: GoalSpec
    target: TargetRef | None = None
    requested_action_type: str = "tap"
    input_text: str = ""
    planning_hints: Dict[str, Any] = field(default_factory=dict)
    is_task_complete: bool = False
    reasoning: str = ""
    from_experience: bool = False


class Planner:
    """Generate next-step intent from memory or LLM."""

    def __init__(self, llm_client, memory_manager: MemoryManager):
        self.llm = llm_client
        self.memory = memory_manager
        self.audit = AuditRecorder(component="planner")
        logger.info("Planner 初始化完成 (V3.1)")

    def plan(self, task: str, ui_state: UIState, step: int | None = None) -> PlanResult:
        logger.info("开始规划(V3.1), task='%s'", task[:80])

        source = "llm"
        result: PlanResult
        experience = self.memory.search_experience(task)
        if experience:
            replay = self._plan_from_experience(experience)
            if replay is not None:
                source = "experience"
                result = replay
                self._record_plan_dataclass(source=source, result=result, step=step)
                return result

        try:
            result = self._plan_with_llm(task=task, ui_state=ui_state, step=step)
        except Exception as exc:
            logger.warning("规划失败，回退启发式策略: %s", exc)
            source = "heuristic_fallback"
            result = self._heuristic_fallback_plan(task=task, ui_state=ui_state, error=str(exc))

        self._record_plan_dataclass(source=source, result=result, step=step)
        return result

    def _plan_from_experience(self, experience) -> Optional[PlanResult]:
        triplets = list(getattr(experience, "step_triplets", None) or [])
        if not triplets:
            logger.info("命中经验但 step_triplets 为空，回退 LLM")
            return None

        progress = self._experience_progress_index()
        if progress >= len(triplets):
            logger.info("经验回放结束(progress=%d,total=%d)，回退 LLM", progress, len(triplets))
            return None

        step = triplets[progress] or {}
        contract_dict = step.get("contract") or {}
        action_dict = step.get("action") or {}

        goal = self._safe_parse_goal(contract_dict.get("goal"))
        if goal is None:
            goal = GoalSpec(
                summary=str(step.get("subgoal") or action_dict.get("description") or "按经验执行下一步").strip()[:160],
                success_definition="完成经验回放步骤",
                tags=["experience"],
            )

        target = None
        target_payload = contract_dict.get("target") or action_dict.get("target")
        if isinstance(target_payload, dict):
            try:
                target = parse_dataclass(target_payload, TargetRef, strict=False)
            except Exception:
                target = None

        action_type = str(action_dict.get("type") or action_dict.get("action_type") or "tap").strip().lower()
        try:
            action_type = validate_action_type(action_type)
        except Exception:
            action_type = "tap"

        input_text = ""
        if isinstance(action_dict.get("params"), dict):
            input_text = str(action_dict["params"].get("text") or "")
        if not input_text:
            input_text = str(action_dict.get("text") or action_dict.get("input_text") or "")

        plan = PlanResult(
            goal=goal,
            target=target,
            requested_action_type=action_type,
            input_text=input_text,
            planning_hints={
                "source": "experience",
                "replay_index": progress,
                "replay_total": len(triplets),
            },
            reasoning=f"复用历史经验步骤 {progress + 1}/{len(triplets)}",
            from_experience=True,
        )

        logger.info("经验回放规划: %s", plan.goal.summary)
        return plan

    def _experience_progress_index(self) -> int:
        progress = 0
        for step in self.memory.short_term.history:
            result = str(step.get("result") or "").lower()
            if step.get("from_experience") and result.startswith("success"):
                progress += 1
        return progress

    def _plan_with_llm(self, task: str, ui_state: UIState, step: int | None = None) -> PlanResult:
        history_text = self.memory.short_term.get_context_summary() or "暂无历史记录"
        ui_text = ui_state.to_prompt_text()

        planner_schema = {
            "type": "object",
            "required": [
                "is_task_complete",
                "reasoning",
                "goal",
                "requested_action_type",
                "input_text",
                "target",
                "planning_hints",
            ],
            "properties": {
                "is_task_complete": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "goal": dataclass_to_json_schema(GoalSpec),
                "requested_action_type": {
                    "enum": ["tap", "input", "swipe", "back", "enter", "long_press", "launch_app"],
                },
                "input_text": {"type": "string"},
                "target": {
                    "anyOf": [
                        dataclass_to_json_schema(TargetRef),
                        {"type": "null"},
                    ]
                },
                "planning_hints": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        }

        prompt = PLANNER_PROMPT.format(
            task=task,
            ui_state=ui_text,
            history=history_text,
            schema_json=json.dumps(planner_schema, ensure_ascii=False),
        )

        response = self.llm.chat(
            prompt,
            audit_meta={
                "artifact_kind": "PlanResult",
                "step": step,
                "stage": "primary",
            },
        )
        result = self._parse_plan_response(response)

        if self._is_redundant(result):
            logger.info("检测到重复规划，触发一次重规划")
            retry_prompt = prompt + (
                "\n\nRecent output looked redundant. Choose a different concrete action/target from history."
            )
            retry = self.llm.chat(
                retry_prompt,
                audit_meta={
                    "artifact_kind": "PlanResult",
                    "step": step,
                    "stage": "retry",
                },
            )
            retry_result = self._parse_plan_response(retry)
            if not self._is_redundant(retry_result):
                result = retry_result

        misaligned, reason = self._is_misaligned(task=task, result=result)
        if misaligned:
            logger.info("检测到主题偏移，触发一次纠偏重规划: %s", reason)
            align_prompt = (
                prompt
                + "\n\nYour last output may be off-topic. Keep the next intent strictly aligned with the final task."
            )
            align_resp = self.llm.chat(
                align_prompt,
                audit_meta={
                    "artifact_kind": "PlanResult",
                    "step": step,
                    "stage": "align",
                },
            )
            align_result = self._parse_plan_response(align_resp)
            still_misaligned, _ = self._is_misaligned(task=task, result=align_result)
            if not still_misaligned:
                result = align_result

        return result

    def _parse_plan_response(self, response: str) -> PlanResult:
        payload = parse_json_object(response)

        allowed_top_level = {
            "is_task_complete",
            "reasoning",
            "goal",
            "requested_action_type",
            "input_text",
            "target",
            "planning_hints",
        }
        extras = sorted(set(payload.keys()) - allowed_top_level)
        if extras:
            raise ContractValidationError(f"Planner response has unexpected fields: {extras}")

        goal = parse_dataclass(payload.get("goal") or {}, GoalSpec, strict=True)

        target_payload = payload.get("target")
        target: Optional[TargetRef] = None
        if target_payload is not None:
            target = parse_dataclass(target_payload, TargetRef, strict=True)

        action_type = validate_action_type(
            str(payload.get("requested_action_type") or "tap").strip().lower()
        )

        input_text = payload.get("input_text")
        if not isinstance(input_text, str):
            raise ContractValidationError("input_text must be a string")

        planning_hints = payload.get("planning_hints")
        if not isinstance(planning_hints, dict):
            raise ContractValidationError("planning_hints must be an object")

        is_task_complete = payload.get("is_task_complete")
        if not isinstance(is_task_complete, bool):
            raise ContractValidationError("is_task_complete must be a boolean")

        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, str):
            raise ContractValidationError("reasoning must be a string")

        result = PlanResult(
            goal=goal,
            target=target,
            requested_action_type=action_type,
            input_text=input_text,
            planning_hints=planning_hints,
            is_task_complete=is_task_complete,
            reasoning=reasoning.strip(),
            from_experience=False,
        )
        return result

    def _safe_parse_goal(self, payload: Any) -> Optional[GoalSpec]:
        if not isinstance(payload, dict):
            return None
        try:
            return parse_dataclass(payload, GoalSpec, strict=False)
        except Exception:
            return None

    def _heuristic_fallback_plan(self, task: str, ui_state: UIState, error: str = "") -> PlanResult:
        lowered = str(task or "").lower()
        target = self._find_widget_target(task=task, ui_state=ui_state)

        if "返回" in task or "back" in lowered:
            action_type = "back"
        elif target and target.resolved and target.resolved.widget_id is not None:
            action_type = "tap"
        elif any(token in lowered for token in ("scroll", "swipe", "下滑", "上滑")):
            action_type = "swipe"
        else:
            action_type = "tap"

        reason = "heuristic_fallback"
        if error:
            reason = f"{reason}:{error[:80]}"

        return PlanResult(
            goal=GoalSpec(
                summary=str(task or "继续下一步").strip()[:160] or "继续下一步",
                success_definition="观察到任务相关可验证变化",
                tags=["fallback"],
            ),
            target=target,
            requested_action_type=action_type,
            input_text="",
            planning_hints={"fallback_reason": reason},
            is_task_complete=False,
            reasoning="LLM 规划失败，使用启发式下一步",
            from_experience=False,
        )

    def _find_widget_target(self, task: str, ui_state: UIState) -> Optional[TargetRef]:
        tokens = [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", str(task or "")) if len(t) >= 2]
        lowered = [t.lower() for t in tokens]

        best_widget: Optional[WidgetInfo] = None
        best_score = -1
        for widget in ui_state.get_prompt_widgets():
            text_blob = " ".join(
                [
                    str(widget.text or ""),
                    str(widget.content_desc or ""),
                    str(widget.resource_id or ""),
                    str(widget.class_name or ""),
                ]
            ).lower()
            score = 0
            for token in lowered:
                if token and token in text_blob:
                    score += 2
            if widget.clickable:
                score += 1
            if widget.enabled:
                score += 1
            if score > best_score:
                best_score = score
                best_widget = widget

        if best_widget is None or best_score <= 0:
            return None
        return self._target_from_widget(best_widget)

    def _target_from_widget(self, widget: WidgetInfo) -> TargetRef:
        selectors = [Selector(kind="widget_id", operator="equals", value=int(widget.widget_id))]
        if widget.resource_id:
            selectors.append(Selector(kind="resource_id", operator="equals", value=widget.resource_id))
        if widget.text:
            selectors.append(Selector(kind="text", operator="contains", value=widget.text[:80]))
        if widget.content_desc:
            selectors.append(Selector(kind="content_desc", operator="contains", value=widget.content_desc[:80]))
        if widget.class_name:
            selectors.append(Selector(kind="class_name", operator="equals", value=widget.class_name))
        if widget.bounds:
            selectors.append(Selector(kind="bounds", operator="overlap", value=list(widget.bounds)))

        return TargetRef(
            ref_id="target:primary",
            role="primary",
            selectors=selectors,
            resolved=TargetResolved(
                widget_id=int(widget.widget_id),
                bounds=tuple(int(v) for v in widget.bounds),
                center=tuple(int(v) for v in widget.center),
                snapshot={
                    "text": widget.text,
                    "resource_id": widget.resource_id,
                    "content_desc": widget.content_desc,
                    "class_name": widget.class_name,
                },
            ),
        )

    def _is_redundant(self, result: PlanResult) -> bool:
        summary = str(result.goal.summary or "").strip().lower()
        action = str(result.requested_action_type or "").strip().lower()
        target_hint = ""
        if result.target and result.target.selectors:
            for selector in result.target.selectors:
                if selector.kind in {"text", "resource_id", "widget_id", "content_desc"}:
                    target_hint = str(selector.value or "").strip().lower()
                    if target_hint:
                        break

        for step in reversed(self.memory.short_term.history[-3:]):
            old_eval = step.get("evaluation") or {}
            old_action = (step.get("action") or {}).get("type") or ""
            old_summary = str((step.get("goal") or {}).get("summary") or step.get("subgoal") or "").strip().lower()
            if old_eval and str(old_eval.get("decision") or "").lower() == "success":
                continue

            old_target_hint = ""
            old_target = (step.get("contract") or {}).get("target") or {}
            if isinstance(old_target, dict):
                for selector in old_target.get("selectors", []) or []:
                    if not isinstance(selector, dict):
                        continue
                    if selector.get("kind") in {"text", "resource_id", "widget_id", "content_desc"}:
                        old_target_hint = str(selector.get("value") or "").strip().lower()
                        if old_target_hint:
                            break

            same_summary = bool(summary and old_summary and summary == old_summary)
            same_action = bool(action and old_action and action == str(old_action).lower())
            same_target = bool(target_hint and old_target_hint and target_hint == old_target_hint)
            if same_action and (same_summary or same_target):
                return True
        return False

    def _is_misaligned(self, task: str, result: PlanResult) -> tuple[bool, str]:
        task_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", task)}
        goal_text = " ".join(
            [
                str(result.goal.summary or ""),
                str(result.goal.success_definition or ""),
            ]
        )
        goal_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", goal_text)}

        if not task_tokens or not goal_tokens:
            return False, ""

        overlap = task_tokens.intersection(goal_tokens)
        if overlap:
            return False, ""

        return True, "token_overlap_zero"

    def _record_plan_dataclass(self, source: str, result: PlanResult, step: int | None = None) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="PlanResult",
                step=int(step),
                payload=result,
            )
        except Exception as exc:
            logger.warning("写入 Planner 审计记录失败: %s", exc)
