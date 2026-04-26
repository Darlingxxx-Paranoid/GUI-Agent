"""Main ReAct loop orchestrator using current minimal modules."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any

from config import AgentConfig
from Execution.action_executor import ActionExecutor
from Oracle.post_oracle import PostOracle
from Oracle.pre_oracle import OraclePre
from Oracle.running_oracle import RunningOracle, RunningOracleResult
from Perception.dump_parser import DumpParser
from Perception.uied_controls import get_uied_visible_widgets_list
from Planning.planner import AnchorResult, PlanResult, Planner
from utils.audit_recorder import AuditRecorder
from utils.llm_client import LLMClient, LLMStructuredOutputError

logger = logging.getLogger(__name__)


@dataclass
class StepObservation:
    screenshot_path: str
    dump_path: str
    dump_tree: dict[str, Any]
    widgets: list[dict[str, Any]]
    activity: str
    package: str
    keyboard_visible: bool


class AgentLoop:
    """A minimal yet complete loop: observe -> plan -> pre-oracle -> act -> post-oracle."""

    def __init__(self, config: AgentConfig):
        self.config = config
        config.ensure_dirs()
        self.audit = AuditRecorder(component="agent_loop")

        self.llm = LLMClient(
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self.executor = ActionExecutor(
            serial=config.adb_serial,
            screenshot_dir=config.screenshot_dir,
            dump_dir=config.dump_dir,
        )
        self.dump_parser = DumpParser()
        self.planner = Planner(
            llm_client=self.llm,
            cv_output_dir=config.cv_output_dir,
            cv_resize_height=config.cv_resize_height,
        )
        self.oracle_pre = OraclePre(llm_client=self.llm)
        self.post_oracle = PostOracle(llm_client=self.llm)
        self.running_oracle = RunningOracle(
            screen_variance_threshold=config.screen_variance_threshold,
        )
        self._app_catalog = self._load_app_catalog()
        self._task_target_app: dict[str, str] | None = None

        self.pending_runtime_replan_hint = ""
        self.action_history: list[dict[str, Any]] = []
        self.progress_context: list[dict[str, str]] = []
        self.experience_context: list[dict[str, str]] = []
        self._cached_before_observation: StepObservation | None = None
        self._cached_before_step: int | None = None
        self._cached_before_source_step: int | None = None
        self._ui_changed_signature_counts: dict[str, int] = {}
        logger.info("AgentLoop 初始化完成(最小闭环), max_steps=%d", config.max_steps)

    def run(self, task: str, dry_run: bool = False) -> bool:
        logger.info("=" * 60)
        logger.info("开始执行任务: '%s'", task)
        logger.info("=" * 60)

        self.running_oracle.reset()
        self.pending_runtime_replan_hint = ""
        self.action_history = []
        self.progress_context = []
        self.experience_context = []
        self._ui_changed_signature_counts = {}
        self._task_target_app = self._resolve_task_target_app(task)
        if self._task_target_app:
            logger.info(
                "任务目标 App 解析: name=%s package=%s activity=%s",
                self._task_target_app.get("name", ""),
                self._task_target_app.get("package", ""),
                self._task_target_app.get("activity", ""),
            )
        else:
            logger.info("任务目标 App 解析为空，将依赖 Planner 视觉判断与运行时回退")
        self._clear_reusable_observation()

        screen_size = (1080, 1920)
        try:
            screen_size = self.executor.get_screen_size()
            self.executor.stabilize_ui_animations()
        except Exception as exc:
            logger.warning("初始化设备状态失败，使用默认分辨率: %s", exc)

        task_completed = False
        for step in range(1, int(self.config.max_steps) + 1):
            logger.info("=" * 40)
            logger.info("第 %d 步 (最大 %d)", step, self.config.max_steps)
            logger.info("=" * 40)
            try:
                step_done, task_completed = self._run_one_step(
                    task=task,
                    step=step,
                    screen_size=screen_size,
                    dry_run=dry_run,
                )
                if task_completed:
                    break
                if not step_done:
                    # 当前步失败，继续下一步重规划。
                    continue
            except KeyboardInterrupt:
                logger.info("用户中断任务")
                break
            except Exception as exc:
                logger.error("步骤 %d 执行异常: %s", step, exc, exc_info=True)
                self._record_plan_exception(
                    step=step,
                    exc=exc,
                    phase="run_one_step",
                    traceback_text=traceback.format_exc(),
                )
                self._clear_reusable_observation()
                continue

        if task_completed:
            logger.info("任务成功完成: '%s'", task)
        else:
            logger.warning(
                "任务未完成: '%s' (step<=%d)",
                task,
                self.config.max_steps,
            )
        return task_completed

    def _run_one_step(
        self,
        task: str,
        step: int,
        screen_size: tuple[int, int],
        dry_run: bool,
    ) -> tuple[bool, bool]:
        before = self._observe_before(step=step)
        if before is None:
            self._record_step_artifact(
                artifact_kind="StepResult",
                step=step,
                payload={"phase": "before", "status": "observe_failed"},
            )
            return False, False

        try:
            plan = self.planner.plan(
                task=task,
                screenshot=before.screenshot_path,
                runtime_exception_hint=str(self.pending_runtime_replan_hint or ""),
                progress_context=self.progress_context,
                experience_context=self.experience_context,
                current_package=before.package,
                task_target_app=self._task_target_app,
                step=step,
            )
        except Exception as exc:
            self._record_plan_exception(
                step=step,
                exc=exc,
                phase="planner.plan",
                traceback_text=traceback.format_exc(),
            )
            raise
        self.pending_runtime_replan_hint = ""
        self._record_step_artifact(
            artifact_kind="PlanResult",
            step=step,
            payload=plan,
        )

        if plan.is_task_complete:
            logger.info("Planner 判断任务已完成")
            self._record_step_artifact(
                artifact_kind="StepResult",
                step=step,
                payload={"phase": "plan", "status": "task_complete"},
            )
            return True, True

        action_type = str(plan.action_type or "").strip().lower()
        anchor_result = AnchorResult(
            target_widget_id=-1,
            anchor_method="none",
            anchor_reason=f"action_type={action_type} does not require widget anchor",
        )
        if self._action_requires_widget(action_type):
            anchor_result = self.planner.anchor(
                plan=plan,
                screenshot=before.screenshot_path,
                visible_widgets_list=before.widgets,
                step=step,
            )
        self._record_step_artifact(
            artifact_kind="AnchorResult",
            step=step,
            payload=anchor_result,
        )

        if self._action_requires_widget(action_type) and int(anchor_result.target_widget_id) < 0:
            self._set_runtime_replan_hint(
                action_text=f"anchor_failed:{plan.target_description}"
            )
            self._append_experience_context(
                plan=plan,
                failure_reason=f"anchor_failed:{anchor_result.anchor_reason}",
            )
            self._cache_reusable_observation(step=step, observation=before)
            logger.warning(
                "锚定失败(step=%d): action_type=%s target_description=%s reason=%s",
                step,
                action_type,
                plan.target_description,
                anchor_result.anchor_reason,
            )
            self._record_step_artifact(
                artifact_kind="StepResult",
                step=step,
                payload={
                    "status": "anchor_failed",
                    "goal": plan.goal,
                    "action_type": plan.action_type,
                    "target_description": plan.target_description,
                    "anchor_reason": anchor_result.anchor_reason,
                },
            )
            return False, False

        if str(plan.action_type or "").strip().lower() == "wait":
            action = self._build_action_from_plan(
                plan=plan,
                anchor_result=anchor_result,
                widgets=before.widgets,
                screen_size=screen_size,
                task=task,
            )
            self._record_step_artifact(
                artifact_kind="ResolvedAction",
                step=step,
                payload=action,
            )
            action_record: dict[str, Any] = {
                "step": step,
                "goal": plan.goal,
                "action_type": action.get("type"),
                "action_params": dict(action.get("params") or {}),
                "dry_run": bool(dry_run),
            }
            self.action_history.append(action_record)
            logger.info("本步为等待动作，跳过 Pre/Post/Running Oracle 与控件动作映射")
            if dry_run:
                logger.info("[DRY RUN] 跳过执行动作: %s", action)
            else:
                self.executor.execute(action)
            wait_progress_item = {
                "goal": str(plan.goal or ""),
                "action_type": "wait",
                "target_description": "",
                "input_description": "",
            }
            self.progress_context.append(wait_progress_item)
            logger.info(
                "等待动作已记录到 progress_context (progress_context=%d)",
                len(self.progress_context),
            )
            self._record_step_artifact(
                artifact_kind="StepResult",
                step=step,
                payload={
                    "status": "wait_executed",
                    "goal": plan.goal,
                    "action_type": "wait",
                    "wait_seconds": float((action.get("params") or {}).get("duration_sec") or 2.0),
                },
            )
            return True, False

        runtime_before = self._run_running_oracle(
            step=step,
            phase="before_action",
            screenshot_path=before.screenshot_path,
        )
        if runtime_before and runtime_before.has_runtime_exception:
            self._clear_reusable_observation()
            self._set_runtime_replan_hint(
                action_text=f"before_action_runtime_exception:{','.join(runtime_before.tags or [])}"
            )
            self._append_experience_context(
                plan=plan,
                failure_reason=f"before_action_runtime_exception:{','.join(runtime_before.tags or [])}",
            )
            if not dry_run:
                logger.warning("Running-Oracle 告警，执行 back 尝试恢复")
                self.executor.back()
            return False, False

        pre_oracle_output = self.oracle_pre.generate_contract(
            plan=plan,
            dump_tree=before.dump_tree,
            screenshot_path=before.screenshot_path,
            anchor_result=anchor_result,
            widgets=before.widgets,
            step=step,
        )
        self._record_step_artifact(
            artifact_kind="SemanticTransitionContract",
            step=step,
            payload=pre_oracle_output.semantic_contract,
        )
        self._record_step_artifact(
            artifact_kind="UIAssertionContract",
            step=step,
            payload=pre_oracle_output.assertion_contract,
        )

        action = self._build_action_from_plan(
            plan=plan,
            anchor_result=anchor_result,
            widgets=before.widgets,
            screen_size=screen_size,
            task=task,
        )
        self._record_step_artifact(
            artifact_kind="ResolvedAction",
            step=step,
            payload=action,
        )

        action_record: dict[str, Any] = {
            "step": step,
            "goal": plan.goal,
            "action_type": action.get("type"),
            "action_params": dict(action.get("params") or {}),
            "dry_run": bool(dry_run),
        }
        self.action_history.append(action_record)

        if dry_run:
            logger.info("[DRY RUN] 跳过执行动作: %s", action)
        else:
            self.executor.execute(action)
        time.sleep(1.8)

        after = self._observe(step=step, phase="after")
        if after is None:
            self._clear_reusable_observation()
            action_record["post_oracle"] = {
                "decision": "semantic_fail_ui_unchanged",
                "needs_back": False,
                "reason": "observe_after_failed",
                "evidence_source": "llm_secondary_retry_fallback",
                "assertion_summary": {"total": 0, "failed": 0},
                "failed_assertions": [],
            }
            self._set_runtime_replan_hint(action_text="observe_after_failed")
            self._append_experience_context(plan=plan, failure_reason="observe_after_failed")
            return False, False

        runtime_after = self._run_running_oracle(
            step=step,
            phase="after_action",
            screenshot_path=after.screenshot_path,
        )
        if runtime_after and runtime_after.has_runtime_exception:
            self._clear_reusable_observation()
            self._set_runtime_replan_hint(
                action_text=f"after_action_runtime_exception:{','.join(runtime_after.tags or [])}"
            )
            self._append_experience_context(
                plan=plan,
                failure_reason=f"after_action_runtime_exception:{','.join(runtime_after.tags or [])}",
            )
            action_record["runtime_oracle"] = {
                "phase": runtime_after.phase,
                "tags": list(runtime_after.tags or []),
            }
            if not dry_run:
                logger.warning("Running-Oracle 告警，执行 back 尝试恢复")
                self.executor.back()
            return False, False

        post_result = self.post_oracle.evaluate(
            before_dump_tree=before.dump_tree,
            after_dump_tree=after.dump_tree,
            action=action,
            semantic_contract=pre_oracle_output.semantic_contract,
            assertion_contract=pre_oracle_output.assertion_contract,
            before_screenshot_path=before.screenshot_path,
            after_screenshot_path=after.screenshot_path,
            step=step,
        )
        post_output = self.post_oracle.to_output(post_result)
        action_record["post_oracle"] = dict(post_output)
        self._record_step_artifact(
            artifact_kind="PostOracle",
            step=step,
            payload={"phase": "post", "value": post_output},
            append=True,
        )

        if post_result.decision == "semantic_success":
            progress_item = {
                "goal": str(plan.goal or ""),
                "action_type": str(plan.action_type or ""),
                "target_description": str(plan.target_description or ""),
                "input_description": str(plan.input_description or ""),
            }
            self.progress_context.append(progress_item)
            self._ui_changed_signature_counts = {}
            self._cache_reusable_observation(step=step, observation=after)
            logger.info(
                "Post-Oracle 通过，子目标完成: %s (progress_context=%d)",
                plan.goal,
                len(self.progress_context),
            )
            self._record_step_artifact(
                artifact_kind="StepResult",
                step=step,
                payload={
                    "status": "subgoal_success",
                    "goal": plan.goal,
                    "action_type": plan.action_type,
                    "input_description": plan.input_description,
                },
            )
            return True, False

        if post_result.decision == "semantic_fail_ui_changed":
            self._clear_reusable_observation()
            self._set_runtime_replan_hint(action_text=f"post_oracle_semantic_fail_ui_changed:{plan.goal}")
            self._append_experience_context(
                plan=plan,
                failure_reason="post_oracle_semantic_fail_ui_changed",
            )
            defer_back, signature, count = self._should_defer_back_on_ui_changed(
                plan=plan,
                before=before,
                after=after,
                post_reason=str(post_result.reason or ""),
            )
            if defer_back:
                logger.warning(
                    "Post-Oracle 语义失败且 UI 有变化(step=%d)，检测到中间态(签名=%s, count=%d)，本次不回退，直接重规划",
                    step,
                    signature,
                    count,
                )
                self._cache_reusable_observation(step=step, observation=after)
                return False, False
            logger.warning(
                "Post-Oracle 语义失败且 UI 有变化(step=%d)，执行回退后重规划(签名=%s, count=%d)",
                step,
                signature,
                count,
            )
            if not dry_run:
                self.executor.back()
            return False, False

        if post_result.decision == "semantic_fail_ui_unchanged":
            self._set_runtime_replan_hint(action_text=f"post_oracle_semantic_fail_ui_unchanged:{plan.goal}")
            self._append_experience_context(
                plan=plan,
                failure_reason=(
                    "post_oracle_llm_fallback_fail"
                    if post_result.evidence_source == "llm_secondary_retry_fallback"
                    else "post_oracle_semantic_fail_ui_unchanged"
                ),
            )
            logger.warning("Post-Oracle 语义失败且 UI 无变化(step=%d)，直接重规划", step)
            self._cache_reusable_observation(step=step, observation=after)
            return False, False

        logger.warning("Post-Oracle 返回未知 decision=%s(step=%d)，按无变化失败处理", post_result.decision, step)
        self._set_runtime_replan_hint(action_text=f"post_oracle_unknown_decision:{plan.goal}")
        self._append_experience_context(
            plan=plan,
            failure_reason="post_oracle_llm_fallback_fail",
        )
        self._cache_reusable_observation(step=step, observation=after)
        return False, False

    def _observe_before(
        self,
        step: int,
    ) -> StepObservation | None:
        reused = self._consume_reusable_observation(step=step)
        if reused is not None:
            return reused
        return self._observe(step=step, phase="before")

    def _consume_reusable_observation(self, step: int) -> StepObservation | None:
        cached = self._cached_before_observation
        target_step = self._cached_before_step
        source_step = self._cached_before_source_step
        if cached is None or target_step is None:
            return None
        if int(target_step) != int(step):
            logger.debug(
                "跳过复用观测: target_step=%s, current_step=%d",
                target_step,
                step,
            )
            self._clear_reusable_observation()
            return None

        self._clear_reusable_observation()
        logger.info(
            "复用观测: step_%d_after -> step_%d_before",
            int(source_step or max(0, step - 1)),
            step,
        )
        self._record_step_artifact(
            artifact_kind="Observation",
            step=step,
            payload={
                "phase": "before",
                "screenshot_path": cached.screenshot_path,
                "dump_path": cached.dump_path,
                "activity": cached.activity,
                "package": cached.package,
                "keyboard_visible": bool(cached.keyboard_visible),
                "dump_node_count": self._count_nodes(cached.dump_tree),
                "widget_count": len(cached.widgets),
                "reused_from_previous_after": True,
                "source_step": int(source_step or max(0, step - 1)),
            },
            append=True,
        )
        return cached

    def _cache_reusable_observation(self, step: int, observation: StepObservation) -> None:
        if not observation.dump_path:
            logger.debug("跳过观测复用缓存: step=%d after dump 为空", step)
            self._clear_reusable_observation()
            return
        next_step = int(step) + 1
        self._cached_before_observation = observation
        self._cached_before_step = next_step
        self._cached_before_source_step = int(step)
        logger.info("缓存观测用于复用: step_%d_after -> step_%d_before", step, next_step)

    def _clear_reusable_observation(self) -> None:
        self._cached_before_observation = None
        self._cached_before_step = None
        self._cached_before_source_step = None

    def _build_ui_changed_signature(
        self,
        plan: PlanResult,
        before: StepObservation,
        after: StepObservation,
    ) -> str:
        goal = str(plan.goal or "").strip().lower()
        action_type = str(plan.action_type or "").strip().lower()
        target = str(plan.target_description or "").strip().lower()
        return "|".join(
            [
                str(before.package or "").strip().lower(),
                str(before.activity or "").strip().lower(),
                str(after.package or "").strip().lower(),
                str(after.activity or "").strip().lower(),
                action_type,
                goal,
                target,
            ]
        )

    def _is_intermediate_ui_changed_failure(
        self,
        plan: PlanResult,
        before: StepObservation,
        after: StepObservation,
        post_reason: str,
    ) -> bool:
        before_pkg = str(before.package or "").strip().lower()
        after_pkg = str(after.package or "").strip().lower()
        before_act = str(before.activity or "").strip().lower()
        after_act = str(after.activity or "").strip().lower()

        if not before_pkg or not after_pkg:
            return False
        if before_pkg != after_pkg:
            return False
        if before_act and after_act and before_act != after_act:
            return False

        reason = str(post_reason or "").strip().lower()
        reason_keywords = [
            "dialog",
            "popup",
            "pop-up",
            "overlay",
            "menu",
            "confirm",
            "confirmation",
            "cancel",
            "send",
            "choose a view",
            "弹窗",
            "菜单",
            "确认",
            "对话框",
        ]
        if any(key in reason for key in reason_keywords):
            return True

        goal = str(plan.goal or "").strip().lower()
        target = str(plan.target_description or "").strip().lower()
        target_keywords = [
            "send",
            "compose",
            "new mail",
            "new message",
            "menu",
            "confirm",
            "弹窗",
            "菜单",
            "确认",
            "发送",
        ]
        if any(key in goal for key in target_keywords) and any(key in target for key in target_keywords):
            return True
        return False

    def _should_defer_back_on_ui_changed(
        self,
        plan: PlanResult,
        before: StepObservation,
        after: StepObservation,
        post_reason: str,
    ) -> tuple[bool, str, int]:
        signature = self._build_ui_changed_signature(
            plan=plan,
            before=before,
            after=after,
        )
        count = int(self._ui_changed_signature_counts.get(signature, 0)) + 1
        self._ui_changed_signature_counts[signature] = count

        is_intermediate = self._is_intermediate_ui_changed_failure(
            plan=plan,
            before=before,
            after=after,
            post_reason=post_reason,
        )
        if is_intermediate and count == 1:
            return True, signature, count
        return False, signature, count

    def _observe(
        self,
        step: int,
        phase: str,
    ) -> StepObservation | None:
        screenshot_name = f"step_{step}_{phase}.png"
        dump_name = f"step_{step}_{phase}.xml"
        try:
            screenshot_path = self.executor.screenshot(screenshot_name)
            dump_path = self.executor.dump_ui(dump_name)
            activity = self.executor.get_current_activity()
            package = self.executor.get_current_package()
            keyboard_visible = self.executor.get_keyboard_visible()
        except Exception as exc:
            logger.warning("环境观测失败(step=%d, phase=%s): %s", step, phase, exc)
            return None

        dump_tree: dict[str, Any] = {}
        if dump_path:
            dump_tree = self.dump_parser.parse_tree(dump_path)

        if not dump_tree:
            # 保证 Pre/Post-Oracle 至少能拿到结构化根节点。
            dump_tree = {"node_id": 1, "children": []}
        if activity:
            dump_tree["activity"] = str(activity)
        if package:
            dump_tree["package"] = str(package)

        widgets = self._extract_widgets(
            screenshot_path=screenshot_path,
        )
        self._record_step_artifact(
            artifact_kind="Observation",
            step=step,
            payload={
                "phase": phase,
                "screenshot_path": screenshot_path,
                "dump_path": dump_path,
                "activity": activity,
                "package": package,
                "keyboard_visible": bool(keyboard_visible),
                "dump_node_count": self._count_nodes(dump_tree),
                "widget_count": len(widgets),
                "reused_from_previous_after": False,
            },
            append=True,
        )

        return StepObservation(
            screenshot_path=screenshot_path,
            dump_path=dump_path,
            dump_tree=dump_tree,
            widgets=widgets,
            activity=str(activity or ""),
            package=str(package or ""),
            keyboard_visible=bool(keyboard_visible),
        )

    def _extract_widgets(
        self,
        screenshot_path: str,
    ) -> list[dict[str, Any]]:
        try:
            widgets = get_uied_visible_widgets_list(
                screenshot_path=screenshot_path,
                cv_output_dir=self.config.cv_output_dir,
                resize_height=self.config.cv_resize_height,
            )
            return list(widgets or [])
        except Exception as exc:
            logger.warning("UIED 可见控件提取失败，降级为空列表: %s", exc)
            return []

    def _build_action_from_plan(
        self,
        plan: PlanResult,
        anchor_result: AnchorResult,
        widgets: list[dict[str, Any]],
        screen_size: tuple[int, int],
        task: str,
    ) -> dict[str, Any]:
        action_type = str(plan.action_type or "").strip().lower()
        if action_type == "wait":
            return {"type": "wait", "params": {"duration_sec": 2.0}}

        if action_type == "launch_app":
            launch_target = self._resolve_launch_target(plan=plan, task=task)
            package = str(launch_target.get("package") or "").strip()
            activity = str(launch_target.get("activity") or "").strip()
            if not package:
                raise ValueError("action_type=launch_app 但无法解析 package，拒绝降级为 tap")
            logger.info(
                "构造 launch_app 动作: package=%s activity=%s",
                package,
                activity,
            )
            return {
                "type": "launch_app",
                "params": {
                    "package": package,
                    "activity": activity,
                },
            }

        target_widget_id = int(getattr(anchor_result, "target_widget_id", -1))
        if self._action_requires_widget(action_type) and target_widget_id < 0:
            raise ValueError(
                f"action_type={action_type} 需要已锚定控件，但 target_widget_id={target_widget_id}"
            )
        widget = self._find_widget_by_id(widgets, target_widget_id)
        center = self._widget_center(widget=widget, screen_size=screen_size)
        bounds = self._widget_bounds(widget=widget)

        if action_type == "tap":
            return {"type": "tap", "params": {"x": center[0], "y": center[1]}}

        if action_type == "long_press":
            return {
                "type": "long_press",
                "params": {"x": center[0], "y": center[1], "duration_ms": 900},
            }

        if action_type == "input":
            return {
                "type": "input",
                "params": {"x": center[0], "y": center[1], "text": str(plan.input_description or "")},
            }

        if action_type == "swipe":
            x1, y1, x2, y2 = self._infer_swipe_points(
                hint_text=f"{task} {plan.goal}",
                bounds=bounds,
                screen_size=screen_size,
            )
            return {
                "type": "swipe",
                "params": {
                    "x": x1,
                    "y": y1,
                    "x2": x2,
                    "y2": y2,
                    "duration_ms": 500,
                },
            }

        if action_type == "back":
            return {"type": "back", "params": {}}

        if action_type == "enter":
            return {"type": "enter", "params": {}}

        logger.warning("未知 action_type=%s，降级为 tap 中心点", action_type)
        return {"type": "tap", "params": {"x": center[0], "y": center[1]}}

    def _action_requires_widget(self, action_type: str) -> bool:
        token = str(action_type or "").strip().lower()
        return token in {"tap", "input", "swipe", "long_press"}

    def _append_experience_context(
        self,
        plan: PlanResult,
        failure_reason: str,
    ) -> None:
        self.experience_context.append(
            {
                "goal": str(plan.goal or ""),
                "action_type": str(plan.action_type or ""),
                "target_description": str(getattr(plan, "target_description", "") or ""),
                "input_description": str(getattr(plan, "input_description", "") or ""),
                "failure_reason": str(failure_reason or "").strip(),
            }
        )
        self.experience_context = self.experience_context[-3:]
        logger.info("experience_context 更新: total=%d", len(self.experience_context))

    def _load_app_catalog(self) -> list[dict[str, str]]:
        conf_path = os.path.join(os.path.dirname(__file__), "conf.json")
        try:
            with open(conf_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            logger.warning("加载 app catalog 失败(conf.json): %s", exc)
            return []

        raw_apps = payload.get("apps")
        if not isinstance(raw_apps, list):
            return []

        apps: list[dict[str, str]] = []
        for item in raw_apps:
            if not isinstance(item, dict):
                continue
            package = str(item.get("package") or "").strip()
            if not package:
                continue
            apps.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "package": package,
                    "activity": str(
                        item.get("launch-activity")
                        or item.get("launch_activity")
                        or ""
                    ).strip(),
                }
            )
        logger.info("加载 app catalog 完成: %d entries", len(apps))
        return apps

    def _resolve_task_target_app(self, task: str) -> dict[str, str] | None:
        task_text = str(task or "").strip()
        if not task_text:
            return None
        task_lower = task_text.lower()

        explicit_package = self._extract_package_from_text(task_text)
        if explicit_package:
            matched = self._lookup_app_by_package(explicit_package)
            if matched is not None:
                return matched
            if explicit_package.count(".") >= 2:
                return {"name": "", "package": explicit_package, "activity": ""}

        best: dict[str, str] | None = None
        best_score = -1
        use_phrase = ""
        use_match = re.search(r"\buse\s+(.+?)\s+to\b", task_lower)
        if use_match:
            use_phrase = str(use_match.group(1) or "").strip()

        for app in self._app_catalog:
            name = str(app.get("name") or "").strip()
            package = str(app.get("package") or "").strip()
            if not package:
                continue

            name_lower = name.lower()
            package_lower = package.lower()
            score = 0

            if package_lower and package_lower in task_lower:
                score += 1000
            if name_lower and name_lower in task_lower:
                score += 220

            tokens = [token for token in re.split(r"[^a-z0-9]+", name_lower) if len(token) >= 3]
            overlap = sum(1 for token in tokens if token and token in task_lower)
            score += overlap * 35

            if use_phrase and name_lower:
                if use_phrase == name_lower:
                    score += 240
                elif use_phrase in name_lower or name_lower in use_phrase:
                    score += 120

            if score > best_score:
                best_score = score
                best = app

        if best is None or best_score < 120:
            return None
        return dict(best)

    def _resolve_launch_target(self, plan: PlanResult, task: str) -> dict[str, str]:
        plan_package = str(getattr(plan, "launch_package", "") or "").strip()
        plan_activity = str(getattr(plan, "launch_activity", "") or "").strip()
        if plan_package:
            matched = self._lookup_app_by_package(plan_package)
            if matched is None:
                return {
                    "name": "",
                    "package": plan_package,
                    "activity": plan_activity,
                }
            if plan_activity:
                matched["activity"] = plan_activity
            return matched

        if self._task_target_app and self._task_target_app.get("package"):
            return dict(self._task_target_app)

        package_from_task = self._extract_package_from_text(task)
        if package_from_task:
            matched = self._lookup_app_by_package(package_from_task)
            if matched is not None:
                return matched
            if package_from_task.count(".") >= 2:
                return {"name": "", "package": package_from_task, "activity": ""}
        return {}

    def _lookup_app_by_package(self, package: str) -> dict[str, str] | None:
        pkg = str(package or "").strip().lower()
        if not pkg:
            return None
        for app in self._app_catalog:
            app_pkg = str(app.get("package") or "").strip().lower()
            if app_pkg == pkg:
                return dict(app)
        return None

    def _extract_package_from_text(self, text: str) -> str:
        token_text = str(text or "")
        pattern = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+\b")
        for match in pattern.finditer(token_text):
            candidate = str(match.group(0) or "").strip()
            start = int(match.start())
            if start > 0 and token_text[start - 1] == "@":
                # Skip email domains like example@example.com.
                continue
            if candidate and "." in candidate:
                return candidate
        return ""

    def _find_widget_by_id(self, widgets: list[dict[str, Any]], widget_id: int) -> dict[str, Any] | None:
        for item in widgets:
            try:
                if int(item.get("widget_id")) == int(widget_id):
                    return item
            except Exception:
                continue
        return None

    def _widget_center(self, widget: dict[str, Any] | None, screen_size: tuple[int, int]) -> tuple[int, int]:
        sw, sh = int(screen_size[0]), int(screen_size[1])
        default = (max(0, sw // 2), max(0, sh // 2))
        if not widget:
            return default
        center = widget.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            try:
                x = int(center[0])
                y = int(center[1])
                return (x, y)
            except Exception:
                return default
        return default

    def _widget_bounds(self, widget: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
        if not widget:
            return None
        bounds = widget.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            try:
                return (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
            except Exception:
                return None
        return None

    def _infer_swipe_points(
        self,
        hint_text: str,
        bounds: tuple[int, int, int, int] | None,
        screen_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        direction = self._infer_direction(hint_text)
        sw, sh = int(screen_size[0]), int(screen_size[1])

        if bounds is None:
            cx = sw // 2
            cy = sh // 2
            dx = max(120, sw // 5)
            dy = max(180, sh // 5)
        else:
            x1, y1, x2, y2 = bounds
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dx = max(80, int((x2 - x1) * 0.6))
            dy = max(120, int((y2 - y1) * 0.7))

        if direction == "down":
            start = (cx, cy - dy // 2)
            end = (cx, cy + dy // 2)
        elif direction == "left":
            start = (cx + dx // 2, cy)
            end = (cx - dx // 2, cy)
        elif direction == "right":
            start = (cx - dx // 2, cy)
            end = (cx + dx // 2, cy)
        else:  # up (default)
            start = (cx, cy + dy // 2)
            end = (cx, cy - dy // 2)

        sx = min(max(0, int(start[0])), max(0, sw - 1))
        sy = min(max(0, int(start[1])), max(0, sh - 1))
        ex = min(max(0, int(end[0])), max(0, sw - 1))
        ey = min(max(0, int(end[1])), max(0, sh - 1))
        return sx, sy, ex, ey

    def _infer_direction(self, text: str) -> str:
        token = str(text or "").lower()
        down_hits = ["swipe down", "pull down", "向下", "下滑", "往下", "下拉", "down"]
        up_hits = ["swipe up", "向上", "上滑", "往上", "up"]
        left_hits = ["向左", "左滑", "往左", "swipe left", "left"]
        right_hits = ["向右", "右滑", "往右", "swipe right", "right"]

        for key in down_hits:
            if key in token:
                return "down"
        for key in up_hits:
            if key in token:
                return "up"
        for key in left_hits:
            if key in token:
                return "left"
        for key in right_hits:
            if key in token:
                return "right"
        return "up"

    def _count_nodes(self, tree: Any) -> int:
        if not isinstance(tree, dict):
            return 0
        count = 1
        children = tree.get("children")
        if isinstance(children, list):
            for child in children:
                count += self._count_nodes(child)
        return count

    def _run_running_oracle(
        self,
        step: int,
        phase: str,
        screenshot_path: str,
    ) -> RunningOracleResult | None:
        try:
            result = self.running_oracle.check(
                screenshot_path=screenshot_path,
                phase=phase,
            )
            self._record_step_artifact(
                artifact_kind="RunningOracle",
                step=step,
                payload={"phase": phase, "value": result},
                append=True,
            )
            return result
        except Exception as exc:
            logger.warning("Running-Oracle 执行失败(step=%d, phase=%s): %s", step, phase, exc)
            return None

    def _set_runtime_replan_hint(self, action_text: str) -> None:
        action_desc = str(action_text or "").strip() or "unknown"
        self.pending_runtime_replan_hint = (
            f"刚刚遇到了异常或未达成目标，最近操作为{action_desc}，请重规划下一步。"
        )

    def _record_plan_exception(
        self,
        step: int,
        exc: Exception,
        phase: str,
        traceback_text: str = "",
    ) -> None:
        if int(step) <= 0:
            return
        plan_path = os.path.join(
            self.audit.base_dir,
            "PlanResult",
            f"step_{int(step)}.json",
        )
        if os.path.exists(plan_path):
            logger.info(
                "步骤 %d 异常发生时 PlanResult 已存在，跳过异常占位写入",
                step,
            )
            return

        if isinstance(exc, LLMStructuredOutputError):
            raw_payload = exc.raw_payload
            if raw_payload is not None:
                self._record_step_artifact(
                    artifact_kind="PlanResult",
                    step=step,
                    payload=raw_payload,
                )
                logger.info(
                    "步骤 %d 记录 LLM 原始结构化输出到 PlanResult（校验失败）",
                    step,
                )
                return

        error_type = type(exc).__name__
        error_message = str(exc or "").strip() or repr(exc)
        tb_tail = str(traceback_text or "").strip()
        if tb_tail and len(tb_tail) > 1800:
            tb_tail = tb_tail[-1800:]

        reasoning = (
            f"status=exception; phase={phase}; "
            f"error_type={error_type}; error_message={error_message}"
        )
        if tb_tail:
            reasoning = f"{reasoning}; traceback_tail={tb_tail}"

        error_plan_payload: dict[str, Any] = {
            "goal": f"[EXCEPTION] step_{int(step)} failed before valid planner output",
            "action_type": "back",
            "target_description": "",
            "input_description": "",
            "launch_package": "",
            "launch_activity": "",
            "is_task_complete": False,
            "reasoning": reasoning,
        }
        self._record_step_artifact(
            artifact_kind="PlanResult",
            step=step,
            payload=error_plan_payload,
        )

    def _record_step_artifact(
        self,
        artifact_kind: str,
        step: int,
        payload: Any,
        append: bool = False,
    ) -> None:
        if int(step) <= 0:
            return
        try:
            self.audit.record_step(
                artifact_kind=artifact_kind,
                step=int(step),
                payload=payload,
                append=append,
            )
        except Exception as exc:
            logger.warning("写入 %s 审计记录失败(step=%d): %s", artifact_kind, step, exc)
