"""Main ReAct loop orchestrator using Oracle V3.1 contracts."""

from __future__ import annotations

import os
import time
import logging

from config import AgentConfig
from utils.llm_client import LLMClient

from Evaluation.evaluator import Evaluator
from Evaluation.replanner import Replanner
from Execution.action_executor import ActionExecutor
from Execution.action_mapper import ActionMapper, MappingFailure
from Execution.oracle_runtime import OracleRuntime
from Memory.memory_manager import MemoryManager
from Oracle.contracts import (
    ACTION_TYPES,
    AdviceParams,
    Assessment,
    RecommendedAction,
    StepEvaluation,
    to_plain_dict,
    export_contract_schema,
)
from Perception.perception_manager import PerceptionManager
from Planning.oracle_pre import OraclePre
from Planning.planner import Planner
from Planning.safety_interceptor import SafetyInterceptor
from utils.audit_recorder import AuditRecorder

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(self, config: AgentConfig):
        self.config = config
        config.ensure_dirs()
        self.audit = AuditRecorder(component="agent_loop")

        self.contract_schema_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "contracts",
            "oracle_contracts_v4_0.json",
        )
        os.makedirs(os.path.dirname(self.contract_schema_path), exist_ok=True)
        try:
            export_contract_schema(self.contract_schema_path)
        except Exception as exc:
            logger.warning("导出 contract schema 失败: %s", exc)

        self.llm = LLMClient(
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )

        self.perception = PerceptionManager(
            cv_output_dir=config.cv_output_dir,
            resize_height=config.cv_resize_height,
        )
        self.memory = MemoryManager()
        self.planner = Planner(
            llm_client=self.llm,
            cv_output_dir=config.cv_output_dir,
            cv_resize_height=config.cv_resize_height,
        )
        self.oracle_pre = OraclePre(llm_client=self.llm)
        self.safety = SafetyInterceptor(high_risk_keywords=config.high_risk_keywords)
        self.executor = ActionExecutor(
            serial=config.adb_serial,
            screenshot_dir=config.screenshot_dir,
            dump_dir=config.dump_dir,
        )
        self.mapper = ActionMapper(llm_client=self.llm)
        self.oracle_runtime = OracleRuntime(
            dead_loop_threshold=config.dead_loop_threshold,
            screen_variance_threshold=config.screen_variance_threshold,
        )
        self.evaluator = Evaluator(llm_client=self.llm)
        self.replanner = Replanner(
            llm_client=self.llm,
            memory_manager=self.memory,
            dead_end_threshold=config.dead_end_threshold,
        )

        logger.info("AgentLoop 初始化完成(V4), max_steps=%d", config.max_steps)

    def run(self, task: str, dry_run: bool = False):
        logger.info("=" * 60)
        logger.info("开始执行任务: '%s'", task)
        logger.info("=" * 60)

        self.memory.reset_short_term()
        self.oracle_runtime.reset()
        self.replanner.reset()

        step = 0
        task_completed = False
        screen_size = (1080, 1920)
        if not dry_run:
            try:
                screen_size = self.executor.get_screen_size()
                self.executor.stabilize_ui_animations()
            except Exception as exc:
                logger.warning("获取屏幕尺寸失败: %s，使用默认值", exc)

        while not task_completed and step < self.config.max_steps:
            step += 1
            logger.info("=" * 40)
            logger.info("第 %d 步 (最大 %d)", step, self.config.max_steps)
            logger.info("=" * 40)

            try:
                result = self._execute_one_step(
                    task=task,
                    step=step,
                    screen_size=screen_size,
                    dry_run=dry_run,
                )
                task_completed = bool(result.get("task_completed", False))
                if bool(result.get("abort", False)):
                    logger.warning("任务中止: %s", result.get("reason", ""))
                    break
            except KeyboardInterrupt:
                logger.info("用户中断任务")
                break
            except Exception as exc:
                logger.error("步骤 %d 执行异常: %s", step, exc, exc_info=True)
                if step < self.config.max_steps:
                    logger.info("尝试继续执行...")
                    continue
                break

        self.replanner.save_task_experience(task, task_completed)

        if task_completed:
            logger.info("任务成功完成: '%s'", task)
        else:
            logger.warning("任务未完成: '%s' (step=%d/%d)", task, step, self.config.max_steps)

        return task_completed

    def _execute_one_step(
        self,
        task: str,
        step: int,
        screen_size: tuple,
        dry_run: bool,
    ) -> dict:
        logger.info("[Step %d] 阶段1: 环境感知", step)
        if dry_run:
            from Perception.context_builder import UIState

            ui_state = UIState(screen_width=screen_size[0], screen_height=screen_size[1])
        else:
            screenshot_path = self.executor.screenshot(f"step_{step}_before.png")
            dump_path = self.executor.dump_ui(f"step_{step}.xml")
            activity = self.executor.get_current_activity()
            package = self.executor.get_current_package()
            keyboard_visible = self.executor.get_keyboard_visible()
            ui_state = self.perception.perceive(
                screenshot_path=screenshot_path,
                dump_path=dump_path,
                screen_size=screen_size,
                activity_name=activity,
                package_name=package,
                keyboard_visible=keyboard_visible,
            )

        logger.info("[Step %d] 阶段2: 认知规划", step)
        planner_screenshot = str(getattr(ui_state, "screenshot_path", "") or "")
        if not planner_screenshot:
            raise RuntimeError("Planner 需要 screenshot_path，但当前 UIState 未提供")
        plan = self.planner.plan(task, planner_screenshot)
        self._record_step_artifact(
            artifact_kind="PlanResult",
            step=step,
            payload=plan,
        )

        if plan.is_task_complete:
            logger.info("Planner 判断任务已完成")
            return {"task_completed": True}

        action_type = str(plan.requested_action_type or "").strip().lower()
        if action_type not in ACTION_TYPES:
            failure_reason = f"planner_invalid_action_type:{action_type}"
            self.memory.short_term.add_failure(failure_reason)
            self.memory.short_term.add_step(
                {
                    "step": step,
                    "goal": to_plain_dict(plan.goal),
                    "result": failure_reason,
                    "from_experience": bool(plan.from_experience),
                }
            )
            return {"task_completed": False}

        self.memory.short_term.current_subgoal = plan.goal.summary
        logger.info("本步目标: '%s' (action=%s)", plan.goal.summary, plan.requested_action_type)

        contract = self.oracle_pre.generate_contract(
            plan=plan,
            ui_state=ui_state,
            task_hint=task,
            step=step,
        )
        self._record_step_artifact(
            artifact_kind="StepContract",
            step=step,
            payload=contract,
        )

        self._record_step_artifact(
            artifact_kind="UIState",
            step=step,
            payload={
                "phase": "before_action",
                "state": ui_state,
            },
            append=True,
        )

        logger.info("[Step %d] 阶段3: 安全检查", step)
        if not self.safety.check(plan=plan, contract=contract):
            self.memory.short_term.add_step(
                {
                    "step": step,
                    "goal": to_plain_dict(plan.goal),
                    "contract": to_plain_dict(contract),
                    "result": "user_rejected",
                    "from_experience": bool(plan.from_experience),
                }
            )
            return {"task_completed": False, "abort": True, "reason": "用户拒绝高风险操作"}

        logger.info("[Step %d] 阶段4: 动作映射", step)
        try:
            action = self.mapper.map_action(plan=plan, ui_state=ui_state, contract=contract)
        except MappingFailure as exc:
            logger.warning("动作映射失败: %s", exc)
            eval_result = self._evaluation_from_mapper_failure(exc)
            self._record_step_artifact(
                artifact_kind="StepEvaluation",
                step=step,
                payload={
                    "phase": "mapper_failure",
                    "value": eval_result,
                },
                append=True,
            )
            decision = self.replanner.handle_failure(
                subgoal_description=plan.goal.summary,
                evaluation=eval_result,
                ui_state=ui_state,
            )
            self._record_step_artifact(
                artifact_kind="ReplanDecision",
                step=step,
                payload={
                    "phase": "mapper_failure",
                    "value": decision,
                },
                append=True,
            )
            self.memory.short_term.add_step(
                {
                    "step": step,
                    "goal": to_plain_dict(plan.goal),
                    "contract": to_plain_dict(contract),
                    "evaluation": to_plain_dict(eval_result),
                    "result": f"failed_mapper:{exc.reason_code}",
                    "replan_decision": decision.action,
                    "from_experience": bool(plan.from_experience),
                }
            )
            if decision.action == "back" and not dry_run:
                self._execute_adaptive_back(
                    reference_activity=str(getattr(ui_state, "activity_name", "") or ""),
                    reference_package=str(getattr(ui_state, "package_name", "") or ""),
                )
            if decision.action == "abort":
                return {"task_completed": False, "abort": True, "reason": decision.reason}
            return {"task_completed": False}

        logger.info("映射动作: type=%s, params=%s", action.type, action.params)
        self._record_step_artifact(
            artifact_kind="action",
            step=step,
            payload=action,
        )
        self._record_step_artifact(
            artifact_kind="ResolvedAction",
            step=step,
            payload=action,
        )

        pre_guard = self.oracle_runtime.pre_guard(action=action, contract=contract, ui_state=ui_state)
        self._record_step_artifact(
            artifact_kind="GuardResult",
            step=step,
            payload={
                "phase": "pre",
                "value": pre_guard,
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="ObservationFact",
            step=step,
            payload={
                "phase": "pre",
                "facts": list(pre_guard.observations or []),
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="Assessment",
            step=step,
            payload={
                "phase": "pre",
                "items": list(pre_guard.assessments or []),
            },
            append=True,
        )
        if not pre_guard.allowed:
            logger.warning("pre-guard 阻断执行: assessments=%d", len(pre_guard.assessments))
            eval_result = self._evaluation_from_pre_guard(pre_guard)
            self._record_step_artifact(
                artifact_kind="StepEvaluation",
                step=step,
                payload={
                    "phase": "pre_guard_blocked",
                    "value": eval_result,
                },
                append=True,
            )
            decision = self.replanner.handle_failure(
                subgoal_description=plan.goal.summary,
                evaluation=eval_result,
                ui_state=ui_state,
            )
            self._record_step_artifact(
                artifact_kind="ReplanDecision",
                step=step,
                payload={
                    "phase": "pre_guard_blocked",
                    "value": decision,
                },
                append=True,
            )

            self.memory.short_term.add_step(
                {
                    "step": step,
                    "goal": to_plain_dict(plan.goal),
                    "contract": to_plain_dict(contract),
                    "action": to_plain_dict(action),
                    "evaluation": to_plain_dict(eval_result),
                    "result": f"failed_pre_guard:{eval_result.decision}",
                    "replan_decision": decision.action,
                    "from_experience": bool(plan.from_experience),
                }
            )

            if decision.action == "back" and not dry_run:
                self._execute_adaptive_back(
                    reference_activity=str(getattr(ui_state, "activity_name", "") or ""),
                    reference_package=str(getattr(ui_state, "package_name", "") or ""),
                )
            if decision.action == "abort":
                return {"task_completed": False, "abort": True, "reason": decision.reason}
            return {"task_completed": False}

        logger.info("[Step %d] 阶段5: 执行动作", step)
        old_activity = str(getattr(ui_state, "activity_name", "") or "")
        old_package = str(getattr(ui_state, "package_name", "") or "")

        if dry_run:
            logger.info("[DRY RUN] 跳过执行动作: %s", action.description)
        else:
            self._execute_action(action, current_state=ui_state)

        self.oracle_runtime.record_action(action)
        self.memory.short_term.add_action(to_plain_dict(action))

        if not dry_run:
            time.sleep(3.0)

        logger.info("[Step %d] 阶段6: 执行后感知 + post-guard", step)
        if dry_run:
            from Perception.context_builder import UIState

            new_state = UIState(screen_width=screen_size[0], screen_height=screen_size[1])
            new_screenshot = ""
        else:
            new_screenshot = self.executor.screenshot(f"step_{step}_after.png")
            new_dump = self.executor.dump_ui(f"step_{step}_after.xml")
            new_activity = self.executor.get_current_activity()
            new_package = self.executor.get_current_package()
            new_keyboard_visible = self.executor.get_keyboard_visible()
            new_state = self.perception.perceive(
                screenshot_path=new_screenshot,
                dump_path=new_dump,
                screen_size=screen_size,
                activity_name=new_activity,
                package_name=new_package,
                keyboard_visible=new_keyboard_visible,
            )

        post_guard = self.oracle_runtime.post_guard(
            action=action,
            contract=contract,
            old_state=ui_state,
            new_state=new_state,
            screenshot_path=new_screenshot,
        )
        self._record_step_artifact(
            artifact_kind="GuardResult",
            step=step,
            payload={
                "phase": "post",
                "value": post_guard,
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="ObservationFact",
            step=step,
            payload={
                "phase": "post",
                "facts": list(post_guard.observations or []),
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="Assessment",
            step=step,
            payload={
                "phase": "post",
                "items": list(post_guard.assessments or []),
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="UIState",
            step=step,
            payload={
                "phase": "after_action",
                "state": new_state,
            },
            append=True,
        )

        logger.info("[Step %d] 阶段7: 事后评估", step)
        evaluation = self.evaluator.evaluate(
            subgoal_description=plan.goal.summary,
            contract=contract,
            old_state=ui_state,
            new_state=new_state,
            action=action,
            post_guard=post_guard,
        )

        if self._should_observe_again(evaluation=evaluation) and not dry_run:
            delay_ms = 1800
            params = evaluation.recommended_action.params if evaluation.recommended_action else None
            if params and params.observe_delay_ms:
                delay_ms = max(400, int(params.observe_delay_ms))

            logger.info("评估建议再次观测，等待 %dms", delay_ms)
            time.sleep(delay_ms / 1000.0)

            new_screenshot_2 = self.executor.screenshot(f"step_{step}_after2.png")
            new_dump_2 = self.executor.dump_ui(f"step_{step}_after2.xml")
            new_activity_2 = self.executor.get_current_activity()
            new_package_2 = self.executor.get_current_package()
            new_keyboard_visible_2 = self.executor.get_keyboard_visible()
            new_state_2 = self.perception.perceive(
                screenshot_path=new_screenshot_2,
                dump_path=new_dump_2,
                screen_size=screen_size,
                activity_name=new_activity_2,
                package_name=new_package_2,
                keyboard_visible=new_keyboard_visible_2,
            )

            post_guard_2 = self.oracle_runtime.post_guard(
                action=action,
                contract=contract,
                old_state=ui_state,
                new_state=new_state_2,
                screenshot_path=new_screenshot_2,
            )
            evaluation = self.evaluator.evaluate(
                subgoal_description=plan.goal.summary,
                contract=contract,
                old_state=ui_state,
                new_state=new_state_2,
                action=action,
                post_guard=post_guard_2,
            )
            self._record_step_artifact(
                artifact_kind="GuardResult",
                step=step,
                payload={
                    "phase": "post_observe_again",
                    "value": post_guard_2,
                },
                append=True,
            )
            self._record_step_artifact(
                artifact_kind="ObservationFact",
                step=step,
                payload={
                    "phase": "post_observe_again",
                    "facts": list(post_guard_2.observations or []),
                },
                append=True,
            )
            self._record_step_artifact(
                artifact_kind="Assessment",
                step=step,
                payload={
                    "phase": "post_observe_again",
                    "items": list(post_guard_2.assessments or []),
                },
                append=True,
            )
            self._record_step_artifact(
                artifact_kind="UIState",
                step=step,
                payload={
                    "phase": "after_observe_again",
                    "state": new_state_2,
                },
                append=True,
            )
            new_state = new_state_2

        self._record_step_artifact(
            artifact_kind="StepEvaluation",
            step=step,
            payload={
                "phase": "post",
                "value": evaluation,
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="ObservationFact",
            step=step,
            payload={
                "phase": "evaluation",
                "facts": list(evaluation.observations or []),
            },
            append=True,
        )
        self._record_step_artifact(
            artifact_kind="Assessment",
            step=step,
            payload={
                "phase": "evaluation",
                "items": list(evaluation.assessments or []),
            },
            append=True,
        )

        if evaluation.decision == "success":
            logger.info("子目标完成: '%s'", plan.goal.summary)
            self.replanner.handle_success(task)
            self.memory.short_term.add_step(
                {
                    "step": step,
                    "goal": to_plain_dict(plan.goal),
                    "contract": to_plain_dict(contract),
                    "action": to_plain_dict(action),
                    "evaluation": to_plain_dict(evaluation),
                    "result": "success",
                    "from_experience": bool(plan.from_experience),
                    "activity_after": str(getattr(new_state, "activity_name", "") or ""),
                    "package_after": str(getattr(new_state, "package_name", "") or ""),
                    "keyboard_visible_after": bool(getattr(new_state, "keyboard_visible", False)),
                }
            )
            return {"task_completed": False}

        logger.warning("子目标未通过: decision=%s", evaluation.decision)
        decision = self.replanner.handle_failure(
            subgoal_description=plan.goal.summary,
            evaluation=evaluation,
            ui_state=new_state,
        )
        self._record_step_artifact(
            artifact_kind="ReplanDecision",
            step=step,
            payload={
                "phase": "post_evaluation_failure",
                "value": decision,
            },
            append=True,
        )

        self.memory.short_term.add_step(
            {
                "step": step,
                "goal": to_plain_dict(plan.goal),
                "contract": to_plain_dict(contract),
                "action": to_plain_dict(action),
                "evaluation": to_plain_dict(evaluation),
                "result": f"failed:{evaluation.decision}",
                "replan_decision": decision.action,
                "from_experience": bool(plan.from_experience),
                "activity_after": str(getattr(new_state, "activity_name", "") or ""),
                "package_after": str(getattr(new_state, "package_name", "") or ""),
                "keyboard_visible_after": bool(getattr(new_state, "keyboard_visible", False)),
            }
        )

        if decision.action == "back" and not dry_run:
            logger.info("执行回溯: adaptive back %d 步", decision.back_steps)
            for _ in range(max(1, int(decision.back_steps or 1))):
                before_back_activity = self.executor.get_current_activity()
                before_back_package = self.executor.get_current_package()
                self._execute_adaptive_back(
                    reference_activity=before_back_activity,
                    reference_package=before_back_package,
                )
                time.sleep(0.5)

        if decision.action == "abort":
            logger.warning("任务中止: %s", decision.reason)
            return {"task_completed": False, "abort": True, "reason": decision.reason}

        return {"task_completed": False}

    def _evaluation_from_pre_guard(self, pre_guard) -> StepEvaluation:
        has_hard = any(item.outcome == "fail" and item.severity == "hard" for item in pre_guard.assessments)
        recommended = RecommendedAction(
            kind="backtrack" if has_hard else "replan",
            params=AdviceParams(backtrack_steps=1 if has_hard else None, reason_tags=["pre_guard_blocked"]),
        )
        return StepEvaluation(
            decision="fail",
            confidence=0.95 if has_hard else 0.85,
            recommended_action=recommended,
            assessments=list(pre_guard.assessments or []),
            observations=list(pre_guard.observations or []),
            metrics=[],
            expectation_matches=[],
        )

    def _record_step_artifact(
        self,
        artifact_kind: str,
        step: int,
        payload,
        append: bool = False,
    ) -> None:
        if step <= 0:
            return
        try:
            self.audit.record_step(
                artifact_kind=artifact_kind,
                step=int(step),
                payload=payload,
                append=append,
            )
        except Exception as exc:
            logger.warning("写入 %s 审计记录失败(step=%s): %s", artifact_kind, step, exc)

    def _evaluation_from_mapper_failure(self, error: MappingFailure) -> StepEvaluation:
        assessment = Assessment(
            name="mapper_failure_check",
            source="runtime",
            applies_to="runtime_guard",
            outcome="fail",
            severity="soft",
            reason_code=str(error.reason_code or "mapper_failure"),
            message=str(error.message or "动作映射失败"),
            evidence_refs=[],
            score=None,
            remedy_hint=AdviceParams(reason_tags=["mapper_failure"]),
        )
        return StepEvaluation(
            decision="fail",
            confidence=0.9,
            recommended_action=RecommendedAction(kind="replan", params=AdviceParams(reason_tags=["mapper_failure"])),
            assessments=[assessment],
            observations=[],
            metrics=[],
            expectation_matches=[],
        )

    def _should_observe_again(self, evaluation: StepEvaluation) -> bool:
        if evaluation.decision != "uncertain":
            return False
        rec = evaluation.recommended_action
        return bool(rec and rec.kind == "observe")

    def _execute_action(self, action, current_state=None):
        action_type = str(action.type or "").strip().lower()
        params = action.params or {}

        if action_type == "tap":
            self.executor.tap(int(params.get("x", 0)), int(params.get("y", 0)))
            return

        if action_type == "input":
            x = int(params.get("x", 0))
            y = int(params.get("y", 0))
            text = str(params.get("text") or "")
            if not self._is_input_target_ready(action, current_state):
                self.executor.tap(x, y)
                time.sleep(0.4)
            self.executor.input_text(text)
            return

        if action_type == "swipe":
            self.executor.swipe(
                int(params.get("x", 0)),
                int(params.get("y", 0)),
                int(params.get("x2", 0)),
                int(params.get("y2", 0)),
                int(params.get("duration_ms", 500)),
            )
            return

        if action_type == "back":
            self.executor.back()
            return

        if action_type == "enter":
            self.executor.enter()
            return

        if action_type == "launch_app":
            package = str(params.get("package") or "").strip()
            activity = str(params.get("activity") or "").strip()
            self.executor.launch_app(package=package, activity=activity)
            return

        if action_type == "long_press":
            self.executor.long_press(
                int(params.get("x", 0)),
                int(params.get("y", 0)),
                int(params.get("duration_ms", 900)),
            )
            return

        logger.warning("未知动作类型: %s，尝试回退 tap", action_type)
        self.executor.tap(int(params.get("x", 0)), int(params.get("y", 0)))

    def _execute_adaptive_back(self, reference_activity: str = "", reference_package: str = "") -> None:
        self.executor.back()
        time.sleep(0.6)
        try:
            current_activity = self.executor.get_current_activity()
            current_package = self.executor.get_current_package()
        except Exception as exc:
            logger.warning("回退后状态探测失败: %s，补发第二次 Back", exc)
            self.executor.back()
            return

        activity_changed = bool(reference_activity) and current_activity != reference_activity
        package_changed = bool(reference_package) and current_package != reference_package
        if activity_changed or package_changed:
            return

        self.executor.back()

    def _is_input_target_ready(self, action, ui_state) -> bool:
        if ui_state is None:
            return False
        if not bool(getattr(ui_state, "keyboard_visible", False)):
            return False

        target_widget_id = None
        target_bounds = None
        if action.target and action.target.resolved:
            target_widget_id = action.target.resolved.widget_id
            target_bounds = action.target.resolved.bounds

        widgets = getattr(ui_state, "widgets", [])
        for widget in widgets:
            same_target = False
            if target_widget_id is not None and int(target_widget_id) == int(getattr(widget, "widget_id", -1)):
                same_target = True
            elif target_bounds is not None and self._bounds_overlap(target_bounds, getattr(widget, "bounds", (0, 0, 0, 0))):
                same_target = True

            if same_target and bool(getattr(widget, "focused", False)) and bool(
                getattr(widget, "editable", False) or getattr(widget, "focusable", False)
            ):
                return True

        return any(
            bool(getattr(widget, "focused", False))
            and bool(getattr(widget, "editable", False) or getattr(widget, "focusable", False))
            for widget in widgets
        )

    def _bounds_overlap(self, box_a: tuple, box_b: tuple) -> bool:
        if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
            return False
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
