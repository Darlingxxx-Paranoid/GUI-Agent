"""
ReAct 主循环编排器
协调 感知→规划→执行→评估 的完整 Agent 循环
"""
import time
import logging

from config import AgentConfig
from utils.llm_client import LLMClient

from Perception.perception_manager import PerceptionManager
from Planning.planner import Planner
from Planning.oracle_pre import OraclePre
from Planning.safety_interceptor import SafetyInterceptor
from Execution.action_executor import ActionExecutor
from Execution.action_mapper import ActionMapper
from Execution.oracle_runtime import OracleRuntime
from Evaluation.evaluator import Evaluator
from Evaluation.replanner import Replanner
from Memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    基于 Oracle 反馈驱动的 GUI Agent 主循环
    实现 ReAct 框架：Observe → Think → Act → Evaluate → Loop

    每一轮循环：
    1. 截屏 + Dump → Perception 感知
    2. Planning 规划子目标 + 事前约束
    3. Safety 拦截检查
    4. ActionMapper 生成动作 + Runtime Oracle 事中检查
    5. ActionExecutor 执行动作
    6. 再次截屏 + 感知
    7. Evaluator 事后评估
    8. 若失败 → Replanner 重规划 / Back 回溯
    9. 若成功 → 继续下一子目标
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        config.ensure_dirs()

        # 初始化 LLM 客户端
        self.llm = LLMClient(
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout,
        )

        # 初始化各模块
        self.perception = PerceptionManager(
            cv_output_dir=config.cv_output_dir,
            resize_height=config.cv_resize_height,
        )
        self.memory = MemoryManager(
            experience_store_path=config.long_term_memory_file,
            similarity_threshold=config.experience_similarity_threshold,
        )
        self.planner = Planner(llm_client=self.llm, memory_manager=self.memory)
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

        logger.info("AgentLoop 初始化完成, 最大步数=%d", config.max_steps)

    def run(self, task: str, dry_run: bool = False):
        """
        执行任务的主循环
        :param task: 任务描述（自然语言）
        :param dry_run: 干跑模式，不实际执行 ADB 命令
        """
        logger.info("=" * 60)
        logger.info("开始执行任务: '%s'", task)
        logger.info("=" * 60)

        self.memory.reset_short_term()
        self.oracle_runtime.reset()
        self.replanner.reset()

        step = 0
        task_completed = False
        screen_size = (1080, 1920)
        task_start_ts = time.time()
        max_task_seconds = int(getattr(self.config, "max_task_seconds", 0) or 0)
        task_deadline_ts = (task_start_ts + max_task_seconds) if max_task_seconds > 0 else None
        self.llm.set_deadline(task_deadline_ts)

        try:
            if not dry_run:
                try:
                    screen_size = self.executor.get_screen_size()
                    self.executor.stabilize_ui_animations()
                except Exception as e:
                    logger.warning("获取屏幕尺寸失败: %s, 使用默认值", e)

            while not task_completed and step < self.config.max_steps:
                try:
                    self._raise_if_task_timeout(
                        task_start_ts=task_start_ts,
                        max_task_seconds=max_task_seconds,
                        stage="loop_boundary",
                    )
                except TimeoutError as e:
                    logger.warning("%s，跳过该任务", str(e))
                    break

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
                        task_start_ts=task_start_ts,
                        max_task_seconds=max_task_seconds,
                    )
                    task_completed = result.get("task_completed", False)

                    if result.get("abort", False):
                        logger.warning("任务中止: %s", result.get("reason", ""))
                        break

                except TimeoutError as e:
                    logger.warning("%s，跳过该任务", str(e))
                    break
                except KeyboardInterrupt:
                    logger.info("用户中断任务")
                    break
                except Exception as e:
                    logger.error("步骤 %d 执行异常: %s", step, e, exc_info=True)
                    # 尝试恢复
                    if step < self.config.max_steps:
                        logger.info("尝试继续执行...")
                        continue
                    break
        finally:
            self.llm.set_deadline(None)

        # 任务结束：沉淀经验
        self.replanner.save_task_experience(task, task_completed)

        if task_completed:
            logger.info("✅ 任务成功完成: '%s'", task)
        else:
            logger.warning("❌ 任务未完成: '%s' (步数=%d/%d)", task, step, self.config.max_steps)

        return task_completed

    def _execute_one_step(
        self,
        task: str,
        step: int,
        screen_size: tuple,
        dry_run: bool,
        task_start_ts: float,
        max_task_seconds: int,
    ) -> dict:
        """
        执行单步 ReAct 循环
        :return: {"task_completed": bool, "abort": bool, "reason": str}
        """

        self._raise_if_task_timeout(
            task_start_ts=task_start_ts,
            max_task_seconds=max_task_seconds,
            stage=f"step_{step}_start",
        )

        # ========================================
        # 1. 感知：截屏 + Dump → UIState
        # ========================================
        logger.info("[Step %d] 阶段1: 环境感知", step)
        if dry_run:
            from Perception.context_builder import UIState
            ui_state = UIState(screen_width=screen_size[0], screen_height=screen_size[1])
            logger.info("[DRY RUN] 使用空 UIState")
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

        # ========================================
        # 2. 规划：生成子目标 + 事前约束
        # ========================================
        self._raise_if_task_timeout(
            task_start_ts=task_start_ts,
            max_task_seconds=max_task_seconds,
            stage=f"step_{step}_planning",
        )
        logger.info("[Step %d] 阶段2: 认知规划", step)
        plan_result = self.planner.plan(task, ui_state)

        # 检查任务是否已完成
        if plan_result.is_task_complete:
            logger.info("LLM 判断任务已完成")
            return {"task_completed": True}

        subgoal = plan_result.subgoal
        action_type = str(getattr(subgoal, "action_type", "") or "").strip().lower()
        if action_type in {"", "none", "unknown", "null"}:
            failure_reason = (
                "planner_invalid_action: "
                f"description='{(subgoal.description or '').strip()[:80]}'"
            )
            logger.warning("规划结果动作类型无效，跳过当前步骤并等待下一轮规划: %s", failure_reason)
            self.memory.short_term.add_failure(failure_reason)
            self.memory.short_term.add_step({
                "step": step,
                "subgoal": subgoal.description,
                "result": failure_reason,
                "from_experience": bool(getattr(subgoal, "from_experience", False)),
            })
            return {"task_completed": False}
        subgoal.action_type = action_type

        self.memory.short_term.current_subgoal = subgoal.description
        logger.info("子目标: '%s' (action=%s)", subgoal.description, subgoal.action_type)

        # 生成事前约束
        constraints = self.oracle_pre.generate_constraints(
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task,
        )
        logger.info("事前约束已生成")

        # ========================================
        # 3. 安全拦截
        # ========================================
        logger.info("[Step %d] 阶段3: 安全检查", step)
        if not self.safety.check(subgoal):
            logger.info("用户拒绝执行高风险操作, 跳过当前子目标")
            self.memory.short_term.add_step({
                "step": step,
                "subgoal": subgoal.description,
                "result": "user_rejected",
            })
            return {"task_completed": False, "abort": True, "reason": "用户拒绝高风险操作"}

        # ========================================
        # 4. 动作映射 + 事中检查(执行前)
        # ========================================
        logger.info("[Step %d] 阶段4: 动作映射", step)
        action = self.mapper.map_action(subgoal, ui_state)
        logger.info("映射动作: %s at (%d, %d)", action.action_type, action.x, action.y)

        if constraints and getattr(constraints, "action_anchor", None) is not None:
            anchor = constraints.action_anchor
            if isinstance(anchor, dict):
                bounds = tuple(getattr(action, "target_bounds", ()) or (0, 0, 0, 0))
                if len(bounds) == 4 and bounds != (0, 0, 0, 0):
                    cx = int((bounds[0] + bounds[2]) / 2)
                    cy = int((bounds[1] + bounds[3]) / 2)
                    anchor["target_bounds_before"] = [int(v) for v in bounds]
                    anchor["target_center_before"] = [cx, cy]
                else:
                    anchor["target_bounds_before"] = [0, 0, 0, 0]
                    anchor["target_center_before"] = [int(getattr(action, "x", 0) or 0), int(getattr(action, "y", 0) or 0)]

        # 事中 Oracle 执行前检查
        pre_check = self.oracle_runtime.pre_execution_check(action)
        if not pre_check["allow"]:
            logger.warning("事中 Oracle 拒绝执行: %s", pre_check["reason"])
            self.memory.short_term.add_failure(pre_check["reason"])
            # 死循环 → 强制重规划
            self.oracle_runtime.reset()
            return {"task_completed": False}

        # ========================================
        # 5. 执行动作
        # ========================================
        logger.info("[Step %d] 阶段5: 执行动作", step)
        old_activity = ui_state.activity_name
        old_package = ui_state.package_name

        if dry_run:
            logger.info("[DRY RUN] 跳过实际执行: %s", action.description)
        else:
            self._execute_action(action, ui_state)

        # 记录动作
        self.oracle_runtime.record_action(action)
        self.memory.short_term.add_action(action.to_dict())

        # 等待 UI 响应
        if not dry_run:
            self._sleep_with_task_timeout(
                seconds=3.0,
                task_start_ts=task_start_ts,
                max_task_seconds=max_task_seconds,
                stage=f"step_{step}_post_action_wait",
            )

        # ========================================
        # 6. 再次感知 + 事中检查(执行后)
        # ========================================
        logger.info("[Step %d] 阶段6: 执行后感知 + 事中检查", step)
        if dry_run:
            from Perception.context_builder import UIState
            new_state = UIState(screen_width=screen_size[0], screen_height=screen_size[1])
            new_activity = ""
            new_package = ""
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

            # 事中 Oracle 执行后检查
            post_check = self.oracle_runtime.post_execution_check(
                screenshot_path=new_screenshot,
                old_activity=old_activity,
                new_activity=new_activity,
                old_package=old_package,
                new_package=new_package,
                constraints=constraints,
            )

            if post_check.get("issues"):
                logger.warning(
                    "事中检查问题(severity=%s): %s, 建议: %s",
                    post_check.get("severity", "none"),
                    post_check.get("issues", []),
                    post_check.get("action_needed", "none"),
                )

            if not post_check["ok"]:
                action_needed = post_check["action_needed"]
                severity = str(post_check.get("severity", "hard"))
                issues = post_check.get("issues", []) or []
                runtime_reason = f"runtime_{severity}: {'; '.join(str(v) for v in issues if v)}"
                self.memory.short_term.add_failure(runtime_reason)

                if action_needed == "back":
                    logger.info("执行自适应拉回操作: Back")
                    self._execute_adaptive_back(
                        reference_activity=new_activity,
                        reference_package=new_package,
                    )
                    return {"task_completed": False}
                if action_needed == "replan":
                    logger.info("Runtime 建议重规划，跳过当前步骤后续评估")
                    return {"task_completed": False}

        # ========================================
        # 7. 事后评估
        # ========================================
        self._raise_if_task_timeout(
            task_start_ts=task_start_ts,
            max_task_seconds=max_task_seconds,
            stage=f"step_{step}_evaluate",
        )
        logger.info("[Step %d] 阶段7: 事后评估", step)
        eval_result = self.evaluator.evaluate(
            subgoal_description=subgoal.description,
            constraints=constraints,
            old_state=ui_state,
            new_state=new_state,
            action=action,
        )

        reason_text = str(getattr(eval_result, "reason", "") or "").lower()
        observe_requested = (
            getattr(eval_result, "suggested_next_action", "") == "observe_again"
            or getattr(eval_result, "needs_more_observation", False)
        )
        # 仅在“低变化证据”类不确定性时触发二次观测；语义争议类不确定性直接交给重规划。
        observe_reason_allowed = any(
            marker in reason_text
            for marker in (
                "low_change_evidence",
                "weak_change_evidence",
                "observe_again_before",
            )
        )
        if observe_requested and observe_reason_allowed and not dry_run:
            remaining = self._remaining_task_seconds(
                task_start_ts=task_start_ts,
                max_task_seconds=max_task_seconds,
            )
            # 二次观测本身需要 sleep + 截图 + dump + 感知，预算不足时直接跳过，避免“观测本身耗尽任务时限”。
            if remaining is not None and remaining < 6.0:
                logger.info(
                    "评估结果不确定，但剩余时限不足(%.1fs)，跳过再次观测",
                    remaining,
                )
            else:
                logger.info("评估结果不确定，等待并再次观测")
                self._sleep_with_task_timeout(
                    seconds=2.0,
                    task_start_ts=task_start_ts,
                    max_task_seconds=max_task_seconds,
                    stage=f"step_{step}_observe_again_wait",
                )

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

                eval_result = self.evaluator.evaluate(
                    subgoal_description=subgoal.description,
                    constraints=constraints,
                    old_state=ui_state,
                    new_state=new_state_2,
                    action=action,
                )
                new_state = new_state_2

        # ========================================
        # 8. 根据评估结果决定下一步
        # ========================================
        if eval_result.success:
            logger.info("✓ 子目标完成: '%s'", subgoal.description)
            self.replanner.handle_success(task)
            self.memory.short_term.add_step({
                "step": step,
                "subgoal": subgoal.description,
                "action": action.to_dict(),
                "result": "success",
                "from_experience": bool(subgoal.from_experience),
                "acceptance_criteria": subgoal.acceptance_criteria,
                "expected_transition": subgoal.expected_transition,
                "eval_reason": eval_result.reason,
                "activity_after": getattr(new_state, "activity_name", ""),
                "package_after": getattr(new_state, "package_name", ""),
                "keyboard_visible_after": bool(getattr(new_state, "keyboard_visible", False)),
            })
        else:
            logger.warning("✗ 子目标失败: '%s', 原因: %s", subgoal.description, eval_result.reason)

            # 重规划决策
            decision = self.replanner.handle_failure(
                subgoal_description=subgoal.description,
                eval_result=eval_result,
                ui_state=new_state,
            )

            self.memory.short_term.add_step({
                "step": step,
                "subgoal": subgoal.description,
                "action": action.to_dict(),
                "result": f"failed: {eval_result.reason}",
                "replan_decision": decision.action,
                "from_experience": bool(subgoal.from_experience),
                "acceptance_criteria": subgoal.acceptance_criteria,
                "expected_transition": subgoal.expected_transition,
                "eval_reason": eval_result.reason,
                "activity_after": getattr(new_state, "activity_name", ""),
                "package_after": getattr(new_state, "package_name", ""),
                "keyboard_visible_after": bool(getattr(new_state, "keyboard_visible", False)),
            })

            if decision.action == "back" and not dry_run:
                logger.info("执行回溯: 自适应 Back %d 步", decision.back_steps)
                for _ in range(decision.back_steps):
                    before_back_activity = self.executor.get_current_activity()
                    before_back_package = self.executor.get_current_package()
                    self._execute_adaptive_back(
                        reference_activity=before_back_activity,
                        reference_package=before_back_package,
                    )
                    self._sleep_with_task_timeout(
                        seconds=0.5,
                        task_start_ts=task_start_ts,
                        max_task_seconds=max_task_seconds,
                        stage=f"step_{step}_back_wait",
                    )

            elif decision.action == "abort":
                logger.warning("任务中止: %s", decision.reason)
                return {"task_completed": False, "abort": True, "reason": decision.reason}

        return {"task_completed": False}

    def _execute_action(self, action, current_state=None):
        """根据 Action 类型执行对应的 ADB 命令"""
        if action.action_type == "tap":
            self.executor.tap(action.x, action.y)

        elif action.action_type == "input":
            # 仅在未就绪时才预点击，避免打断已聚焦输入域
            if not self._is_input_target_ready(action, current_state):
                self.executor.tap(action.x, action.y)
                import time as _time
                _time.sleep(0.5)
            else:
                logger.info("输入目标已聚焦且键盘可见，跳过预点击")
            self.executor.input_text(action.text)

        elif action.action_type == "swipe":
            self.executor.swipe(action.x, action.y, action.x2, action.y2)

        elif action.action_type == "back":
            self.executor.back()

        elif action.action_type == "enter":
            self.executor.enter()

        elif action.action_type == "long_press":
            self.executor.long_press(action.x, action.y)

        elif action.action_type == "noop":
            logger.warning("无效动作: %s", action.description)

        else:
            logger.warning("未知动作类型: %s, 尝试作为 tap 执行", action.action_type)
            self.executor.tap(action.x, action.y)

    def _execute_adaptive_back(self, reference_activity: str = "", reference_package: str = "") -> None:
        """
        自适应 Back:
        1) 先执行一次 Back
        2) 若前台 activity/package 未变化，再补一次 Back
        """
        logger.info("执行自适应回退: 先 Back 1 次")
        self.executor.back()
        time.sleep(0.6)

        try:
            current_activity = self.executor.get_current_activity()
            current_package = self.executor.get_current_package()
        except Exception as e:
            logger.warning("回退后状态探测失败: %s，补发第 2 次 Back", e)
            self.executor.back()
            return

        activity_changed = bool(reference_activity) and current_activity != reference_activity
        package_changed = bool(reference_package) and current_package != reference_package
        if activity_changed or package_changed:
            logger.info(
                "首次 Back 生效: activity %s -> %s, package %s -> %s",
                reference_activity or "unknown",
                current_activity or "unknown",
                reference_package or "unknown",
                current_package or "unknown",
            )
            return

        logger.info("首次 Back 未观察到状态变化，补发第 2 次 Back")
        self.executor.back()

    def _is_input_target_ready(self, action, ui_state) -> bool:
        """判断目标输入域是否已可直接输入。"""
        if ui_state is None:
            return False
        if not getattr(ui_state, "keyboard_visible", False):
            return False

        widgets = getattr(ui_state, "widgets", [])
        target_bounds = tuple(getattr(action, "target_bounds", ()) or (0, 0, 0, 0))

        for widget in widgets:
            same_target = False
            if action.target_resource_id and action.target_resource_id == getattr(widget, "resource_id", ""):
                same_target = True
            elif action.widget_id is not None and action.widget_id == getattr(widget, "widget_id", None):
                same_target = True
            elif target_bounds != (0, 0, 0, 0) and self._bounds_overlap(target_bounds, getattr(widget, "bounds", (0, 0, 0, 0))):
                same_target = True

            if same_target and (getattr(widget, "focused", False) or getattr(widget, "editable", False)):
                return True

        return any(
            getattr(widget, "focused", False) and (getattr(widget, "editable", False) or getattr(widget, "focusable", False))
            for widget in widgets
        )

    def _bounds_overlap(self, box_a: tuple, box_b: tuple) -> bool:
        if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
            return False
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _raise_if_task_timeout(
        self,
        task_start_ts: float,
        max_task_seconds: int,
        stage: str = "",
    ) -> None:
        if max_task_seconds <= 0:
            return
        elapsed = time.time() - task_start_ts
        if elapsed >= max_task_seconds:
            stage_hint = f", stage={stage}" if stage else ""
            raise TimeoutError(
                f"任务超时: {elapsed:.1f}s >= {max_task_seconds}s{stage_hint}"
            )

    def _remaining_task_seconds(
        self,
        task_start_ts: float,
        max_task_seconds: int,
    ):
        if max_task_seconds <= 0:
            return None
        elapsed = time.time() - task_start_ts
        return max(0.0, float(max_task_seconds - elapsed))

    def _sleep_with_task_timeout(
        self,
        seconds: float,
        task_start_ts: float,
        max_task_seconds: int,
        stage: str = "",
    ) -> None:
        duration = max(0.0, float(seconds or 0.0))
        if duration <= 0:
            return
        if max_task_seconds <= 0:
            time.sleep(duration)
            return

        elapsed = time.time() - task_start_ts
        remaining = max_task_seconds - elapsed
        if remaining <= 0:
            self._raise_if_task_timeout(task_start_ts, max_task_seconds, stage=stage)
            return

        sleep_time = min(duration, remaining)
        if sleep_time > 0:
            time.sleep(sleep_time)
        if sleep_time < duration:
            self._raise_if_task_timeout(task_start_ts, max_task_seconds, stage=stage)
