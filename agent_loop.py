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

        if not dry_run:
            try:
                screen_size = self.executor.get_screen_size()
            except Exception as e:
                logger.warning("获取屏幕尺寸失败: %s, 使用默认值", e)

        while not task_completed and step < self.config.max_steps:
            step += 1
            logger.info("=" * 40)
            logger.info("第 %d 步 (最大 %d)", step, self.config.max_steps)
            logger.info("=" * 40)

            try:
                result = self._execute_one_step(
                    task, step, screen_size, dry_run
                )
                task_completed = result.get("task_completed", False)

                if result.get("abort", False):
                    logger.warning("任务中止: %s", result.get("reason", ""))
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

        # 任务结束：沉淀经验
        self.replanner.save_task_experience(task, task_completed)

        if task_completed:
            logger.info("✅ 任务成功完成: '%s'", task)
        else:
            logger.warning("❌ 任务未完成: '%s' (步数=%d/%d)", task, step, self.config.max_steps)

        return task_completed

    def _execute_one_step(
        self, task: str, step: int, screen_size: tuple, dry_run: bool
    ) -> dict:
        """
        执行单步 ReAct 循环
        :return: {"task_completed": bool, "abort": bool, "reason": str}
        """

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
        logger.info("[Step %d] 阶段2: 认知规划", step)
        plan_result = self.planner.plan(task, ui_state)

        # 检查任务是否已完成
        if plan_result.is_task_complete:
            logger.info("LLM 判断任务已完成")
            return {"task_completed": True}

        subgoal = plan_result.subgoal
        self.memory.short_term.current_subgoal = subgoal.description
        logger.info("子目标: '%s' (action=%s)", subgoal.description, subgoal.action_type)

        # 生成事前约束
        constraints = self.oracle_pre.generate_constraints(subgoal, ui_state)
        logger.info("事前约束: transition=%s", constraints.transition_type)

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

        if dry_run:
            logger.info("[DRY RUN] 跳过实际执行: %s", action.description)
        else:
            self._execute_action(action, ui_state)

        # 记录动作
        self.oracle_runtime.record_action(action)
        self.memory.short_term.add_action(action.to_dict())

        # 等待 UI 响应
        if not dry_run:
            import time as _time
            _time.sleep(3)

        # ========================================
        # 6. 再次感知 + 事中检查(执行后)
        # ========================================
        logger.info("[Step %d] 阶段6: 执行后感知 + 事中检查", step)
        if dry_run:
            from Perception.context_builder import UIState
            new_state = UIState(screen_width=screen_size[0], screen_height=screen_size[1])
            new_activity = ""
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
                constraints=constraints,
            )

            if not post_check["ok"]:
                action_needed = post_check["action_needed"]
                logger.warning("事中检查异常: %s, 建议: %s", post_check["issues"], action_needed)

                if action_needed == "back":
                    logger.info("执行拉回操作: Back")
                    self.executor.back()
                    return {"task_completed": False}

        # ========================================
        # 7. 事后评估
        # ========================================
        logger.info("[Step %d] 阶段7: 事后评估", step)
        eval_result = self.evaluator.evaluate(
            subgoal_description=subgoal.description,
            constraints=constraints,
            old_state=ui_state,
            new_state=new_state,
            action=action,
        )

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
                logger.info("执行回溯: Back %d 步", decision.back_steps)
                for _ in range(decision.back_steps):
                    self.executor.back()
                    import time as _time
                    _time.sleep(0.5)

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
