"""
事中 Oracle 监控模块
在动作执行期间提供"条件反射"式的实时监控
- 死循环检测：连续相同动作熔断
- 异常兜底：白屏/黑屏/崩溃检测
- 跳变校验：与事前预期比对
"""
import logging
from typing import List, Dict, Optional

import cv2
import numpy as np

from Execution.action_mapper import Action
from Planning.oracle_pre import PreConstraints

logger = logging.getLogger(__name__)


class OracleRuntime:
    """
    事中 Oracle 监控器
    在动作执行的前后提供实时安全检查
    """

    def __init__(self, dead_loop_threshold: int = 3, screen_variance_threshold: float = 10.0):
        """
        :param dead_loop_threshold: 连续相同动作次数阈值，超过则熔断
        :param screen_variance_threshold: 屏幕像素方差阈值（低于则判定为白屏/黑屏）
        """
        self.dead_loop_threshold = dead_loop_threshold
        self.screen_variance_threshold = screen_variance_threshold
        self.action_history: List[Dict] = []  # 记录最近的动作
        logger.info(
            "OracleRuntime 初始化完成: dead_loop_threshold=%d, screen_var_threshold=%.1f",
            dead_loop_threshold, screen_variance_threshold,
        )

    def pre_execution_check(self, action: Action) -> dict:
        """
        动作执行前检查
        :param action: 即将执行的动作
        :return: {"allow": True/False, "reason": "..."}
        """
        # 死循环检测
        if self._is_dead_loop(action):
            logger.warning(
                "🔄 死循环检测触发! 连续 %d 次相同动作: %s at (%d, %d)",
                self.dead_loop_threshold, action.action_type, action.x, action.y,
            )
            return {
                "allow": False,
                "reason": f"连续{self.dead_loop_threshold}次执行相同动作({action.action_type}@({action.x},{action.y}))，判定为死循环，需要重规划",
                "error_type": "dead_loop",
            }

        logger.debug("执行前检查通过: %s at (%d, %d)", action.action_type, action.x, action.y)
        return {"allow": True, "reason": ""}

    def post_execution_check(
        self,
        screenshot_path: str,
        old_activity: str,
        new_activity: str,
        constraints: Optional[PreConstraints] = None,
    ) -> dict:
        """
        动作执行后立即检查
        :param screenshot_path: 执行后的截图路径
        :param old_activity: 执行前的 Activity
        :param new_activity: 执行后的 Activity
        :param constraints: 事前约束（跳变预期）
        :return: {"ok": True/False, "issues": [...], "action_needed": "none/back/replan"}
        """
        issues = []
        action_needed = "none"

        # 1. 异常兜底：检测白屏/黑屏
        if screenshot_path:
            screen_issue = self._check_screen_anomaly(screenshot_path)
            if screen_issue:
                issues.append(screen_issue)
                action_needed = "replan"

        # 2. 跳变校验
        if constraints and old_activity != new_activity:
            transition_issue = self._check_transition(
                old_activity, new_activity, constraints
            )
            if transition_issue:
                issues.append(transition_issue)
                action_needed = "back"

        if issues:
            logger.warning("事中检查发现问题: %s, 建议: %s", issues, action_needed)
        else:
            logger.debug("事中检查通过")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "action_needed": action_needed,
        }

    def record_action(self, action: Action):
        """记录已执行的动作（用于死循环检测）"""
        self.action_history.append({
            "action_type": action.action_type,
            "x": action.x,
            "y": action.y,
            "text": action.text,
        })
        # 只保留最近 N 条
        if len(self.action_history) > self.dead_loop_threshold * 2:
            self.action_history = self.action_history[-self.dead_loop_threshold * 2:]

    def reset(self):
        """重置历史记录"""
        self.action_history.clear()
        logger.debug("事中 Oracle 历史已重置")

    def _is_dead_loop(self, action: Action) -> bool:
        """
        检测是否陷入死循环：
        连续 N 次对同一坐标/元素执行相同动作
        """
        if len(self.action_history) < self.dead_loop_threshold:
            return False

        recent = self.action_history[-self.dead_loop_threshold:]
        current = {
            "action_type": action.action_type,
            "x": action.x,
            "y": action.y,
        }

        # 检查最近 N 次是否全部相同
        for past in recent:
            if (
                past["action_type"] != current["action_type"]
                or abs(past["x"] - current["x"]) > 10
                or abs(past["y"] - current["y"]) > 10
            ):
                return False
        return True

    def _check_screen_anomaly(self, screenshot_path: str) -> Optional[str]:
        """
        检测屏幕异常（白屏、黑屏、崩溃）
        通过计算像素 HSV 方差来判断
        """
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return "无法读取截图文件"

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            total_var = (
                np.var(hsv[:, :, 0])
                + np.var(hsv[:, :, 1])
                + np.var(hsv[:, :, 2])
            )

            if total_var < self.screen_variance_threshold:
                logger.warning("检测到屏幕异常! 像素方差=%.2f (阈值=%.2f), 可能白屏/黑屏", total_var, self.screen_variance_threshold)
                return f"屏幕异常: 像素方差极低({total_var:.2f}), 可能App崩溃/白屏/黑屏"

            return None
        except Exception as e:
            logger.error("屏幕异常检测失败: %s", e)
            return None

    def _check_transition(
        self,
        old_activity: str,
        new_activity: str,
        constraints: PreConstraints,
    ) -> Optional[str]:
        """
        跳变校验：比对实际页面跳转与事前预期
        """
        expected_type = constraints.transition_type

        if expected_type == "none" and old_activity != new_activity:
            # 预期不跳转但发生了跳转 → 非预期剧变
            logger.warning(
                "非预期页面跳转! 预期=不跳转, 实际: %s -> %s",
                old_activity, new_activity,
            )
            return f"非预期页面跳转: {old_activity} -> {new_activity} (预期不跳转)"

        if expected_type == "partial_refresh" and old_activity != new_activity:
            # 预期局部刷新但跳到新页面
            logger.warning(
                "非预期全页跳转! 预期=局部刷新, 实际: %s -> %s",
                old_activity, new_activity,
            )
            return f"非预期全页跳转: {old_activity} -> {new_activity} (预期局部刷新)"

        if constraints.expected_activity and new_activity:
            if constraints.expected_activity not in new_activity:
                logger.warning(
                    "跳转目标不符! 预期=%s, 实际=%s",
                    constraints.expected_activity, new_activity,
                )
                return f"跳转目标不符: 预期={constraints.expected_activity}, 实际={new_activity}"

        return None
