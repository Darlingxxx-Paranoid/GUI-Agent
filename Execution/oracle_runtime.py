"""
事中 Oracle 监控模块
在动作执行期间提供"条件反射"式的实时监控
- 死循环检测：连续相同动作熔断
- 异常兜底：白屏/黑屏/崩溃检测
- 跳变校验：与事前预期比对
"""
import logging
import re
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
        old_package: str = "",
        new_package: str = "",
        constraints: Optional[PreConstraints] = None,
    ) -> dict:
        """
        动作执行后立即检查
        :param screenshot_path: 执行后的截图路径
        :param old_activity: 执行前的 Activity
        :param new_activity: 执行后的 Activity
        :param old_package: 执行前包名
        :param new_package: 执行后包名
        :param constraints: 事前约束（边界 + 变化证据计划）
        :return: {"ok": bool, "issues": [...], "action_needed": "none/back/replan", "severity": "none/soft/hard"}
        """
        issues = []
        action_needed = "none"
        severity = "none"

        # 1. 异常兜底：检测白屏/黑屏
        if screenshot_path:
            screen_issue = self._check_screen_anomaly(screenshot_path)
            if screen_issue:
                issues.append(screen_issue)
                severity = "hard"
                action_needed = "replan"

        # 2. 跳变与边界校验
        if constraints:
            transition_issue = self._check_transition(
                old_activity=old_activity,
                new_activity=new_activity,
                old_package=old_package,
                new_package=new_package,
                constraints=constraints,
            )
            if transition_issue:
                issues.append(str(transition_issue.get("message") or ""))
                issue_severity = str(transition_issue.get("severity") or "soft")
                if issue_severity == "hard":
                    severity = "hard"
                    if action_needed != "replan":
                        action_needed = str(transition_issue.get("action_needed") or "back")
                elif severity == "none":
                    severity = "soft"

        if issues:
            if severity == "hard":
                logger.warning("事中检查发现硬问题: %s, 建议: %s", issues, action_needed)
            else:
                logger.info("事中检查发现软问题(继续评估): %s", issues)
        else:
            logger.debug("事中检查通过")

        return {
            "ok": severity != "hard",
            "issues": issues,
            "action_needed": action_needed,
            "severity": severity,
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
        old_package: str,
        new_package: str,
        constraints: PreConstraints,
    ) -> Optional[Dict[str, str]]:
        """
        跳变校验：
        - 仅将边界违规作为硬问题（阻断）
        - 变化范围偏离作为软问题（继续交由评估器判断）
        """
        success_plan = constraints.success_evidence_plan or {}
        boundary = constraints.boundary_constraints or {}

        expected_scope = str(success_plan.get("expected_change_scope") or "local").strip().lower()
        must_stay_in_app = bool(boundary.get("must_stay_in_app", False))
        expected_package = str(boundary.get("expected_package") or "").strip()
        forbidden_packages = boundary.get("forbidden_packages") or []
        if not isinstance(forbidden_packages, list):
            forbidden_packages = []
        forbidden_packages = [str(p).strip() for p in forbidden_packages if str(p).strip()]
        expected_activity_contains = str(boundary.get("expected_activity_contains") or "").strip()
        mismatch_severity = str(boundary.get("package_mismatch_severity") or "soft").strip().lower()
        if mismatch_severity not in {"soft", "hard"}:
            mismatch_severity = "soft"
        related_tokens = boundary.get("related_package_tokens") or []
        if not isinstance(related_tokens, list):
            related_tokens = []
        related_tokens = [str(v).strip().lower() for v in related_tokens if str(v).strip()]

        if new_package and new_package in forbidden_packages:
            message = f"进入禁止包名: {new_package}"
            logger.warning(message)
            return {"severity": "hard", "message": message, "action_needed": "back"}

        # 若已进入显式期望包，视为有效目标变化，不应被 must_stay 规则先行误拦截。
        entered_expected_package = bool(
            expected_package
            and new_package
            and new_package == expected_package
            and old_package
            and old_package != new_package
        )

        if must_stay_in_app and old_package and new_package and old_package != new_package and not entered_expected_package:
            relation = self._classify_package_relation(
                reference_package=old_package,
                new_package=new_package,
                related_tokens=related_tokens,
            )
            if relation == "related":
                message = f"must_stay_in_app package drift(soft, related): {old_package} -> {new_package}"
                logger.info(message)
                return {"severity": "soft", "message": message, "action_needed": "none"}
            if mismatch_severity == "hard":
                message = f"must_stay_in_app violated(hard, {relation}): {old_package} -> {new_package}"
                logger.warning(message)
                return {"severity": "hard", "message": message, "action_needed": "back"}
            message = f"must_stay_in_app drift(soft, {relation}): {old_package} -> {new_package}"
            logger.info(message)
            return {"severity": "soft", "message": message, "action_needed": "none"}

        if expected_package and new_package and expected_package != new_package:
            relation = self._classify_package_relation(
                reference_package=expected_package,
                new_package=new_package,
                related_tokens=related_tokens,
            )
            if relation == "related":
                message = f"expected_package drift(soft, related): expected={expected_package}, actual={new_package}"
                logger.info(message)
                return {"severity": "soft", "message": message, "action_needed": "none"}
            if mismatch_severity == "hard":
                message = f"expected_package mismatch(hard, {relation}): expected={expected_package}, actual={new_package}"
                logger.warning(message)
                return {"severity": "hard", "message": message, "action_needed": "back"}
            message = f"expected_package mismatch(soft, {relation}): expected={expected_package}, actual={new_package}"
            logger.info(message)
            return {"severity": "soft", "message": message, "action_needed": "none"}

        if expected_activity_contains and new_activity and expected_activity_contains not in new_activity:
            message = f"跳转目标不符: 预期包含={expected_activity_contains}, 实际={new_activity}"
            logger.warning(message)
            return {"severity": "hard", "message": message, "action_needed": "back"}

        if old_activity == new_activity and old_package == new_package:
            return None

        if expected_scope in {"anchor", "local"}:
            message = (
                f"变化范围偏离预期(scope={expected_scope}): "
                f"activity {old_activity} -> {new_activity}, package {old_package} -> {new_package}"
            )
            logger.info(
                "变化范围偏离预期(软问题): 预期scope=%s, activity=%s -> %s, package=%s -> %s",
                expected_scope, old_activity, new_activity,
                old_package, new_package,
            )
            return {"severity": "soft", "message": message, "action_needed": "none"}

        return None

    def _classify_package_relation(self, reference_package: str, new_package: str, related_tokens: List[str]) -> str:
        ref = str(reference_package or "").strip().lower()
        new = str(new_package or "").strip().lower()
        if not ref or not new:
            return "unknown"
        if ref == new:
            return "same"

        normalized_tokens = []
        for token in related_tokens:
            value = str(token or "").strip().lower()
            if value and value not in normalized_tokens:
                normalized_tokens.append(value)
        if not normalized_tokens:
            normalized_tokens = self._extract_package_tokens(ref)

        for token in normalized_tokens:
            if token in ref and token in new:
                return "related"
        return "unrelated"

    def _extract_package_tokens(self, package_name: str) -> List[str]:
        pkg = str(package_name or "").strip().lower()
        if not pkg:
            return []
        ignored = {
            "com",
            "org",
            "net",
            "android",
            "google",
            "apps",
            "app",
            "activity",
        }
        tokens: List[str] = []
        normalized = pkg.replace("-", ".").replace("_", ".")
        for token in normalized.split("."):
            for t in self._expand_package_token(token):
                if not t or t in ignored or len(t) <= 2:
                    continue
                if t not in tokens:
                    tokens.append(t)
        return tokens[:8]

    def _expand_package_token(self, token: str) -> List[str]:
        raw = str(token or "").strip().lower()
        if not raw:
            return []

        variants: List[str] = [raw]
        without_tail_digits = re.sub(r"\d+$", "", raw)
        if without_tail_digits and without_tail_digits != raw:
            variants.append(without_tail_digits)

        for prefix in ("google", "android"):
            if raw.startswith(prefix) and len(raw) > len(prefix) + 2:
                variants.append(raw[len(prefix):])

        alpha_parts = re.findall(r"[a-z]+", raw)
        if len(alpha_parts) >= 2:
            merged_alpha = "".join(alpha_parts)
            if merged_alpha:
                variants.append(merged_alpha)
            for part in alpha_parts:
                if len(part) > 2:
                    variants.append(part)

        deduped: List[str] = []
        for v in variants:
            value = str(v or "").strip().lower()
            if value and value not in deduped:
                deduped.append(value)
        return deduped
