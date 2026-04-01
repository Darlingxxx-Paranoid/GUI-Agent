"""Runtime oracle guards (pre/post) using V3.1 contracts."""

from __future__ import annotations

import logging
from typing import Dict, List

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None
    np = None

from Execution.observation_extractor import ObservationExtractor
from Execution.policy_engine import PolicyEngine
from Oracle.contracts import (
    AdviceParams,
    Assessment,
    GuardResult,
    ObservationFact,
    ResolvedAction,
    StepContract,
    normalize_subject_ref,
)
from Perception.context_builder import UIState

logger = logging.getLogger(__name__)


class OracleRuntime:
    def __init__(self, dead_loop_threshold: int = 3, screen_variance_threshold: float = 10.0):
        self.dead_loop_threshold = max(2, int(dead_loop_threshold or 3))
        self.screen_variance_threshold = float(screen_variance_threshold or 10.0)
        self.action_history: List[Dict] = []
        self.policy_engine = PolicyEngine()
        self.extractor = ObservationExtractor()
        logger.info(
            "OracleRuntime 初始化完成(V3.1): dead_loop_threshold=%d, screen_var=%.1f",
            self.dead_loop_threshold,
            self.screen_variance_threshold,
        )

    def pre_guard(
        self,
        action: ResolvedAction,
        contract: StepContract,
        ui_state: UIState,
    ) -> GuardResult:
        observations: list[ObservationFact] = []
        assessments: list[Assessment] = []

        loop_detected = self._is_dead_loop(action)
        if loop_detected:
            observations.append(
                ObservationFact(
                    fact_id="obs_pre_loop_detected",
                    type="risk_detected",
                    scope="global",
                    subject_ref=normalize_subject_ref(action.target.ref_id if action.target else None),
                    attributes={"risk": "loop_detected", "history_len": len(self.action_history)},
                    confidence=1.0,
                    source="runtime_guard",
                    is_derived=True,
                    evidence_refs=None,
                )
            )

        context = {
            "phase": "pre",
            "loop_detected": loop_detected,
            "action_history": list(self.action_history),
            "retry_count": self._recent_failure_count(),
            "time_budget": None,
            "old_package": str(getattr(ui_state, "package_name", "") or ""),
            "new_package": str(getattr(ui_state, "package_name", "") or ""),
            "new_activity": str(getattr(ui_state, "activity_name", "") or ""),
        }

        assessments.extend(
            self.policy_engine.evaluate(
                phase="pre",
                policies=list(contract.policies or []),
                observations=observations,
                context=context,
            )
        )

        if loop_detected and not any(a.reason_code == "loop_detected" for a in assessments):
            assessments.append(
                Assessment(
                    name="loop_guard_check",
                    source="runtime",
                    applies_to="runtime_guard",
                    outcome="fail",
                    severity="hard",
                    reason_code="loop_detected",
                    message=f"连续动作重复达到阈值({self.dead_loop_threshold})",
                    evidence_refs=[obs.fact_id for obs in observations],
                    score=None,
                    remedy_hint=AdviceParams(backtrack_steps=1, reason_tags=["loop_guard"]),
                )
            )

        allowed = not any(a.outcome == "fail" and a.severity == "hard" for a in assessments)
        return GuardResult(
            phase="pre",
            allowed=allowed,
            assessments=assessments,
            observations=observations,
        )

    def post_guard(
        self,
        action: ResolvedAction,
        contract: StepContract,
        old_state: UIState,
        new_state: UIState,
        screenshot_path: str = "",
    ) -> GuardResult:
        observations = self.extractor.extract(old_state=old_state, new_state=new_state, action=action)

        screen_issue = self._check_screen_anomaly(screenshot_path)
        if screen_issue:
            observations.append(
                ObservationFact(
                    fact_id="obs_post_screen_anomaly",
                    type="risk_detected",
                    scope="global",
                    subject_ref=normalize_subject_ref(action.target.ref_id if action.target else None),
                    attributes={"risk": "screen_anomaly", "message": screen_issue},
                    confidence=0.95,
                    source="runtime_guard",
                    is_derived=True,
                    evidence_refs=None,
                )
            )

        context = {
            "phase": "post",
            "loop_detected": False,
            "action_history": list(self.action_history),
            "retry_count": self._recent_failure_count(),
            "time_budget": None,
            "old_package": str(getattr(old_state, "package_name", "") or ""),
            "new_package": str(getattr(new_state, "package_name", "") or ""),
            "old_activity": str(getattr(old_state, "activity_name", "") or ""),
            "new_activity": str(getattr(new_state, "activity_name", "") or ""),
        }

        assessments = self.policy_engine.evaluate(
            phase="post",
            policies=list(contract.policies or []),
            observations=observations,
            context=context,
        )

        if screen_issue:
            assessments.append(
                Assessment(
                    name="visual_guard_check",
                    source="runtime",
                    applies_to="runtime_guard",
                    outcome="fail",
                    severity="hard",
                    reason_code="screen_anomaly",
                    message=screen_issue,
                    evidence_refs=["obs_post_screen_anomaly"],
                    score=None,
                    remedy_hint=AdviceParams(reason_tags=["screen_anomaly"], backtrack_steps=1),
                )
            )

        # Post guard should always return observations/assessments; final success is decided by evaluator.
        return GuardResult(
            phase="post",
            allowed=True,
            assessments=assessments,
            observations=observations,
        )

    def record_action(self, action: ResolvedAction):
        params = action.params or {}
        self.action_history.append(
            {
                "type": str(action.type or ""),
                "x": int(params.get("x", 0) or 0),
                "y": int(params.get("y", 0) or 0),
                "text": str(params.get("text") or ""),
            }
        )
        max_history = self.dead_loop_threshold * 3
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]

    def reset(self):
        self.action_history.clear()

    def _is_dead_loop(self, action: ResolvedAction) -> bool:
        if len(self.action_history) < self.dead_loop_threshold:
            return False

        params = action.params or {}
        current = {
            "type": str(action.type or ""),
            "x": int(params.get("x", 0) or 0),
            "y": int(params.get("y", 0) or 0),
            "text": str(params.get("text") or ""),
        }

        recent = self.action_history[-self.dead_loop_threshold :]
        for past in recent:
            if past.get("type") != current["type"]:
                return False
            if abs(int(past.get("x", 0)) - current["x"]) > 12:
                return False
            if abs(int(past.get("y", 0)) - current["y"]) > 12:
                return False
            if current["text"] and str(past.get("text") or "") != current["text"]:
                return False
        return True

    def _check_screen_anomaly(self, screenshot_path: str) -> str:
        if not screenshot_path:
            return ""
        if cv2 is None or np is None:
            return ""
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return ""
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            total_var = (
                float(np.var(hsv[:, :, 0]))
                + float(np.var(hsv[:, :, 1]))
                + float(np.var(hsv[:, :, 2]))
            )
            if total_var < self.screen_variance_threshold:
                return f"屏幕异常: HSV方差过低({total_var:.2f})"
            return ""
        except Exception:
            return ""

    def _recent_failure_count(self) -> int:
        # Runtime 不持有失败历史，这里保留预留字段。
        return 0
