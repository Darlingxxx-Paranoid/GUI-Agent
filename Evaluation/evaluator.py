"""Post evaluator that outputs StepEvaluation (V4)."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from Evaluation.expectation_matcher import ExpectationMatcher
from Execution.observation_extractor import ObservationExtractor
from Execution.policy_engine import PolicyEngine
from Oracle.contracts import (
    AdviceParams,
    Assessment,
    GuardResult,
    NumericMetric,
    RecommendedAction,
    ResolvedAction,
    StepContract,
    StepEvaluation,
    to_plain_dict,
)
from Perception.context_builder import UIState

logger = logging.getLogger(__name__)


class Evaluator:
    """Combine observation/policy/expectation to produce step-level decision."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.matcher = ExpectationMatcher()
        self.policy_engine = PolicyEngine()
        self.extractor = ObservationExtractor()
        logger.info("Evaluator 初始化完成 (V4)")

    def evaluate(
        self,
        subgoal_description: str,
        contract: StepContract,
        old_state: UIState,
        new_state: UIState,
        action: Optional[ResolvedAction] = None,
        post_guard: Optional[GuardResult] = None,
    ) -> StepEvaluation:
        logger.info("开始事后评估(V4): '%s'", subgoal_description)

        runtime_observations = list(post_guard.observations) if post_guard else []
        runtime_assessments = list(post_guard.assessments) if post_guard else []

        if action is None:
            action = ResolvedAction(type="tap", params={}, target=contract.target, description="")

        observations = runtime_observations
        if not observations:
            observations = self.extractor.extract(old_state=old_state, new_state=new_state, action=action)

        policy_assessments = runtime_assessments
        if not policy_assessments:
            policy_assessments = self.policy_engine.evaluate(
                phase="post",
                policies=list(contract.policies or []),
                observations=observations,
                context={
                    "phase": "post",
                    "old_package": str(getattr(old_state, "package_name", "") or ""),
                    "new_package": str(getattr(new_state, "package_name", "") or ""),
                    "old_activity": str(getattr(old_state, "activity_name", "") or ""),
                    "new_activity": str(getattr(new_state, "activity_name", "") or ""),
                    "loop_detected": False,
                    "action_history": [],
                    "retry_count": 0,
                    "time_budget": None,
                },
            )

        expectation_matches = self.matcher.match(
            expectations=list(contract.expectations or []),
            observations=observations,
        )

        match_by_id = {item.expectation_id: item for item in expectation_matches}
        required_expectations = [
            exp
            for exp in (contract.expectations or [])
            if str(exp.tier or "required") == "required" and not bool(exp.optional)
        ]
        required_total = len(required_expectations)
        required_matched = 0
        for exp in required_expectations:
            matched = bool((match_by_id.get(exp.id) or None) and match_by_id[exp.id].matched)
            if matched:
                required_matched += 1

        total_expectations = len(contract.expectations or [])
        total_matched = sum(1 for item in expectation_matches if item.matched)

        hard_fail_count = sum(1 for item in policy_assessments if item.outcome == "fail" and item.severity == "hard")
        soft_fail_count = sum(1 for item in policy_assessments if item.outcome == "fail" and item.severity == "soft")
        required_ok = bool(required_total > 0 and required_matched == required_total)
        required_missing = bool(required_total > 0 and required_matched < required_total)

        meaningful_change = self._has_meaningful_change(observations)
        soft_boundary_fail = any(
            item.outcome == "fail"
            and item.severity == "soft"
            and str(item.name or "").strip().lower() in {"app_boundary_check", "activity_boundary_check"}
            for item in policy_assessments
        )
        app_boundary_mode = self._get_app_boundary_mode(contract)
        switch_boundary_pass = any(
            item.outcome == "pass"
            and str(item.name or "").strip().lower() == "app_boundary_check"
            and str(item.reason_code or "").strip().lower() == "switch_boundary_pass"
            for item in policy_assessments
        )

        if hard_fail_count > 0:
            decision = "fail"
            confidence = 0.95
            recommended = RecommendedAction(
                kind="backtrack",
                params=AdviceParams(backtrack_steps=1, reason_tags=["policy_hard_fail"]),
            )
            eval_reason = "硬策略失败，触发回退"
        else:
            if required_ok:
                if soft_boundary_fail:
                    decision = "uncertain"
                    confidence = 0.58
                    recommended = RecommendedAction(
                        kind="observe",
                        params=AdviceParams(
                            observe_delay_ms=1800,
                            reason_tags=["semantic_conflict_required_vs_boundary_soft"],
                        ),
                    )
                    eval_reason = "required 证据满足，但 boundary 存在软失败"
                else:
                    decision = "success"
                    confidence = min(0.95, 0.65 + 0.3 * self._ratio(total_matched, max(1, total_expectations)))
                    recommended = RecommendedAction(kind="continue", params=None)
                    eval_reason = "核心 required 期望匹配"
            elif app_boundary_mode == "switch" and switch_boundary_pass and required_missing:
                decision = "uncertain"
                confidence = 0.52
                recommended = RecommendedAction(
                    kind="observe",
                    params=AdviceParams(
                        observe_delay_ms=1800,
                        reason_tags=["semantic_conflict_switch_pass_but_missing_required"],
                    ),
                )
                eval_reason = "switch boundary 已通过，但 required effect 证据不足"
            elif meaningful_change:
                decision = "uncertain"
                confidence = 0.48
                recommended = RecommendedAction(
                    kind="observe",
                    params=AdviceParams(observe_delay_ms=1800, reason_tags=["insufficient_expectation_match"]),
                )
                eval_reason = "有变化但证据不足"
            else:
                decision = "fail"
                confidence = 0.72
                next_kind = "replan" if soft_fail_count > 0 else "retry"
                recommended = RecommendedAction(
                    kind=next_kind,
                    params=AdviceParams(retry_count=1 if next_kind == "retry" else None, reason_tags=["no_meaningful_change"]),
                )
                eval_reason = "缺少有效变化"

        coverage_score = self._ratio(total_matched, max(1, total_expectations))
        expectation_assessment = Assessment(
            name="expectation_match_check",
            source="evaluator",
            applies_to="effect",
            outcome="pass" if decision == "success" else ("uncertain" if decision == "uncertain" else "fail"),
            severity="none" if decision == "success" else ("soft" if decision == "uncertain" else "soft"),
            reason_code="expectation_coverage",
            message=(
                f"{eval_reason}; required={required_matched}/{required_total}; total={total_matched}/{total_expectations}"
            ),
            evidence_refs=[fid for item in expectation_matches for fid in item.matched_fact_ids],
            score=coverage_score,
            remedy_hint=None,
        )

        assessments = list(policy_assessments) + [expectation_assessment]

        metrics = [
            NumericMetric(key="expectation_match_ratio", kind="ratio", value=coverage_score, unit=None),
            NumericMetric(
                key="required_expectation_ratio",
                kind="ratio",
                value=self._ratio(required_matched, max(1, required_total)),
                unit=None,
            ),
            NumericMetric(key="observation_count", kind="count", value=float(len(observations)), unit="facts"),
            NumericMetric(key="policy_fail_count", kind="count", value=float(hard_fail_count + soft_fail_count), unit="count"),
            NumericMetric(key="policy_hard_fail_count", kind="count", value=float(hard_fail_count), unit="count"),
        ]

        similarity = self._max_similarity(observations)
        if similarity is not None:
            metrics.append(
                NumericMetric(key="visual_similarity", kind="similarity", value=float(similarity), unit=None)
            )

        result = StepEvaluation(
            decision=decision,
            confidence=float(confidence),
            recommended_action=recommended,
            assessments=assessments,
            observations=observations,
            metrics=metrics,
            expectation_matches=expectation_matches,
        )

        self._save_eval_artifact(
            subgoal_description=subgoal_description,
            contract=contract,
            old_state=old_state,
            new_state=new_state,
            action=action,
            result=result,
        )

        logger.info(
            "评估完成: decision=%s, confidence=%.2f, recommended=%s",
            result.decision,
            result.confidence,
            result.recommended_action.kind if result.recommended_action else "none",
        )
        return result

    def _has_meaningful_change(self, observations) -> bool:
        changed_types = {
            "package_changed",
            "activity_changed",
            "text_appeared",
            "text_disappeared",
            "widget_appeared",
            "widget_disappeared",
            "keyboard_changed",
            "focus_changed",
        }
        for obs in observations:
            if str(obs.type) in changed_types:
                return True
            if str(obs.type) == "visual_similarity_state":
                try:
                    similarity = float((obs.attributes or {}).get("similarity") or 1.0)
                except Exception:
                    similarity = 1.0
                if similarity < 0.995:
                    return True
        return False

    def _max_similarity(self, observations) -> Optional[float]:
        values = []
        for obs in observations:
            if str(obs.type) != "visual_similarity_state":
                continue
            try:
                values.append(float((obs.attributes or {}).get("similarity") or 0.0))
            except Exception:
                continue
        if not values:
            return None
        return max(values)

    def _ratio(self, left: int, right: int) -> float:
        if right <= 0:
            return 0.0
        return round(float(left) / float(right), 4)

    def _save_eval_artifact(
        self,
        subgoal_description: str,
        contract: StepContract,
        old_state: UIState,
        new_state: UIState,
        action: ResolvedAction,
        result: StepEvaluation,
    ) -> None:
        screenshot_path = str(getattr(new_state, "screenshot_path", "") or "")
        if screenshot_path:
            base = os.path.splitext(os.path.basename(screenshot_path))[0]
        else:
            base = "step_unknown_after"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(project_root, "data", "context")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.eval_v4.json")

        payload = {
            "subgoal_description": subgoal_description,
            "contract": to_plain_dict(contract),
            "action": to_plain_dict(action),
            "result": to_plain_dict(result),
            "old_state": {
                "activity": str(getattr(old_state, "activity_name", "") or ""),
                "package": str(getattr(old_state, "package_name", "") or ""),
                "keyboard_visible": bool(getattr(old_state, "keyboard_visible", False)),
                "widget_count": len(getattr(old_state, "widgets", []) or []),
            },
            "new_state": {
                "activity": str(getattr(new_state, "activity_name", "") or ""),
                "package": str(getattr(new_state, "package_name", "") or ""),
                "keyboard_visible": bool(getattr(new_state, "keyboard_visible", False)),
                "widget_count": len(getattr(new_state, "widgets", []) or []),
            },
        }

        try:
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("写入评估 artifact 失败: %s", exc)

    def _get_app_boundary_mode(self, contract: StepContract) -> str:
        for policy in (contract.policies or []):
            if str(policy.kind or "").strip().lower() != "app_boundary":
                continue
            mode = str(policy.boundary_mode or "").strip().lower()
            if mode in {"stay", "switch", "either"}:
                return mode
        return ""
