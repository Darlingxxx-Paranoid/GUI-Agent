"""Unified policy evaluation for runtime and post phases."""

from __future__ import annotations

import math
import re
from typing import Any

from Oracle.contracts import AdviceParams, Assessment, ObservationFact, PolicyRule


class PolicyEngine:
    def evaluate(
        self,
        phase: str,
        policies: list[PolicyRule],
        observations: list[ObservationFact],
        context: dict,
    ) -> list[Assessment]:
        assessments: list[Assessment] = []
        for policy in policies:
            assessments.append(
                self._evaluate_single(
                    phase=phase,
                    policy=policy,
                    observations=observations,
                    context=context,
                )
            )
        return assessments

    def _evaluate_single(
        self,
        phase: str,
        policy: PolicyRule,
        observations: list[ObservationFact],
        context: dict,
    ) -> Assessment:
        kind = str(policy.kind or "").strip().lower()
        level = str(policy.level or "soft").strip().lower()
        if level not in {"none", "soft", "hard"}:
            level = "soft"

        violation = False
        evidence_refs: list[str] = []
        reason_code = ""
        message = ""

        if kind == "loop_guard":
            loop_detected = bool(context.get("loop_detected", False))
            violation = loop_detected
            reason_code = "loop_detected" if violation else "loop_clear"
            message = "检测到重复动作循环" if violation else "循环检测通过"

        elif kind == "app_boundary":
            expected_pkg = str((policy.extra or {}).get("expected_package") or "").strip()
            must_stay = bool((policy.extra or {}).get("must_stay_in_app", False))
            old_pkg = str(context.get("old_package") or "")
            new_pkg = str(context.get("new_package") or "")
            forbidden = [str(v).strip() for v in ((policy.extra or {}).get("forbidden_packages") or []) if str(v).strip()]

            if new_pkg and forbidden and new_pkg in forbidden:
                violation = True
                reason_code = "forbidden_package"
                message = f"进入禁止包名: {new_pkg}"
            elif must_stay and old_pkg and new_pkg and old_pkg != new_pkg and (not expected_pkg or expected_pkg != new_pkg):
                violation = True
                reason_code = "must_stay_in_app"
                message = f"跨包漂移: {old_pkg} -> {new_pkg}"
            elif expected_pkg and new_pkg and new_pkg != expected_pkg and must_stay:
                violation = True
                reason_code = "expected_package_mismatch"
                message = f"包名不匹配: expected={expected_pkg}, actual={new_pkg}"
            else:
                reason_code = "app_boundary_pass"
                message = "应用边界检查通过"

        elif kind == "activity_boundary":
            expected_fragment = str((policy.extra or {}).get("expected_activity_contains") or "").strip()
            new_activity = str(context.get("new_activity") or "")
            if expected_fragment and new_activity and expected_fragment not in new_activity:
                violation = True
                reason_code = "activity_mismatch"
                message = f"Activity 不符合预期: {new_activity}"
            else:
                reason_code = "activity_boundary_pass"
                message = "Activity 边界检查通过"

        elif kind == "visual_guard":
            threshold = float((policy.extra or {}).get("max_similarity") or 0.999)
            sim_facts = [obs for obs in observations if str(obs.type) == "visual_similarity_state"]
            similarity = None
            if sim_facts:
                similarity = max(
                    float((obs.attributes or {}).get("similarity") or 0.0)
                    for obs in sim_facts
                )
                evidence_refs = [obs.fact_id for obs in sim_facts]
            if similarity is not None and similarity >= threshold:
                violation = True
                reason_code = "visual_similarity_too_high"
                message = f"界面变化过小: similarity={similarity:.4f}"
            else:
                reason_code = "visual_guard_pass"
                message = "视觉变化检查通过"

        else:
            violation, evidence_refs = self._match_generic_policy(policy=policy, observations=observations, context=context)
            reason_code = f"{kind}_violation" if violation else f"{kind}_pass"
            message = f"策略命中: {kind}" if violation else f"策略通过: {kind}"

        outcome = "fail" if violation else "pass"
        severity = level if violation else "none"
        applies_to = self._applies_to_from_kind(kind)
        remedy = None
        if violation:
            remedy = AdviceParams(
                backtrack_steps=1 if level == "hard" else None,
                reason_tags=[kind, reason_code],
                target_ref=policy.subject_ref,
            )

        if not evidence_refs:
            evidence_refs = self._collect_policy_evidence(policy=policy, observations=observations)

        return Assessment(
            name=f"{kind}_check",
            source="policy_engine",
            applies_to=applies_to,
            outcome=outcome,
            severity=severity,
            reason_code=reason_code,
            message=message,
            evidence_refs=evidence_refs,
            score=None,
            remedy_hint=remedy,
        )

    def _match_generic_policy(
        self,
        policy: PolicyRule,
        observations: list[ObservationFact],
        context: dict,
    ) -> tuple[bool, list[str]]:
        if not policy.predicates:
            return False, []

        evidence_refs: list[str] = []
        for obs in observations:
            if policy.subject_ref and obs.subject_ref and policy.subject_ref != obs.subject_ref:
                continue
            matched = True
            for predicate in policy.predicates:
                if not self._predicate_match(predicate.field, predicate.operator, predicate.value, obs, context):
                    matched = False
                    break
            if matched:
                evidence_refs.append(obs.fact_id)

        return bool(evidence_refs), evidence_refs

    def _collect_policy_evidence(self, policy: PolicyRule, observations: list[ObservationFact]) -> list[str]:
        refs: list[str] = []
        for obs in observations:
            if policy.subject_ref and obs.subject_ref and policy.subject_ref != obs.subject_ref:
                continue
            if str(policy.kind).startswith("app_") and str(obs.type).startswith("package"):
                refs.append(obs.fact_id)
            if str(policy.kind).startswith("activity") and str(obs.type).startswith("activity"):
                refs.append(obs.fact_id)
            if str(policy.kind).startswith("visual") and str(obs.type) == "visual_similarity_state":
                refs.append(obs.fact_id)
        return refs

    def _applies_to_from_kind(self, kind: str) -> str:
        if kind in {"app_boundary", "activity_boundary", "risk_boundary"}:
            return "boundary"
        if kind in {"loop_guard", "visual_guard"}:
            return "runtime_guard"
        if kind in {"input_guard", "focus_guard"}:
            return "effect"
        return "safety"

    def _predicate_match(
        self,
        field_path: str,
        operator: str,
        expected: Any,
        obs: ObservationFact,
        context: dict,
    ) -> bool:
        actual = self._resolve_field(field_path=field_path, obs=obs, context=context)
        op = str(operator or "").strip().lower()

        if op == "equals":
            return actual == expected
        if op == "not_equals":
            return actual != expected
        if op == "contains":
            if isinstance(actual, (list, tuple, set)):
                return any(str(expected) in str(item) for item in actual)
            return str(expected) in str(actual)
        if op == "not_contains":
            if isinstance(actual, (list, tuple, set)):
                return all(str(expected) not in str(item) for item in actual)
            return str(expected) not in str(actual)
        if op == "in":
            if isinstance(expected, (list, tuple, set)):
                return actual in expected
            return str(actual) in str(expected)
        if op == "not_in":
            if isinstance(expected, (list, tuple, set)):
                return actual not in expected
            return str(actual) not in str(expected)
        if op == "any_of":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return any(self._value_equal(actual, item) for item in expected)
        if op == "all_of":
            if not isinstance(expected, (list, tuple, set)):
                return False
            if not isinstance(actual, (list, tuple, set)):
                return False
            actual_norm = [str(v) for v in actual]
            return all(str(item) in actual_norm for item in expected)
        if op == "exists":
            return actual is not None
        if op == "not_exists":
            return actual is None
        if op == "regex":
            try:
                return bool(re.search(str(expected), str(actual or "")))
            except re.error:
                return False
        if op in {"gt", "gte", "lt", "lte"}:
            try:
                left = float(actual)
                right = float(expected)
            except Exception:
                return False
            if op == "gt":
                return left > right
            if op == "gte":
                return left >= right
            if op == "lt":
                return left < right
            return left <= right
        if op == "between":
            if not isinstance(expected, (list, tuple)) or len(expected) != 2:
                return False
            try:
                left = float(actual)
                lo = float(expected[0])
                hi = float(expected[1])
            except Exception:
                return False
            return lo <= left <= hi
        if op == "overlap":
            if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
                return False
            actual_set = {str(v) for v in actual}
            expected_set = {str(v) for v in expected}
            return bool(actual_set.intersection(expected_set))
        if op == "near":
            if not isinstance(actual, (list, tuple)) or len(actual) != 2:
                return False
            if not isinstance(expected, (list, tuple)) or len(expected) < 2:
                return False
            threshold = float(expected[2]) if len(expected) > 2 else 120.0
            try:
                ax, ay = float(actual[0]), float(actual[1])
                ex, ey = float(expected[0]), float(expected[1])
            except Exception:
                return False
            dist = math.hypot(ax - ex, ay - ey)
            return dist <= threshold

        return False

    def _resolve_field(self, field_path: str, obs: ObservationFact, context: dict) -> Any:
        path = str(field_path or "").strip()
        if not path:
            return None

        if path.startswith("context."):
            return self._get_nested(context, path[len("context.") :])

        if path.startswith("attributes."):
            return self._get_nested(obs.attributes or {}, path[len("attributes.") :])

        if hasattr(obs, path):
            return getattr(obs, path)

        if path == "attributes":
            return obs.attributes

        return self._get_nested(obs.attributes or {}, path)

    def _get_nested(self, data: dict, dotted: str) -> Any:
        current: Any = data
        for key in str(dotted or "").split("."):
            if key == "":
                continue
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _value_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, float) or isinstance(right, float):
            try:
                return float(left) == float(right)
            except Exception:
                return False
        return left == right
