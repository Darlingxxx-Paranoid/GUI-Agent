"""Expectation matcher for V3.1 contracts."""

from __future__ import annotations

import math
import re
from typing import Any

from Oracle.contracts import Expectation, ExpectationMatch, ObservationFact, Predicate


class ExpectationMatcher:
    def match(
        self,
        expectations: list[Expectation],
        observations: list[ObservationFact],
    ) -> list[ExpectationMatch]:
        results: list[ExpectationMatch] = []
        for expectation in expectations:
            candidates = self._select_candidates(expectation=expectation, observations=observations)
            matched_fact_ids: list[str] = []

            for fact in candidates:
                if self._predicates_match(expectation.predicates or [], fact):
                    matched_fact_ids.append(fact.fact_id)

            if expectation.polarity == "negative":
                matched = len(matched_fact_ids) == 0
                score = float(expectation.weight if matched else 0.0)
                reason = "negative_expectation_satisfied" if matched else "negative_expectation_hit"
                message = (
                    "未命中反证信号"
                    if matched
                    else f"命中反证信号: {len(matched_fact_ids)}"
                )
            else:
                matched = len(matched_fact_ids) > 0
                score = float(expectation.weight if matched else 0.0)
                reason = "expectation_matched" if matched else "expectation_not_matched"
                message = (
                    f"命中证据: {len(matched_fact_ids)}"
                    if matched
                    else "未找到匹配证据"
                )

            results.append(
                ExpectationMatch(
                    expectation_id=str(expectation.id),
                    matched=matched,
                    matched_fact_ids=matched_fact_ids,
                    score=score,
                    reason_code=reason,
                    message=message,
                )
            )

        return results

    def _select_candidates(
        self,
        expectation: Expectation,
        observations: list[ObservationFact],
    ) -> list[ObservationFact]:
        selected: list[ObservationFact] = []
        for obs in observations:
            if str(obs.type) != str(expectation.fact_type):
                continue
            if expectation.scope and obs.scope and str(obs.scope) != str(expectation.scope):
                continue
            if expectation.subject_ref and obs.subject_ref and str(obs.subject_ref) != str(expectation.subject_ref):
                continue
            selected.append(obs)
        return selected

    def _predicates_match(self, predicates: list[Predicate], fact: ObservationFact) -> bool:
        if not predicates:
            return True
        for predicate in predicates:
            if not self._predicate_match(predicate=predicate, fact=fact):
                return False
        return True

    def _predicate_match(self, predicate: Predicate, fact: ObservationFact) -> bool:
        actual = self._resolve_field(path=str(predicate.field or ""), fact=fact)
        operator = str(predicate.operator or "").strip().lower()
        expected = predicate.value

        if operator == "equals":
            return actual == expected
        if operator == "not_equals":
            return actual != expected
        if operator == "contains":
            if isinstance(actual, (list, tuple, set)):
                return any(str(expected) in str(item) for item in actual)
            return str(expected) in str(actual)
        if operator == "not_contains":
            if isinstance(actual, (list, tuple, set)):
                return all(str(expected) not in str(item) for item in actual)
            return str(expected) not in str(actual)
        if operator == "in":
            if isinstance(expected, (list, tuple, set)):
                return actual in expected
            return str(actual) in str(expected)
        if operator == "not_in":
            if isinstance(expected, (list, tuple, set)):
                return actual not in expected
            return str(actual) not in str(expected)
        if operator == "any_of":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return any(self._eq(actual, item) for item in expected)
        if operator == "all_of":
            if not isinstance(expected, (list, tuple, set)) or not isinstance(actual, (list, tuple, set)):
                return False
            actual_norm = [str(v) for v in actual]
            return all(str(item) in actual_norm for item in expected)
        if operator == "exists":
            return actual is not None
        if operator == "not_exists":
            return actual is None
        if operator == "regex":
            try:
                return bool(re.search(str(expected), str(actual or "")))
            except re.error:
                return False
        if operator in {"gt", "gte", "lt", "lte"}:
            try:
                left = float(actual)
                right = float(expected)
            except Exception:
                return False
            if operator == "gt":
                return left > right
            if operator == "gte":
                return left >= right
            if operator == "lt":
                return left < right
            return left <= right
        if operator == "between":
            if not isinstance(expected, (list, tuple)) or len(expected) != 2:
                return False
            try:
                left = float(actual)
                lo = float(expected[0])
                hi = float(expected[1])
            except Exception:
                return False
            return lo <= left <= hi
        if operator == "overlap":
            if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
                return False
            return bool({str(v) for v in actual}.intersection({str(v) for v in expected}))
        if operator == "near":
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
            return math.hypot(ax - ex, ay - ey) <= threshold

        return False

    def _resolve_field(self, path: str, fact: ObservationFact) -> Any:
        if not path:
            return None
        if path.startswith("attributes."):
            return self._get_nested(fact.attributes or {}, path[len("attributes.") :])
        if hasattr(fact, path):
            return getattr(fact, path)
        return self._get_nested(fact.attributes or {}, path)

    def _get_nested(self, data: dict, dotted: str) -> Any:
        current: Any = data
        for key in str(dotted or "").split("."):
            if not key:
                continue
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _eq(self, left: Any, right: Any) -> bool:
        if isinstance(left, float) or isinstance(right, float):
            try:
                return float(left) == float(right)
            except Exception:
                return False
        return left == right
