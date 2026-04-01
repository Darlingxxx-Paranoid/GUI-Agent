"""Pre-oracle: build StepContract from planning intent."""

from __future__ import annotations

import logging
import re
from typing import List

from Oracle.contracts import (
    Expectation,
    GoalSpec,
    PolicyRule,
    Predicate,
    Selector,
    StepContract,
    TargetRef,
    normalize_subject_ref,
)
from Perception.context_builder import UIState
from Planning.planner import PlanResult

logger = logging.getLogger(__name__)


class OraclePre:
    """Generate V3.1 step contract before execution."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        logger.info("OraclePre 初始化完成 (V3.1)")

    def generate_contract(
        self,
        plan: PlanResult,
        ui_state: UIState,
        task_hint: str = "",
    ) -> StepContract:
        goal = self._normalize_goal(plan.goal)
        target = self._normalize_target(plan.target)
        expectations = self._build_expectations(plan=plan, target=target)
        policies = self._build_policies(plan=plan, ui_state=ui_state, task_hint=task_hint, target=target)

        hints = dict(plan.planning_hints or {})
        hints["requested_action_type"] = plan.requested_action_type
        hints["from_experience"] = bool(plan.from_experience)

        contract = StepContract(
            goal=goal,
            target=target,
            expectations=expectations,
            policies=policies,
            planning_hints=hints,
        )
        logger.info(
            "生成 StepContract: goal='%s', expectations=%d, policies=%d",
            contract.goal.summary,
            len(contract.expectations),
            len(contract.policies),
        )
        return contract

    def _normalize_goal(self, goal: GoalSpec) -> GoalSpec:
        summary = str(goal.summary or "").strip() or "执行下一步"
        success_definition = str(goal.success_definition or "").strip() or "观察到预期变化"
        tags = [str(tag).strip() for tag in (goal.tags or []) if str(tag).strip()]
        return GoalSpec(summary=summary[:180], success_definition=success_definition[:240], tags=tags)

    def _normalize_target(self, target: TargetRef | None) -> TargetRef | None:
        if target is None:
            return None

        ref_id = normalize_subject_ref(target.ref_id, fallback_widget_id=(target.resolved.widget_id if target.resolved else None))
        role = target.role or "primary"
        selectors: List[Selector] = []
        for selector in target.selectors:
            kind = str(selector.kind or "").strip()
            operator = str(selector.operator or "").strip()
            if not kind or not operator:
                continue
            selectors.append(Selector(kind=kind, operator=operator, value=selector.value))

        target.ref_id = ref_id
        target.role = role
        target.selectors = selectors
        return target

    def _build_expectations(self, plan: PlanResult, target: TargetRef | None) -> list[Expectation]:
        action_type = str(plan.requested_action_type or "tap").strip().lower()
        subject_ref = normalize_subject_ref(target.ref_id if target else None, fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None))

        expectations: list[Expectation] = []

        if action_type in {"tap", "long_press", "swipe"}:
            expectations.append(
                Expectation(
                    id="effect.visual_change.required",
                    fact_type="visual_similarity_state",
                    scope="local",
                    subject_ref=subject_ref,
                    predicates=[Predicate(field="attributes.similarity", operator="lt", value=0.995)],
                    weight=1.0,
                    optional=False,
                    polarity="positive",
                    tier="required",
                )
            )

        if action_type == "input":
            token = str(plan.input_text or "").strip()
            if token:
                expectations.append(
                    Expectation(
                        id="effect.input_text.required",
                        fact_type="text_appeared",
                        scope="local",
                        subject_ref=subject_ref,
                        predicates=[Predicate(field="attributes.text", operator="contains", value=token[:80])],
                        weight=1.0,
                        optional=False,
                        polarity="positive",
                        tier="required",
                    )
                )
            expectations.append(
                Expectation(
                    id="effect.focus_or_keyboard.supporting",
                    fact_type="keyboard_changed",
                    scope="global",
                    subject_ref=subject_ref,
                    predicates=[],
                    weight=0.6,
                    optional=True,
                    polarity="positive",
                    tier="supporting",
                )
            )

        if action_type == "back":
            expectations.append(
                Expectation(
                    id="effect.back_navigation.required",
                    fact_type="activity_changed",
                    scope="global",
                    subject_ref=subject_ref,
                    predicates=[],
                    weight=1.0,
                    optional=False,
                    polarity="positive",
                    tier="required",
                )
            )

        expectations.append(
            Expectation(
                id="safety.no_risk.required",
                fact_type="risk_detected",
                scope="global",
                subject_ref=subject_ref,
                predicates=[],
                weight=1.0,
                optional=True,
                polarity="negative",
                tier="required",
            )
        )

        return expectations

    def _build_policies(
        self,
        plan: PlanResult,
        ui_state: UIState,
        task_hint: str,
        target: TargetRef | None,
    ) -> list[PolicyRule]:
        current_package = str(getattr(ui_state, "package_name", "") or "").strip()
        current_activity = str(getattr(ui_state, "activity_name", "") or "").strip()
        expected_package = self._infer_expected_package(plan=plan, current_package=current_package, task_hint=task_hint)

        subject_ref = normalize_subject_ref(target.ref_id if target else None, fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None))

        must_stay = bool(expected_package and expected_package == current_package)
        policies = [
            PolicyRule(
                id="loop_guard",
                kind="loop_guard",
                level="hard",
                subject_ref=None,
                predicates=[],
                tags=["runtime"],
                extra={"threshold": 3},
            ),
            PolicyRule(
                id="app_boundary",
                kind="app_boundary",
                level="hard" if must_stay else "soft",
                subject_ref=subject_ref,
                predicates=[],
                tags=["boundary"],
                extra={
                    "must_stay_in_app": must_stay,
                    "expected_package": expected_package,
                    "forbidden_packages": [],
                },
            ),
            PolicyRule(
                id="activity_boundary",
                kind="activity_boundary",
                level="soft",
                subject_ref=subject_ref,
                predicates=[],
                tags=["boundary"],
                extra={
                    "expected_activity_contains": current_activity if plan.requested_action_type == "back" else "",
                },
            ),
            PolicyRule(
                id="visual_guard",
                kind="visual_guard",
                level="soft",
                subject_ref=subject_ref,
                predicates=[Predicate(field="attributes.similarity", operator="gte", value=0.999)],
                tags=["runtime", "effect"],
                extra={"max_similarity": 0.999},
            ),
        ]
        return policies

    def _infer_expected_package(self, plan: PlanResult, current_package: str, task_hint: str) -> str:
        text = " ".join(
            [
                str(plan.goal.summary or ""),
                str(plan.goal.success_definition or ""),
                str(task_hint or ""),
            ]
        ).lower()

        if any(token in text for token in ("open ", "launch", "打开", "启动")) and "package:" in text:
            match = re.search(r"package:([a-z0-9_.]+)", text)
            if match:
                return match.group(1)

        return current_package
