"""Pre-oracle: build StepContract from planning intent."""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from Oracle.contracts import (
    Expectation,
    GoalSpec,
    PolicyRule,
    Predicate,
    Selector,
    StepContract,
    TargetRef,
    dataclass_to_json_schema,
    normalize_subject_ref,
    parse_dataclass,
    parse_json_object,
)
from Perception.context_builder import UIState
from Planning.planner import PlanResult
from prompt.oracle_pre_prompt import ORACLE_PRE_PROMPT
from utils.audit_recorder import AuditRecorder

logger = logging.getLogger(__name__)


class OraclePre:
    """Generate V4 step contract before execution."""

    _APP_ALIAS_MAP = {
        "gmail": "com.google.android.gm",
        "chrome": "com.android.chrome",
        "settings": "com.android.settings",
        "clock": "com.google.android.deskclock",
        "youtube": "com.google.android.youtube",
        "maps": "com.google.android.apps.maps",
        "photos": "com.google.android.apps.photos",
        "play store": "com.android.vending",
        "camera": "com.android.camera",
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.audit = AuditRecorder(component="oracle_pre")
        logger.info("OraclePre 初始化完成 (V4)")

    def generate_contract(
        self,
        plan: PlanResult,
        ui_state: UIState,
        task_hint: str = "",
        step: int | None = None,
    ) -> StepContract:
        normalized_goal = self._normalize_goal(plan.goal)
        normalized_target = self._normalize_target(plan.target)

        fallback_expectations = self._build_expectations(plan=plan, target=normalized_target)
        fallback_policies = self._build_policies(
            plan=plan,
            ui_state=ui_state,
            task_hint=task_hint,
            target=normalized_target,
        )

        semantic_source = "fallback_rules"
        llm_contract = self._generate_contract_with_llm(
            plan=plan,
            ui_state=ui_state,
            task_hint=task_hint,
            step=step,
        )
        if llm_contract is not None:
            semantic_source = "oracle_pre_llm"
            goal = self._normalize_goal(llm_contract.goal)
            target = self._normalize_target(llm_contract.target if llm_contract.target is not None else normalized_target)
            expectations = self._sanitize_expectations(
                expectations=list(llm_contract.expectations or []),
                fallback=fallback_expectations,
                subject_ref=normalize_subject_ref(
                    target.ref_id if target else None,
                    fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None),
                ),
            )
            policies = self._sanitize_policies(
                policies=list(llm_contract.policies or []),
                fallback=fallback_policies,
                subject_ref=normalize_subject_ref(
                    target.ref_id if target else None,
                    fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None),
                ),
                current_package=str(getattr(ui_state, "package_name", "") or "").strip(),
                current_activity=str(getattr(ui_state, "activity_name", "") or "").strip(),
                action_type=str(plan.requested_action_type or "tap").strip().lower(),
            )
            llm_hints = dict(llm_contract.planning_hints or {})
        else:
            goal = normalized_goal
            target = normalized_target
            expectations = fallback_expectations
            policies = fallback_policies
            llm_hints = {}

        hints = dict(plan.planning_hints or {})
        hints.update(llm_hints)
        hints["task_hint"] = str(task_hint or "")
        hints["requested_action_type"] = plan.requested_action_type
        hints["from_experience"] = bool(plan.from_experience)
        hints["semantic_contract_source"] = semantic_source
        hints["primary_expectation_ids"] = [
            str(exp.id)
            for exp in expectations
            if str(exp.tier or "required") == "required" and not bool(exp.optional)
        ]

        contract = StepContract(
            goal=goal,
            target=target,
            expectations=expectations,
            policies=policies,
            planning_hints=hints,
        )
        self._record_oracle_pre_artifact(step=step, payload=contract)
        logger.info(
            "生成 StepContract: goal='%s', expectations=%d, policies=%d, source=%s",
            contract.goal.summary,
            len(contract.expectations),
            len(contract.policies),
            semantic_source,
        )
        return contract

    def _generate_contract_with_llm(
        self,
        plan: PlanResult,
        ui_state: UIState,
        task_hint: str,
        step: int | None = None,
    ) -> Optional[StepContract]:
        if self.llm is None:
            return None

        schema = dataclass_to_json_schema(StepContract)
        plan_payload = {
            "goal": {
                "summary": str(plan.goal.summary or ""),
                "success_definition": str(plan.goal.success_definition or ""),
                "tags": list(plan.goal.tags or []),
            },
            "requested_action_type": str(plan.requested_action_type or "tap"),
            "input_text": str(plan.input_text or ""),
            "target": {
                "ref_id": (plan.target.ref_id if plan.target else None),
                "role": (plan.target.role if plan.target else None),
                "selectors": [
                    {"kind": str(s.kind), "operator": str(s.operator), "value": s.value}
                    for s in (plan.target.selectors if plan.target else [])
                ],
            },
            "planning_hints": dict(plan.planning_hints or {}),
        }

        prompt = ORACLE_PRE_PROMPT.format(
            task_hint=str(task_hint or ""),
            plan_json=json.dumps(plan_payload, ensure_ascii=False),
            ui_state=ui_state.to_prompt_text(),
            schema_json=json.dumps(schema, ensure_ascii=False),
        )

        try:
            response = self.llm.chat(
                prompt,
                audit_meta={
                    "artifact_kind": "StepContract",
                    "step": step,
                    "stage": "generate_contract",
                },
            )
            payload = parse_json_object(response)
            contract = parse_dataclass(payload, StepContract, strict=False)
            return contract
        except Exception as exc:
            self._record_oracle_pre_parse_error(step=step, error=str(exc))
            logger.warning("OraclePre LLM 生成失败，回退规则合同: %s", exc)
            return None

    def _normalize_goal(self, goal: GoalSpec) -> GoalSpec:
        summary = str(goal.summary or "").strip() or "执行下一步"
        success_definition = str(goal.success_definition or "").strip() or "观察到预期变化"
        tags = [str(tag).strip() for tag in (goal.tags or []) if str(tag).strip()]
        return GoalSpec(summary=summary[:180], success_definition=success_definition[:240], tags=tags)

    def _normalize_target(self, target: TargetRef | None) -> TargetRef | None:
        if target is None:
            return None

        ref_id = normalize_subject_ref(
            target.ref_id,
            fallback_widget_id=(target.resolved.widget_id if target.resolved else None),
        )
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

    def _sanitize_expectations(
        self,
        expectations: list[Expectation],
        fallback: list[Expectation],
        subject_ref: str,
    ) -> list[Expectation]:
        if not expectations:
            return fallback

        out: list[Expectation] = []
        for idx, exp in enumerate(expectations, start=1):
            exp_id = str(exp.id or "").strip() or f"effect.auto_{idx}"
            tier = str(exp.tier or "required").strip().lower()
            if tier not in {"required", "supporting"}:
                tier = "required"

            fact_type = str(exp.fact_type or "").strip()
            if not fact_type:
                continue

            # visual similarity can only be supporting evidence.
            optional = bool(exp.optional)
            if fact_type == "visual_similarity_state" and tier == "required":
                tier = "supporting"
                optional = True

            out.append(
                Expectation(
                    id=exp_id,
                    fact_type=fact_type,
                    scope=exp.scope,
                    subject_ref=exp.subject_ref or subject_ref,
                    predicates=list(exp.predicates or []),
                    weight=float(exp.weight if exp.weight is not None else 1.0),
                    optional=optional,
                    polarity=exp.polarity if exp.polarity in {"positive", "negative"} else "positive",
                    tier=tier,
                )
            )

        if not out:
            return fallback

        if not any(str(item.id) == "safety.no_risk.required" for item in out):
            out.append(
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
        return out

    def _sanitize_policies(
        self,
        policies: list[PolicyRule],
        fallback: list[PolicyRule],
        subject_ref: str,
        current_package: str,
        current_activity: str,
        action_type: str,
    ) -> list[PolicyRule]:
        by_kind = {str(item.kind): item for item in policies if str(item.kind)}
        fallback_by_kind = {str(item.kind): item for item in fallback if str(item.kind)}

        for required_kind in ("loop_guard", "app_boundary", "activity_boundary", "visual_guard"):
            if required_kind not in by_kind:
                by_kind[required_kind] = fallback_by_kind[required_kind]

        out: list[PolicyRule] = []
        for kind in ("loop_guard", "app_boundary", "activity_boundary", "visual_guard"):
            p = by_kind[kind]
            base = PolicyRule(
                id=str(p.id or kind),
                kind=kind,
                level=p.level if p.level in {"none", "soft", "hard"} else "soft",
                subject_ref=p.subject_ref or subject_ref,
                predicates=list(p.predicates or []),
                tags=list(p.tags or []),
                boundary_mode=p.boundary_mode if p.boundary_mode in {"stay", "switch", "either"} else None,
                expected_packages=[
                    str(v).strip()
                    for v in (p.expected_packages or [])
                    if str(v).strip()
                ],
                forbidden_packages=[
                    str(v).strip()
                    for v in (p.forbidden_packages or [])
                    if str(v).strip()
                ],
                expected_activity_contains=str(p.expected_activity_contains or "").strip() or None,
                loop_threshold=(int(p.loop_threshold) if p.loop_threshold is not None else None),
                max_similarity=(float(p.max_similarity) if p.max_similarity is not None else None),
            )

            if kind == "app_boundary":
                if base.boundary_mode is None:
                    base.boundary_mode = self._default_boundary_mode(
                        action_type=action_type,
                        expected_packages=base.expected_packages,
                        current_package=current_package,
                    )
                if base.boundary_mode == "switch" and not base.expected_packages:
                    fallback_app = fallback_by_kind.get("app_boundary")
                    fallback_expected = list((fallback_app.expected_packages if fallback_app else []) or [])
                    base.expected_packages = [str(v).strip() for v in fallback_expected if str(v).strip()]
                if not base.expected_packages and base.boundary_mode == "stay" and current_package:
                    base.expected_packages = [current_package]
                if base.level == "soft" and base.boundary_mode in {"stay", "switch"}:
                    base.level = "hard"

            if kind == "activity_boundary" and not base.expected_activity_contains and action_type == "back":
                base.expected_activity_contains = current_activity or None

            if kind == "loop_guard" and base.loop_threshold is None:
                base.loop_threshold = 3
                if base.level == "none":
                    base.level = "hard"

            if kind == "visual_guard" and base.max_similarity is None:
                base.max_similarity = 0.999
                if base.level == "none":
                    base.level = "soft"

            out.append(base)

        return out

    def _build_expectations(self, plan: PlanResult, target: TargetRef | None) -> list[Expectation]:
        action_type = str(plan.requested_action_type or "tap").strip().lower()
        subject_ref = normalize_subject_ref(
            target.ref_id if target else None,
            fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None),
        )

        expectations: list[Expectation] = []

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
                    id="effect.keyboard_or_focus.supporting",
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

        if action_type == "launch_app":
            expectations.append(
                Expectation(
                    id="effect.package_switched.required",
                    fact_type="package_changed",
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
                id="effect.visual_change.supporting",
                fact_type="visual_similarity_state",
                scope="local",
                subject_ref=subject_ref,
                predicates=[Predicate(field="attributes.similarity", operator="lt", value=0.995)],
                weight=0.4,
                optional=True,
                polarity="positive",
                tier="supporting",
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
        expected_packages = self._infer_expected_packages(
            plan=plan,
            task_hint=task_hint,
        )

        action_type = str(plan.requested_action_type or "tap").strip().lower()
        boundary_mode = self._default_boundary_mode(
            action_type=action_type,
            expected_packages=expected_packages,
            current_package=current_package,
        )

        subject_ref = normalize_subject_ref(
            target.ref_id if target else None,
            fallback_widget_id=(target.resolved.widget_id if target and target.resolved else None),
        )

        return [
            PolicyRule(
                id="loop_guard",
                kind="loop_guard",
                level="hard",
                subject_ref=None,
                predicates=[],
                tags=["runtime"],
                loop_threshold=3,
            ),
            PolicyRule(
                id="app_boundary",
                kind="app_boundary",
                level="hard" if boundary_mode in {"stay", "switch"} else "soft",
                subject_ref=subject_ref,
                predicates=[],
                tags=["boundary"],
                boundary_mode=boundary_mode,
                expected_packages=expected_packages,
                forbidden_packages=[],
            ),
            PolicyRule(
                id="activity_boundary",
                kind="activity_boundary",
                level="soft",
                subject_ref=subject_ref,
                predicates=[],
                tags=["boundary"],
                expected_activity_contains=current_activity if action_type == "back" else None,
            ),
            PolicyRule(
                id="visual_guard",
                kind="visual_guard",
                level="soft",
                subject_ref=subject_ref,
                predicates=[Predicate(field="attributes.similarity", operator="gte", value=0.999)],
                tags=["runtime", "effect"],
                max_similarity=0.999,
            ),
        ]

    def _infer_expected_packages(self, plan: PlanResult, task_hint: str) -> list[str]:
        package_values: list[str] = []

        hint_pkg = str((plan.planning_hints or {}).get("target_package") or "").strip().lower()
        if self._is_valid_package(hint_pkg):
            package_values.append(hint_pkg)

        text_blob = " ".join(
            [
                str(plan.goal.summary or ""),
                str(plan.goal.success_definition or ""),
                str(task_hint or ""),
            ]
        ).lower()

        for match in re.findall(r"package:([a-z0-9_.]+)", text_blob):
            pkg = str(match or "").strip().lower()
            if self._is_valid_package(pkg):
                package_values.append(pkg)

        for alias, pkg in self._APP_ALIAS_MAP.items():
            if alias in text_blob:
                package_values.append(pkg)

        deduped: list[str] = []
        for pkg in package_values:
            if pkg and pkg not in deduped:
                deduped.append(pkg)
        return deduped

    def _default_boundary_mode(self, action_type: str, expected_packages: list[str], current_package: str) -> str:
        if action_type == "launch_app":
            return "switch" if expected_packages else "either"
        if expected_packages:
            if len(expected_packages) == 1 and current_package and expected_packages[0] == current_package:
                return "stay"
            return "switch"
        return "stay" if current_package else "either"

    def _is_valid_package(self, value: str) -> bool:
        token = str(value or "").strip().lower()
        if not token:
            return False
        if token.count(".") < 1:
            return False
        return bool(re.match(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$", token))

    def _record_oracle_pre_artifact(self, step: int | None, payload) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="StepContract",
                step=int(step),
                payload=payload,
            )
        except Exception as exc:
            logger.warning("写入 OraclePre 审计记录失败: %s", exc)

    def _record_oracle_pre_parse_error(self, step: int | None, error: str) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="StepContract",
                step=int(step),
                payload={
                    "stage": "llm_parse_error",
                    "llm_response": "",
                    "error": error,
                },
                llm=True,
                append=True,
            )
        except Exception as exc:
            logger.warning("写入 OraclePre 解析错误审计记录失败: %s", exc)
