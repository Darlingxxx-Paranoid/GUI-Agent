"""
事后评估模块
对动作结果进行约束检测和语义确认
决定是继续前进还是标记失败
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from Execution.action_mapper import Action
from Perception.context_builder import UIState
from Planning.oracle_pre import PreConstraints
from prompt.evaluator_prompt import EVALUATOR_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果"""
    success: bool
    reason: str = ""
    constraint_passed: bool = True
    semantic_passed: bool = True
    decision: str = ""
    confidence: float = 0.0
    needs_more_observation: bool = False
    constraint_status: str = ""
    semantic_status: str = ""
    suggested_next_action: str = ""
    matched_signals: List[Dict[str, Any]] = None
    unmatched_signals: List[Dict[str, Any]] = None
    counter_signal_hits: List[Dict[str, Any]] = None
    delta_facts: Dict[str, Any] = None
    boundary_evidence: Dict[str, Any] = None


class Evaluator:
    """
    事后评估器
    1. 基础约束检测（环境边界、控件存在/消失、UI变化、弱证据打分）
    2. 约束通过后调用 LLM 做语义确认
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("Evaluator 初始化完成")
        self._semantic_timeout_sec = 10
        self._semantic_min_remaining_sec = 7
        self._input_stopwords = {
            "this", "that", "have", "with", "from", "there", "using",
            "please", "hello", "best", "regards", "subject", "body",
        }

    def evaluate(
        self,
        subgoal_description: str,
        constraints: PreConstraints,
        old_state: UIState,
        new_state: UIState,
        action: Optional[Action] = None,
    ) -> EvalResult:
        """
        评估子目标执行结果
        """
        logger.info("开始事后评估: '%s'", subgoal_description)

        constraint_result, evidence = self._check_constraints(
            constraints,
            old_state,
            new_state,
            action=action,
        )
        if not constraint_result.constraint_passed:
            logger.info("基础约束未通过: %s", constraint_result.reason)
            self._save_eval_artifact(
                subgoal_description=subgoal_description,
                acceptance_criteria=constraints.semantic_goal or subgoal_description,
                old_state=old_state,
                new_state=new_state,
                constraint_evidence=evidence,
                llm_result=None,
                raw_response=None,
                final_result=constraint_result,
            )
            return constraint_result

        delta_facts = self._extract_delta_facts(
            old_state=old_state,
            new_state=new_state,
            action=action,
            constraints=constraints,
            constraint_evidence=evidence,
        )
        evidence["delta_facts"] = delta_facts
        evidence["success_evidence_plan"] = constraints.success_evidence_plan or {}
        evidence["action_anchor"] = constraints.action_anchor or {}
        evidence["boundary_constraints"] = constraints.boundary_constraints or {}

        match = self._match_success_evidence_plan(
            success_evidence_plan=constraints.success_evidence_plan or {},
            delta_facts=delta_facts,
        )
        evidence["matched_signals"] = match.get("matched_signals") or []
        evidence["unmatched_signals"] = match.get("unmatched_signals") or []
        evidence["counter_signal_hits"] = match.get("counter_signal_hits") or []

        machine_result = self._build_machine_eval_result(
            constraints=constraints,
            delta_facts=delta_facts,
            match=match,
            base_evidence=evidence,
        )

        defer_semantic_for_observation = (
            machine_result.decision == "uncertain"
            and bool(machine_result.needs_more_observation)
            and any(
                marker in str(machine_result.reason or "")
                for marker in ("low_change_evidence", "weak_change_evidence")
            )
        )

        if machine_result.decision != "uncertain" or defer_semantic_for_observation:
            self._save_eval_artifact(
                subgoal_description=subgoal_description,
                acceptance_criteria=constraints.semantic_goal or subgoal_description,
                old_state=old_state,
                new_state=new_state,
                constraint_evidence=evidence,
                llm_result=None,
                raw_response=None,
                final_result=machine_result,
            )
            return machine_result

        llm_payload = None
        raw_response = None
        should_skip_semantic, remaining_semantic = self._should_skip_semantic_check()
        if should_skip_semantic:
            logger.info(
                "剩余时限不足(%ss < %ss)，跳过语义确认并使用机器证据回退",
                remaining_semantic,
                self._semantic_min_remaining_sec,
            )
            semantic_result = self._fallback_from_evidence(
                evidence,
                error=f"semantic_skipped_budget_remaining={remaining_semantic}",
            )
        else:
            semantic_result, llm_payload, raw_response = self._semantic_check(
                subgoal_description=subgoal_description,
                acceptance_criteria=constraints.semantic_goal or subgoal_description,
                old_state=old_state,
                new_state=new_state,
                constraint_evidence=evidence,
            )
        if semantic_result.decision:
            semantic_result.constraint_status = machine_result.constraint_status or ("passed" if semantic_result.constraint_passed else "failed")
            semantic_result.delta_facts = delta_facts
            semantic_result.boundary_evidence = (delta_facts.get("boundary_evidence") or {}) if isinstance(delta_facts, dict) else {}
            semantic_result.matched_signals = evidence.get("matched_signals") or []
            semantic_result.unmatched_signals = evidence.get("unmatched_signals") or []
            semantic_result.counter_signal_hits = evidence.get("counter_signal_hits") or []
            if semantic_result.decision == "success":
                semantic_result.semantic_status = "passed"
                semantic_result.suggested_next_action = "continue"
            elif semantic_result.decision == "uncertain":
                semantic_result.semantic_status = "uncertain"
                semantic_result.suggested_next_action = "observe_again"
            else:
                semantic_result.semantic_status = "failed"
                semantic_result.suggested_next_action = "replan"
        self._save_eval_artifact(
            subgoal_description=subgoal_description,
            acceptance_criteria=constraints.semantic_goal or subgoal_description,
            old_state=old_state,
            new_state=new_state,
            constraint_evidence=evidence,
            llm_result=llm_payload,
            raw_response=raw_response,
            final_result=semantic_result,
        )

        if not semantic_result.success:
            logger.info("语义确认未通过: %s", semantic_result.reason)
            return semantic_result

        logger.info("事后评估通过: '%s'", subgoal_description)
        return semantic_result


    def _check_constraints(
        self,
        constraints: PreConstraints,
        old_state: UIState,
        new_state: UIState,
        action: Optional[Action] = None,
    ) -> Tuple[EvalResult, Dict[str, Any]]:

        old_package = self._get_package(old_state)
        new_package = self._get_package(new_state)
        old_activity = self._get_activity(old_state)
        new_activity = self._get_activity(new_state)

        boundary = constraints.boundary_constraints or {}
        must_stay_in_app = bool(boundary.get("must_stay_in_app", False))
        expected_package = str(boundary.get("expected_package") or "").strip()
        forbidden_packages = list(boundary.get("forbidden_packages") or [])
        expected_activity_contains = str(boundary.get("expected_activity_contains") or "").strip()
        package_mismatch_severity = str(boundary.get("package_mismatch_severity") or "soft").strip().lower()
        if package_mismatch_severity not in {"soft", "hard"}:
            package_mismatch_severity = "soft"
        related_package_tokens = boundary.get("related_package_tokens") or []
        if not isinstance(related_package_tokens, list):
            related_package_tokens = []
        related_package_tokens = self._normalize_package_tokens(related_package_tokens)
        if not related_package_tokens:
            related_package_tokens = self._extract_package_tokens(expected_package or old_package)

        hard_violations: List[str] = []
        soft_warnings: List[str] = []

        if must_stay_in_app and old_package and new_package and old_package != new_package:
            relation = self._classify_package_relation(old_package, new_package, related_package_tokens)
            if package_mismatch_severity == "hard" and relation == "unrelated":
                hard_violations.append(f"must_stay_in_app violated(hard,{relation}): '{old_package}' -> '{new_package}'")
            else:
                soft_warnings.append(f"must_stay_in_app drift(soft,{relation}): '{old_package}' -> '{new_package}'")

        if expected_package and new_package and new_package != expected_package:
            relation = self._classify_package_relation(expected_package, new_package, related_package_tokens)
            if package_mismatch_severity == "hard" and relation == "unrelated":
                hard_violations.append(f"expected_package mismatch(hard,{relation}): expected='{expected_package}', actual='{new_package}'")
            else:
                soft_warnings.append(f"expected_package mismatch(soft,{relation}): expected='{expected_package}', actual='{new_package}'")

        if new_package in forbidden_packages:
            hard_violations.append(f"forbidden_package opened: '{new_package}'")

        if expected_activity_contains and new_activity and expected_activity_contains not in new_activity:
            hard_violations.append(
                f"expected_activity_contains mismatch: expected_contains='{expected_activity_contains}', actual='{new_activity}'"
            )

        ui_changed = self._has_meaningful_ui_change(old_state, new_state)

        input_validation = None
        if action and action.action_type == "input" and action.text:
            input_validation = self._check_input_action_effect(
                action=action,
                old_state=old_state,
                new_state=new_state,
            )
            if input_validation.get("ok"):
                pass
            else:
                reason = str(input_validation.get("reason") or "input_validation_failed")
                mode = str(input_validation.get("mode") or "")
                if mode == "empty_input":
                    hard_violations.append(reason)
                else:
                    soft_warnings.append(f"input_validation_soft: {reason}")

        focus_validation = None
        if action and action.action_type in ("tap", "long_press"):
            focus_validation = self._check_focus_tap_effect(
                action=action,
                old_state=old_state,
                new_state=new_state,
            )

        evidence: Dict[str, Any] = {
            "old_package": old_package,
            "new_package": new_package,
            "old_activity": old_activity,
            "new_activity": new_activity,
            "boundary_constraints": boundary,
            "ui_changed": bool(ui_changed),
            "action_context": self._action_context(action),
            "input_validation": input_validation,
            "focus_validation": focus_validation,
            "hard_violations": hard_violations,
            "soft_warnings": soft_warnings,
            "package_relation": self._classify_package_relation(old_package, new_package, related_package_tokens),
            "package_mismatch_severity": package_mismatch_severity,
        }

        if hard_violations:
            return (
                EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason="; ".join(hard_violations),
                    decision="fail",
                    confidence=1.0,
                    constraint_status="failed",
                    semantic_status="failed",
                    suggested_next_action="backtrack",
                    matched_signals=[],
                    unmatched_signals=[],
                    counter_signal_hits=[],
                    delta_facts={},
                    boundary_evidence={},
                ),
                evidence,
            )

        return (
            EvalResult(
                success=True,
                constraint_passed=True,
                reason="evidence_collected",
                constraint_status="passed",
            ),
            evidence,
        )

    def _match_success_evidence_plan(self, success_evidence_plan: Dict[str, Any], delta_facts: Dict[str, Any]) -> Dict[str, Any]:
        plan = success_evidence_plan or {}
        required = plan.get("required_signals_any_of") or []
        supporting = plan.get("supporting_signals_any_of") or []
        counter = plan.get("counter_signals_any_of") or []

        if not isinstance(required, list):
            required = []
        if not isinstance(supporting, list):
            supporting = []
        if not isinstance(counter, list):
            counter = []

        matched_signals: List[Dict[str, Any]] = []
        unmatched_signals: List[Dict[str, Any]] = []
        counter_hits: List[Dict[str, Any]] = []

        required_weight = 0.0
        supporting_weight = 0.0
        counter_weight = 0.0
        required_records: List[Dict[str, Any]] = []

        for sig in required:
            ok, score, reason = self._match_signal(sig, delta_facts)
            record = {"signal": sig, "matched": bool(ok), "score": float(score), "reason": reason}
            required_records.append(record)
            if ok:
                matched_signals.append(record)
                required_weight += score
            else:
                unmatched_signals.append(record)

        for sig in supporting:
            ok, score, reason = self._match_signal(sig, delta_facts)
            record = {"signal": sig, "matched": bool(ok), "score": float(score), "reason": reason}
            if ok:
                matched_signals.append(record)
                supporting_weight += score
            else:
                unmatched_signals.append(record)

        for sig in counter:
            ok, score, reason = self._match_signal(sig, delta_facts)
            record = {"signal": sig, "matched": bool(ok), "score": float(score), "reason": reason}
            if ok:
                counter_hits.append(record)
                counter_weight += score

        required_met = required_weight >= 1.0 or any(
            rec.get("matched") and not bool((rec.get("signal") or {}).get("optional", False))
            for rec in required_records
        )

        return {
            "required_weight": required_weight,
            "supporting_weight": supporting_weight,
            "counter_weight": counter_weight,
            "required_met": bool(required_met),
            "matched_signals": matched_signals,
            "unmatched_signals": unmatched_signals,
            "counter_signal_hits": counter_hits,
        }

    def _match_signal(self, signal: Dict[str, Any], delta_facts: Dict[str, Any]) -> Tuple[bool, float, str]:
        if not isinstance(signal, dict):
            return False, 0.0, "invalid_signal"

        type_ = str(signal.get("type") or "").strip()
        operator = str(signal.get("operator") or "").strip()
        scope = str(signal.get("scope") or "").strip().lower()
        target = signal.get("target") or {}
        value = str(signal.get("value") or "")
        try:
            weight = float(signal.get("weight", 1.0))
        except Exception:
            weight = 1.0

        if not isinstance(target, dict):
            target = {}

        anchor_evidence = (delta_facts.get("anchor_evidence") or {}) if isinstance(delta_facts, dict) else {}
        delta_evidence = (delta_facts.get("delta_evidence") or {}) if isinstance(delta_facts, dict) else {}
        boundary_evidence = (delta_facts.get("boundary_evidence") or {}) if isinstance(delta_facts, dict) else {}
        spatial = (delta_facts.get("spatial_anchor_evidence") or {}) if isinstance(delta_facts, dict) else {}

        spatial_usable = bool(spatial.get("spatial_anchor_usable", True))
        if scope == "anchor" and not spatial_usable:
            return False, 0.0, "spatial_anchor_unusable"

        def _text_contains(haystack: List[str], needle: str) -> bool:
            n = (needle or "").strip().lower()
            if not n:
                return False
            return any(n in (t or "").lower() for t in (haystack or []))

        if type_ == "region_changed":
            if operator == "changed":
                target_changed = bool(anchor_evidence.get("target_region_changed", False))
                local_diff = float(spatial.get("local_visual_change_score", 0.0) or 0.0)
                global_diff = float(spatial.get("global_visual_change_score", 0.0) or 0.0)
                ui_changed = bool(delta_evidence.get("ui_changed_meaningfully", False))

                if scope == "anchor":
                    ok = target_changed
                    return ok, weight if ok else 0.0, "target_region_changed" if ok else "target_region_unchanged"
                if scope == "local":
                    # 本地视觉差异必须伴随“语义变化”才可计分，避免把动画/噪声当成功证据。
                    ok = target_changed or (ui_changed and local_diff >= 8.0)
                    return ok, weight if ok else 0.0, "local_region_changed" if ok else "local_region_unchanged"

                ok = target_changed or ui_changed or (ui_changed and global_diff >= 8.0)
                return ok, weight if ok else 0.0, "global_region_changed" if ok else "global_region_unchanged"
            return False, 0.0, "unsupported_operator"

        if type_ == "overlay_changed":
            if operator == "changed":
                ok = bool(delta_evidence.get("overlay_changed", False))
                return ok, weight if ok else 0.0, "overlay_changed" if ok else "overlay_unchanged"
            return False, 0.0, "unsupported_operator"

        if type_ == "package_changed":
            if operator == "changed":
                ok = bool(delta_evidence.get("package_changed", False))
                return ok, weight if ok else 0.0, "package_changed" if ok else "package_unchanged"
            return False, 0.0, "unsupported_operator"

        if type_ == "activity_changed":
            if operator == "changed":
                ok = bool(delta_evidence.get("activity_changed", False))
                return ok, weight if ok else 0.0, "activity_changed" if ok else "activity_unchanged"
            return False, 0.0, "unsupported_operator"

        if type_ == "keyboard_changed":
            if operator == "appeared":
                ok = bool(delta_evidence.get("keyboard_appeared", False))
                return ok, weight if ok else 0.0, "keyboard_appeared" if ok else "keyboard_not_appeared"
            if operator == "disappeared":
                ok = bool(delta_evidence.get("keyboard_disappeared", False))
                return ok, weight if ok else 0.0, "keyboard_disappeared" if ok else "keyboard_not_disappeared"
            if operator == "changed":
                ok = bool(delta_evidence.get("keyboard_before", False)) != bool(delta_evidence.get("keyboard_after", False))
                return ok, weight if ok else 0.0, "keyboard_changed" if ok else "keyboard_unchanged"
            return False, 0.0, "unsupported_operator"

        if type_ == "focus_changed":
            if operator == "changed":
                ok = bool(delta_evidence.get("focus_changed", False))
                if target.get("class"):
                    ok = ok and bool(delta_evidence.get("focused_editable_exists_after", False))
                return ok, weight if ok else 0.0, "focus_changed" if ok else "focus_not_changed"
            return False, 0.0, "unsupported_operator"

        if type_ == "text_appeared":
            appeared = delta_evidence.get("appeared_texts") or []
            token = str((target.get("text") or value) or "").strip()
            if operator in {"exists", "contains"}:
                ok = _text_contains(appeared, token)
                return ok, weight if ok else 0.0, "text_appeared" if ok else "text_not_appeared"
            return False, 0.0, "unsupported_operator"

        if type_ == "text_disappeared":
            vanished = delta_evidence.get("vanished_texts") or []
            token = str((target.get("text") or value) or "").strip()
            if operator in {"exists", "contains", "disappeared"}:
                ok = _text_contains(vanished, token)
                return ok, weight if ok else 0.0, "text_disappeared" if ok else "text_not_disappeared"
            return False, 0.0, "unsupported_operator"

        if type_ in {"widget_appeared", "widget_disappeared"}:
            appeared = delta_evidence.get("new_interactive_widgets") or []
            vanished = delta_evidence.get("vanished_interactive_widgets") or []
            clazz = str((target.get("class") or value) or "").strip()
            if operator in {"appeared", "exists", "contains"} and type_ == "widget_appeared":
                ok = _text_contains(appeared, clazz)
                return ok, weight if ok else 0.0, "widget_appeared" if ok else "widget_not_appeared"
            if operator in {"disappeared", "exists", "contains"} and type_ == "widget_disappeared":
                ok = _text_contains(vanished, clazz)
                return ok, weight if ok else 0.0, "widget_disappeared" if ok else "widget_not_disappeared"
            return False, 0.0, "unsupported_operator"

        if type_ == "risk_ui_detected":
            risks = boundary_evidence.get("risk_types_detected") or []
            if not isinstance(risks, list):
                risks = []
            risk_target = str((target.get("risk") or value) or "").strip().lower()
            if operator in {"exists", "contains"}:
                if risk_target:
                    ok = any(risk_target == str(r).lower() for r in risks)
                else:
                    ok = bool(boundary_evidence.get("risk_ui_detected", False))
                return ok, weight if ok else 0.0, "risk_ui_detected" if ok else "risk_not_detected"
            return False, 0.0, "unsupported_operator"

        return False, 0.0, "unknown_type"

    def _build_machine_eval_result(
        self,
        constraints: PreConstraints,
        delta_facts: Dict[str, Any],
        match: Dict[str, Any],
        base_evidence: Dict[str, Any],
    ) -> EvalResult:
        boundary = constraints.boundary_constraints or {}
        forbidden_risks = boundary.get("forbidden_ui_risks") or []
        if not isinstance(forbidden_risks, list):
            forbidden_risks = []
        forbidden_risks = {str(v).strip().lower() for v in forbidden_risks if str(v).strip()}

        boundary_evidence = (delta_facts.get("boundary_evidence") or {}) if isinstance(delta_facts, dict) else {}
        risk_types = boundary_evidence.get("risk_types_detected") or []
        if not isinstance(risk_types, list):
            risk_types = []
        risk_types_norm = {str(v).strip().lower() for v in risk_types if str(v).strip()}

        violated_risks = sorted(list(forbidden_risks & risk_types_norm))
        if violated_risks:
            return EvalResult(
                success=False,
                reason=f"forbidden_ui_risks hit: {violated_risks[:5]}",
                constraint_passed=False,
                semantic_passed=False,
                decision="fail",
                confidence=1.0,
                needs_more_observation=False,
                constraint_status="failed",
                semantic_status="failed",
                suggested_next_action="backtrack",
                matched_signals=match.get("matched_signals") or [],
                unmatched_signals=match.get("unmatched_signals") or [],
                counter_signal_hits=match.get("counter_signal_hits") or [],
                delta_facts=delta_facts,
                boundary_evidence=boundary_evidence,
            )

        counter_hits = match.get("counter_signal_hits") or []
        counter_weight = float(match.get("counter_weight", 0.0) or 0.0)
        required_met = bool(match.get("required_met", False))
        required_weight = float(match.get("required_weight", 0.0) or 0.0)
        supporting_weight = float(match.get("supporting_weight", 0.0) or 0.0)
        soft_warnings = list(base_evidence.get("soft_warnings") or [])
        delta_evidence = (delta_facts.get("delta_evidence") or {}) if isinstance(delta_facts, dict) else {}
        anchor_evidence = (delta_facts.get("anchor_evidence") or {}) if isinstance(delta_facts, dict) else {}
        old_package = str(base_evidence.get("old_package") or "")
        new_package = str(base_evidence.get("new_package") or "")
        expected_package = str(boundary.get("expected_package") or "").strip()
        action_context = base_evidence.get("action_context") or {}
        action_type = str(action_context.get("action_type") or "").strip().lower()
        semantic_goal_text = str(getattr(constraints, "semantic_goal", "") or "").strip().lower()

        context_transition_bonus = 0.0
        activity_transition_bonus = 0.0
        toggle_state_bonus = 0.0
        focus_state_bonus = 0.0
        control_toggle_bonus = 0.0
        support = required_weight + 0.6 * supporting_weight
        against = 0.9 * counter_weight + min(1.2, 0.3 * len(soft_warnings))

        no_change = bool(delta_evidence.get("no_meaningful_change", False))
        activity_or_package_changed = bool(delta_evidence.get("activity_changed", False)) or bool(delta_evidence.get("package_changed", False))
        focus_or_keyboard_changed = bool(delta_evidence.get("focus_changed", False)) or bool(delta_evidence.get("keyboard_appeared", False)) or bool(delta_evidence.get("keyboard_disappeared", False))
        focused_editable_before = bool(delta_evidence.get("focused_editable_exists_before", False))
        focused_editable_after = bool(delta_evidence.get("focused_editable_exists_after", False))
        anchor_changed = bool(anchor_evidence.get("target_region_changed", False))
        switch_state_changed = bool(delta_evidence.get("switch_state_changed", False))
        try:
            switch_checked_delta = int(delta_evidence.get("switch_checked_delta", 0) or 0)
        except Exception:
            switch_checked_delta = 0
        low_observability_actions = {"tap", "long_press", "swipe", "scroll", "scroll_up", "scroll_down"}
        should_soften_no_change = action_type in low_observability_actions

        # 不再按场景特判 success：仅把“到达预期包”作为变化证据加权。
        transitioned_into_expected_package = bool(
            expected_package
            and old_package
            and old_package != expected_package
            and new_package == expected_package
        )
        if transitioned_into_expected_package:
            context_transition_bonus = 0.8
            support += context_transition_bonus

        # 导航型动作在同包内切换 activity 也属于有效变化证据，避免被焦点类信号过严否决。
        if bool(delta_evidence.get("activity_changed", False)) and action_type in {"tap", "back", "long_press"}:
            activity_transition_bonus = 0.55
            nav_intent_markers = (
                "open",
                "enter",
                "go to",
                "navigate",
                "return to",
                "进入",
                "打开",
                "前往",
                "返回",
                "add phone",
                "new contact",
            )
            if any(marker in semantic_goal_text for marker in nav_intent_markers):
                activity_transition_bonus += 0.25
            support += activity_transition_bonus

        focus_intent = self._infer_focus_intent(
            semantic_goal_text=semantic_goal_text,
            action_context=action_context,
        )
        focus_intent_satisfied = False
        if focus_intent and focused_editable_after:
            if bool(delta_evidence.get("focus_changed", False)) or bool(delta_evidence.get("keyboard_appeared", False)):
                focus_state_bonus += 0.75
                focus_intent_satisfied = True
            elif focused_editable_before and (bool(delta_evidence.get("keyboard_before", False)) or bool(delta_evidence.get("keyboard_after", False))):
                # 允许“状态已满足”的幂等点击：即便变化很小，也不应反复重试同一焦点动作。
                focus_state_bonus += 0.65
                focus_intent_satisfied = True
        support += focus_state_bonus

        toggle_expectation = self._infer_toggle_expectation(semantic_goal_text)
        if toggle_expectation:
            if switch_state_changed:
                toggle_state_bonus += 0.65
            if toggle_expectation == "on":
                if switch_checked_delta > 0:
                    toggle_state_bonus += 0.9
                elif switch_checked_delta < 0:
                    against += 0.9
            elif toggle_expectation == "off":
                if switch_checked_delta < 0:
                    toggle_state_bonus += 0.9
                elif switch_checked_delta > 0:
                    against += 0.9
            support += toggle_state_bonus

        # 通用控制态切换证据：例如 pause -> resume / resume -> pause。
        appeared_texts = [str(v).strip().lower() for v in (delta_evidence.get("appeared_texts") or []) if str(v).strip()]
        vanished_texts = [str(v).strip().lower() for v in (delta_evidence.get("vanished_texts") or []) if str(v).strip()]
        if action_type in {"tap", "long_press"}:
            pause_markers = ("pause", "paused", "暂停", "stopwatch", "秒表")
            resume_markers = ("resume", "start", "继续", "开始")
            if any(marker in semantic_goal_text for marker in pause_markers):
                if self._text_list_contains_any(appeared_texts, resume_markers):
                    control_toggle_bonus += 0.95
                if self._text_list_contains_any(vanished_texts, ("pause", "暂停", "stop")):
                    control_toggle_bonus += 0.55
            if any(marker in semantic_goal_text for marker in ("resume", "继续", "start stopwatch", "开始秒表")):
                if self._text_list_contains_any(appeared_texts, ("pause", "暂停", "stop")):
                    control_toggle_bonus += 0.9
            support += control_toggle_bonus

        # 对“控制态切换”子目标，允许在变化证据充分时直接成功，避免被语义补判误拖入不确定循环。
        control_toggle_intent = (
            action_type in {"tap", "long_press"}
            and any(
                marker in semantic_goal_text
                for marker in (
                    "pause",
                    "resume",
                    "stopwatch",
                    "暂停",
                    "继续",
                    "开始秒表",
                )
            )
        )
        if control_toggle_intent and support >= 0.75 and against < 1.2:
            return EvalResult(
                success=True,
                reason=(
                    f"control_toggle_verified: support={support:.2f}, against={against:.2f}, "
                    f"control_toggle_bonus={control_toggle_bonus:.2f}, "
                    f"ui_changed={bool(delta_evidence.get('ui_changed_meaningfully', False))}"
                ),
                constraint_passed=True,
                semantic_passed=True,
                decision="success",
                confidence=min(1.0, 0.62 + 0.1 * support),
                needs_more_observation=False,
                constraint_status="passed",
                semantic_status="passed",
                suggested_next_action="continue",
                matched_signals=match.get("matched_signals") or [],
                unmatched_signals=match.get("unmatched_signals") or [],
                counter_signal_hits=match.get("counter_signal_hits") or [],
                delta_facts=delta_facts,
                boundary_evidence=boundary_evidence,
            )

        # 变化验证优先：若关键变化证据几乎为零，直接判失败，避免无意义语义补判。
        if (
            no_change
            and not activity_or_package_changed
            and not focus_or_keyboard_changed
            and not anchor_changed
            and not switch_state_changed
            and not focus_intent_satisfied
            and required_weight < 0.55
            and supporting_weight < 0.45
            and counter_weight <= 0.0
        ):
            if should_soften_no_change:
                return EvalResult(
                    success=False,
                    reason=(
                        "low_change_evidence: core change signals absent; "
                        "observe_again_before_fail"
                    ),
                    constraint_passed=True,
                    semantic_passed=False,
                    decision="uncertain",
                    confidence=0.58,
                    needs_more_observation=True,
                    constraint_status="passed",
                    semantic_status="uncertain",
                    suggested_next_action="observe_again",
                    matched_signals=match.get("matched_signals") or [],
                    unmatched_signals=match.get("unmatched_signals") or [],
                    counter_signal_hits=counter_hits,
                    delta_facts=delta_facts,
                    boundary_evidence=boundary_evidence,
                )
            return EvalResult(
                success=False,
                reason=(
                    "no_meaningful_change: core change signals absent; "
                    "skip semantic fallback"
                ),
                constraint_passed=False,
                semantic_passed=False,
                decision="fail",
                confidence=0.9,
                needs_more_observation=False,
                constraint_status="failed",
                semantic_status="failed",
                suggested_next_action="retry",
                matched_signals=match.get("matched_signals") or [],
                unmatched_signals=match.get("unmatched_signals") or [],
                counter_signal_hits=counter_hits,
                delta_facts=delta_facts,
                boundary_evidence=boundary_evidence,
            )

        weak_unattributed_change = (
            no_change
            and not activity_or_package_changed
            and not focus_or_keyboard_changed
            and not anchor_changed
            and not switch_state_changed
            and not focus_intent_satisfied
            and required_met
            and required_weight <= 1.05
            and supporting_weight < 0.6
            and counter_weight <= 0.0
        )
        if weak_unattributed_change:
            if should_soften_no_change:
                return EvalResult(
                    success=False,
                    reason=(
                        "weak_change_evidence: only weak local delta observed; "
                        "observe_again_before_replan"
                    ),
                    constraint_passed=True,
                    semantic_passed=False,
                    decision="uncertain",
                    confidence=0.56,
                    needs_more_observation=True,
                    constraint_status="passed",
                    semantic_status="uncertain",
                    suggested_next_action="observe_again",
                    matched_signals=match.get("matched_signals") or [],
                    unmatched_signals=match.get("unmatched_signals") or [],
                    counter_signal_hits=counter_hits,
                    delta_facts=delta_facts,
                    boundary_evidence=boundary_evidence,
                )
            return EvalResult(
                success=False,
                reason=(
                    "no_action_attributed_change: only weak local visual delta matched; "
                    "no meaningful state transition observed"
                ),
                constraint_passed=False,
                semantic_passed=False,
                decision="fail",
                confidence=0.8,
                needs_more_observation=False,
                constraint_status="failed",
                semantic_status="failed",
                suggested_next_action="replan",
                matched_signals=match.get("matched_signals") or [],
                unmatched_signals=match.get("unmatched_signals") or [],
                counter_signal_hits=counter_hits,
                delta_facts=delta_facts,
                boundary_evidence=boundary_evidence,
            )

        if required_met and support >= 0.8 and against < 1.4:
            decision = "success"
            confidence = min(1.0, 0.55 + 0.15 * support)
            suggested = "continue"
        elif support >= 1.2 and against < (support + 0.6):
            decision = "success"
            confidence = min(1.0, 0.52 + 0.12 * support)
            suggested = "continue"
        elif against >= 2.2:
            decision = "fail"
            confidence = min(1.0, 0.75 + 0.08 * against)
            suggested = "backtrack" if boundary.get("must_stay_in_app", False) else "replan"
        elif against >= 1.8 and support < 0.6:
            decision = "fail"
            confidence = min(1.0, 0.65 + 0.08 * against)
            suggested = "replan"
        else:
            decision = "uncertain"
            confidence = max(0.0, min(1.0, 0.45 + 0.1 * (support - against)))
            suggested = "observe_again"

        semantic_status = "passed" if decision == "success" else ("failed" if decision == "fail" else "uncertain")

        return EvalResult(
            success=decision == "success",
            reason=(
                f"evidence: support={support:.2f}, against={against:.2f}, "
                f"required_met={required_met}, soft_warnings={len(soft_warnings)}, "
                f"context_transition_bonus={context_transition_bonus:.2f}, "
                f"activity_transition_bonus={activity_transition_bonus:.2f}, "
                f"focus_state_bonus={focus_state_bonus:.2f}, "
                f"toggle_state_bonus={toggle_state_bonus:.2f}, "
                f"control_toggle_bonus={control_toggle_bonus:.2f}, "
                f"switch_checked_delta={switch_checked_delta}"
            ),
            constraint_passed=True,
            semantic_passed=decision == "success",
            decision=decision,
            confidence=confidence,
            needs_more_observation=decision == "uncertain",
            constraint_status="passed",
            semantic_status=semantic_status,
            suggested_next_action=suggested,
            matched_signals=match.get("matched_signals") or [],
            unmatched_signals=match.get("unmatched_signals") or [],
            counter_signal_hits=match.get("counter_signal_hits") or [],
            delta_facts=delta_facts,
            boundary_evidence=boundary_evidence,
        )

    def _semantic_check(
        self,
        subgoal_description: str,
        acceptance_criteria: str,
        old_state: UIState,
        new_state: UIState,
        constraint_evidence: Dict[str, Any],
    ) -> Tuple[EvalResult, Optional[Dict[str, Any]], Optional[str]]:
        success_evidence_plan = constraint_evidence.get("success_evidence_plan") or {}
        delta_facts = constraint_evidence.get("delta_facts") or {}
        prompt = EVALUATOR_PROMPT.format(
            subgoal_description=subgoal_description,
            acceptance_criteria=acceptance_criteria,
            success_evidence_plan=json.dumps(success_evidence_plan, ensure_ascii=False),
            delta_facts=json.dumps(delta_facts, ensure_ascii=False),
            constraint_evidence=json.dumps(constraint_evidence, ensure_ascii=False),
            old_state_summary=old_state.to_prompt_text()[:1400],
            new_state_summary=new_state.to_prompt_text()[:1400],
        )

        try:
            semantic_timeout = self._select_semantic_timeout(
                default_timeout=self._semantic_timeout_sec,
                reserve_seconds=6,
            )
            response = self.llm.chat(prompt, timeout=semantic_timeout)
            result, payload = self._parse_semantic_response(response)
            return result, payload, response
        except Exception as e:
            logger.error("LLM 语义确认失败: %s", e)
            fallback = self._fallback_from_evidence(constraint_evidence, error=str(e))
            return fallback, None, None

    def _remaining_task_seconds(self) -> Optional[int]:
        getter = getattr(self.llm, "remaining_seconds", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:
            return None
        if value is None:
            return None
        try:
            remaining = int(value)
        except Exception:
            return None
        return max(0, remaining)

    def _select_semantic_timeout(self, default_timeout: int, reserve_seconds: int = 6) -> int:
        timeout = max(1, int(default_timeout or 1))
        remaining = self._remaining_task_seconds()
        if remaining is None:
            return timeout
        budget = max(1, remaining - max(0, int(reserve_seconds or 0)))
        return max(1, min(timeout, budget))

    def _should_skip_semantic_check(self) -> Tuple[bool, Optional[int]]:
        remaining = self._remaining_task_seconds()
        if remaining is None:
            return False, None
        return remaining < self._semantic_min_remaining_sec, remaining

    def _parse_semantic_response(self, response: str) -> Tuple[EvalResult, Optional[Dict[str, Any]]]:
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
            decision = str(data.get("decision") or "").strip().lower()
            if decision not in {"success", "fail", "uncertain"}:
                return (
                    EvalResult(
                        success=False,
                        semantic_passed=False,
                        reason="语义确认返回缺少合法 decision 字段",
                        decision="fail",
                        confidence=0.0,
                    ),
                    data,
                )

            confidence_raw = data.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0

            is_success = decision == "success"
            return (
                EvalResult(
                    success=is_success,
                    semantic_passed=is_success,
                    reason=str(data.get("reason", "")),
                    decision=decision,
                    confidence=max(0.0, min(1.0, confidence)),
                    needs_more_observation=decision == "uncertain",
                ),
                data,
            )
        except json.JSONDecodeError as e:
            logger.warning("语义确认 JSON 解析失败: %s", e)
            return (
                EvalResult(
                    success=False,
                    semantic_passed=False,
                    reason="JSON解析失败",
                    decision="fail",
                    confidence=0.0,
                ),
                None,
            )

    def _fallback_from_evidence(self, evidence: Dict[str, Any], error: str = "") -> EvalResult:
        hard_violations = evidence.get("hard_violations") or []
        if hard_violations:
            return EvalResult(
                success=False,
                semantic_passed=False,
                reason="; ".join([*hard_violations, error] if error else hard_violations),
                decision="fail",
                confidence=0.9,
                constraint_status="failed",
                semantic_status="failed",
                suggested_next_action="backtrack",
                matched_signals=evidence.get("matched_signals") or [],
                unmatched_signals=evidence.get("unmatched_signals") or [],
                counter_signal_hits=evidence.get("counter_signal_hits") or [],
                delta_facts=evidence.get("delta_facts") or {},
                boundary_evidence=(evidence.get("delta_facts") or {}).get("boundary_evidence") if isinstance(evidence.get("delta_facts"), dict) else {},
            )

        ui_changed = bool(evidence.get("ui_changed", False))
        delta_facts = evidence.get("delta_facts") or {}
        delta_evidence = (delta_facts.get("delta_evidence") or {}) if isinstance(delta_facts, dict) else {}
        anchor_evidence = (delta_facts.get("anchor_evidence") or {}) if isinstance(delta_facts, dict) else {}
        boundary_evidence = (delta_facts.get("boundary_evidence") or {}) if isinstance(delta_facts, dict) else {}

        support = 0.0
        against = 0.0

        if ui_changed or bool(delta_evidence.get("ui_changed_meaningfully", False)):
            support += 0.5
        if bool(anchor_evidence.get("target_region_changed", False)):
            support += 0.5
        if bool(delta_evidence.get("keyboard_appeared", False)):
            support += 0.5
        if bool(delta_evidence.get("focused_editable_exists_after", False)):
            support += 0.5
        if bool(delta_evidence.get("activity_changed", False)):
            support += 0.3

        if bool(delta_evidence.get("no_meaningful_change", False)) or (not ui_changed):
            against += 0.6
        if bool(boundary_evidence.get("risk_ui_detected", False)):
            against += 1.2
        if boundary_evidence.get("stayed_in_app") is False:
            against += 1.2
        if bool(boundary_evidence.get("entered_forbidden_package", False)):
            against += 1.2

        if against >= 1.2:
            decision = "fail"
            confidence = 0.8
        elif support >= 1.0:
            decision = "success"
            confidence = 0.55
        elif (not ui_changed) and support < 0.5:
            decision = "fail"
            confidence = 0.55
        else:
            decision = "uncertain"
            confidence = 0.4

        reason_parts = [
            f"fallback: ui_changed={ui_changed}",
            f"support={support:.1f}",
            f"against={against:.1f}",
        ]
        if error:
            reason_parts.append(f"llm_error={error}")

        return EvalResult(
            success=decision == "success",
            semantic_passed=decision == "success",
            reason="; ".join(reason_parts),
            decision=decision,
            confidence=confidence,
            needs_more_observation=decision == "uncertain",
            constraint_status="passed",
            semantic_status="passed" if decision == "success" else ("failed" if decision == "fail" else "uncertain"),
            suggested_next_action="continue" if decision == "success" else ("observe_again" if decision == "uncertain" else "replan"),
            matched_signals=evidence.get("matched_signals") or [],
            unmatched_signals=evidence.get("unmatched_signals") or [],
            counter_signal_hits=evidence.get("counter_signal_hits") or [],
            delta_facts=evidence.get("delta_facts") or {},
            boundary_evidence=(evidence.get("delta_facts") or {}).get("boundary_evidence") if isinstance(evidence.get("delta_facts"), dict) else {},
        )

    def _save_eval_artifact(
        self,
        subgoal_description: str,
        acceptance_criteria: str,
        old_state: UIState,
        new_state: UIState,
        constraint_evidence: Dict[str, Any],
        llm_result: Optional[Dict[str, Any]],
        raw_response: Optional[str],
        final_result: EvalResult,
    ) -> None:
        screenshot_path = getattr(new_state, "screenshot_path", "") or ""
        if not screenshot_path:
            return

        data_dir = os.path.dirname(os.path.dirname(screenshot_path))
        context_dir = os.path.join(data_dir, "context")
        os.makedirs(context_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(screenshot_path))[0] or "eval"
        out_path = os.path.join(context_dir, f"{stem}.eval.json")

        payload = {
            "subgoal": subgoal_description,
            "acceptance_criteria": acceptance_criteria,
            "old_screenshot_path": getattr(old_state, "screenshot_path", "") or "",
            "new_screenshot_path": screenshot_path,
            "constraint_evidence": constraint_evidence,
            "llm_result": llm_result,
            "raw_response": raw_response,
            "final": {
                "success": bool(final_result.success),
                "reason": final_result.reason,
                "constraint_passed": bool(final_result.constraint_passed),
                "semantic_passed": bool(final_result.semantic_passed),
                "decision": (final_result.decision or ("success" if final_result.success else "fail")),
                "confidence": float(getattr(final_result, "confidence", 0.0) or 0.0),
                "needs_more_observation": bool(getattr(final_result, "needs_more_observation", False)),
                "constraint_status": str(getattr(final_result, "constraint_status", "") or ""),
                "semantic_status": str(getattr(final_result, "semantic_status", "") or ""),
                "suggested_next_action": str(getattr(final_result, "suggested_next_action", "") or ""),
            },
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存评估复盘文件失败: %s", e)

    # -----------------------
    # 辅助函数
    # -----------------------

    def _get_package(self, state: UIState) -> str:
        return getattr(state, "package", "") or getattr(state, "package_name", "") or ""

    def _get_activity(self, state: UIState) -> str:
        return getattr(state, "activity", "") or getattr(state, "activity_name", "") or ""

    def _collect_texts(self, state: UIState) -> List[str]:
        texts = []
        for w in getattr(state, "widgets", []):
            if getattr(w, "text", ""):
                texts.append(w.text)
            if getattr(w, "content_desc", ""):
                texts.append(w.content_desc)
        return texts

    def _text_exists(self, target: str, texts: List[str]) -> bool:
        target = (target or "").strip().lower()
        if not target:
            return False
        return any(target in (t or "").lower() for t in texts)

    def _text_list_contains_any(self, texts: List[str], needles: tuple) -> bool:
        values = [str(t or "").strip().lower() for t in (texts or []) if str(t or "").strip()]
        if not values:
            return False
        for needle in needles:
            token = str(needle or "").strip().lower()
            if not token:
                continue
            if any(token in value for value in values):
                return True
        return False

    def _infer_toggle_expectation(self, semantic_goal_text: str) -> str:
        text = str(semantic_goal_text or "").strip().lower()
        if not text:
            return ""

        toggle_markers = (
            "switch",
            "toggle",
            "wifi",
            "wi-fi",
            "wlan",
            "bluetooth",
            "开关",
            "蓝牙",
            "无线网",
        )
        if not any(marker in text for marker in toggle_markers):
            return ""

        on_markers = (
            "turn on",
            "switch on",
            "enable",
            "开启",
            "打开",
            "启用",
        )
        off_markers = (
            "turn off",
            "switch off",
            "disable",
            "关闭",
            "关掉",
            "停用",
        )

        has_on = any(marker in text for marker in on_markers)
        has_off = any(marker in text for marker in off_markers)
        if has_on and not has_off:
            return "on"
        if has_off and not has_on:
            return "off"
        return ""

    def _infer_focus_intent(self, semantic_goal_text: str, action_context: Dict[str, Any]) -> bool:
        text = str(semantic_goal_text or "").strip().lower()
        action_type = str((action_context or {}).get("action_type") or "").strip().lower()
        target_class = str((action_context or {}).get("target_class_name") or "").strip().lower()
        if action_type not in {"tap", "long_press"}:
            return False
        if "edittext" in target_class:
            return True
        markers = (
            "focus",
            "cursor",
            "input field",
            "text field",
            "editable",
            "聚焦",
            "光标",
            "输入框",
            "编辑框",
        )
        return any(marker in text for marker in markers)

    def _collect_switch_state_map(self, widgets: List[Any]) -> Dict[str, bool]:
        states: Dict[str, bool] = {}
        for w in widgets or []:
            cls = str(getattr(w, "class_name", "") or "").lower()
            rid = str(getattr(w, "resource_id", "") or "").lower()
            text = str(getattr(w, "text", "") or getattr(w, "content_desc", "") or "").lower()

            switch_like = (
                bool(getattr(w, "checkable", False))
                or "switch" in cls
                or "switch" in rid
                or "toggle" in rid
            )
            if not switch_like:
                continue

            try:
                cx, cy = getattr(w, "center", (0, 0))
                x_bucket = int(int(cx) // 80)
                y_bucket = int(int(cy) // 80)
            except Exception:
                x_bucket, y_bucket = 0, 0

            rid_tail = rid.split("/")[-1] if rid else ""
            cls_tail = cls.split(".")[-1] if cls else ""
            text_token = self._compact_text_token(text)[:24].lower() if text else ""
            key = "|".join(
                [
                    rid_tail or "no_rid",
                    cls_tail or "no_cls",
                    text_token or "no_text",
                    f"{x_bucket}:{y_bucket}",
                ]
            )
            states[key] = bool(getattr(w, "checked", False))
        return states

    def _widget_feature_exists(self, feature: str, state: UIState) -> bool:
        feature = (feature or "").strip().lower()
        if not feature:
            return False

        for w in getattr(state, "widgets", []):
            text = getattr(w, "text", "") or ""
            desc = getattr(w, "content_desc", "") or ""
            rid = getattr(w, "resource_id", "") or ""
            clazz = getattr(w, "class_name", "") or ""

            merged = " | ".join([text, desc, rid, clazz]).lower()
            if feature in merged:
                return True
        return False

    def _action_context(self, action: Optional[Action]) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        return {
            "action_type": action.action_type,
            "widget_id": action.widget_id,
            "target_widget_text": action.target_widget_text,
            "target_resource_id": action.target_resource_id,
            "target_class_name": action.target_class_name,
            "target_content_desc": action.target_content_desc,
            "target_bounds": list(action.target_bounds) if action.target_bounds else [0, 0, 0, 0],
            "text_preview": (action.text or "")[:120],
            "description": action.description,
        }

    def _normalize_package_tokens(self, tokens: List[Any]) -> List[str]:
        normalized: List[str] = []
        for token in tokens or []:
            value = str(token or "").strip().lower()
            if not value or value in normalized:
                continue
            normalized.append(value)
        return normalized[:8]

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

    def _classify_package_relation(self, reference_package: str, new_package: str, tokens: List[str]) -> str:
        ref = str(reference_package or "").strip().lower()
        new = str(new_package or "").strip().lower()
        if not ref or not new:
            return "unknown"
        if ref == new:
            return "same"

        normalized = self._normalize_package_tokens(tokens)
        if not normalized:
            normalized = self._extract_package_tokens(ref)
        for token in normalized:
            if token in ref and token in new:
                return "related"
        return "unrelated"

    def _extract_delta_facts(
        self,
        old_state: UIState,
        new_state: UIState,
        action: Optional[Action],
        constraints: PreConstraints,
        constraint_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        old_package = str(constraint_evidence.get("old_package") or self._get_package(old_state) or "")
        new_package = str(constraint_evidence.get("new_package") or self._get_package(new_state) or "")
        old_activity = str(constraint_evidence.get("old_activity") or self._get_activity(old_state) or "")
        new_activity = str(constraint_evidence.get("new_activity") or self._get_activity(new_state) or "")

        old_keyboard = bool(getattr(old_state, "keyboard_visible", False))
        new_keyboard = bool(getattr(new_state, "keyboard_visible", False))

        old_texts_raw = [self._compact_text_token(t) for t in self._collect_texts(old_state)]
        new_texts_raw = [self._compact_text_token(t) for t in self._collect_texts(new_state)]
        old_texts = {t for t in old_texts_raw if t}
        new_texts = {t for t in new_texts_raw if t}
        old_texts = self._filter_text_tokens(old_texts)
        new_texts = self._filter_text_tokens(new_texts)

        appeared_texts = sorted(list(new_texts - old_texts))[:30]
        vanished_texts = sorted(list(old_texts - new_texts))[:30]

        old_widgets = list(getattr(old_state, "widgets", []) or [])
        new_widgets = list(getattr(new_state, "widgets", []) or [])

        old_switch_states = self._collect_switch_state_map(old_widgets)
        new_switch_states = self._collect_switch_state_map(new_widgets)
        shared_switch_keys = sorted(list(set(old_switch_states.keys()) & set(new_switch_states.keys())))
        switch_changed_keys = [
            key for key in shared_switch_keys
            if bool(old_switch_states.get(key, False)) != bool(new_switch_states.get(key, False))
        ]
        old_switch_checked_count = sum(1 for v in old_switch_states.values() if bool(v))
        new_switch_checked_count = sum(1 for v in new_switch_states.values() if bool(v))
        switch_checked_delta = int(new_switch_checked_count - old_switch_checked_count)
        switch_state_changed = bool(switch_changed_keys)
        if not switch_state_changed and old_switch_states and new_switch_states:
            switch_state_changed = old_switch_checked_count != new_switch_checked_count

        def _count(pred, values):
            return sum(1 for v in values if pred(v))

        old_counts = {
            "widgets": len(old_widgets),
            "clickable": _count(lambda w: bool(getattr(w, "clickable", False)), old_widgets),
            "editable": _count(lambda w: bool(getattr(w, "editable", False)), old_widgets),
            "focused": _count(lambda w: bool(getattr(w, "focused", False)), old_widgets),
            "scrollable": _count(lambda w: bool(getattr(w, "scrollable", False)), old_widgets),
        }
        new_counts = {
            "widgets": len(new_widgets),
            "clickable": _count(lambda w: bool(getattr(w, "clickable", False)), new_widgets),
            "editable": _count(lambda w: bool(getattr(w, "editable", False)), new_widgets),
            "focused": _count(lambda w: bool(getattr(w, "focused", False)), new_widgets),
            "scrollable": _count(lambda w: bool(getattr(w, "scrollable", False)), new_widgets),
        }

        def _interactive_key(w) -> str:
            clazz = str(getattr(w, "class_name", "") or getattr(w, "category", "") or "")
            if "/" in clazz:
                clazz = clazz.split("/")[-1]
            clazz = clazz.split(".")[-1]
            return clazz[:60]

        old_interactive = {_interactive_key(w) for w in old_widgets if getattr(w, "clickable", False) or getattr(w, "focusable", False) or getattr(w, "editable", False)}
        new_interactive = {_interactive_key(w) for w in new_widgets if getattr(w, "clickable", False) or getattr(w, "focusable", False) or getattr(w, "editable", False)}
        new_interactive_widgets = sorted(list(new_interactive - old_interactive))[:30]
        vanished_interactive_widgets = sorted(list(old_interactive - new_interactive))[:30]

        focused_editable_exists_before = any(bool(getattr(w, "editable", False)) and bool(getattr(w, "focused", False)) for w in old_widgets)
        focused_editable_exists = any(bool(getattr(w, "editable", False)) and bool(getattr(w, "focused", False)) for w in new_widgets)
        focus_changed = (old_counts["focused"] != new_counts["focused"]) or (
            focused_editable_exists_before
            != focused_editable_exists
        )

        overlay_changed = self._detect_overlay_changed(old_state, new_state)

        ui_changed_meaningfully = self._has_meaningful_ui_change(old_state, new_state)
        no_meaningful_change = not ui_changed_meaningfully

        anchor = constraints.action_anchor or {}
        anchor_center = None
        anchor_bounds = None
        if isinstance(anchor, dict):
            center = anchor.get("target_center_before")
            bounds = anchor.get("target_bounds_before")
            if isinstance(center, list) and len(center) == 2:
                try:
                    anchor_center = (int(center[0]), int(center[1]))
                except Exception:
                    anchor_center = None
            if isinstance(bounds, list) and len(bounds) == 4:
                try:
                    anchor_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
                except Exception:
                    anchor_bounds = None

        target_region_changed, target_region_stats = self._detect_target_region_changed(
            old_state=old_state,
            new_state=new_state,
            action=action,
            anchor_center=anchor_center,
            anchor_bounds=anchor_bounds,
        )

        anchor_features = (anchor.get("target_widget_features") or {}) if isinstance(anchor, dict) else {}
        anchor_text = str(anchor_features.get("text") or "").strip()
        anchor_rid = str(anchor_features.get("resource_id") or "").strip()
        anchor_desc = str(anchor_features.get("content_desc") or "").strip()
        anchor_class = str(anchor_features.get("class") or "").strip()

        anchor_tokens = [anchor_rid, anchor_text, anchor_desc, anchor_class]
        anchor_tokens = [t for t in anchor_tokens if t]

        target_widget_found_before = any(self._widget_feature_exists(t[:30], old_state) for t in anchor_tokens)
        target_widget_found_after = any(self._widget_feature_exists(t[:30], new_state) for t in anchor_tokens)
        target_widget_disappeared = target_widget_found_before and (not target_widget_found_after)

        screenshot_diff = self._visual_diff_score(
            getattr(old_state, "screenshot_path", "") or "",
            getattr(new_state, "screenshot_path", "") or "",
        )

        local_screenshot_diff = self._local_visual_diff_score(
            getattr(old_state, "screenshot_path", "") or "",
            getattr(new_state, "screenshot_path", "") or "",
            anchor_center=anchor_center,
            anchor_bounds=anchor_bounds,
        )

        interactive_union = len(old_interactive | new_interactive) or 1
        interactive_intersection = len(old_interactive & new_interactive)
        matched_widget_ratio = float(interactive_intersection / interactive_union)

        risk_detected, risk_signals, risk_types = self._detect_risk_signals(
            appeared_texts=appeared_texts,
            new_package=new_package,
            new_activity=new_activity,
        )

        boundary = constraints.boundary_constraints or {}
        must_stay_in_app = bool(boundary.get("must_stay_in_app", False))
        forbidden_packages = list(boundary.get("forbidden_packages") or [])
        package_mismatch_severity = str(boundary.get("package_mismatch_severity") or "soft").strip().lower()
        if package_mismatch_severity not in {"soft", "hard"}:
            package_mismatch_severity = "soft"
        related_package_tokens = boundary.get("related_package_tokens") or []
        if not isinstance(related_package_tokens, list):
            related_package_tokens = []
        related_package_tokens = self._normalize_package_tokens(related_package_tokens)
        if not related_package_tokens:
            related_package_tokens = self._extract_package_tokens(str(boundary.get("expected_package") or old_package))

        package_relation = self._classify_package_relation(old_package, new_package, related_package_tokens)
        stayed_in_app = True
        if must_stay_in_app and old_package and new_package and old_package != new_package:
            stayed_in_app = package_relation in {"same", "related"}
        package_mismatch_hard = (
            must_stay_in_app
            and old_package
            and new_package
            and old_package != new_package
            and package_mismatch_severity == "hard"
            and package_relation == "unrelated"
        )
        package_mismatch_soft = (
            must_stay_in_app
            and old_package
            and new_package
            and old_package != new_package
            and not package_mismatch_hard
        )

        spatial_anchor_usable = True
        spatial_reason = "ok"
        if bool(old_package and new_package and old_package != new_package) or bool(old_activity and new_activity and old_activity != new_activity):
            spatial_anchor_usable = False
            spatial_reason = "app_context_changed"
        elif overlay_changed:
            spatial_anchor_usable = False
            spatial_reason = "overlay_changed"
        elif screenshot_diff >= 22.0:
            spatial_anchor_usable = False
            spatial_reason = "global_visual_change"
        elif matched_widget_ratio < 0.2:
            spatial_anchor_usable = False
            spatial_reason = "layout_shift_or_recomposition"

        return {
            "anchor_evidence": {
                "target_widget_found_before": bool(target_widget_found_before),
                "target_widget_still_exists_after": bool(target_widget_found_after),
                "target_widget_disappeared": bool(target_widget_disappeared),
                "target_region_changed": bool(target_region_changed),
                "target_region_stats": target_region_stats,
            },
            "spatial_anchor_evidence": {
                "spatial_anchor_usable": bool(spatial_anchor_usable),
                "spatial_anchor_reason": spatial_reason,
                "matched_widget_ratio": round(matched_widget_ratio, 3),
                "global_visual_change_score": screenshot_diff,
                "local_visual_change_score": local_screenshot_diff,
            },
            "delta_evidence": {
                "package_changed": bool(old_package and new_package and old_package != new_package),
                "activity_changed": bool(old_activity and new_activity and old_activity != new_activity),
                "keyboard_before": old_keyboard,
                "keyboard_after": new_keyboard,
                "keyboard_appeared": (not old_keyboard) and new_keyboard,
                "keyboard_disappeared": old_keyboard and (not new_keyboard),
                "overlay_changed": bool(overlay_changed),
                "ui_changed_meaningfully": bool(ui_changed_meaningfully),
                "no_meaningful_change": bool(no_meaningful_change),
                "widget_counts_before": old_counts,
                "widget_counts_after": new_counts,
                "appeared_texts": appeared_texts,
                "vanished_texts": vanished_texts,
                "new_interactive_widgets": new_interactive_widgets,
                "vanished_interactive_widgets": vanished_interactive_widgets,
                "focused_editable_exists_before": bool(focused_editable_exists_before),
                "focused_editable_exists_after": bool(focused_editable_exists),
                "focus_changed": bool(focus_changed),
                "switch_count_before": len(old_switch_states),
                "switch_count_after": len(new_switch_states),
                "switch_checked_count_before": old_switch_checked_count,
                "switch_checked_count_after": new_switch_checked_count,
                "switch_checked_delta": switch_checked_delta,
                "switch_state_changed": bool(switch_state_changed),
                "switch_changed_keys": switch_changed_keys[:6],
                "screenshot_diff_score": screenshot_diff,
            },
            "boundary_evidence": {
                "stayed_in_app": bool(stayed_in_app),
                "package_relation": package_relation,
                "package_mismatch_severity": package_mismatch_severity,
                "package_mismatch_soft": bool(package_mismatch_soft),
                "package_mismatch_hard": bool(package_mismatch_hard),
                "related_package_tokens": related_package_tokens[:8],
                "entered_forbidden_package": bool(new_package in forbidden_packages),
                "risk_ui_detected": bool(risk_detected),
                "risk_signals": risk_signals,
                "risk_types_detected": risk_types,
            },
        }

    def _compact_text_token(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        value = re.sub(r"\s+", " ", value)
        return value[:40]

    def _filter_text_tokens(self, tokens: set) -> set:
        filtered = set()
        for token in tokens:
            value = (token or "").strip()
            if not value:
                continue
            low = value.lower()
            if low in self._input_stopwords:
                continue
            if len(low) <= 1:
                continue
            if re.fullmatch(r"[0-9]+", low):
                continue
            filtered.add(value)
        return filtered

    def _visual_diff_score(self, old_path: str, new_path: str) -> float:
        old_path = (old_path or "").strip()
        new_path = (new_path or "").strip()
        if not old_path or not new_path:
            return 0.0
        try:
            import cv2
            import numpy as np
        except Exception:
            return 0.0

        try:
            img1 = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                return 0.0
            img1 = cv2.resize(img1, (48, 48))
            img2 = cv2.resize(img2, (48, 48))
            diff = cv2.absdiff(img1, img2)
            return float(np.mean(diff))
        except Exception:
            return 0.0

    def _local_visual_diff_score(
        self,
        old_path: str,
        new_path: str,
        anchor_center: Optional[tuple],
        anchor_bounds: Optional[tuple],
    ) -> float:
        old_path = (old_path or "").strip()
        new_path = (new_path or "").strip()
        if not old_path or not new_path:
            return 0.0
        if anchor_center is None and anchor_bounds is None:
            return 0.0
        try:
            import cv2
            import numpy as np
        except Exception:
            return 0.0

        try:
            img1 = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                return 0.0

            h, w = img1.shape[:2]
            if img2.shape[:2] != (h, w):
                img2 = cv2.resize(img2, (w, h))

            if anchor_bounds is not None and len(anchor_bounds) == 4:
                x1, y1, x2, y2 = anchor_bounds
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(max(120, 0.6 * max(abs(x2 - x1), abs(y2 - y1))))
            else:
                cx, cy = anchor_center
                radius = 180

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w, cx + radius)
            y2 = min(h, cy + radius)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            crop1 = img1[y1:y2, x1:x2]
            crop2 = img2[y1:y2, x1:x2]
            crop1 = cv2.resize(crop1, (48, 48))
            crop2 = cv2.resize(crop2, (48, 48))
            diff = cv2.absdiff(crop1, crop2)
            return float(np.mean(diff))
        except Exception:
            return 0.0

    def _detect_overlay_changed(self, old_state: UIState, new_state: UIState) -> bool:
        def _overlay_score(state: UIState) -> int:
            w = int(getattr(state, "screen_width", 0) or 0)
            h = int(getattr(state, "screen_height", 0) or 0)
            if w <= 0 or h <= 0:
                w, h = 1080, 1920
            screen_area = float(w * h)
            score = 0
            for widget in getattr(state, "widgets", []) or []:
                x1, y1, x2, y2 = getattr(widget, "bounds", (0, 0, 0, 0)) or (0, 0, 0, 0)
                area = float(max(0, x2 - x1) * max(0, y2 - y1))
                clazz = (getattr(widget, "class_name", "") or "").lower()
                cat = (getattr(widget, "category", "") or "").lower()
                if area >= 0.7 * screen_area:
                    score += 1
                if "dialog" in clazz or "popup" in clazz or "dialog" in cat or "popup" in cat:
                    score += 1
            return score

        return _overlay_score(old_state) != _overlay_score(new_state)

    def _detect_target_region_changed(
        self,
        old_state: UIState,
        new_state: UIState,
        action: Optional[Action],
        anchor_center: Optional[tuple] = None,
        anchor_bounds: Optional[tuple] = None,
    ) -> tuple:
        x, y = 0, 0
        bounds = None

        if anchor_bounds is not None and len(anchor_bounds) == 4:
            bounds = anchor_bounds
            x = int((bounds[0] + bounds[2]) / 2)
            y = int((bounds[1] + bounds[3]) / 2)
        elif anchor_center is not None and len(anchor_center) == 2:
            x, y = int(anchor_center[0]), int(anchor_center[1])
        elif action is not None:
            x, y = int(getattr(action, "x", 0) or 0), int(getattr(action, "y", 0) or 0)
            action_bounds = tuple(getattr(action, "target_bounds", ()) or (0, 0, 0, 0))
            if action_bounds and len(action_bounds) == 4 and action_bounds != (0, 0, 0, 0):
                x = int((action_bounds[0] + action_bounds[2]) / 2)
                y = int((action_bounds[1] + action_bounds[3]) / 2)
                bounds = action_bounds
        else:
            return False, {"mode": "no_anchor"}

        radius = 180
        if bounds is not None:
            bw = max(0, int(bounds[2] - bounds[0]))
            bh = max(0, int(bounds[3] - bounds[1]))
            radius = max(radius, int(0.7 * max(bw, bh)))
        region = (x - radius, y - radius, x + radius, y + radius)

        def _sig(state: UIState) -> set:
            items = set()
            for w in getattr(state, "widgets", []) or []:
                cx, cy = getattr(w, "center", (0, 0)) or (0, 0)
                if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                    continue
                text = (getattr(w, "text", "") or getattr(w, "content_desc", "") or "")[:20]
                rid = (getattr(w, "resource_id", "") or "")[:40]
                clazz = (getattr(w, "class_name", "") or getattr(w, "category", "") or "")[:40]
                items.add(f"{clazz}|{rid}|{text}")
            return items

        a = _sig(old_state)
        b = _sig(new_state)
        if not a and not b:
            return False, {"mode": "empty_region", "center": [x, y], "radius": radius}
        inter = len(a & b)
        union = max(1, len(a | b))
        similarity = inter / union
        changed = similarity < 0.75
        return bool(changed), {"mode": "sig_jaccard", "center": [x, y], "radius": radius, "similarity": round(similarity, 3), "size_before": len(a), "size_after": len(b)}

    def _detect_risk_signals(self, appeared_texts: List[str], new_package: str, new_activity: str) -> tuple:
        signals: List[str] = []
        types: List[str] = []

        joined = " ".join([str(t).lower() for t in (appeared_texts or [])])
        pkg_low = (new_package or "").lower()
        act_low = (new_activity or "").lower()

        def _hit(risk_type: str, key: str, channel: str):
            signals.append(f"{channel}:{key}")
            types.append(risk_type)

        payment_keys = ["支付", "付款", "收款", "钱包", "充值", "订单", "pay", "wallet"]
        permission_keys = ["授权", "允许", "permission", "allow"]
        destructive_keys = ["删除", "清空", "移除", "确定删除", "永久删除", "提交", "确认提交"]
        auth_keys = ["登录", "sign in", "log in", "verify", "验证码"]
        share_keys = ["分享", "share", "发送给", "send to"]
        webview_keys = ["http", "www.", "chrome", "webview", "browser"]

        for key in payment_keys:
            if key.lower() in joined or key.lower() in pkg_low or key.lower() in act_low:
                _hit("payment", key, "hit")
        for key in permission_keys:
            if key.lower() in joined or key.lower() in pkg_low or key.lower() in act_low:
                _hit("permission", key, "hit")
        for key in destructive_keys:
            if key.lower() in joined:
                _hit("destructive_action", key, "text")
        for key in auth_keys:
            if key.lower() in joined:
                _hit("external_auth", key, "text")
        for key in share_keys:
            if key.lower() in joined:
                _hit("share_sheet", key, "text")
        for key in webview_keys:
            if key.lower() in joined or key.lower() in pkg_low or key.lower() in act_low:
                _hit("browser_or_webview_escape", key, "hit")

        if "com.android.settings" in pkg_low or "settings" in act_low:
            _hit("system_ui", "com.android.settings", "context")

        deduped_types = []
        seen = set()
        for t in types:
            if t not in seen:
                deduped_types.append(t)
                seen.add(t)

        deduped_signals = []
        seen_sig = set()
        for s in signals:
            if s not in seen_sig:
                deduped_signals.append(s)
                seen_sig.add(s)

        detected = bool(deduped_types or deduped_signals)
        return detected, deduped_signals[:12], deduped_types[:8]

    def _check_input_action_effect(
        self,
        action: Action,
        old_state: UIState,
        new_state: UIState,
    ) -> Dict[str, Any]:
        raw_text = (action.text or "").replace("\r\n", "\n").replace("\r", "\n")
        expected_fragments = self._build_input_fragments(action.text)

        if raw_text == "":
            return {
                "ok": False,
                "mode": "empty_input",
                "strict_target": self._has_target_anchor(action),
                "target_scope_found": False,
                "required_hits": 1,
                "fragments": [],
                "old_hits": [],
                "new_hits": [],
                "effective_new_hits": [],
                "old_target_hits": [],
                "all_target_hits": [],
                "new_target_hits": [],
                "target_old_texts": [],
                "target_new_texts": [],
                "reason": "输入动作文本为空，无法验证输入效果",
            }

        if raw_text.strip() == "" and not expected_fragments:
            return {
                "ok": True,
                "mode": "control_input",
                "strict_target": self._has_target_anchor(action),
                "target_scope_found": False,
                "required_hits": 0,
                "fragments": [],
                "old_hits": [],
                "new_hits": [],
                "effective_new_hits": [],
                "old_target_hits": [],
                "all_target_hits": [],
                "new_target_hits": [],
                "target_old_texts": [],
                "target_new_texts": [],
                "reason": "控制型输入（无可比对文本片段），交由语义评估判定",
            }

        old_texts = self._collect_texts(old_state)
        new_texts = self._collect_texts(new_state)
        old_norm = [self._normalize_text(text) for text in old_texts]
        new_norm = [self._normalize_text(text) for text in new_texts]

        target_old = self._collect_target_area_texts(old_state, action)
        target_new = self._collect_target_area_texts(new_state, action)
        target_old_norm = [self._normalize_text(text) for text in target_old]
        target_new_norm = [self._normalize_text(text) for text in target_new]

        def _hit(fragment: str, values: List[str]) -> bool:
            return any(fragment in value for value in values)

        old_hits = [frag for frag in expected_fragments if _hit(frag, old_norm)]
        new_hits = [frag for frag in expected_fragments if _hit(frag, new_norm) and frag not in old_hits]
        old_target_hits = [frag for frag in expected_fragments if _hit(frag, target_old_norm)]
        all_target_hits = [frag for frag in expected_fragments if _hit(frag, target_new_norm)]
        new_target_hits = [
            frag for frag in expected_fragments
            if _hit(frag, target_new_norm) and frag not in old_target_hits
        ]

        required_hits = self._required_input_hits(action.text, expected_fragments)
        strict_target = self._has_target_anchor(action)
        target_scope_found = bool(target_old_norm or target_new_norm)

        mode = "global_fallback"
        effective_new_hits = new_hits
        ok = len(new_hits) >= required_hits
        reason = ""

        if strict_target and target_scope_found:
            mode = "target_only"
            effective_new_hits = new_target_hits
            ok = len(new_target_hits) >= required_hits
            # 允许幂等输入: 目标控件前后都已包含关键片段，不要求必须新增
            if (not ok) and len(all_target_hits) >= required_hits and len(old_target_hits) >= required_hits:
                ok = True
                mode = "target_stable"
            if not ok:
                reason = (
                    f"输入结果未命中目标控件: 需要至少 {required_hits} 个新片段, "
                    f"目标区域新增={new_target_hits[:5]}, 全局新增(已忽略)={new_hits[:5]}"
                )
        elif strict_target:
            mode = "target_unresolved"
            effective_new_hits = []
            ok = False
            reason = "输入动作缺少可验证的目标控件文本快照，拒绝使用全页命中作为成功依据"
        elif not ok:
            reason = (
                f"输入结果未充分反映在新界面: 需要至少 {required_hits} 个新片段命中, "
                f"全局新增={new_hits[:5]}"
            )

        return {
            "ok": ok,
            "mode": mode,
            "strict_target": strict_target,
            "target_scope_found": target_scope_found,
            "required_hits": required_hits,
            "fragments": expected_fragments,
            "old_hits": old_hits,
            "new_hits": new_hits,
            "effective_new_hits": effective_new_hits,
            "old_target_hits": old_target_hits,
            "all_target_hits": all_target_hits,
            "new_target_hits": new_target_hits,
            "target_old_texts": target_old[:5],
            "target_new_texts": target_new[:5],
            "reason": reason,
        }

    def _build_input_fragments(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        fragments: List[str] = []
        lines = [
            self._normalize_text(line)
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if self._normalize_text(line)
        ]
        fragments.extend(lines[:3])

        for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            fragments.append(self._normalize_text(email))

        for token in re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}:\d{2}\b", text):
            fragments.append(self._normalize_text(token))

        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text):
            norm = self._normalize_text(token)
            if norm and norm not in self._input_stopwords:
                fragments.append(norm)

        if normalized not in fragments and len(lines) <= 1 and len(normalized) <= 80:
            fragments.insert(0, normalized)

        deduped: List[str] = []
        seen = set()
        for frag in fragments:
            if not frag or frag in seen:
                continue
            deduped.append(frag)
            seen.add(frag)
        return deduped

    def _required_input_hits(self, text: str, fragments: List[str]) -> int:
        lines = [
            line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        if not fragments:
            return 1
        if len(lines) >= 3:
            return min(3, len(fragments))
        if len(lines) == 2:
            return min(2, len(fragments))
        normalized = self._normalize_text(text)
        if len(normalized) <= 40:
            return 1
        return min(2, len(fragments))

    def _has_target_anchor(self, action: Action) -> bool:
        target_bounds = tuple(action.target_bounds or (0, 0, 0, 0))
        return bool(
            action.target_resource_id
            or action.widget_id is not None
            or target_bounds != (0, 0, 0, 0)
        )

    def _match_target_widgets(self, state: UIState, action: Action) -> List[Any]:
        widgets = list(getattr(state, "widgets", []))
        if not widgets:
            return []

        if action.target_resource_id:
            exact_rid = [w for w in widgets if action.target_resource_id == getattr(w, "resource_id", "")]
            if exact_rid:
                return exact_rid

        if action.widget_id is not None:
            by_id = [w for w in widgets if action.widget_id == getattr(w, "widget_id", None)]
            if by_id:
                return by_id

        target_bounds = tuple(action.target_bounds or (0, 0, 0, 0))
        if target_bounds != (0, 0, 0, 0):
            screen_w = max(0, getattr(state, "screen_width", 0))
            screen_h = max(0, getattr(state, "screen_height", 0))
            expanded = self._expand_bounds(
                target_bounds,
                screen_w,
                screen_h,
                pad_x=40,
                pad_y_top=40,
                pad_y_bottom=60,
            )
            scored = []
            for w in widgets:
                ratio = self._bounds_overlap_ratio(expanded, getattr(w, "bounds", (0, 0, 0, 0)))
                if ratio >= 0.45:
                    scored.append((ratio, w))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                return [w for _, w in scored]

        if action.target_class_name:
            class_matches = [w for w in widgets if action.target_class_name == getattr(w, "class_name", "")]
            if len(class_matches) == 1:
                return class_matches

        return []

    def _collect_target_area_texts(self, state: UIState, action: Action) -> List[str]:
        target_widgets = self._match_target_widgets(state, action)
        if not target_widgets:
            return []

        screen_w = max(0, getattr(state, "screen_width", 0))
        screen_h = max(0, getattr(state, "screen_height", 0))
        target_bounds = self._merge_bounds([getattr(w, "bounds", (0, 0, 0, 0)) for w in target_widgets])
        if target_bounds == (0, 0, 0, 0):
            target_bounds = tuple(action.target_bounds or (0, 0, 0, 0))
        expanded_bounds = self._expand_bounds(
            target_bounds,
            screen_w,
            screen_h,
            pad_x=24,
            pad_y_top=24,
            pad_y_bottom=24,
        )

        texts: List[str] = []
        for w in getattr(state, "widgets", []):
            merged_text = [getattr(w, "text", ""), getattr(w, "content_desc", "")]
            if not any(merged_text):
                continue
            overlap_ratio = self._bounds_overlap_ratio(
                expanded_bounds,
                getattr(w, "bounds", (0, 0, 0, 0)),
            )
            if overlap_ratio < 0.55:
                continue
            if getattr(w, "text", ""):
                texts.append(w.text)
            if getattr(w, "content_desc", ""):
                texts.append(w.content_desc)

        deduped: List[str] = []
        seen = set()
        for text in texts:
            norm = self._normalize_text(text)
            if not norm or norm in seen:
                continue
            deduped.append(text)
            seen.add(norm)
        return deduped

    def _merge_bounds(self, bounds_list: List[tuple]) -> tuple:
        valid = [b for b in bounds_list if b and len(b) == 4 and b != (0, 0, 0, 0)]
        if not valid:
            return (0, 0, 0, 0)
        x1 = min(b[0] for b in valid)
        y1 = min(b[1] for b in valid)
        x2 = max(b[2] for b in valid)
        y2 = max(b[3] for b in valid)
        return (x1, y1, x2, y2)

    def _expand_bounds(
        self,
        bounds: tuple,
        screen_w: int,
        screen_h: int,
        pad_x: int = 0,
        pad_y_top: int = 0,
        pad_y_bottom: int = 0,
    ) -> tuple:
        if not bounds or len(bounds) != 4:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = bounds
        if x1 == x2 == y1 == y2 == 0:
            return (0, 0, 0, 0)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y_top),
            min(screen_w or x2, x2 + pad_x),
            min(screen_h or y2, y2 + pad_y_bottom),
        )

    def _bounds_overlap(self, box_a: tuple, box_b: tuple) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _bounds_overlap_ratio(self, box_a: tuple, box_b: tuple) -> float:
        if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter_area = inter_w * inter_h
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / max(1, min(area_a, area_b))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def _check_focus_tap_effect(
        self,
        action: Action,
        old_state: UIState,
        new_state: UIState,
    ) -> Dict[str, Any]:
        old_targets = self._match_target_widgets(old_state, action)
        new_targets = self._match_target_widgets(new_state, action)
        applicable = bool(old_targets or new_targets)
        if not applicable:
            return {
                "applicable": False,
                "ok": False,
                "decisive_success": False,
                "status": "target_missing",
                "reason": "未定位到可验证的目标控件",
            }

        old_focus = any(getattr(w, "focused", False) for w in old_targets)
        new_focus = any(getattr(w, "focused", False) for w in new_targets)
        old_editable = any(getattr(w, "editable", False) or getattr(w, "focusable", False) for w in old_targets)
        new_editable = any(getattr(w, "editable", False) or getattr(w, "focusable", False) for w in new_targets)
        old_keyboard = bool(getattr(old_state, "keyboard_visible", False))
        new_keyboard = bool(getattr(new_state, "keyboard_visible", False))

        status = "not_focused"
        ok = False
        decisive = False
        if new_focus and not old_focus:
            ok = True
            decisive = True
            status = "focused_after_tap"
        elif old_focus and new_focus:
            ok = True
            decisive = True
            status = "already_focused"
        elif new_keyboard and (new_editable or old_editable):
            ok = True
            status = "keyboard_visible"
        elif old_keyboard and new_keyboard and old_focus:
            ok = True
            decisive = True
            status = "keyboard_and_focus_stable"

        reason = (
            ""
            if ok else
            f"焦点未建立: old_focus={old_focus}, new_focus={new_focus}, "
            f"old_keyboard={old_keyboard}, new_keyboard={new_keyboard}"
        )
        return {
            "applicable": True,
            "ok": ok,
            "decisive_success": decisive,
            "status": status,
            "old_focus": old_focus,
            "new_focus": new_focus,
            "old_keyboard": old_keyboard,
            "new_keyboard": new_keyboard,
            "reason": reason,
        }

    def _has_meaningful_ui_change(self, old_state: UIState, new_state: UIState) -> bool:
        old_package = self._get_package(old_state)
        new_package = self._get_package(new_state)
        if old_package != new_package:
            return True

        old_activity = self._get_activity(old_state)
        new_activity = self._get_activity(new_state)
        if old_activity != new_activity:
            return True

        old_keyboard = bool(getattr(old_state, "keyboard_visible", False))
        new_keyboard = bool(getattr(new_state, "keyboard_visible", False))
        if old_keyboard != new_keyboard:
            return True

        old_focused = sum(1 for w in getattr(old_state, "widgets", []) if getattr(w, "focused", False))
        new_focused = sum(1 for w in getattr(new_state, "widgets", []) if getattr(w, "focused", False))
        if old_focused != new_focused:
            return True

        old_texts = set(self._collect_texts(old_state))
        new_texts = set(self._collect_texts(new_state))
        text_delta = len(old_texts.symmetric_difference(new_texts))

        old_widget_count = len(getattr(old_state, "widgets", []))
        new_widget_count = len(getattr(new_state, "widgets", []))
        widget_delta = abs(old_widget_count - new_widget_count)

        # 阈值可后续再调
        return text_delta >= 2 or widget_delta >= 2

    def _match_target_state_type(
        self,
        target_state_type: str,
        old_state: UIState,
        new_state: UIState,
    ) -> tuple[float, str]:
        """
        对通用页面状态做弱判断，返回 (score, reason)
        """
        target = (target_state_type or "").strip().lower()
        if not target or target == "unknown":
            return 0.0, ""

        widgets = getattr(new_state, "widgets", [])
        editable_count = 0
        focused_editable_count = 0
        clickable_count = 0
        checkable_count = 0
        scrollable_count = 0

        for w in widgets:
            if getattr(w, "editable", False):
                editable_count += 1
                if getattr(w, "focused", False):
                    focused_editable_count += 1
            if getattr(w, "clickable", False):
                clickable_count += 1
            if getattr(w, "checkable", False):
                checkable_count += 1
            if getattr(w, "scrollable", False):
                scrollable_count += 1

        new_texts = self._collect_texts(new_state)
        old_texts = self._collect_texts(old_state)
        keyboard_visible = bool(getattr(new_state, "keyboard_visible", False))

        # 可以逐步增强，这里先给实用规则
        if target == "form":
            if editable_count >= 1 or focused_editable_count >= 1 or keyboard_visible:
                return (
                    1.0,
                    "target_state=form 命中: "
                    f"editable_count={editable_count}, focused_editable={focused_editable_count}, "
                    f"keyboard_visible={keyboard_visible}",
                )
            return 0.0, "target_state=form 未命中"

        if target == "dialog":
            # 粗略：文本/控件较少但可点击集中，activity/package 常不变
            if len(widgets) <= 20 and clickable_count >= 1:
                return 0.8, f"target_state=dialog 弱命中: widgets={len(widgets)}, clickable={clickable_count}"
            return 0.0, "target_state=dialog 未命中"

        if target == "list":
            if scrollable_count >= 1 and clickable_count >= 3:
                return 0.8, f"target_state=list 命中: scrollable={scrollable_count}, clickable={clickable_count}"
            return 0.0, "target_state=list 未命中"

        if target == "detail":
            # 粗略：比旧页面更聚焦，文本变化较多，但不一定有可编辑项
            delta = len(set(new_texts).symmetric_difference(set(old_texts)))
            if delta >= 2:
                return 0.8, f"target_state=detail 弱命中: text_delta={delta}"
            return 0.0, "target_state=detail 未命中"

        if target == "search":
            # 有输入框 + 内容变化
            delta = len(set(new_texts).symmetric_difference(set(old_texts)))
            if (editable_count >= 1 or focused_editable_count >= 1 or keyboard_visible) and delta >= 1:
                return (
                    1.0,
                    "target_state=search 命中: "
                    f"editable_count={editable_count}, focused_editable={focused_editable_count}, "
                    f"keyboard_visible={keyboard_visible}, text_delta={delta}",
                )
            return 0.0, "target_state=search 未命中"

        if target == "selection":
            if checkable_count >= 1:
                return 0.8, f"target_state=selection 命中: checkable_count={checkable_count}"
            return 0.0, "target_state=selection 未命中"

        if target == "menu":
            if len(widgets) <= 15 and clickable_count >= 2:
                return 0.8, f"target_state=menu 弱命中: widgets={len(widgets)}, clickable={clickable_count}"
            return 0.0, "target_state=menu 未命中"

        if target == "tab":
            # tab 很难纯规则判定，给弱分，主要依赖其它证据
            return 0.3, "target_state=tab 默认弱分"

        return 0.0, f"未知target_state_type: {target}"

    def _match_source_state_exit(
        self,
        source_state_type: str,
        old_state: UIState,
        new_state: UIState,
    ) -> tuple[float, str]:
        """
        判断是否离开了原始状态类型，作为辅助证据
        """
        source = (source_state_type or "").strip().lower()
        if not source or source == "unknown":
            return 0.0, ""

        if source == "form":
            old_editable = sum(1 for w in getattr(old_state, "widgets", []) if getattr(w, "editable", False))
            new_editable = sum(1 for w in getattr(new_state, "widgets", []) if getattr(w, "editable", False))
            if new_editable < old_editable:
                return 0.5, f"source_state=form 退出弱命中: editable {old_editable}->{new_editable}"
            return 0.0, "source_state=form 未明显退出"

        if source == "dialog":
            old_count = len(getattr(old_state, "widgets", []))
            new_count = len(getattr(new_state, "widgets", []))
            if new_count != old_count:
                return 0.3, f"source_state=dialog 可能退出: widgets {old_count}->{new_count}"
            return 0.0, "source_state=dialog 未明显退出"

        if source == "list":
            old_click = sum(1 for w in getattr(old_state, "widgets", []) if getattr(w, "clickable", False))
            new_click = sum(1 for w in getattr(new_state, "widgets", []) if getattr(w, "clickable", False))
            if new_click < old_click:
                return 0.3, f"source_state=list 可能退出: clickable {old_click}->{new_click}"
            return 0.0, "source_state=list 未明显退出"

        return 0.0, ""
