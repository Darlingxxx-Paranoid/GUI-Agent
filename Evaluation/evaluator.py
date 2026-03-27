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


class Evaluator:
    """
    事后评估器
    1. 基础约束检测（环境边界、控件存在/消失、UI变化、弱证据打分）
    2. 约束通过后调用 LLM 做语义确认
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("Evaluator 初始化完成")
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
                acceptance_criteria=constraints.semantic_criteria or subgoal_description,
                old_state=old_state,
                new_state=new_state,
                constraint_evidence=evidence,
                llm_result=None,
                raw_response=None,
                final_result=constraint_result,
            )
            return constraint_result

        focus_validation = evidence.get("focus_validation") if isinstance(evidence, dict) else None
        if focus_validation and focus_validation.get("decisive_success"):
            shortcut_reason = str(
                focus_validation.get("reason")
                or focus_validation.get("status")
                or "focus_ready"
            )
            shortcut = EvalResult(
                success=True,
                reason=shortcut_reason,
                constraint_passed=True,
                semantic_passed=True,
            )
            self._save_eval_artifact(
                subgoal_description=subgoal_description,
                acceptance_criteria=constraints.semantic_criteria or subgoal_description,
                old_state=old_state,
                new_state=new_state,
                constraint_evidence=evidence,
                llm_result=None,
                raw_response=None,
                final_result=shortcut,
            )
            logger.info("基于焦点证据直接通过: %s", shortcut.reason)
            return shortcut

        semantic_result, llm_payload, raw_response = self._semantic_check(
            subgoal_description=subgoal_description,
            acceptance_criteria=constraints.semantic_criteria or subgoal_description,
            old_state=old_state,
            new_state=new_state,
            constraint_evidence=evidence,
        )
        self._save_eval_artifact(
            subgoal_description=subgoal_description,
            acceptance_criteria=constraints.semantic_criteria or subgoal_description,
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

        old_texts = self._collect_texts(old_state)
        new_texts = self._collect_texts(new_state)

        hard_violations: List[str] = []

        # 1. app/package 边界检查
        if constraints.must_stay_in_app and old_package and new_package and old_package != new_package:
            hard_violations.append(f"must_stay_in_app violated: '{old_package}' -> '{new_package}'")

        if constraints.expected_package and new_package and new_package != constraints.expected_package:
            hard_violations.append(
                f"expected_package mismatch: expected='{constraints.expected_package}', actual='{new_package}'"
            )

        if new_package in constraints.forbidden_packages:
            hard_violations.append(f"forbidden_package opened: '{new_package}'")

        missing_should_exist: List[str] = []
        for feature in constraints.widget_should_exist:
            if not self._widget_feature_exists(feature, new_state):
                missing_should_exist.append(feature)

        present_should_vanish: List[str] = []
        for feature in constraints.widget_should_vanish:
            if self._widget_feature_exists(feature, new_state):
                present_should_vanish.append(feature)

        ui_changed = self._has_meaningful_ui_change(old_state, new_state)

        state_score = 0.0
        state_reasons: List[str] = []

        target_state_score, target_reason = self._match_target_state_type(
            constraints.target_state_type, old_state, new_state
        )
        state_score += target_state_score
        if target_reason:
            state_reasons.append(target_reason)

        source_exit_score, source_exit_reason = self._match_source_state_exit(
            constraints.source_state_type, old_state, new_state
        )
        state_score += source_exit_score
        if source_exit_reason:
            state_reasons.append(source_exit_reason)

        # 7. supporting_texts 弱证据
        support_score = 0.0
        support_hits = []

        for text in constraints.supporting_texts:
            if self._text_exists(text, new_texts):
                support_score += 0.5
                support_hits.append(f"text:{text}")

        for feature in constraints.supporting_widgets:
            if self._widget_feature_exists(feature, new_state):
                support_score += 0.5
                support_hits.append(f"widget:{feature}")

        input_validation = None
        if action and action.action_type == "input" and action.text:
            input_validation = self._check_input_action_effect(
                action=action,
                old_state=old_state,
                new_state=new_state,
            )
            if input_validation.get("ok"):
                effective_hits = input_validation.get("effective_new_hits", [])
                support_hits.extend(
                    [f"input_new:{frag}" for frag in effective_hits[:3]]
                )
                support_score += min(1.0, 0.5 * len(effective_hits))
            else:
                hard_violations.append(input_validation["reason"])

        focus_validation = None
        if action and action.action_type in ("tap", "long_press"):
            focus_validation = self._check_focus_tap_effect(
                action=action,
                old_state=old_state,
                new_state=new_state,
            )
            if focus_validation.get("ok"):
                support_hits.append(f"focus:{focus_validation.get('status', 'ok')}")
                support_score += 0.8

        total_support = state_score + support_score

        evidence: Dict[str, Any] = {
            "old_package": old_package,
            "new_package": new_package,
            "old_activity": old_activity,
            "new_activity": new_activity,
            "expected_activity": constraints.expected_activity or "",
            "expected_activity_matched": bool(
                constraints.expected_activity and (constraints.expected_activity in (new_activity or ""))
            ),
            "require_ui_change": bool(constraints.require_ui_change),
            "ui_changed": bool(ui_changed),
            "widget_should_exist": list(constraints.widget_should_exist or []),
            "missing_widget_should_exist": missing_should_exist,
            "widget_should_vanish": list(constraints.widget_should_vanish or []),
            "present_widget_should_vanish": present_should_vanish,
            "support_hits": support_hits,
            "support_score": total_support,
            "min_support_score": float(getattr(constraints, "min_support_score", 0.0) or 0.0),
            "state_reasons": state_reasons,
            "action_context": self._action_context(action),
            "input_validation": input_validation,
            "focus_validation": focus_validation,
            "hard_violations": hard_violations,
        }

        if hard_violations:
            return (
                EvalResult(success=False, constraint_passed=False, reason="; ".join(hard_violations)),
                evidence,
            )

        return (
            EvalResult(
                success=True,
                constraint_passed=True,
                reason=f"evidence_collected support_score={total_support:.1f}",
            ),
            evidence,
        )

    def _semantic_check(
        self,
        subgoal_description: str,
        acceptance_criteria: str,
        old_state: UIState,
        new_state: UIState,
        constraint_evidence: Dict[str, Any],
    ) -> Tuple[EvalResult, Optional[Dict[str, Any]], Optional[str]]:
        prompt = EVALUATOR_PROMPT.format(
            subgoal_description=subgoal_description,
            acceptance_criteria=acceptance_criteria,
            constraint_evidence=json.dumps(constraint_evidence, ensure_ascii=False),
            old_state_summary=old_state.to_prompt_text()[:1400],
            new_state_summary=new_state.to_prompt_text()[:1400],
        )

        try:
            response = self.llm.chat(prompt)
            result, payload = self._parse_semantic_response(response)
            return result, payload, response
        except Exception as e:
            logger.error("LLM 语义确认失败: %s", e)
            fallback = self._fallback_from_evidence(constraint_evidence, error=str(e))
            return fallback, None, None

    def _parse_semantic_response(self, response: str) -> Tuple[EvalResult, Optional[Dict[str, Any]]]:
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
            return (
                EvalResult(
                    success=bool(data.get("success", False)),
                    semantic_passed=bool(data.get("success", False)),
                    reason=str(data.get("reason", "")),
                ),
                data,
            )
        except json.JSONDecodeError as e:
            logger.warning("语义确认 JSON 解析失败: %s", e)
            return (
                EvalResult(success=False, semantic_passed=False, reason="JSON解析失败"),
                None,
            )

    def _fallback_from_evidence(self, evidence: Dict[str, Any], error: str = "") -> EvalResult:
        hard_violations = evidence.get("hard_violations") or []
        if hard_violations:
            return EvalResult(
                success=False,
                semantic_passed=False,
                reason="; ".join([*hard_violations, error] if error else hard_violations),
            )

        ui_changed = bool(evidence.get("ui_changed", False))
        support_score = float(evidence.get("support_score", 0.0) or 0.0)
        missing_should_exist = evidence.get("missing_widget_should_exist") or []
        present_should_vanish = evidence.get("present_widget_should_vanish") or []

        if missing_should_exist or present_should_vanish:
            ok = ui_changed and support_score >= 0.8
        else:
            ok = ui_changed or support_score >= 0.8

        reason_parts = [
            f"fallback: ui_changed={ui_changed}",
            f"support_score={support_score:.1f}",
        ]
        if missing_should_exist:
            reason_parts.append(f"missing_exist={missing_should_exist[:5]}")
        if present_should_vanish:
            reason_parts.append(f"present_vanish={present_should_vanish[:5]}")
        if error:
            reason_parts.append(f"llm_error={error}")

        return EvalResult(
            success=ok,
            semantic_passed=ok,
            reason="; ".join(reason_parts),
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
