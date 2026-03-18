"""
事后评估模块
对动作结果进行约束检测和语义确认
决定是继续前进还是标记失败
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional, List

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

    def evaluate(
        self,
        subgoal_description: str,
        constraints: PreConstraints,
        old_state: UIState,
        new_state: UIState,
    ) -> EvalResult:
        """
        评估子目标执行结果
        """
        logger.info("开始事后评估: '%s'", subgoal_description)

        constraint_result = self._check_constraints(constraints, old_state, new_state)
        if not constraint_result.constraint_passed:
            logger.info("基础约束未通过: %s", constraint_result.reason)
            return constraint_result

        if constraints.semantic_criteria:
            semantic_result = self._semantic_check(
                subgoal_description,
                constraints.semantic_criteria,
                old_state,
                new_state,
            )
            if not semantic_result.success:
                logger.info("语义确认未通过: %s", semantic_result.reason)
                return semantic_result

        logger.info("事后评估通过: '%s'", subgoal_description)
        return EvalResult(success=True, reason="基础约束和语义确认均通过")

    def _check_constraints(
        self,
        constraints: PreConstraints,
        old_state: UIState,
        new_state: UIState,
    ) -> EvalResult:
        """基础约束检测"""

        old_package = self._get_package(old_state)
        new_package = self._get_package(new_state)
        old_activity = self._get_activity(old_state)
        new_activity = self._get_activity(new_state)

        old_texts = self._collect_texts(old_state)
        new_texts = self._collect_texts(new_state)

        # 1. app/package 边界检查
        if constraints.must_stay_in_app and old_package and new_package and old_package != new_package:
            return EvalResult(
                success=False,
                constraint_passed=False,
                reason=f"要求留在当前App，但package发生变化: '{old_package}' -> '{new_package}'",
            )

        if constraints.expected_package and new_package and new_package != constraints.expected_package:
            return EvalResult(
                success=False,
                constraint_passed=False,
                reason=f"当前package不符合预期: expected='{constraints.expected_package}', actual='{new_package}'",
            )

        if new_package in constraints.forbidden_packages:
            return EvalResult(
                success=False,
                constraint_passed=False,
                reason=f"跳转到了禁止package: '{new_package}'",
            )

        # 2. activity 检查
        if constraints.expected_activity and constraints.expected_activity not in new_activity:
            return EvalResult(
                success=False,
                constraint_passed=False,
                reason=f"当前Activity不符合预期: expected contains '{constraints.expected_activity}', actual='{new_activity}'",
            )

        # 3. 是否发生UI变化
        if constraints.require_ui_change:
            if not self._has_meaningful_ui_change(old_state, new_state):
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason="期望发生UI变化，但执行前后界面无明显变化",
                )

        # 4. 强约束：期望存在的控件
        for feature in constraints.widget_should_exist:
            if not self._widget_feature_exists(feature, new_state):
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason=f"期望控件未出现: '{feature}'",
                )

        # 5. 强约束：期望消失的控件
        for feature in constraints.widget_should_vanish:
            if self._widget_feature_exists(feature, new_state):
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason=f"期望消失的控件仍然存在: '{feature}'",
                )

        # 6. 目标状态类型弱判断
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

        total_support = state_score + support_score

        if constraints.min_support_score > 0 and total_support < constraints.min_support_score:
            return EvalResult(
                success=False,
                constraint_passed=False,
                reason=(
                    f"弱证据不足: score={total_support:.1f} < required={constraints.min_support_score:.1f}; "
                    f"hits={support_hits}; state_reasons={state_reasons}"
                ),
            )

        return EvalResult(
            success=True,
            constraint_passed=True,
            reason=(
                f"基础约束通过; support_score={total_support:.1f}; "
                f"support_hits={support_hits}; state_reasons={state_reasons}"
            ),
        )

    def _semantic_check(
        self,
        subgoal_description: str,
        acceptance_criteria: str,
        old_state: UIState,
        new_state: UIState,
    ) -> EvalResult:
        """通过 LLM 进行高阶语义确认"""
        prompt = EVALUATOR_PROMPT.format(
            subgoal_description=subgoal_description,
            acceptance_criteria=acceptance_criteria,
            old_state_summary=old_state.to_prompt_text()[:800],
            new_state_summary=new_state.to_prompt_text()[:800],
        )

        try:
            response = self.llm.chat(prompt)
            return self._parse_semantic_response(response)
        except Exception as e:
            logger.error("LLM 语义确认失败: %s, 默认通过", e)
            return EvalResult(success=True, reason=f"LLM调用失败({e}), 默认通过")

    def _parse_semantic_response(self, response: str) -> EvalResult:
        """解析 LLM 语义确认结果"""
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
            return EvalResult(
                success=data.get("success", False),
                semantic_passed=data.get("success", False),
                reason=data.get("reason", ""),
            )
        except json.JSONDecodeError:
            logger.warning("语义确认 JSON 解析失败, 默认通过")
            return EvalResult(success=True, reason="JSON解析失败, 默认通过")

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

    def _has_meaningful_ui_change(self, old_state: UIState, new_state: UIState) -> bool:
        old_package = self._get_package(old_state)
        new_package = self._get_package(new_state)
        if old_package != new_package:
            return True

        old_activity = self._get_activity(old_state)
        new_activity = self._get_activity(new_state)
        if old_activity != new_activity:
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
        clickable_count = 0
        checkable_count = 0
        scrollable_count = 0

        for w in widgets:
            if getattr(w, "editable", False):
                editable_count += 1
            if getattr(w, "clickable", False):
                clickable_count += 1
            if getattr(w, "checkable", False):
                checkable_count += 1
            if getattr(w, "scrollable", False):
                scrollable_count += 1

        new_texts = self._collect_texts(new_state)
        old_texts = self._collect_texts(old_state)

        # 可以逐步增强，这里先给实用规则
        if target == "form":
            if editable_count >= 1:
                return 1.0, f"target_state=form 命中: editable_count={editable_count}"
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
            if editable_count >= 1 and delta >= 1:
                return 1.0, f"target_state=search 命中: editable_count={editable_count}, text_delta={delta}"
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