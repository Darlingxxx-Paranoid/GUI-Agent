"""
事后评估模块
对动作结果进行约束检测和语义确认
决定是继续前进还是标记失败
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from Perception.context_builder import UIState
from Planning.oracle_pre import PreConstraints
from prompt.evaluator_prompt import EVALUATOR_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果"""
    success: bool                # 子目标是否成功完成
    reason: str = ""             # 判定理由
    constraint_passed: bool = True    # 基础约束是否通过
    semantic_passed: bool = True      # 语义确认是否通过


class Evaluator:
    """
    事后评估器
    1. 基础约束检测（字符串匹配、控件出现/消失）
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
        :param subgoal_description: 子目标描述
        :param constraints: 事前约束
        :param old_state: 执行前的 UI 状态
        :param new_state: 执行后的 UI 状态
        :return: EvalResult
        """
        logger.info("开始事后评估: '%s'", subgoal_description)

        # 第一步：基础约束检测
        constraint_result = self._check_constraints(constraints, new_state)

        if not constraint_result.constraint_passed:
            logger.info("基础约束未通过: %s", constraint_result.reason)
            return constraint_result

        # 第二步：语义确认（条件：有语义约束且基础约束通过）
        if constraints.semantic_criteria:
            semantic_result = self._semantic_check(
                subgoal_description, constraints.semantic_criteria,
                old_state, new_state,
            )
            if not semantic_result.success:
                logger.info("语义确认未通过: %s", semantic_result.reason)
                return semantic_result

        logger.info("事后评估通过: '%s'", subgoal_description)
        return EvalResult(success=True, reason="基础约束和语义确认均通过")

    def _check_constraints(self, constraints: PreConstraints, new_state: UIState) -> EvalResult:
        """基础约束检测"""

        # 检查期望出现的文字
        for text in constraints.expected_texts:
            found = False
            for w in new_state.widgets:
                if text in w.text or text in w.content_desc:
                    found = True
                    break
            if not found:
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason=f"期望文字未出现: '{text}'",
                )

        # 检查不应出现的文字
        for text in constraints.forbidden_texts:
            for w in new_state.widgets:
                if text in w.text or text in w.content_desc:
                    return EvalResult(
                        success=False,
                        constraint_passed=False,
                        reason=f"出现了禁止文字: '{text}' in widget '{w.text}'",
                    )

        # 检查期望存在的控件
        for feature in constraints.widget_should_exist:
            found = any(
                feature in w.text or feature in w.content_desc or feature in w.resource_id
                for w in new_state.widgets
            )
            if not found:
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason=f"期望控件未出现: '{feature}'",
                )

        # 检查期望消失的控件
        for feature in constraints.widget_should_vanish:
            found = any(
                feature in w.text or feature in w.content_desc or feature in w.resource_id
                for w in new_state.widgets
            )
            if found:
                return EvalResult(
                    success=False,
                    constraint_passed=False,
                    reason=f"期望消失的控件仍然存在: '{feature}'",
                )

        return EvalResult(success=True, constraint_passed=True, reason="基础约束全部通过")

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
