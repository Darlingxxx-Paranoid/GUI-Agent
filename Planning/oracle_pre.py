"""
事前 Oracle 模块
为子目标生成验收标准和跳变预期
与规划模块配合，保证子目标天生带有约束条件
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from Perception.context_builder import UIState
from Planning.planner import SubGoal
from prompt.oracle_pre_prompt import ORACLE_PRE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PreConstraints:
    """事前约束数据"""
    # 基础约束 —— 可编程检测
    expected_texts: List[str] = field(default_factory=list)     # 期望出现的文字
    forbidden_texts: List[str] = field(default_factory=list)    # 不应出现的文字
    expected_activity: str = ""                                  # 期望跳转的 Activity
    widget_should_exist: List[str] = field(default_factory=list)  # 期望存在的控件特征
    widget_should_vanish: List[str] = field(default_factory=list)  # 期望消失的控件特征

    # 跳变预期
    transition_type: str = "partial_refresh"  # partial_refresh/new_page/dialog/external_app/none
    transition_description: str = ""           # 跳变预期的文字描述

    # 高阶语义约束（由 LLM 在事后评估时判断）
    semantic_criteria: str = ""                # 自然语言描述的验收标准


class OraclePre:
    """
    事前 Oracle
    在动作执行前，为子目标生成可验证的约束条件
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("OraclePre 初始化完成")

    def generate_constraints(self, subgoal: SubGoal, ui_state: UIState) -> PreConstraints:
        """
        为子目标生成事前约束
        :param subgoal: 当前子目标
        :param ui_state: 当前 UI 状态
        :return: 事前约束数据
        """
        logger.info("生成事前约束: subgoal='%s'", subgoal.description)

        # 如果子目标自带约束信息（来自经验），直接使用
        if subgoal.from_experience and subgoal.acceptance_criteria:
            logger.info("使用经验中的约束信息")
            return PreConstraints(
                transition_type=subgoal.expected_transition,
                semantic_criteria=subgoal.acceptance_criteria,
            )

        # 通过 LLM 生成约束
        return self._generate_with_llm(subgoal, ui_state)

    def _generate_with_llm(self, subgoal: SubGoal, ui_state: UIState) -> PreConstraints:
        """通过 LLM 生成约束"""
        prompt = ORACLE_PRE_PROMPT.format(
            description=subgoal.description,
            action_type=subgoal.action_type,
            target_widget=subgoal.target_widget_text,
            input_text=subgoal.input_text or "无",
            ui_summary=ui_state.to_prompt_text()[:1500],  # 截断防超长
        )

        try:
            response = self.llm.chat(prompt)
            constraints = self._parse_response(response)
            logger.info(
                "事前约束生成完成: transition=%s, expected_texts=%s",
                constraints.transition_type, constraints.expected_texts,
            )
            return constraints
        except Exception as e:
            logger.error("事前约束生成失败: %s, 使用默认约束", e)
            return PreConstraints(
                transition_type=subgoal.expected_transition or "partial_refresh",
                semantic_criteria=subgoal.acceptance_criteria or subgoal.description,
            )

    def _parse_response(self, response: str) -> PreConstraints:
        """解析 LLM 返回的约束 JSON"""
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.warning("约束 JSON 解析失败: %s", e)
            return PreConstraints(semantic_criteria=response[:200])

        return PreConstraints(
            expected_texts=data.get("expected_texts", []),
            forbidden_texts=data.get("forbidden_texts", []),
            expected_activity=data.get("expected_activity", ""),
            widget_should_exist=data.get("widget_should_exist", []),
            widget_should_vanish=data.get("widget_should_vanish", []),
            transition_type=data.get("transition_type", "partial_refresh"),
            transition_description=data.get("transition_description", ""),
            semantic_criteria=data.get("semantic_criteria", ""),
        )
