"""
ReAct 动态规划器
评估当前UI状态与目标差距，生成单个子目标
支持长期记忆经验复用
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from Perception.context_builder import UIState
from Memory.memory_manager import MemoryManager
from prompt.planner_prompt import PLANNER_PROMPT

logger = logging.getLogger(__name__)


# ========================
# 数据类定义
# ========================


@dataclass
class SubGoal:
    """单个子目标"""
    description: str                           # 子目标描述
    target_widget_text: str = ""               # 目标控件文本特征
    target_widget_id: Optional[int] = None     # 目标控件 ID
    action_type: str = ""                      # 预期动作类型 (tap/swipe/input/back)
    input_text: str = ""                       # 输入文本（action_type=input 时）
    acceptance_criteria: str = ""              # 验收标准描述
    expected_transition: str = ""              # 跳变预期类型 (partial_refresh/new_page/external_app/dialog)
    from_experience: bool = False              # 是否来自长期记忆复用


@dataclass
class PlanResult:
    """规划结果"""
    subgoal: SubGoal
    is_task_complete: bool = False             # LLM 判断任务是否已完成
    reasoning: str = ""                        # 推理过程


class Planner:
    """
    ReAct 动态规划器
    - 优先检索长期记忆中的经验
    - 无命中时通过 LLM 生成单个子目标
    - 子目标包含验收标准和跳变预期
    """

    def __init__(self, llm_client, memory_manager: MemoryManager):
        """
        :param llm_client: LLM 调用客户端 (需有 chat 方法)
        :param memory_manager: 记忆管理器
        """
        self.llm = llm_client
        self.memory = memory_manager
        logger.info("Planner 初始化完成")

    def plan(self, task: str, ui_state: UIState) -> PlanResult:
        """
        生成下一个子目标
        :param task: 最终任务描述
        :param ui_state: 当前 UI 状态
        :return: PlanResult
        """
        logger.info("开始规划, 任务: '%s'", task[:80])

        # 第一步：尝试从长期记忆检索经验
        experience = self.memory.search_experience(task)
        if experience:
            logger.info("命中长期记忆经验, 将复用历史动作链路")
            return self._plan_from_experience(experience)

        # 第二步：通过 LLM 动态规划
        return self._plan_with_llm(task, ui_state)

    def _plan_from_experience(self, experience) -> PlanResult:
        """从历史经验中复用动作"""
        if not experience.action_sequence:
            logger.warning("经验记录中动作序列为空, 回退到 LLM 规划")
            return PlanResult(
                subgoal=SubGoal(description="经验为空，需要LLM重新规划"),
                reasoning="经验记录中无动作序列",
            )

        # 取第一个未执行的动作
        action = experience.action_sequence[0]
        subgoal = SubGoal(
            description=action.get("description", "来自经验的动作"),
            target_widget_text=action.get("target_widget_text", ""),
            action_type=action.get("action_type", "tap"),
            input_text=action.get("input_text", ""),
            acceptance_criteria=action.get("acceptance_criteria", ""),
            expected_transition=action.get("expected_transition", "partial_refresh"),
            from_experience=True,
        )
        logger.info("从经验复用子目标: '%s'", subgoal.description)

        return PlanResult(
            subgoal=subgoal,
            reasoning="复用长期记忆中的成功经验",
        )

    def _plan_with_llm(self, task: str, ui_state: UIState) -> PlanResult:
        """通过 LLM 生成子目标"""
        history_text = self.memory.short_term.get_context_summary()
        ui_text = ui_state.to_prompt_text()

        prompt = PLANNER_PROMPT.format(
            task=task,
            ui_state=ui_text,
            history=history_text or "暂无历史记录",
        )

        logger.debug("发送规划请求到 LLM")
        try:
            response = self.llm.chat(prompt)
            result = self._parse_plan_response(response)
            logger.info(
                "LLM 规划完成: subgoal='%s', action=%s, complete=%s",
                result.subgoal.description, result.subgoal.action_type, result.is_task_complete,
            )
            return result
        except Exception as e:
            logger.error("LLM 规划失败: %s", e)
            return PlanResult(
                subgoal=SubGoal(description="LLM规划失败，需要重试"),
                reasoning=f"LLM调用异常: {e}",
            )

    def _parse_plan_response(self, response: str) -> PlanResult:
        """解析 LLM 返回的 JSON 规划结果"""
        # 提取 JSON 块
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.warning("JSON 解析失败: %s, 原始响应: %s", e, response[:200])
            return PlanResult(
                subgoal=SubGoal(description=response[:100]),
                reasoning="JSON解析失败，使用原始响应",
            )

        sg_data = data.get("subgoal", {})
        subgoal = SubGoal(
            description=sg_data.get("description", ""),
            target_widget_text=sg_data.get("target_widget_text", ""),
            target_widget_id=sg_data.get("target_widget_id"),
            action_type=sg_data.get("action_type", "tap"),
            input_text=sg_data.get("input_text", ""),
            acceptance_criteria=sg_data.get("acceptance_criteria", ""),
            expected_transition=sg_data.get("expected_transition", "partial_refresh"),
        )

        return PlanResult(
            subgoal=subgoal,
            is_task_complete=data.get("is_task_complete", False),
            reasoning=data.get("reasoning", ""),
        )
