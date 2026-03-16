"""
重规划与状态回溯模块
处理失败情况：分析原因、决定回溯或重试、沉淀经验
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from Evaluation.evaluator import EvalResult
from Memory.memory_manager import MemoryManager
from Perception.context_builder import UIState
from prompt.replanner_prompt import REPLANNER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ReplanDecision:
    """重规划决策结果"""
    action: str           # retry / back / replan / abort
    reason: str = ""
    back_steps: int = 1   # 回退步数


class Replanner:
    """
    重规划器
    - 失败原因分析 → 加入短期记忆
    - 死胡同检测 → 触发 Back 回溯
    - 经验沉淀 → 成功时保存到长期记忆
    """

    def __init__(self, llm_client, memory_manager: MemoryManager, dead_end_threshold: int = 3):
        """
        :param llm_client: LLM 客户端
        :param memory_manager: 记忆管理器
        :param dead_end_threshold: 连续失败次数阈值
        """
        self.llm = llm_client
        self.memory = memory_manager
        self.dead_end_threshold = dead_end_threshold
        self.consecutive_failures = 0
        logger.info("Replanner 初始化完成, 死胡同阈值=%d", dead_end_threshold)

    def handle_failure(
        self,
        subgoal_description: str,
        eval_result: EvalResult,
        ui_state: UIState,
    ) -> ReplanDecision:
        """
        处理子目标执行失败
        :param subgoal_description: 子目标描述
        :param eval_result: 评估结果
        :param ui_state: 当前 UI 状态
        :return: 重规划决策
        """
        self.consecutive_failures += 1
        logger.info(
            "处理失败: subgoal='%s', reason='%s', 连续失败次数=%d",
            subgoal_description, eval_result.reason, self.consecutive_failures,
        )

        # 记录失败到短期记忆
        self.memory.short_term.add_failure(
            f"[{subgoal_description}] {eval_result.reason}"
        )

        # 死胡同检测：连续失败次数超过阈值
        if self.consecutive_failures >= self.dead_end_threshold:
            logger.warning(
                "检测到死胡同! 连续失败 %d 次, 触发回溯",
                self.consecutive_failures,
            )
            self.consecutive_failures = 0  # 重置计数
            return ReplanDecision(
                action="back",
                reason=f"连续失败{self.dead_end_threshold}次，判定为死胡同，回溯重新观测",
                back_steps=1,
            )

        # 使用 LLM 分析失败原因
        return self._analyze_with_llm(subgoal_description, eval_result, ui_state)

    def handle_success(self, task_description: str):
        """
        处理子目标成功
        重置连续失败计数
        """
        self.consecutive_failures = 0
        logger.debug("子目标成功, 重置连续失败计数")

    def save_task_experience(self, task_description: str, success: bool):
        """
        任务结束时沉淀经验
        将正确的动作序列转化为长期记忆
        """
        actions = self.memory.short_term.action_log
        if success and actions:
            self.memory.save_experience(
                task_description=task_description,
                action_sequence=[a for a in actions],
                success=True,
            )
            logger.info("任务经验已沉淀: %d 个动作", len(actions))
        elif not success:
            logger.info("任务失败, 不沉淀经验")

    def _analyze_with_llm(
        self,
        subgoal_description: str,
        eval_result: EvalResult,
        ui_state: UIState,
    ) -> ReplanDecision:
        """通过 LLM 分析失败原因"""
        prompt = REPLANNER_PROMPT.format(
            subgoal_description=subgoal_description,
            failure_reason=eval_result.reason,
            ui_state=ui_state.to_prompt_text()[:1000],
            history=self.memory.short_term.get_context_summary(),
        )

        try:
            response = self.llm.chat(prompt)
            decision = self._parse_response(response)
            logger.info("LLM 失败分析: action=%s, reason='%s'", decision.action, decision.reason)
            return decision
        except Exception as e:
            logger.error("LLM 失败分析出错: %s, 默认重试", e)
            return ReplanDecision(action="retry", reason=f"LLM分析失败({e}), 默认重试")

    def _parse_response(self, response: str) -> ReplanDecision:
        """解析 LLM 失败分析结果"""
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
            action = data.get("suggestion", "retry")
            if action not in ("retry", "back", "replan", "abort"):
                action = "retry"
            return ReplanDecision(
                action=action,
                reason=data.get("analysis", "") + " | " + data.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            logger.warning("失败分析 JSON 解析失败, 默认重试")
            return ReplanDecision(action="retry", reason="JSON解析失败, 默认重试")

    def reset(self):
        """重置状态"""
        self.consecutive_failures = 0
        logger.debug("Replanner 状态已重置")
