"""
记忆管理模块
统一管理短期记忆（当前Task内）和长期记忆（跨Task持久化经验）
"""
import logging
from typing import List, Dict, Any, Optional

from utils.utils import cosine_similarity
from Memory.experience_store import ExperienceStore, ExperienceRecord

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆 —— 当前 Task 范围内的即时上下文
    包含：历史动作、子目标执行记录、失败原因
    随 Task 结束而清空
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []        # 每一步的完整记录
        self.action_log: List[Dict[str, Any]] = []     # 动作序列（坐标+类型）
        self.failure_reasons: List[str] = []            # 失败原因列表
        self.current_subgoal: Optional[str] = None      # 当前子目标描述
        logger.debug("短期记忆已初始化")

    def add_step(self, step_record: Dict[str, Any]):
        """记录一步完整执行信息"""
        self.history.append(step_record)
        logger.debug("短期记忆新增步骤 #%d", len(self.history))

    def add_action(self, action: Dict[str, Any]):
        """记录一个动作"""
        self.action_log.append(action)

    def add_failure(self, reason: str):
        """记录一次失败原因"""
        self.failure_reasons.append(reason)
        logger.info("记录失败原因: %s", reason)

    def get_recent_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近 n 个动作"""
        return self.action_log[-n:]

    def get_context_summary(self) -> str:
        """生成供 LLM 使用的上下文摘要"""
        lines = []
        for i, step in enumerate(self.history[-5:], 1):
            subgoal = step.get("subgoal", "未知")
            result = step.get("result", "未知")
            lines.append(f"步骤{i}: 子目标='{subgoal}' 结果={result}")
        if self.failure_reasons:
            lines.append(f"最近失败原因: {self.failure_reasons[-1]}")
        return "\n".join(lines)

    def clear(self):
        """清空短期记忆（Task结束时调用）"""
        self.history.clear()
        self.action_log.clear()
        self.failure_reasons.clear()
        self.current_subgoal = None
        logger.info("短期记忆已清空")


class MemoryManager:
    """
    统一记忆管理器
    - 短期记忆：当前Task内维护，Task结束清空
    - 长期记忆：通过 ExperienceStore 持久化存储和检索
    """

    def __init__(self, experience_store_path: str, similarity_threshold: float = 0.75):
        self.short_term = ShortTermMemory()
        self.experience_store = ExperienceStore(experience_store_path)
        self.similarity_threshold = similarity_threshold
        logger.info("记忆管理器初始化完成, 经验库路径: %s", experience_store_path)

    def search_experience(self, task_description: str) -> Optional[ExperienceRecord]:
        """
        根据任务描述在长期记忆中检索最相似的成功经验
        使用字符级余弦相似度进行匹配
        :return: 匹配的经验记录，若无则返回 None
        """
        best_match: Optional[ExperienceRecord] = None
        best_score = 0.0

        for exp in self.experience_store.get_successful():
            score = cosine_similarity(task_description, exp.task_description)
            if score > best_score:
                best_score = score
                best_match = exp

        if best_match and best_score >= self.similarity_threshold:
            logger.info(
                "经验命中! 相似度=%.3f, 历史任务='%s'",
                best_score, best_match.task_description[:50]
            )
            return best_match
        else:
            logger.debug("未找到匹配经验, 最高相似度=%.3f", best_score)
            return None

    def save_experience(self, task_description: str, action_sequence: List[Dict], success: bool):
        """将当前Task的执行经验沉淀到长期记忆"""
        record = ExperienceRecord(
            task_description=task_description,
            action_sequence=action_sequence,
            success=success,
        )
        self.experience_store.add(record)
        logger.info("经验已沉淀: success=%s, actions=%d", success, len(action_sequence))

    def reset_short_term(self):
        """重置短期记忆（新Task开始时调用）"""
        self.short_term.clear()
