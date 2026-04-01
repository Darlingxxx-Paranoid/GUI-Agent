"""Memory manager for short-term context only."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ShortTermMemory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.action_log: List[Dict[str, Any]] = []
        self.failure_reasons: List[str] = []
        self.current_subgoal: Optional[str] = None
        logger.debug("短期记忆已初始化")

    def add_step(self, step_record: Dict[str, Any]):
        self.history.append(step_record)

    def add_action(self, action: Dict[str, Any]):
        self.action_log.append(action)

    def add_failure(self, reason: str):
        self.failure_reasons.append(str(reason or ""))

    def get_recent_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.action_log[-max(1, int(n or 1)) :]

    def get_context_summary(self) -> str:
        lines = []
        for idx, step in enumerate(self.history[-5:], 1):
            goal = (step.get("goal") or {}).get("summary") if isinstance(step.get("goal"), dict) else ""
            goal = goal or step.get("subgoal") or "未知"
            result = step.get("result", "未知")
            lines.append(f"步骤{idx}: 目标='{goal}' 结果={result}")
        if self.failure_reasons:
            lines.append(f"最近失败原因: {self.failure_reasons[-1]}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
        self.action_log.clear()
        self.failure_reasons.clear()
        self.current_subgoal = None


class MemoryManager:
    def __init__(self, experience_store_path: str = "", similarity_threshold: float = 0.75):
        self.short_term = ShortTermMemory()
        self.similarity_threshold = float(similarity_threshold or 0.75)
        logger.info("记忆管理器初始化完成(v4, long_term_disabled)")

    def search_experience(self, task_description: str):
        _ = task_description
        # 长期记忆链路已停用，统一返回空命中。
        return None

    def save_experience(
        self,
        task_description: str,
        step_triplets: List[Dict[str, Any]],
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        _ = (task_description, step_triplets, success, metadata)
        # 长期记忆链路已停用。
        return None

    def reset_short_term(self):
        self.short_term.clear()
