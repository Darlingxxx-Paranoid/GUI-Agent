"""
经验持久化存储模块
将成功的任务执行经验以JSON格式持久化到磁盘，支持增删查
"""
import json
import os
import logging
import time
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ExperienceRecord:
    """单条经验记录"""

    def __init__(
        self,
        task_description: str,
        action_sequence: List[Dict[str, Any]],
        success: bool,
        timestamp: float = None,
        metadata: Dict[str, Any] = None,
    ):
        self.task_description = task_description
        self.action_sequence = action_sequence
        self.success = success
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "action_sequence": self.action_sequence,
            "success": self.success,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceRecord":
        return cls(
            task_description=data["task_description"],
            action_sequence=data.get("action_sequence", []),
            success=data.get("success", False),
            timestamp=data.get("timestamp", 0),
            metadata=data.get("metadata", {}),
        )


class ExperienceStore:
    """
    基于JSON文件的经验持久化存储
    每条经验包含：任务描述、动作序列、成功标记、时间戳
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._experiences: List[ExperienceRecord] = []
        self._load()

    def _load(self):
        """从磁盘加载经验库"""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._experiences = [ExperienceRecord.from_dict(d) for d in data]
                logger.info("从 %s 加载了 %d 条经验记录", self.store_path, len(self._experiences))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("经验库文件损坏，将重建: %s", e)
                self._experiences = []
        else:
            logger.info("经验库文件不存在，将创建新文件: %s", self.store_path)
            self._experiences = []

    def _save(self):
        """持久化到磁盘"""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._experiences], f, ensure_ascii=False, indent=2)
        logger.debug("经验库已保存，共 %d 条记录", len(self._experiences))

    def add(self, record: ExperienceRecord):
        """添加一条经验记录并持久化"""
        self._experiences.append(record)
        self._save()
        logger.info("新增经验记录: task='%s', success=%s", record.task_description[:50], record.success)

    def get_successful(self) -> List[ExperienceRecord]:
        """获取所有成功的经验"""
        return [e for e in self._experiences if e.success]

    def get_all(self) -> List[ExperienceRecord]:
        """获取所有经验"""
        return list(self._experiences)

    def clear(self):
        """清空经验库"""
        self._experiences = []
        self._save()
        logger.info("经验库已清空")
