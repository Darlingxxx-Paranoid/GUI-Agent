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
        success: bool,
        action_sequence: Optional[List[Dict[str, Any]]] = None,
        timestamp: float = None,
        metadata: Dict[str, Any] = None,
        semantic_steps: Optional[List[Dict[str, Any]]] = None,
        schema_version: int = 1,
    ):
        self.task_description = task_description
        self.semantic_steps = semantic_steps if semantic_steps is not None else (action_sequence or [])
        # 向后兼容：旧逻辑仍通过 action_sequence 访问
        self.action_sequence = self.semantic_steps
        self.success = success
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.schema_version = schema_version

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "schema_version": self.schema_version,
            "semantic_steps": self.semantic_steps,
            # 保留旧字段，便于旧版本读取
            "action_sequence": self.semantic_steps,
            "success": self.success,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceRecord":
        semantic_steps = data.get("semantic_steps")
        if semantic_steps is None:
            semantic_steps = data.get("action_sequence", [])

        return cls(
            task_description=data["task_description"],
            action_sequence=semantic_steps,
            success=data.get("success", False),
            timestamp=data.get("timestamp", 0),
            metadata=data.get("metadata", {}),
            semantic_steps=semantic_steps,
            schema_version=int(data.get("schema_version", 1) or 1),
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
