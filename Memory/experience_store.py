"""Persistent experience storage for schema v3 triplets."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperienceRecord:
    """Single persisted experience record (schema v3)."""

    def __init__(
        self,
        task_description: str,
        success: bool,
        step_triplets: Optional[List[Dict[str, Any]]] = None,
        timestamp: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        schema_version: int = 3,
    ):
        self.task_description = str(task_description or "")
        self.step_triplets = list(step_triplets or [])
        self.success = bool(success)
        self.timestamp = float(timestamp or time.time())
        self.metadata = dict(metadata or {})
        self.schema_version = int(schema_version or 3)

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "schema_version": int(self.schema_version),
            "step_triplets": self.step_triplets,
            "success": bool(self.success),
            "timestamp": float(self.timestamp),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceRecord":
        schema_version = int(data.get("schema_version", 1) or 1)
        if schema_version < 3:
            raise ValueError(f"unsupported_schema_version={schema_version}")

        triplets = data.get("step_triplets")
        if triplets is None:
            triplets = []

        if not isinstance(triplets, list):
            triplets = []

        return cls(
            task_description=str(data.get("task_description") or ""),
            step_triplets=triplets,
            success=bool(data.get("success", False)),
            timestamp=float(data.get("timestamp") or time.time()),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            schema_version=3,
        )


class ExperienceStore:
    """JSON-backed store for v3 experience records."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._experiences: List[ExperienceRecord] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.store_path):
            logger.info("经验库文件不存在，将创建新文件: %s", self.store_path)
            self._experiences = []
            return

        try:
            with open(self.store_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("经验库读取失败，将重建为空: %s", exc)
            self._experiences = []
            self._save()
            return

        if not isinstance(payload, list):
            logger.warning("经验库结构异常(非列表)，将重建为空")
            self._experiences = []
            self._save()
            return

        loaded: List[ExperienceRecord] = []
        dropped = 0
        for item in payload:
            if not isinstance(item, dict):
                dropped += 1
                continue
            try:
                loaded.append(ExperienceRecord.from_dict(item))
            except Exception:
                dropped += 1

        self._experiences = loaded
        if dropped > 0:
            logger.warning("经验库已丢弃 %d 条非 v3 记录", dropped)
            self._save()

        logger.info("经验库加载完成: %d 条(v3)", len(self._experiences))

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as file:
            json.dump([item.to_dict() for item in self._experiences], file, ensure_ascii=False, indent=2)

    def add(self, record: ExperienceRecord):
        self._experiences.append(record)
        self._save()

    def get_successful(self) -> List[ExperienceRecord]:
        return [item for item in self._experiences if item.success]

    def get_all(self) -> List[ExperienceRecord]:
        return list(self._experiences)

    def clear(self):
        self._experiences = []
        self._save()
        logger.info("经验库已清空")
