"""Audit artifact recorder for LLM outputs and stitched dataclass instances."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from Oracle.contracts import to_plain_dict


class AuditRecorder:
    """Write structured audit records under data/audit for manual inspection."""

    _lock = threading.Lock()
    _sequence = 0

    def __init__(self, component: str = "runtime", base_dir: str = ""):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.component = self._sanitize_token(component or "runtime")
        self.base_dir = base_dir or os.path.join(project_root, "data", "audit")
        os.makedirs(self.base_dir, exist_ok=True)

    @classmethod
    def _next_sequence(cls) -> int:
        with cls._lock:
            cls._sequence += 1
            return cls._sequence

    def record(
        self,
        module: str,
        stage: str,
        event: str,
        payload: Any,
    ) -> str:
        module_token = self._sanitize_token(module or self.component)
        stage_token = self._sanitize_token(stage or "default")
        event_token = self._sanitize_token(event or "event")

        folder = os.path.join(self.base_dir, module_token)
        os.makedirs(folder, exist_ok=True)

        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        millis = int((now - int(now)) * 1000)
        seq = self._next_sequence()
        path = os.path.join(
            folder,
            f"{stamp}_{millis:03d}_{seq:06d}_{stage_token}_{event_token}.json",
        )

        body = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "component": self.component,
            "module": module_token,
            "stage": stage,
            "event": event,
            "payload": to_plain_dict(payload),
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(body, file, ensure_ascii=False, indent=2)
        return path

    def _sanitize_token(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9_.-]+", "_", text)
        return text.strip("._-") or "default"
