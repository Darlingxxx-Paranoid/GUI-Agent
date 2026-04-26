"""Step-aligned artifact recorder: data/<kind>/step_<n>[_llm].json."""

from __future__ import annotations

import json
import os
import re
import threading
import dataclasses
from typing import Any


class AuditRecorder:
    """Write artifacts grouped by kind and aligned to agent step index."""

    _lock = threading.Lock()

    def __init__(self, component: str = "runtime", base_dir: str = ""):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.component = self._sanitize_token(component or "runtime")
        self.base_dir = os.path.abspath(base_dir or os.path.join(project_root, "data"))
        os.makedirs(self.base_dir, exist_ok=True)

    def record_step(
        self,
        artifact_kind: str,
        step: int,
        payload: Any,
        llm: bool = False,
        append: bool = False,
    ) -> str:
        step_num = self._normalize_step(step)
        kind = self._sanitize_token(artifact_kind)
        folder = os.path.join(self.base_dir, kind)
        os.makedirs(folder, exist_ok=True)

        suffix = "_llm" if llm else ""
        path = os.path.join(folder, f"step_{step_num}{suffix}.json")
        body = _to_plain_dict(payload)

        if append:
            with self._lock:
                merged = self._append_payload(path=path, payload=body)
                self._write_json(path=path, payload=merged)
        else:
            self._write_json(path=path, payload=body)
        return path

    def record_step_phase(
        self,
        artifact_kind: str,
        step: int,
        phase: str,
        payload: Any,
        llm: bool = False,
    ) -> str:
        step_num = self._normalize_step(step)
        kind = self._sanitize_token(artifact_kind)
        phase_token = self._sanitize_token(phase)
        folder = os.path.join(self.base_dir, kind)
        os.makedirs(folder, exist_ok=True)

        suffix = "_llm" if llm else ""
        path = os.path.join(folder, f"step_{step_num}_{phase_token}{suffix}.json")
        body = _to_plain_dict(payload)
        self._write_json(path=path, payload=body)
        return path

    def _append_payload(self, path: str, payload: Any) -> list[Any]:
        current: list[Any] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as file:
                    old = json.load(file)
                if isinstance(old, list):
                    current = old
                else:
                    current = [old]
            except Exception:
                current = []
        current.append(payload)
        return current

    def _write_json(self, path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _normalize_step(self, step: int) -> int:
        try:
            value = int(step)
        except Exception as exc:
            raise ValueError(f"invalid step index: {step}") from exc
        if value <= 0:
            raise ValueError(f"step must be >=1, got {value}")
        return value

    def _sanitize_token(self, value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
        text = text.strip("._-")
        return text or "default"


def _to_plain_dict(value: Any) -> Any:
    """Best-effort serialization for dataclass/pydantic/native types."""
    if dataclasses.is_dataclass(value):
        output: dict[str, Any] = {}
        for item in dataclasses.fields(value):
            output[item.name] = _to_plain_dict(getattr(value, item.name))
        return output

    if hasattr(value, "model_dump"):
        try:
            return _to_plain_dict(value.model_dump())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _to_plain_dict(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain_dict(v) for v in value]

    return value
