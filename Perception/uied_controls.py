"""Utilities for extracting UIED visible controls from a screenshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from Perception.uied.detect import WidgetDetector

logger = logging.getLogger(__name__)


def get_uied_visible_controls(
    screenshot_path: str,
    cv_output_dir: str,
    resize_height: int = 800,
) -> list[dict]:
    """
    Run UIED on a screenshot and return normalized visible controls.

    The returned controls always include:
    - widget_id
    - class
    - text
    - bounds
    - center
    - width
    - height
    """
    screenshot = _resolve_existing_file(screenshot_path)
    cv_root = Path(str(cv_output_dir or "").strip()).expanduser().resolve()
    cv_root.mkdir(parents=True, exist_ok=True)

    detector = WidgetDetector()
    detector.resize_length = int(resize_height or 800)
    detector.output_root = str(cv_root)

    try:
        _, resize_ratio, raw_controls = detector.detect(
            img_path=str(screenshot),
            debug=False,
        )
    except Exception as exc:
        raise RuntimeError(f"UIED detection failed: {exc}") from exc

    controls: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_controls or []):
        normalized = _normalize_control(item=item, fallback_id=idx)
        if normalized is None:
            logger.warning("Skip invalid UIED control at index=%d", idx)
            continue
        controls.append(normalized)

    payload = {
        "screenshot_path": str(screenshot),
        "resize_ratio": float(resize_ratio or 1.0),
        "count": len(controls),
        "controls": controls,
    }
    out_path = _context_output_path(screenshot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("UIED controls saved: %s (count=%d)", out_path, len(controls))
    return controls


def _normalize_control(item: dict[str, Any], fallback_id: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    position = item.get("position")
    if not isinstance(position, dict):
        return None

    try:
        x1 = int(position.get("column_min"))
        y1 = int(position.get("row_min"))
        x2 = int(position.get("column_max"))
        y2 = int(position.get("row_max"))
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    try:
        widget_id = int(item.get("id"))
    except Exception:
        widget_id = int(fallback_id)

    try:
        width = int(item.get("width"))
    except Exception:
        width = x2 - x1

    try:
        height = int(item.get("height"))
    except Exception:
        height = y2 - y1

    return {
        "widget_id": widget_id,
        "class": str(item.get("class") or ""),
        "text": str(item.get("text_content") or ""),
        "bounds": [x1, y1, x2, y2],
        "center": [(x1 + x2) // 2, (y1 + y2) // 2],
        "width": width,
        "height": height,
    }


def _resolve_existing_file(path_text: str) -> Path:
    text = str(path_text or "").strip()
    if not text:
        raise FileNotFoundError("screenshot_path is empty")

    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"screenshot file not found: {candidate}")
    return candidate


def _context_output_path(screenshot: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    context_dir = project_root / "data" / "context"
    return context_dir / f"{screenshot.stem}.uied_controls.json"
