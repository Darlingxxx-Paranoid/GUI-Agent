"""Utilities for extracting UIED visible controls from a screenshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2

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
        merged_preview_path, resize_ratio, raw_controls = detector.detect(
            img_path=str(screenshot),
            debug=False,
        )
    except Exception as exc:
        raise RuntimeError(f"UIED detection failed: {exc}") from exc

    src_img = cv2.imread(str(screenshot))
    if src_img is None:
        raise RuntimeError(f"Failed to read screenshot: {screenshot}")
    src_h, src_w = src_img.shape[:2]

    merged_img = cv2.imread(str(merged_preview_path))
    if merged_img is not None:
        det_h, det_w = merged_img.shape[:2]
    else:
        # Fallback to isotropic scale when the merge preview cannot be read.
        ratio = float(resize_ratio or 1.0)
        inv_ratio = 1.0 / ratio if ratio > 0 else 1.0
        det_h = max(1, int(round(src_h / inv_ratio)))
        det_w = max(1, int(round(src_w / inv_ratio)))

    scale_x = float(src_w) / float(det_w) if det_w > 0 else 1.0
    scale_y = float(src_h) / float(det_h) if det_h > 0 else 1.0

    controls: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_controls or []):
        normalized = _normalize_control(
            item=item,
            fallback_id=idx,
            scale_x=scale_x,
            scale_y=scale_y,
            screen_width=src_w,
            screen_height=src_h,
        )
        if normalized is None:
            logger.warning("Skip invalid UIED control at index=%d", idx)
            continue
        controls.append(normalized)

    payload = {
        "screenshot_path": str(screenshot),
        "coordinate_space": "original_screenshot",
        "screen_width": int(src_w),
        "screen_height": int(src_h),
        "detected_width": int(det_w),
        "detected_height": int(det_h),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "resize_ratio": float(resize_ratio or 1.0),
        "count": len(controls),
        "controls": controls,
    }
    out_path = _context_output_path(screenshot=screenshot, cv_root=cv_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("UIED controls saved: %s (count=%d)", out_path, len(controls))
    return controls


def _normalize_control(
    item: dict[str, Any],
    fallback_id: int,
    scale_x: float,
    scale_y: float,
    screen_width: int,
    screen_height: int,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    position = item.get("position")
    if not isinstance(position, dict):
        return None

    try:
        raw_x1 = float(position.get("column_min"))
        raw_y1 = float(position.get("row_min"))
        raw_x2 = float(position.get("column_max"))
        raw_y2 = float(position.get("row_max"))
    except Exception:
        return None

    x1 = int(round(raw_x1 * scale_x))
    y1 = int(round(raw_y1 * scale_y))
    x2 = int(round(raw_x2 * scale_x))
    y2 = int(round(raw_y2 * scale_y))

    if screen_width > 0:
        x1 = max(0, min(x1, screen_width - 1))
        x2 = max(0, min(x2, screen_width))
    if screen_height > 0:
        y1 = max(0, min(y1, screen_height - 1))
        y2 = max(0, min(y2, screen_height))

    if x2 <= x1 or y2 <= y1:
        return None

    try:
        widget_id = int(item.get("id"))
    except Exception:
        widget_id = int(fallback_id)

    width = x2 - x1
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


def _context_output_path(screenshot: Path, cv_root: Path) -> Path:
    context_dir = cv_root / "context"
    return context_dir / f"{screenshot.stem}.uied_controls.json"
