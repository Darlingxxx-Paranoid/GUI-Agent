"""Utilities for extracting UIED visible widgets list from a screenshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

import cv2

from Perception.uied.detect import WidgetDetector

logger = logging.getLogger(__name__)


def get_uied_visible_widgets_list(
    screenshot_path: str,
    cv_output_dir: str,
    resize_height: int = 800,
) -> list[dict]:
    """
    Run UIED on a screenshot and return normalized visible widgets list.

    The returned widgets always include:
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

    visible_widgets_list: list[dict[str, Any]] = []
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
            logger.warning("Skip invalid UIED widget at index=%d", idx)
            continue
        visible_widgets_list.append(normalized)

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
        "count": len(visible_widgets_list),
        "visible_widgets_list": visible_widgets_list,
        "controls": visible_widgets_list,
    }
    out_path = _context_output_path(screenshot=screenshot, cv_root=cv_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("UIED visible widgets list saved: %s (count=%d)", out_path, len(visible_widgets_list))
    return visible_widgets_list


def build_uied_numbered_anchor_image(
    screenshot_path: str,
    widgets: list[dict[str, Any]],
    cv_output_dir: str,
    step: int | None = None,
    bbox_output_dir: str | None = None,
) -> str:
    """
    Render numbered UIED boxes on top of the original screenshot for LLM anchoring.

    The overlaid number is widget_id, which should be aligned with the executor-side
    widget list used after anchoring.
    """
    screenshot = _resolve_existing_file(screenshot_path)
    cv_root = Path(str(cv_output_dir or "").strip()).expanduser().resolve()
    context_dir = cv_root / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(screenshot))
    if image is None:
        raise RuntimeError(f"Failed to read screenshot for anchor overlay: {screenshot}")

    drawn = 0
    img_h, img_w = image.shape[:2]
    for widget in list(widgets or []):
        parsed = _parse_widget_for_overlay(widget=widget, img_w=img_w, img_h=img_h)
        if parsed is None:
            continue
        widget_id, (x1, y1, x2, y2) = parsed
        # Keep the same style as UIED Element.visualize_element.
        x1 = max(0, x1 - 3)
        y1 = max(0, y1 - 3)
        x2 = min(img_w - 1, x2 + 3)
        y2 = min(img_h - 1, y2 + 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        text = f"{widget_id}"
        tx = max(0, x1 - (20 if widget_id >= 10 else 10))
        ty = min(max(16, y1 + 12), max(16, img_h - 4))
        cv2.putText(
            image,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        drawn += 1

    if drawn <= 0:
        raise RuntimeError("No valid widgets can be drawn on anchor overlay image")

    context_out_path = context_dir / f"{screenshot.stem}.uied_anchor_numbered.jpg"
    ok = cv2.imwrite(str(context_out_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write anchor overlay image: {context_out_path}")

    bbox_dir = _resolve_bbox_output_dir(
        cv_root=cv_root,
        bbox_output_dir=bbox_output_dir,
    )
    bbox_dir.mkdir(parents=True, exist_ok=True)
    bbox_name = _build_bbox_filename(screenshot=screenshot, step=step)
    bbox_out_path = bbox_dir / bbox_name
    ok = cv2.imwrite(str(bbox_out_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write bbox screenshot image: {bbox_out_path}")

    logger.info(
        "UIED anchor numbered image saved: context=%s, bbox=%s (drawn=%d)",
        context_out_path,
        bbox_out_path,
        drawn,
    )
    return str(bbox_out_path)


def get_uied_visible_controls(
    screenshot_path: str,
    cv_output_dir: str,
    resize_height: int = 800,
) -> list[dict]:
    """Backward-compatible alias for get_uied_visible_widgets_list."""
    return get_uied_visible_widgets_list(
        screenshot_path=screenshot_path,
        cv_output_dir=cv_output_dir,
        resize_height=resize_height,
    )


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


def _parse_widget_for_overlay(
    widget: dict[str, Any],
    img_w: int,
    img_h: int,
) -> tuple[int, tuple[int, int, int, int]] | None:
    if not isinstance(widget, dict):
        return None

    try:
        widget_id = int(widget.get("widget_id"))
    except Exception:
        return None

    bounds = widget.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        x1 = int(bounds[0])
        y1 = int(bounds[1])
        x2 = int(bounds[2])
        y2 = int(bounds[3])
    except Exception:
        return None

    if img_w > 0:
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w))
    if img_h > 0:
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return widget_id, (x1, y1, x2, y2)


def _resolve_bbox_output_dir(cv_root: Path, bbox_output_dir: str | None) -> Path:
    text = str(bbox_output_dir or "").strip()
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    if cv_root.name == "cv_output":
        return cv_root.parent / "bbox_screenshots"
    return cv_root / "bbox_screenshots"


def _build_bbox_filename(screenshot: Path, step: int | None) -> str:
    stem = str(screenshot.stem or "")
    match = re.match(r"^step_(\d+)(?:_(before|after))?$", stem)
    phase = "anchor"
    parsed_step: int | None = None
    if match:
        try:
            parsed_step = int(match.group(1))
        except Exception:
            parsed_step = None
        phase_token = str(match.group(2) or "").strip()
        if phase_token in {"before", "after"}:
            phase = phase_token

    step_num = int(step) if step is not None else parsed_step
    if step_num is not None and step_num >= 0:
        return f"step_{step_num}_{phase}_bbox.jpg"
    return f"{stem}_bbox.jpg"


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
