"""Map planning intent to executable ResolvedAction."""

from __future__ import annotations

import logging
from typing import Optional

from Oracle.contracts import (
    ResolvedAction,
    Selector,
    StepContract,
    TargetRef,
    TargetResolved,
    normalize_subject_ref,
    validate_action_type,
)
from Perception.context_builder import UIState, WidgetInfo
from Planning.planner import PlanResult
from utils.utils import calc_iou

logger = logging.getLogger(__name__)


class ActionMapper:
    """Resolve target and build concrete action parameters."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        logger.info("ActionMapper 初始化完成 (V3.1)")

    def map_action(
        self,
        plan: PlanResult,
        ui_state: UIState,
        contract: Optional[StepContract] = None,
    ) -> ResolvedAction:
        requested = str(plan.requested_action_type or "tap").strip().lower()
        try:
            action_type = validate_action_type(requested)
        except Exception:
            action_type = "tap"
        if action_type == "launch_app":
            action_type = "tap"

        target = self._resolve_target(
            target=(contract.target if contract and contract.target is not None else plan.target),
            ui_state=ui_state,
        )

        if action_type == "back":
            return ResolvedAction(type="back", params={}, target=target, description="按返回键")

        if action_type == "enter":
            return ResolvedAction(type="enter", params={}, target=target, description="按回车键")

        if action_type == "swipe":
            params, desc = self._build_swipe_params(plan=plan, target=target, ui_state=ui_state)
            return ResolvedAction(type="swipe", params=params, target=target, description=desc)

        if action_type in {"tap", "input", "long_press"}:
            x, y = self._pick_point(target=target, ui_state=ui_state)
            params = {"x": x, "y": y}
            if action_type == "input":
                params["text"] = str(plan.input_text or "")
                desc = f"在({x},{y})输入文本"
            elif action_type == "long_press":
                params["duration_ms"] = 900
                desc = f"长按({x},{y})"
            else:
                desc = f"点击({x},{y})"
            return ResolvedAction(type=action_type, params=params, target=target, description=desc)

        # fallback
        x, y = self._pick_point(target=target, ui_state=ui_state)
        return ResolvedAction(
            type="tap",
            params={"x": x, "y": y},
            target=target,
            description=f"未知动作回退为点击({x},{y})",
        )

    def _resolve_target(self, target: TargetRef | None, ui_state: UIState) -> TargetRef | None:
        if target is None:
            return None

        resolved_widget = None

        selectors = list(target.selectors or [])
        for selector in selectors:
            resolved_widget = self._match_selector(selector, ui_state)
            if resolved_widget is not None:
                break

        if resolved_widget is None:
            # final fallback: reuse resolved widget id if still valid
            if target.resolved and target.resolved.widget_id is not None:
                resolved_widget = ui_state.find_widget_by_id(int(target.resolved.widget_id))

        ref_id = normalize_subject_ref(
            target.ref_id,
            fallback_widget_id=(resolved_widget.widget_id if resolved_widget else None),
        )

        resolved = None
        if resolved_widget is not None:
            resolved = TargetResolved(
                widget_id=int(resolved_widget.widget_id),
                bounds=tuple(int(v) for v in resolved_widget.bounds),
                center=tuple(int(v) for v in resolved_widget.center),
                snapshot={
                    "text": resolved_widget.text,
                    "resource_id": resolved_widget.resource_id,
                    "content_desc": resolved_widget.content_desc,
                    "class_name": resolved_widget.class_name,
                    "clickable": bool(resolved_widget.clickable),
                    "enabled": bool(resolved_widget.enabled),
                },
            )

        return TargetRef(
            ref_id=ref_id,
            role=target.role or "primary",
            selectors=selectors,
            resolved=resolved,
        )

    def _match_selector(self, selector: Selector, ui_state: UIState) -> Optional[WidgetInfo]:
        kind = str(selector.kind or "").strip().lower()
        operator = str(selector.operator or "").strip().lower()
        value = selector.value

        if kind == "widget_id":
            try:
                return ui_state.find_widget_by_id(int(value))
            except Exception:
                return None

        if kind == "text":
            token = str(value or "").strip()
            if not token:
                return None
            return ui_state.find_widget_by_text(token)

        if kind == "resource_id":
            token = str(value or "").strip().lower()
            if not token:
                return None
            return self._best_match(
                ui_state.widgets,
                lambda w: token in str(w.resource_id or "").lower(),
            )

        if kind == "content_desc":
            token = str(value or "").strip().lower()
            if not token:
                return None
            return self._best_match(
                ui_state.widgets,
                lambda w: token in str(w.content_desc or "").lower(),
            )

        if kind == "class_name":
            token = str(value or "").strip().lower()
            if not token:
                return None
            return self._best_match(
                ui_state.widgets,
                lambda w: str(w.class_name or "").lower() == token,
            )

        if kind == "bounds" and isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                bounds = tuple(int(v) for v in value)
            except Exception:
                return None
            best = None
            best_iou = 0.0
            for widget in ui_state.widgets:
                iou = calc_iou(bounds, widget.bounds)
                if iou > best_iou:
                    best = widget
                    best_iou = iou
            if best is not None and (best_iou > 0.1 or operator in {"near", "overlap"}):
                return best
            return None

        if kind == "point" and isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                x = int(value[0])
                y = int(value[1])
            except Exception:
                return None
            for widget in ui_state.widgets:
                x1, y1, x2, y2 = widget.bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return widget
            # point near fallback
            return self._best_match(
                ui_state.widgets,
                lambda w: abs(w.center[0] - x) + abs(w.center[1] - y) <= 120,
            )

        return None

    def _best_match(self, widgets: list[WidgetInfo], cond) -> Optional[WidgetInfo]:
        scored: list[tuple[int, WidgetInfo]] = []
        for widget in widgets:
            try:
                ok = bool(cond(widget))
            except Exception:
                ok = False
            if not ok:
                continue
            score = 0
            if widget.enabled:
                score += 1
            if widget.clickable:
                score += 2
            if widget.focusable:
                score += 1
            if widget.editable:
                score += 1
            scored.append((score, widget))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _pick_point(self, target: TargetRef | None, ui_state: UIState) -> tuple[int, int]:
        if target and target.resolved and target.resolved.center:
            x, y = target.resolved.center
            return self._clamp(x, y, ui_state)

        # fallback to middle area to reduce top status bar tapping.
        x = int((ui_state.screen_width or 1080) * 0.5)
        y = int((ui_state.screen_height or 1920) * 0.58)
        return self._clamp(x, y, ui_state)

    def _build_swipe_params(self, plan: PlanResult, target: TargetRef | None, ui_state: UIState) -> tuple[dict, str]:
        base_x, base_y = self._pick_point(target=target, ui_state=ui_state)
        width = int(ui_state.screen_width or 1080)
        height = int(ui_state.screen_height or 1920)

        desc = str(plan.goal.summary or "").lower()
        hints = str((plan.planning_hints or {}).get("direction") or "").lower()
        down_markers = ("down", "下", "pull", "refresh")
        is_down = any(marker in desc for marker in down_markers) or any(
            marker in hints for marker in down_markers
        )

        delta = max(180, int(height * 0.18))
        if is_down:
            start_x, start_y = self._clamp(base_x, base_y - delta, ui_state)
            end_x, end_y = self._clamp(base_x, base_y + delta, ui_state)
            action_desc = "向下滑动"
        else:
            start_x, start_y = self._clamp(base_x, base_y + delta, ui_state)
            end_x, end_y = self._clamp(base_x, base_y - delta, ui_state)
            action_desc = "向上滑动"

        params = {
            "x": start_x,
            "y": start_y,
            "x2": end_x,
            "y2": end_y,
            "duration_ms": 420,
            "screen_width": width,
            "screen_height": height,
        }
        return params, action_desc

    def _clamp(self, x: int, y: int, ui_state: UIState) -> tuple[int, int]:
        width = max(1, int(ui_state.screen_width or 1080))
        height = max(1, int(ui_state.screen_height or 1920))
        safe_x = min(max(1, int(x)), width - 1)
        safe_y = min(max(1, int(y)), height - 1)
        return safe_x, safe_y
