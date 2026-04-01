"""Map planning intent to executable ResolvedAction."""

from __future__ import annotations

import logging
import re
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


class MappingFailure(ValueError):
    """Raised when mapper cannot provide a safe executable action."""

    def __init__(self, reason_code: str, message: str):
        self.reason_code = str(reason_code or "mapper_failure")
        self.message = str(message or "")
        super().__init__(f"{self.reason_code}:{self.message}")


class ActionMapper:
    """Resolve target and build concrete action parameters."""

    _APP_ALIAS_MAP = {
        "gmail": "com.google.android.gm",
        "chrome": "com.android.chrome",
        "settings": "com.android.settings",
        "clock": "com.google.android.deskclock",
        "youtube": "com.google.android.youtube",
        "maps": "com.google.android.apps.maps",
        "photos": "com.google.android.apps.photos",
        "play store": "com.android.vending",
        "camera": "com.android.camera",
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client
        logger.info("ActionMapper 初始化完成 (V4)")

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

        original_target = contract.target if contract and contract.target is not None else plan.target
        target = self._resolve_target(target=original_target, ui_state=ui_state)

        if action_type == "launch_app":
            return self._map_launch_app(
                plan=plan,
                ui_state=ui_state,
                contract=contract,
                original_target=original_target,
                resolved_target=target,
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

    def _map_launch_app(
        self,
        plan: PlanResult,
        ui_state: UIState,
        contract: Optional[StepContract],
        original_target: TargetRef | None,
        resolved_target: TargetRef | None,
    ) -> ResolvedAction:
        package, activity = self._resolve_launch_target(
            plan=plan,
            contract=contract,
            target=original_target,
        )

        if package:
            params = {"package": package}
            if activity:
                params["activity"] = activity
            return ResolvedAction(
                type="launch_app",
                params=params,
                target=resolved_target,
                description=f"启动应用 {package}",
            )

        # Safe downgrade rule 1: resolved target center tap.
        if resolved_target and resolved_target.resolved and resolved_target.resolved.center:
            x, y = self._clamp(resolved_target.resolved.center[0], resolved_target.resolved.center[1], ui_state)
            return ResolvedAction(
                type="tap",
                params={
                    "x": x,
                    "y": y,
                    "degraded_from_launch_app": True,
                    "degrade_reason": "launch_app_missing_package_resolved_target",
                },
                target=resolved_target,
                description=f"launch_app 解析失败，安全降级为点击({x},{y})",
            )

        # Safe downgrade rule 2: unique selector remap target tap.
        unique_widget = self._resolve_unique_widget_from_selectors(target=original_target, ui_state=ui_state)
        if unique_widget is not None:
            downgrade_target = self._target_from_widget(widget=unique_widget)
            x, y = self._clamp(unique_widget.center[0], unique_widget.center[1], ui_state)
            return ResolvedAction(
                type="tap",
                params={
                    "x": x,
                    "y": y,
                    "degraded_from_launch_app": True,
                    "degrade_reason": "launch_app_missing_package_unique_selector",
                },
                target=downgrade_target,
                description=f"launch_app 解析失败，按唯一 selector 降级点击({x},{y})",
            )

        raise MappingFailure(
            reason_code="launch_app_unresolved_package_no_safe_target",
            message="launch_app 目标包解析失败且不存在安全可点击目标",
        )

    def _resolve_launch_target(
        self,
        plan: PlanResult,
        contract: Optional[StepContract],
        target: TargetRef | None,
    ) -> tuple[str, str]:
        activity = ""
        package_candidates: list[str] = []

        plan_hints = dict(plan.planning_hints or {})
        contract_hints = dict((contract.planning_hints if contract else {}) or {})

        for hints in (plan_hints, contract_hints):
            pkg = str(hints.get("target_package") or "").strip().lower()
            if self._is_valid_package(pkg):
                package_candidates.append(pkg)
            act = str(hints.get("target_activity") or "").strip()
            if act and not activity:
                activity = act

        text_blob = " ".join(
            [
                str(plan.goal.summary or ""),
                str(plan.goal.success_definition or ""),
                str(plan.reasoning or ""),
                str(contract_hints.get("task_hint") or ""),
            ]
        ).lower()

        for match in re.findall(r"package:([a-z0-9_.]+)", text_blob):
            pkg = str(match or "").strip().lower()
            if self._is_valid_package(pkg):
                package_candidates.append(pkg)

        if target is not None:
            for selector in target.selectors:
                if str(selector.kind or "").strip().lower() != "resource_id":
                    continue
                value = str(selector.value or "").strip().lower()
                if ":id/" in value:
                    pkg = value.split(":id/", 1)[0]
                    if self._is_valid_package(pkg):
                        package_candidates.append(pkg)

        for alias, pkg in self._APP_ALIAS_MAP.items():
            if alias in text_blob:
                package_candidates.append(pkg)

        deduped: list[str] = []
        for pkg in package_candidates:
            if pkg and pkg not in deduped:
                deduped.append(pkg)

        return (deduped[0] if deduped else ""), activity

    def _resolve_unique_widget_from_selectors(self, target: TargetRef | None, ui_state: UIState) -> Optional[WidgetInfo]:
        if target is None:
            return None

        matched: dict[int, WidgetInfo] = {}
        for selector in (target.selectors or []):
            for widget in self._match_selector_all(selector=selector, ui_state=ui_state):
                matched[int(widget.widget_id)] = widget

        if len(matched) == 1:
            return list(matched.values())[0]
        return None

    def _target_from_widget(self, widget: WidgetInfo) -> TargetRef:
        return TargetRef(
            ref_id=normalize_subject_ref(None, fallback_widget_id=int(widget.widget_id)),
            role="primary",
            selectors=[Selector(kind="widget_id", operator="equals", value=int(widget.widget_id))],
            resolved=TargetResolved(
                widget_id=int(widget.widget_id),
                bounds=tuple(int(v) for v in widget.bounds),
                center=tuple(int(v) for v in widget.center),
                snapshot={
                    "text": widget.text,
                    "resource_id": widget.resource_id,
                    "content_desc": widget.content_desc,
                    "class_name": widget.class_name,
                    "clickable": bool(widget.clickable),
                    "enabled": bool(widget.enabled),
                },
            ),
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

    def _match_selector_all(self, selector: Selector, ui_state: UIState) -> list[WidgetInfo]:
        kind = str(selector.kind or "").strip().lower()
        value = selector.value

        if kind == "widget_id":
            try:
                widget = ui_state.find_widget_by_id(int(value))
            except Exception:
                widget = None
            return [widget] if widget is not None else []

        if kind == "text":
            token = str(value or "").strip().lower()
            if not token:
                return []
            return [w for w in ui_state.widgets if token in str(w.text or "").lower() or token in str(w.content_desc or "").lower()]

        if kind == "resource_id":
            token = str(value or "").strip().lower()
            if not token:
                return []
            return [w for w in ui_state.widgets if token in str(w.resource_id or "").lower()]

        if kind == "content_desc":
            token = str(value or "").strip().lower()
            if not token:
                return []
            return [w for w in ui_state.widgets if token in str(w.content_desc or "").lower()]

        if kind == "class_name":
            token = str(value or "").strip().lower()
            if not token:
                return []
            return [w for w in ui_state.widgets if str(w.class_name or "").lower() == token]

        if kind == "bounds" and isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                bounds = tuple(int(v) for v in value)
            except Exception:
                return []
            out: list[WidgetInfo] = []
            for widget in ui_state.widgets:
                iou = calc_iou(bounds, widget.bounds)
                if iou > 0.1:
                    out.append(widget)
            return out

        if kind == "point" and isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                x = int(value[0])
                y = int(value[1])
            except Exception:
                return []
            out: list[WidgetInfo] = []
            for widget in ui_state.widgets:
                x1, y1, x2, y2 = widget.bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    out.append(widget)
            return out

        return []

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

    def _is_valid_package(self, value: str) -> bool:
        token = str(value or "").strip().lower()
        if not token:
            return False
        if token.count(".") < 1:
            return False
        return bool(re.match(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$", token))
