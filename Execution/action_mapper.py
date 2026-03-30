"""
动作映射模块
将抽象的子目标转化为具体的物理动作（坐标 + 类型）
通过 LLM 解析意图，在控件列表中寻址匹配
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from Perception.context_builder import UIState, WidgetInfo
from Planning.planner import SubGoal
from prompt.action_mapper_prompt import ACTION_MAPPER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """物理动作描述"""

    action_type: str  # tap / swipe / input / back / scroll_up / scroll_down / enter
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    text: str = ""
    widget_id: Optional[int] = None
    target_widget_text: str = ""
    target_resource_id: str = ""
    target_class_name: str = ""
    target_content_desc: str = ""
    target_bounds: tuple = (0, 0, 0, 0)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "x": self.x,
            "y": self.y,
            "x2": self.x2,
            "y2": self.y2,
            "text": self.text,
            "input_text": self.text,
            "widget_id": self.widget_id,
            "target_widget_text": self.target_widget_text,
            "target_resource_id": self.target_resource_id,
            "target_class_name": self.target_class_name,
            "target_content_desc": self.target_content_desc,
            "target_bounds": list(self.target_bounds),
            "description": self.description,
        }


class ActionMapper:
    """
    意图→物理动作映射器
    1. 通过 LLM 解析子目标，得到目标控件描述和动作类型
    2. 在 UIState 控件列表中寻址匹配，获取精确坐标
    3. 输出可执行的 Action
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("ActionMapper 初始化完成")

    def map_action(self, subgoal: SubGoal, ui_state: UIState) -> Action:
        """
        将子目标映射为物理动作
        :param subgoal: 当前子目标
        :param ui_state: 当前 UI 状态
        :return: 可执行的 Action
        """
        logger.info("映射动作: subgoal='%s'", subgoal.description)

        # 简单动作直接映射，无需 LLM
        if subgoal.action_type == "back":
            return Action(action_type="back", description="按返回键")
        if subgoal.action_type == "enter":
            return Action(action_type="enter", description="按回车键")
        if subgoal.action_type in ("swipe", "scroll", "scroll_up", "scroll_down") and (
            self._should_force_default_swipe(subgoal)
            or (subgoal.target_widget_id is None and not subgoal.target_widget_text)
        ):
            return self._create_default_swipe(subgoal, ui_state)

        # 如果子目标已指定控件 ID，直接使用
        if subgoal.target_widget_id is not None:
            widget = ui_state.find_widget_by_id(subgoal.target_widget_id)
            if widget:
                widget = self._refine_widget_for_subgoal(subgoal, ui_state, widget)
                return self._create_action_from_widget(subgoal, widget, ui_state)

        # 尝试通过文本匹配控件
        if subgoal.target_widget_text:
            widget = ui_state.find_widget_by_text(subgoal.target_widget_text)
            if widget:
                widget = self._refine_widget_for_subgoal(subgoal, ui_state, widget)
                logger.info("通过文本匹配到控件: id=%d, text='%s'", widget.widget_id, widget.text)
                return self._create_action_from_widget(subgoal, widget, ui_state)

        # 文本匹配失败，使用 LLM 映射
        return self._map_with_llm(subgoal, ui_state)

    def _create_action_from_widget(self, subgoal: SubGoal, widget: WidgetInfo, ui_state: UIState) -> Action:
        """基于匹配到的控件创建动作"""
        cx, cy = widget.center
        action_type = str(subgoal.action_type or "tap").lower()
        tap_x, tap_y = self._clamp_to_screen(cx, cy, ui_state)

        if action_type == "input":
            return Action(
                action_type="input",
                x=tap_x,
                y=tap_y,
                text=subgoal.input_text,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 中输入 '{subgoal.input_text}'",
            )

        if action_type in ("scroll_up", "swipe_up"):
            start_x, start_y = self._clamp_to_screen(cx, cy + 200, ui_state)
            end_x, end_y = self._clamp_to_screen(cx, cy - 200, ui_state)
            return Action(
                action_type="swipe",
                x=start_x,
                y=start_y,
                x2=end_x,
                y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 上向上滑动",
            )

        if action_type in ("scroll_down", "swipe_down"):
            start_x, start_y = self._clamp_to_screen(cx, cy - 200, ui_state)
            end_x, end_y = self._clamp_to_screen(cx, cy + 200, ui_state)
            return Action(
                action_type="swipe",
                x=start_x,
                y=start_y,
                x2=end_x,
                y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 上向下滑动",
            )

        if action_type == "scroll":
            desc = (subgoal.description or "").lower()
            down_markers = ("scroll down", "swipe down", "pull down", "向下", "下滑", "下拉")
            if any(marker in desc for marker in down_markers):
                start_x, start_y = self._clamp_to_screen(cx, cy - 220, ui_state)
                end_x, end_y = self._clamp_to_screen(cx, cy + 220, ui_state)
                action_desc = f"在 '{widget.text or widget.resource_id}' 上向下滚动"
            else:
                start_x, start_y = self._clamp_to_screen(cx, cy + 220, ui_state)
                end_x, end_y = self._clamp_to_screen(cx, cy - 220, ui_state)
                action_desc = f"在 '{widget.text or widget.resource_id}' 上向上滚动"
            return Action(
                action_type="swipe",
                x=start_x,
                y=start_y,
                x2=end_x,
                y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=action_desc,
            )

        if action_type == "swipe":
            start_x, start_y = self._clamp_to_screen(cx, cy + 260, ui_state)
            end_x, end_y = self._clamp_to_screen(cx, max(80, cy - 460), ui_state)
            return Action(
                action_type="swipe",
                x=start_x,
                y=start_y,
                x2=end_x,
                y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 上执行滑动",
            )

        if action_type in ("long_press", "long-press"):
            return Action(
                action_type="long_press",
                x=tap_x,
                y=tap_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"长按 '{widget.text or widget.resource_id}'",
            )

        return Action(
            action_type="tap",
            x=tap_x,
            y=tap_y,
            widget_id=widget.widget_id,
            target_widget_text=widget.text,
            target_resource_id=widget.resource_id,
            target_class_name=widget.class_name,
            target_content_desc=widget.content_desc,
            target_bounds=widget.bounds,
            description=f"点击 '{widget.text or widget.resource_id}'",
        )

    def _map_with_llm(self, subgoal: SubGoal, ui_state: UIState) -> Action:
        """通过 LLM 映射动作"""
        widget_list = ui_state.to_prompt_text()
        prompt = ACTION_MAPPER_PROMPT.format(
            subgoal_description=subgoal.description,
            widget_list=widget_list,
        )

        try:
            response = self.llm.chat(prompt)
            action = self._parse_response(response, ui_state, subgoal)
            logger.info("LLM 映射结果: %s at (%d, %d)", action.action_type, action.x, action.y)
            return action
        except Exception as e:
            logger.error("LLM 动作映射失败: %s", e)
            return Action(action_type="noop", description=f"映射失败: {e}")

    def _parse_response(self, response: str, ui_state: UIState, subgoal: SubGoal) -> Action:
        """解析 LLM 映射结果"""
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            logger.warning("动作映射 JSON 解析失败")
            return Action(action_type="noop", description="JSON解析失败")

        action_type = str(data.get("action_type", "tap") or "tap").strip().lower()
        if action_type in ("longpress", "long-press", "press_hold", "press-and-hold"):
            action_type = "long_press"
        if action_type in ("swipe_up", "scroll_up"):
            action_type = "scroll_up"
        elif action_type in ("swipe_down", "scroll_down"):
            action_type = "scroll_down"
        allowed_actions = {"tap", "input", "swipe", "scroll", "scroll_up", "scroll_down", "back", "enter", "long_press"}
        if action_type not in allowed_actions:
            action_type = "tap"

        widget_id = data.get("target_widget_id")
        input_text = data.get("input_text", "")

        if action_type == "back":
            return Action(action_type="back", description="按返回键")
        if action_type == "enter":
            return Action(action_type="enter", description="按回车键")

        if action_type in {"swipe", "scroll", "scroll_up", "scroll_down"} and self._should_force_default_swipe(subgoal):
            return self._create_default_swipe(
                SubGoal(
                    description=subgoal.description,
                    action_type=action_type,
                    input_text=input_text,
                ),
                ui_state,
            )

        if widget_id is not None:
            widget = ui_state.find_widget_by_id(widget_id)
            if widget:
                sub = SubGoal(
                    description=subgoal.description,
                    action_type=action_type,
                    input_text=input_text,
                )
                widget = self._refine_widget_for_subgoal(sub, ui_state, widget)
                return self._create_action_from_widget(sub, widget, ui_state)

        if action_type in {"swipe", "scroll", "scroll_up", "scroll_down"}:
            return self._create_default_swipe(
                SubGoal(
                    description=subgoal.description,
                    action_type=action_type,
                    input_text=input_text,
                ),
                ui_state,
            )

        return Action(
            action_type=action_type,
            x=int(ui_state.screen_width // 2),
            y=int(ui_state.screen_height // 2),
            text=input_text,
            target_widget_text=subgoal.target_widget_text,
            description=data.get("reasoning", "LLM映射结果"),
        )

    def _create_default_swipe(self, subgoal: SubGoal, ui_state: UIState) -> Action:
        """无目标控件时的通用滑动映射。"""
        action_type = (subgoal.action_type or "").lower()
        desc = (subgoal.description or "").lower()
        x = int(ui_state.screen_width * 0.5)

        explicit_up_markers = (
            "swipe up",
            "scroll up",
            "pull up",
            "upward",
            "上滑",
            "向上滑",
            "向上",
        )
        explicit_down_markers = (
            "swipe down",
            "scroll down",
            "pull down",
            "downward",
            "from the top",
            "top edge",
            "下滑",
            "下拉",
            "向下滑",
            "向下",
        )

        force_up = action_type in ("scroll_up", "swipe_up") or any(marker in desc for marker in explicit_up_markers)
        force_down = action_type in ("scroll_down", "swipe_down") or any(marker in desc for marker in explicit_down_markers)

        if force_down:
            y_start = int(ui_state.screen_height * 0.2)
            y_end = int(ui_state.screen_height * 0.78)
            action_desc = "默认向下滑动"
        else:
            y_start = int(ui_state.screen_height * 0.82)
            y_end = int(ui_state.screen_height * 0.28)
            action_desc = "默认向上滑动"

        start_x, start_y = self._clamp_to_screen(x, y_start, ui_state)
        end_x, end_y = self._clamp_to_screen(x, y_end, ui_state)
        return Action(
            action_type="swipe",
            x=start_x,
            y=start_y,
            x2=end_x,
            y2=end_y,
            description=action_desc,
        )

    def _should_force_default_swipe(self, subgoal: SubGoal) -> bool:
        desc = (subgoal.description or "").lower()
        markers = (
            "from the bottom",
            "bottom of the screen",
            "from the top",
            "top edge",
            "screen edge",
            "上滑",
            "从底部",
            "下拉",
            "从顶部",
        )
        return any(marker in desc for marker in markers)

    def _refine_widget_for_subgoal(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        widget: WidgetInfo,
    ) -> WidgetInfo:
        """针对输入/聚焦子目标，避免点到标签文本而非实际输入域。"""
        action = (subgoal.action_type or "").lower()
        desc = (subgoal.description or "").lower()

        is_toggle_like = action == "tap" and any(marker in desc for marker in ("switch", "toggle", "开关"))
        if is_toggle_like:
            widget_cls = (getattr(widget, "class_name", "") or "").lower()
            widget_rid = (getattr(widget, "resource_id", "") or "").lower()
            if bool(getattr(widget, "checkable", False)) or "switch" in widget_cls or "switch" in widget_rid:
                return widget

            toggle_candidate = self._find_nearby_toggle_widget(widget, ui_state)
            if toggle_candidate and toggle_candidate.widget_id != widget.widget_id:
                logger.info("开关型子目标重定向控件: %s -> %s", widget.widget_id, toggle_candidate.widget_id)
                return toggle_candidate

        is_focus_like = action in ("input", "tap") and (
            action == "input"
            or "focus" in desc
            or "输入" in desc
            or "聚焦" in desc
        )
        if not is_focus_like:
            return widget

        if getattr(widget, "editable", False) or getattr(widget, "focused", False):
            return widget

        candidate = self._find_nearby_editable_widget(widget, ui_state)
        if candidate and candidate.widget_id != widget.widget_id:
            logger.info("聚焦型子目标重定向控件: %s -> %s", widget.widget_id, candidate.widget_id)
            return candidate
        return widget

    def _find_nearby_toggle_widget(self, anchor: WidgetInfo, ui_state: UIState) -> Optional[WidgetInfo]:
        candidates = [
            w
            for w in ui_state.widgets
            if getattr(w, "enabled", True)
            and (
                bool(getattr(w, "checkable", False))
                or "switch" in (getattr(w, "class_name", "") or "").lower()
                or "switch" in (getattr(w, "resource_id", "") or "").lower()
            )
        ]
        if not candidates:
            return None

        best = None
        best_score = float("-inf")
        for w in candidates:
            y_overlap = self._y_overlap_ratio(anchor.bounds, w.bounds)
            if y_overlap < 0.18:
                continue

            dx = abs(anchor.center[0] - w.center[0])
            dy = abs(anchor.center[1] - w.center[1])
            right_bonus = 120 if w.center[0] >= anchor.center[0] else 0
            checkable_bonus = 160 if bool(getattr(w, "checkable", False)) else 0
            score = y_overlap * 1000 - (dx + dy) + right_bonus + checkable_bonus
            if score > best_score:
                best = w
                best_score = score
        return best

    def _find_nearby_editable_widget(self, anchor: WidgetInfo, ui_state: UIState) -> Optional[WidgetInfo]:
        """查找与锚点同一行附近的可编辑控件。"""
        candidates = [
            w
            for w in ui_state.widgets
            if getattr(w, "enabled", True)
            and (getattr(w, "editable", False) or getattr(w, "focusable", False))
        ]
        if not candidates:
            return None

        best = None
        best_score = float("-inf")
        for w in candidates:
            y_overlap = self._y_overlap_ratio(anchor.bounds, w.bounds)
            if y_overlap < 0.2:
                continue

            dx = abs(anchor.center[0] - w.center[0])
            dy = abs(anchor.center[1] - w.center[1])
            left_penalty = 250 if w.center[0] + 20 < anchor.center[0] else 0
            score = y_overlap * 1000 - (dx + dy + left_penalty)
            if score > best_score:
                best = w
                best_score = score

        return best

    def _y_overlap_ratio(self, box_a: tuple, box_b: tuple) -> float:
        ay1, ay2 = box_a[1], box_a[3]
        by1, by2 = box_b[1], box_b[3]
        overlap = max(0, min(ay2, by2) - max(ay1, by1))
        if overlap <= 0:
            return 0.0
        min_height = max(1, min(ay2 - ay1, by2 - by1))
        return overlap / min_height

    def _clamp_to_screen(self, x: int, y: int, ui_state: UIState) -> tuple:
        width = max(1, int(getattr(ui_state, "screen_width", 1080) or 1080))
        height = max(1, int(getattr(ui_state, "screen_height", 1920) or 1920))
        x_i = max(0, min(width - 1, int(x)))
        y_i = max(0, min(height - 1, int(y)))
        return x_i, y_i
