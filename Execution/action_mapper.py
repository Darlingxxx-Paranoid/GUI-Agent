"""
动作映射模块
将抽象的子目标转化为具体的物理动作（坐标 + 类型）
通过 LLM 解析意图，在控件列表中寻址匹配
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from Perception.context_builder import UIState, WidgetInfo
from Planning.planner import SubGoal
from prompt.action_mapper_prompt import ACTION_MAPPER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """物理动作描述"""
    action_type: str         # tap / swipe / input / back / scroll_up / scroll_down / enter
    x: int = 0               # 点击/滑动起始 x
    y: int = 0               # 点击/滑动起始 y
    x2: int = 0              # 滑动终点 x
    y2: int = 0              # 滑动终点 y
    text: str = ""            # 输入文本
    widget_id: Optional[int] = None  # 对应的控件 ID
    target_widget_text: str = ""
    target_resource_id: str = ""
    target_class_name: str = ""
    target_content_desc: str = ""
    target_bounds: tuple = (0, 0, 0, 0)
    description: str = ""     # 动作描述

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
            logger.info("映射为返回操作")
            return Action(action_type="back", description="按返回键")
        if subgoal.action_type in ("swipe", "scroll", "scroll_up", "scroll_down") and (
            self._should_force_default_swipe(subgoal)
            or (subgoal.target_widget_id is None and not subgoal.target_widget_text)
        ):
            return self._create_default_swipe(subgoal, ui_state)

        timer_keypad_action = self._map_timer_keypad_intent(subgoal, ui_state)
        if timer_keypad_action is not None:
            return timer_keypad_action

        clock_tab_action = self._map_clock_bottom_tab(subgoal, ui_state)
        if clock_tab_action is not None:
            return clock_tab_action

        launcher_search_action = self._map_launcher_search_bar_intent(subgoal, ui_state)
        if launcher_search_action is not None:
            return launcher_search_action

        launcher_app_action = self._map_launcher_open_app_intent(subgoal, ui_state)
        if launcher_app_action is not None:
            return launcher_app_action

        settings_toggle_action = self._map_settings_toggle_intent(subgoal, ui_state)
        if settings_toggle_action is not None:
            return settings_toggle_action

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
        action_type = subgoal.action_type or "tap"
        tap_x, tap_y = self._clamp_to_screen(cx, cy, ui_state)

        if action_type == "input":
            return Action(
                action_type="input",
                x=tap_x, y=tap_y,
                text=subgoal.input_text,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 中输入 '{subgoal.input_text}'",
            )
        elif action_type in ("scroll_up", "swipe_up"):
            start_x, start_y = self._clamp_to_screen(cx, cy + 200, ui_state)
            end_x, end_y = self._clamp_to_screen(cx, cy - 200, ui_state)
            return Action(
                action_type="swipe",
                x=start_x, y=start_y,
                x2=end_x, y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 上向上滑动",
            )
        elif action_type == "scroll":
            desc = (subgoal.description or "").lower()
            down_markers = (
                "scroll down",
                "swipe down",
                "pull down",
                "向下",
                "下滑",
                "下拉",
            )
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
        elif action_type in ("scroll_down", "swipe_down"):
            start_x, start_y = self._clamp_to_screen(cx, cy - 200, ui_state)
            end_x, end_y = self._clamp_to_screen(cx, cy + 200, ui_state)
            return Action(
                action_type="swipe",
                x=start_x, y=start_y,
                x2=end_x, y2=end_y,
                widget_id=widget.widget_id,
                target_widget_text=widget.text,
                target_resource_id=widget.resource_id,
                target_class_name=widget.class_name,
                target_content_desc=widget.content_desc,
                target_bounds=widget.bounds,
                description=f"在 '{widget.text or widget.resource_id}' 上向下滑动",
            )
        elif action_type == "swipe":
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
        elif action_type in ("long_press", "long-press"):
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
        else:
            return Action(
                action_type="tap",
                x=tap_x, y=tap_y,
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
            # 兜底：返回无效动作
            return Action(
                action_type="noop",
                description=f"映射失败: {e}",
            )

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
        widget_id = data.get("target_widget_id")
        input_text = data.get("input_text", "")

        if action_type == "back":
            return Action(action_type="back", description="按返回键")

        desc = (subgoal.description or "").lower()
        if action_type in ("swipe", "scroll", "scroll_up", "swipe_up", "scroll_down", "swipe_down"):
            # “从底部上滑”等手势意图优先走屏幕级默认滑动，避免 LLM 误绑到底部控件导致超界坐标。
            bottom_swipe_markers = (
                "from the bottom",
                "bottom of the screen",
                "bottom of the home screen",
                "swipe up",
                "上滑",
                "从底部",
            )
            if any(marker in desc for marker in bottom_swipe_markers):
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

        # 使用屏幕中心作为兜底
        if action_type in ("swipe", "scroll", "scroll_up", "swipe_up", "scroll_down", "swipe_down"):
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
            "dismiss",
            "close quick settings",
            "close notification shade",
            "return to home",
            "go back to home",
            "上滑",
            "向上滑",
            "向上",
            "收起",
            "关闭通知栏",
            "关闭快捷设置",
        )
        explicit_down_markers = (
            "swipe down",
            "pull down",
            "from the top",
            "top edge",
            "open quick settings",
            "open notification shade",
            "expand quick settings",
            "expand notification shade",
            "下滑",
            "下拉",
            "向下滑",
            "向下",
            "打开通知栏",
            "打开快捷设置",
        )
        context_down_markers = (
            "quick settings",
            "notification shade",
            "通知栏",
            "快捷设置",
        )
        force_up = (
            action_type in ("scroll_up", "swipe_up")
            or any(marker in desc for marker in explicit_up_markers)
        )
        force_down = (
            action_type in ("scroll_down", "swipe_down")
            or any(marker in desc for marker in explicit_down_markers)
        )
        if force_up:
            y_start = int(ui_state.screen_height * 0.82)
            y_end = int(ui_state.screen_height * 0.28)
            action_desc = "默认向上滑动"
        elif force_down or any(marker in desc for marker in context_down_markers):
            # 通知栏/快捷设置下拉必须从顶部边缘起手，否则常常无法触发面板展开。
            y_start = int(ui_state.screen_height * 0.06)
            y_end = int(ui_state.screen_height * 0.68)
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
            "bottom of the home screen",
            "open the app drawer",
            "all apps",
            "swipe down",
            "pull down",
            "from the top",
            "top edge",
            "status bar",
            "notification shade",
            "quick settings",
            "上滑",
            "从底部",
            "应用抽屉",
            "下拉",
            "通知栏",
            "快捷设置",
        )
        return any(marker in desc for marker in markers)

    def _is_launcher_context(self, ui_state: UIState) -> bool:
        pkg = (getattr(ui_state, "package_name", "") or "").lower()
        activity = (getattr(ui_state, "activity_name", "") or "").lower()
        if "launcher" in pkg or "quickstep" in pkg:
            return True
        return "launcher" in activity or "home" in activity

    def _map_launcher_search_bar_intent(self, subgoal: SubGoal, ui_state: UIState) -> Optional[Action]:
        """Launcher/AppDrawer 搜索栏的确定性映射，避免误点到左上角应用图标。"""
        if (subgoal.action_type or "").lower() != "tap":
            return None
        if not self._is_launcher_context(ui_state):
            return None

        desc = (subgoal.description or "").lower()
        if "search" not in desc and "搜索" not in desc:
            return None
        if not any(marker in desc for marker in ("app drawer", "all apps", "search bar at the top", "top search", "顶部搜索", "应用抽屉")):
            return None

        x = int(ui_state.screen_width * 0.5)
        y = int(ui_state.screen_height * 0.08)
        y = max(80, y)
        logger.info("命中 launcher 搜索栏确定性映射: point=(%d,%d)", x, y)
        return Action(
            action_type="tap",
            x=x,
            y=y,
            description="点击 Launcher 顶部搜索栏",
        )

    def _map_launcher_open_app_intent(self, subgoal: SubGoal, ui_state: UIState) -> Optional[Action]:
        """Launcher 场景下打开 App 的确定性映射，优先按应用名匹配文本控件。"""
        if (subgoal.action_type or "").lower() != "tap":
            return None
        if not self._is_launcher_context(ui_state):
            return None

        desc = " ".join(
            [
                subgoal.description or "",
                subgoal.acceptance_criteria or "",
                subgoal.target_widget_text or "",
            ]
        ).lower()
        if not any(marker in desc for marker in ("open", "launch", "app icon", "打开", "启动")):
            return None
        quick_panel_markers = (
            "quick settings",
            "notification shade",
            "internet tile",
            "wi-fi tile",
            "wifi tile",
            "bluetooth tile",
            "control center",
            "status bar",
            "快捷设置",
            "通知栏",
        )
        if any(marker in desc for marker in quick_panel_markers):
            return None

        target_name, target_terms, target_package = self._infer_launcher_target_app(desc)
        if not target_name:
            return None

        best_widget = None
        best_score = float("-inf")
        for w in ui_state.widgets:
            text_blob = " ".join(
                [
                    str(getattr(w, "text", "") or ""),
                    str(getattr(w, "content_desc", "") or ""),
                    str(getattr(w, "resource_id", "") or ""),
                ]
            ).lower()
            if not text_blob:
                continue
            if not any(term in text_blob for term in target_terms):
                continue

            score = 0.0
            if getattr(w, "clickable", False):
                score += 2.0
            if getattr(w, "focusable", False):
                score += 1.0
            if (getattr(w, "text", "") or "").strip().lower() in target_terms:
                score += 2.0
            # 主屏/抽屉 icon 标签通常在中下区域，顶部状态栏文本应降权。
            if int(w.center[1]) > int(ui_state.screen_height * 0.22):
                score += 1.0
            if score > best_score:
                best_score = score
                best_widget = w

        if best_widget is None:
            if target_package:
                logger.info(
                    "launcher 打开应用未命中文字控件，使用包名兜底启动: app=%s package=%s",
                    target_name,
                    target_package,
                )
                return Action(
                    action_type="launch_app",
                    text=target_package,
                    description=f"通过包名启动应用: {target_name}",
                )
            return None

        cx, cy = best_widget.center
        logger.info(
            "命中 launcher 打开应用确定性映射: app=%s, widget_id=%s, center=(%d,%d)",
            target_name,
            getattr(best_widget, "widget_id", None),
            cx,
            cy,
        )
        return Action(
            action_type="tap",
            x=int(cx),
            y=int(cy),
            widget_id=getattr(best_widget, "widget_id", None),
            target_widget_text=getattr(best_widget, "text", "") or "",
            target_resource_id=getattr(best_widget, "resource_id", "") or "",
            target_class_name=getattr(best_widget, "class_name", "") or "",
            target_content_desc=getattr(best_widget, "content_desc", "") or "",
            target_bounds=getattr(best_widget, "bounds", (0, 0, 0, 0)),
            description=f"点击 Launcher 应用图标: {target_name}",
        )

    def _infer_launcher_target_app(self, desc: str) -> tuple:
        if any(
            marker in (desc or "").lower()
            for marker in (
                "quick settings",
                "notification shade",
                "internet tile",
                "wi-fi tile",
                "wifi tile",
                "bluetooth tile",
                "快捷设置",
                "通知栏",
            )
        ):
            return "", (), ""
        app_aliases = [
            ("chrome", ("chrome",), "com.android.chrome"),
            ("settings", ("settings",), "com.android.settings"),
            ("clock", ("clock", "deskclock"), "com.google.android.deskclock"),
            ("camera", ("camera",), "com.google.android.GoogleCamera"),
            ("photos", ("photos",), "com.google.android.apps.photos"),
            ("gmail", ("gmail",), "com.google.android.gm"),
            ("calendar", ("calendar",), "com.google.android.calendar"),
            ("contacts", ("contacts",), "com.google.android.contacts"),
            ("youtube", ("youtube",), "com.google.android.youtube"),
        ]
        for name, terms, pkg in app_aliases:
            if any(re.search(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)", desc) for term in terms):
                return name, terms, pkg
        return "", (), ""

    def _map_settings_toggle_intent(self, subgoal: SubGoal, ui_state: UIState) -> Optional[Action]:
        """Settings 页面开关类子目标的确定性映射，优先命中真实 Switch 控件。"""
        if (subgoal.action_type or "").lower() != "tap":
            return None

        desc = " ".join(
            [
                subgoal.description or "",
                subgoal.acceptance_criteria or "",
                subgoal.target_widget_text or "",
            ]
        ).lower()
        if not any(token in desc for token in ("switch", "toggle", "开关")):
            return None

        package_name = (getattr(ui_state, "package_name", "") or "").lower()
        activity_name = (getattr(ui_state, "activity_name", "") or "").lower()
        if "settings" not in package_name and "settings" not in activity_name:
            return None

        target_terms = []
        if any(token in desc for token in ("wifi", "wi-fi", "wi fi", "wlan", "无线网", "网络")):
            target_terms = ["wifi", "wi-fi", "wi fi", "wlan", "internet"]
        elif any(token in desc for token in ("bluetooth", "蓝牙")):
            target_terms = ["bluetooth", "蓝牙"]

        best_widget = None
        best_score = float("-inf")
        for w in ui_state.widgets:
            cls = (getattr(w, "class_name", "") or "").lower()
            rid = (getattr(w, "resource_id", "") or "").lower()
            text_blob = " ".join(
                [
                    str(getattr(w, "text", "") or ""),
                    str(getattr(w, "content_desc", "") or ""),
                    str(getattr(w, "resource_id", "") or ""),
                ]
            ).lower()

            switch_like = (
                bool(getattr(w, "checkable", False))
                or "switch" in cls
                or "switch" in rid
                or "toggle" in rid
            )
            if not switch_like or not getattr(w, "enabled", True):
                continue

            score = 0.0
            if bool(getattr(w, "checkable", False)):
                score += 3.0
            if "switch" in cls:
                score += 2.0
            if "switch" in rid or "toggle" in rid:
                score += 2.0
            if int(getattr(w, "center", (0, 0))[0]) >= int(ui_state.screen_width * 0.62):
                score += 1.2
            if getattr(w, "clickable", False) or getattr(w, "focusable", False):
                score += 0.8
            if target_terms and any(term in text_blob for term in target_terms):
                score += 2.0
            if target_terms and self._has_nearby_terms(ui_state, w, target_terms):
                score += 2.5

            if score > best_score:
                best_score = score
                best_widget = w

        if best_widget is None:
            return None

        cx, cy = best_widget.center
        logger.info(
            "命中 Settings 开关确定性映射: widget_id=%s, center=(%d,%d), score=%.2f",
            getattr(best_widget, "widget_id", None),
            cx,
            cy,
            best_score,
        )
        return Action(
            action_type="tap",
            x=int(cx),
            y=int(cy),
            widget_id=getattr(best_widget, "widget_id", None),
            target_widget_text=getattr(best_widget, "text", "") or "",
            target_resource_id=getattr(best_widget, "resource_id", "") or "",
            target_class_name=getattr(best_widget, "class_name", "") or "",
            target_content_desc=getattr(best_widget, "content_desc", "") or "",
            target_bounds=getattr(best_widget, "bounds", (0, 0, 0, 0)),
            description="点击 Settings 开关控件",
        )

    def _has_nearby_terms(self, ui_state: UIState, anchor_widget: WidgetInfo, terms: tuple) -> bool:
        if not terms:
            return False
        ax1, ay1, ax2, ay2 = getattr(anchor_widget, "bounds", (0, 0, 0, 0))
        for w in ui_state.widgets:
            if getattr(w, "widget_id", None) == getattr(anchor_widget, "widget_id", None):
                continue
            text_blob = " ".join(
                [
                    str(getattr(w, "text", "") or ""),
                    str(getattr(w, "content_desc", "") or ""),
                    str(getattr(w, "resource_id", "") or ""),
                ]
            ).lower()
            if not text_blob:
                continue
            if not any(term in text_blob for term in terms):
                continue

            wx1, wy1, wx2, wy2 = getattr(w, "bounds", (0, 0, 0, 0))
            vertical_close = not (wy2 < ay1 - 140 or wy1 > ay2 + 140)
            horizontal_close = wx1 <= ax2 + 120
            if vertical_close and horizontal_close:
                return True
        return False

    def _map_clock_bottom_tab(self, subgoal: SubGoal, ui_state: UIState) -> Optional[Action]:
        """Clock 底部导航 tab 的确定性映射，规避 CV-only 场景下 LLM 误选。"""
        if (subgoal.action_type or "").lower() != "tap":
            return None

        desc = (subgoal.description or "").lower()
        if "tab" not in desc and "导航" not in desc:
            return None

        package_name = (getattr(ui_state, "package_name", "") or "").lower()
        if "deskclock" not in package_name and "clock" not in desc:
            return None

        tab_aliases = [
            ("alarm", ("alarm", "闹钟")),
            ("clock", ("clock", "时钟")),
            ("timer", ("timer", "计时器")),
            ("stopwatch", ("stopwatch", "秒表")),
            ("bedtime", ("bedtime", "就寝")),
        ]

        target_tab = ""
        target_terms = ()
        for name, aliases in tab_aliases:
            if any(term in desc for term in aliases):
                target_tab = name
                target_terms = aliases
                break
        if not target_tab:
            return None

        bottom_threshold = int(ui_state.screen_height * 0.72)
        best_widget = None
        best_score = float("-inf")
        for w in ui_state.widgets:
            text_blob = " ".join(
                [
                    str(getattr(w, "text", "") or ""),
                    str(getattr(w, "content_desc", "") or ""),
                    str(getattr(w, "resource_id", "") or ""),
                ]
            ).lower()
            if not text_blob:
                continue
            if not any(term in text_blob for term in target_terms):
                continue

            score = 0.0
            if w.center[1] >= bottom_threshold:
                score += 5.0
            if getattr(w, "clickable", False):
                score += 2.0
            if getattr(w, "focusable", False):
                score += 1.0
            if (getattr(w, "text", "") or "").strip().lower() in target_terms:
                score += 2.0
            if score > best_score:
                best_score = score
                best_widget = w

        if best_widget is not None:
            cx, cy = best_widget.center
            logger.info(
                "命中 Clock tab 确定性映射: tab=%s, widget_id=%s, center=(%d,%d)",
                target_tab,
                getattr(best_widget, "widget_id", None),
                cx,
                cy,
            )
            return Action(
                action_type="tap",
                x=int(cx),
                y=int(cy),
                widget_id=getattr(best_widget, "widget_id", None),
                target_widget_text=getattr(best_widget, "text", "") or "",
                target_resource_id=getattr(best_widget, "resource_id", "") or "",
                target_class_name=getattr(best_widget, "class_name", "") or "",
                target_content_desc=getattr(best_widget, "content_desc", "") or "",
                target_bounds=getattr(best_widget, "bounds", (0, 0, 0, 0)),
                description=f"点击 Clock 底部标签: {target_tab}",
            )

        slot_ratio = {
            "alarm": 0.10,
            "clock": 0.30,
            "timer": 0.50,
            "stopwatch": 0.70,
            "bedtime": 0.90,
        }.get(target_tab, 0.50)
        x = int(ui_state.screen_width * slot_ratio)
        y = int(ui_state.screen_height * 0.905)
        logger.info(
            "使用 Clock tab 坐标兜底: tab=%s, point=(%d,%d)",
            target_tab,
            x,
            y,
        )
        return Action(
            action_type="tap",
            x=x,
            y=y,
            description=f"点击 Clock 底部标签(兜底): {target_tab}",
        )

    def _map_timer_keypad_intent(self, subgoal: SubGoal, ui_state: UIState) -> Optional[Action]:
        """Timer 数字键盘动作的确定性映射，避免 LLM 把数字映射到错误行。"""
        if (subgoal.action_type or "").lower() != "tap":
            return None

        desc = (subgoal.description or "").lower()
        package_name = (getattr(ui_state, "package_name", "") or "").lower()
        if "deskclock" not in package_name and "timer" not in desc and "计时器" not in desc:
            return None

        if not any(
            token in desc
            for token in ("timer", "计时器", "seconds", "秒")
        ):
            return None

        digit = self._extract_keypad_digit(desc)
        special = self._extract_keypad_special(desc)
        if digit is None and special is None:
            return None

        key_map = {
            "1": (0.26, 0.315),
            "2": (0.50, 0.315),
            "3": (0.73, 0.315),
            "4": (0.26, 0.429),
            "5": (0.50, 0.429),
            "6": (0.73, 0.429),
            "7": (0.26, 0.539),
            "8": (0.50, 0.539),
            "9": (0.73, 0.539),
            "0": (0.50, 0.656),
            "00": (0.26, 0.656),
            "backspace": (0.73, 0.656),
        }

        key = digit if digit is not None else special
        if key not in key_map:
            return None
        long_press_clear = bool(
            key == "backspace"
            and any(
                token in desc
                for token in ("clear", "reset", "清空", "重置", "clear all", "full reset")
            )
        )

        ratio_x, ratio_y = key_map[key]
        expected_x = int(ui_state.screen_width * ratio_x)
        expected_y = int(ui_state.screen_height * ratio_y)
        widget = self._find_nearest_keypad_widget(expected_x, expected_y, ui_state)
        if widget is not None:
            x, y = widget.center
            logger.info(
                "命中 Timer 键盘确定性映射: key=%s, widget_id=%s, center=(%d,%d), action=%s",
                key,
                getattr(widget, "widget_id", None),
                x,
                y,
                "long_press" if long_press_clear else "tap",
            )
            return Action(
                action_type="long_press" if long_press_clear else "tap",
                x=int(x),
                y=int(y),
                widget_id=getattr(widget, "widget_id", None),
                target_widget_text=getattr(widget, "text", "") or "",
                target_resource_id=getattr(widget, "resource_id", "") or "",
                target_class_name=getattr(widget, "class_name", "") or "",
                target_content_desc=getattr(widget, "content_desc", "") or "",
                target_bounds=getattr(widget, "bounds", (0, 0, 0, 0)),
                description=f"{'长按' if long_press_clear else '点击'} Timer 键盘按键: {key}",
            )

        logger.info(
            "使用 Timer 键盘坐标兜底: key=%s, point=(%d,%d), action=%s",
            key,
            expected_x,
            expected_y,
            "long_press" if long_press_clear else "tap",
        )
        return Action(
            action_type="long_press" if long_press_clear else "tap",
            x=expected_x,
            y=expected_y,
            description=f"{'长按' if long_press_clear else '点击'} Timer 键盘按键(兜底): {key}",
        )

    def _extract_keypad_digit(self, desc: str) -> Optional[str]:
        if not desc:
            return None
        if not any(
            token in desc
            for token in ("keypad", "digit", "number", "num", "key", "数字", "按键", "键盘")
        ):
            return None

        patterns = [
            r"(?:number|digit|num(?:ber)?|key|数字|按键)\s*['\"]?([0-9])['\"]?",
            r"tap\s+['\"]?([0-9])['\"]?",
            r"点击\s*['\"]?([0-9])['\"]?",
        ]
        for pattern in patterns:
            match = re.search(pattern, desc)
            if match:
                return match.group(1)
        return None

    def _extract_keypad_special(self, desc: str) -> Optional[str]:
        if not desc:
            return None
        if any(token in desc for token in ("double zero", "00", "双零", "两个0")):
            return "00"
        if any(token in desc for token in ("backspace", "delete", "clear", "reset", "删除", "清空", "重置")):
            return "backspace"
        return None

    def _find_nearest_keypad_widget(
        self,
        expected_x: int,
        expected_y: int,
        ui_state: UIState,
    ) -> Optional[WidgetInfo]:
        candidates = []
        y_min = int(ui_state.screen_height * 0.24)
        y_max = int(ui_state.screen_height * 0.75)
        for w in ui_state.widgets:
            if not (y_min <= w.center[1] <= y_max):
                continue
            width = max(1, w.bounds[2] - w.bounds[0])
            height = max(1, w.bounds[3] - w.bounds[1])
            area = width * height
            if area < 300 or area > 90000:
                continue
            candidates.append(w)

        if not candidates:
            return None

        best = None
        best_dist = float("inf")
        for w in candidates:
            dx = abs(int(w.center[0]) - expected_x)
            dy = abs(int(w.center[1]) - expected_y)
            dist = dx + dy
            if dist < best_dist:
                best = w
                best_dist = dist

        if best is None:
            return None
        max_dist = int(min(ui_state.screen_width, ui_state.screen_height) * 0.22)
        if best_dist > max_dist:
            return None
        return best

    def _refine_widget_for_subgoal(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        widget: WidgetInfo,
    ) -> WidgetInfo:
        """针对输入/聚焦子目标，避免点到标签文本而非实际输入域。"""
        action = (subgoal.action_type or "").lower()
        desc = (subgoal.description or "").lower()
        is_toggle_like = action == "tap" and any(
            marker in desc for marker in ("switch", "toggle", "开关")
        )
        if is_toggle_like:
            widget_cls = (getattr(widget, "class_name", "") or "").lower()
            widget_rid = (getattr(widget, "resource_id", "") or "").lower()
            if bool(getattr(widget, "checkable", False)) or "switch" in widget_cls or "switch" in widget_rid:
                return widget

            toggle_candidate = self._find_nearby_toggle_widget(widget, ui_state)
            if toggle_candidate and toggle_candidate.widget_id != widget.widget_id:
                logger.info(
                    "开关型子目标重定向控件: %s -> %s",
                    widget.widget_id,
                    toggle_candidate.widget_id,
                )
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
            logger.info(
                "聚焦型子目标重定向控件: %s -> %s",
                widget.widget_id,
                candidate.widget_id,
            )
            return candidate
        return widget

    def _find_nearby_toggle_widget(
        self,
        anchor: WidgetInfo,
        ui_state: UIState,
    ) -> Optional[WidgetInfo]:
        candidates = [
            w for w in ui_state.widgets
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

    def _find_nearby_editable_widget(
        self,
        anchor: WidgetInfo,
        ui_state: UIState,
    ) -> Optional[WidgetInfo]:
        """查找与锚点同一行附近的可编辑控件。"""
        candidates = [
            w for w in ui_state.widgets
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
