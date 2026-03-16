"""
上下文构建模块
将 CV 检测结果与 Dump 结构树融合，并进行上下文瘦身
输出精简的 UIState 供后续模块使用
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from Perception.dump_parser import UIElement
from utils.utils import calc_iou

logger = logging.getLogger(__name__)

# ========================
# 数据类定义
# ========================


@dataclass
class WidgetInfo:
    """
    融合后的控件信息
    同时包含 CV 检测的视觉属性和 Dump 的语义属性
    """
    widget_id: int
    bounds: tuple                    # (x1, y1, x2, y2)
    center: tuple                    # (cx, cy)
    category: str = ""               # CV 检测类别 (Button, EditText, etc.)
    text: str = ""                   # OCR 或 Dump 提取的文字
    resource_id: str = ""            # Dump resource-id
    class_name: str = ""             # Dump class name
    content_desc: str = ""           # Dump content-desc
    clickable: bool = False
    scrollable: bool = False
    cv_confidence: float = 0.0       # CV 检测置信度

    def to_dict(self) -> dict:
        return {
            "id": self.widget_id,
            "bounds": list(self.bounds),
            "center": list(self.center),
            "category": self.category,
            "text": self.text,
            "resource_id": self.resource_id,
            "class": self.class_name,
            "content_desc": self.content_desc,
            "clickable": self.clickable,
            "scrollable": self.scrollable,
        }

    def get_description(self) -> str:
        """生成控件的文本摘要，用于 LLM prompt"""
        parts = [f"[{self.widget_id}]"]
        if self.category:
            parts.append(self.category)
        if self.text:
            parts.append(f'text="{self.text}"')
        elif self.content_desc:
            parts.append(f'desc="{self.content_desc}"')
        elif self.resource_id:
            parts.append(f"id={self.resource_id.split('/')[-1]}")
        if self.clickable:
            parts.append("clickable")
        if self.scrollable:
            parts.append("scrollable")
        parts.append(f"position={self.center}")
        return " ".join(parts)


@dataclass
class UIState:
    """
    当前 UI 的完整状态快照
    作为所有模块间传递的核心数据结构
    """
    widgets: List[WidgetInfo] = field(default_factory=list)
    activity_name: str = ""                  # 当前 Activity
    package_name: str = ""                   # 当前包名
    screen_width: int = 0
    screen_height: int = 0
    screenshot_path: str = ""                # 对应截图路径
    raw_cv_elements: List[Dict] = field(default_factory=list)  # CV 原始检测结果

    def to_prompt_text(self) -> str:
        """生成供 LLM 使用的状态描述文本"""
        lines = [
            f"Current Activity: {self.activity_name or 'unknown'}",
            f"Package: {self.package_name or 'unknown'}",
            f"Screen: {self.screen_width}x{self.screen_height}",
            f"Widgets Count: {len(self.widgets)}",
            "--- Widgets List ---",
        ]
        for w in self.widgets:
            lines.append(w.get_description())
        return "\n".join(lines)

    def find_widget_by_text(self, text: str) -> Optional[WidgetInfo]:
        """根据文本内容查找控件"""
        for w in self.widgets:
            if text in w.text or text in w.content_desc:
                return w
        return None

    def find_widget_by_id(self, widget_id: int) -> Optional[WidgetInfo]:
        """根据 widget_id 查找控件"""
        for w in self.widgets:
            if w.widget_id == widget_id:
                return w
        return None


class ContextBuilder:
    """
    上下文构建器
    1. 将 CV 元素与 Dump 控件通过空间 IoU 进行匹配融合
    2. 进行上下文瘦身：过滤不可见/面积过小/冗余节点
    3. 输出精简的 UIState
    """

    # IoU 匹配阈值
    IOU_THRESHOLD = 0.3
    # 最小控件面积阈值（像素^2）
    MIN_AREA = 100
    # 最大控件占屏比例
    MAX_SCREEN_RATIO = 0.9

    def __init__(self):
        logger.debug("ContextBuilder 初始化完成")

    def build(
        self,
        cv_elements: List[Dict[str, Any]],
        dump_elements: List[UIElement],
        screenshot_path: str = "",
        screen_size: tuple = (1080, 1920),
        resize_ratio: float = 1.0,
        activity_name: str = "",
        package_name: str = "",
    ) -> UIState:
        """
        融合 CV 检测结果和 Dump 控件，构建 UIState
        :param cv_elements: CV 检测得到的元素列表 (来自 merge.py 输出)
        :param dump_elements: Dump 解析得到的 UIElement 列表
        :param screenshot_path: 截图文件路径
        :param screen_size: (width, height)
        :param resize_ratio: CV 检测的缩放比例
        :param activity_name: 当前 Activity 名
        :param package_name: 当前应用包名
        :return: 构建的 UIState
        """
        logger.info(
            "开始构建 UIState: CV元素=%d, Dump控件=%d",
            len(cv_elements), len(dump_elements),
        )

        screen_w, screen_h = screen_size
        screen_area = screen_w * screen_h

        # 第一步：将 CV 元素坐标还原到原始分辨率
        restored_cv = self._restore_cv_coords(cv_elements, resize_ratio)

        # 第二步：融合 CV 元素和 Dump 控件
        merged_widgets = self._merge_sources(restored_cv, dump_elements)

        # 第三步：上下文瘦身 — 过滤冗余
        filtered = self._filter_widgets(merged_widgets, screen_area)

        # 重新分配 ID
        for i, w in enumerate(filtered):
            w.widget_id = i

        state = UIState(
            widgets=filtered,
            activity_name=activity_name,
            package_name=package_name,
            screen_width=screen_w,
            screen_height=screen_h,
            screenshot_path=screenshot_path,
            raw_cv_elements=cv_elements,
        )

        logger.info(
            "UIState 构建完成: 最终控件数=%d (CV原始=%d, Dump原始=%d)",
            len(filtered), len(cv_elements), len(dump_elements),
        )
        return state

    def _restore_cv_coords(
        self, cv_elements: List[Dict], resize_ratio: float
    ) -> List[Dict]:
        """将 CV 检测坐标还原到原始分辨率"""
        if resize_ratio == 1.0 or resize_ratio == 0:
            return cv_elements

        restored = []
        for elem in cv_elements:
            pos = elem.get("position", {})
            new_elem = dict(elem)
            new_elem["position"] = {
                "column_min": int(pos.get("column_min", 0) / resize_ratio),
                "row_min": int(pos.get("row_min", 0) / resize_ratio),
                "column_max": int(pos.get("column_max", 0) / resize_ratio),
                "row_max": int(pos.get("row_max", 0) / resize_ratio),
            }
            restored.append(new_elem)
        return restored

    def _merge_sources(
        self,
        cv_elements: List[Dict],
        dump_elements: List[UIElement],
    ) -> List[WidgetInfo]:
        """
        融合两路数据源
        - CV 提供视觉 bbox + OCR 文字 + 类别
        - Dump 提供 resource-id, clickable 等语义属性
        """
        widgets: List[WidgetInfo] = []
        matched_dump_indices = set()

        # 遍历 CV 元素，尝试匹配 Dump 控件
        for cv_idx, cv_elem in enumerate(cv_elements):
            pos = cv_elem.get("position", {})
            cv_box = (
                pos.get("column_min", 0),
                pos.get("row_min", 0),
                pos.get("column_max", 0),
                pos.get("row_max", 0),
            )

            best_dump = None
            best_iou = 0.0
            best_dump_idx = -1

            # 在 Dump 中寻找 IoU 最高的匹配
            for d_idx, dump_elem in enumerate(dump_elements):
                if d_idx in matched_dump_indices:
                    continue
                iou = calc_iou(cv_box, dump_elem.bounds)
                if iou > best_iou:
                    best_iou = iou
                    best_dump = dump_elem
                    best_dump_idx = d_idx

            cx = (cv_box[0] + cv_box[2]) // 2
            cy = (cv_box[1] + cv_box[3]) // 2

            widget = WidgetInfo(
                widget_id=cv_idx,
                bounds=cv_box,
                center=(cx, cy),
                category=cv_elem.get("class", ""),
                text=cv_elem.get("text_content", ""),
            )

            # 如果匹配成功，融合 Dump 信息
            if best_dump is not None and best_iou >= self.IOU_THRESHOLD:
                matched_dump_indices.add(best_dump_idx)
                widget.resource_id = best_dump.resource_id
                widget.class_name = best_dump.class_name
                widget.clickable = best_dump.clickable
                widget.scrollable = best_dump.scrollable
                widget.content_desc = best_dump.content_desc
                # 优先使用 Dump 文本（更准确）
                if best_dump.text:
                    widget.text = best_dump.text
                logger.debug(
                    "CV[%d] <-> Dump[%s] 匹配成功, IoU=%.3f",
                    cv_idx, best_dump.resource_id, best_iou,
                )

            widgets.append(widget)

        # 添加未匹配的 Dump 控件（CV 可能遗漏的）
        for d_idx, dump_elem in enumerate(dump_elements):
            if d_idx not in matched_dump_indices and (dump_elem.is_interactive or dump_elem.text or dump_elem.content_desc):
                cx, cy = dump_elem.center
                widget = WidgetInfo(
                    widget_id=len(widgets),
                    bounds=dump_elem.bounds,
                    center=(cx, cy),
                    class_name=dump_elem.class_name,
                    text=dump_elem.text,
                    resource_id=dump_elem.resource_id,
                    content_desc=dump_elem.content_desc,
                    clickable=dump_elem.clickable,
                    scrollable=dump_elem.scrollable,
                )
                widgets.append(widget)

        return widgets

    def _filter_widgets(
        self, widgets: List[WidgetInfo], screen_area: int
    ) -> List[WidgetInfo]:
        """
        上下文瘦身：过滤不可见/面积过小/过大/冗余节点
        有效压缩 LLM 的上下文窗口占用
        """
        filtered = []
        for w in widgets:
            area = (w.bounds[2] - w.bounds[0]) * (w.bounds[3] - w.bounds[1])

            # 过滤面积过小
            if area < self.MIN_AREA:
                continue

            # 过滤占屏过大（通常是背景容器）
            if screen_area > 0 and area / screen_area > self.MAX_SCREEN_RATIO:
                continue

            # 过滤没有任何有用信息的空控件
            if (
                not w.text
                and not w.content_desc
                and not w.resource_id
                and not w.clickable
                and not w.scrollable
                and w.category in ("", "Block")
            ):
                continue

            filtered.append(w)

        logger.debug(
            "上下文瘦身: %d -> %d 控件 (过滤了 %d 个冗余控件)",
            len(widgets), len(filtered), len(widgets) - len(filtered),
        )
        return filtered
