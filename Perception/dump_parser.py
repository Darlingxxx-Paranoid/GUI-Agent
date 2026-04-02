"""
Android UI Dump 解析模块
解析 UI 层级 XML（由 uiautomator2 等来源生成），提取所有可见控件属性
"""
import xml.etree.ElementTree as ET
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class UIElement:
    """解析后的单个UI控件"""
    _EDITABLE_CLASS_HINTS = ("edittext", "autocompletetextview", "textinputedittext")

    def __init__(
        self,
        resource_id: str = "",
        class_name: str = "",
        text: str = "",
        content_desc: str = "",
        bounds: tuple = (0, 0, 0, 0),
        clickable: bool = False,
        scrollable: bool = False,
        enabled: bool = True,
        focusable: bool = False,
        editable: bool = False,
        focused: bool = False,
        checkable: bool = False,
        checked: bool = False,
        selected: bool = False,
        package: str = "",
        index: int = 0,
    ):
        self.resource_id = resource_id
        self.class_name = class_name
        self.text = text
        self.content_desc = content_desc
        self.bounds = bounds  # (x1, y1, x2, y2)
        self.clickable = clickable
        self.scrollable = scrollable
        self.enabled = enabled
        self.focusable = focusable
        self.editable = editable
        self.focused = focused
        self.checkable = checkable
        self.checked = checked
        self.selected = selected
        self.package = package
        self.index = index

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )

    @property
    def is_interactive(self) -> bool:
        """判断控件是否可交互"""
        return self.clickable or self.scrollable or self.focusable or self.checkable

    @property
    def is_editable(self) -> bool:
        """判断控件是否为输入类组件"""
        class_name = (self.class_name or "").lower()
        if self.editable:
            return True
        return any(hint in class_name for hint in self._EDITABLE_CLASS_HINTS)

    def to_dict(self) -> dict:
        return {
            "resource_id": self.resource_id,
            "class": self.class_name,
            "text": self.text,
            "content_desc": self.content_desc,
            "bounds": list(self.bounds),
            "center": list(self.center),
            "clickable": self.clickable,
            "scrollable": self.scrollable,
            "enabled": self.enabled,
            "focusable": self.focusable,
            "focused": self.focused,
            "checkable": self.checkable,
            "checked": self.checked,
            "selected": self.selected,
            "editable": self.is_editable,
        }

    def __repr__(self) -> str:
        label = self.text or self.content_desc or self.resource_id or self.class_name
        return f"UIElement({label}, bounds={self.bounds}, clickable={self.clickable})"


class DumpParser:
    """
    Android UI Dump 解析器
    解析 UI 层级 XML 文件
    """

    # bounds 正则匹配: [x1,y1][x2,y2]
    BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

    def __init__(self):
        logger.debug("DumpParser 初始化完成")
        self._raw_node_counter = 0

    def parse(self, dump_path: str) -> List[UIElement]:
        """
        解析 dump XML 文件
        :param dump_path: XML 文件路径
        :return: 解析后的 UIElement 列表
        """
        logger.info("解析 UI Dump: %s", dump_path)
        try:
            tree = ET.parse(dump_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error("XML 解析失败: %s", e)
            return []

        elements: List[UIElement] = []
        self._traverse(root, elements)
        logger.info("Dump 解析完成, 提取 %d 个控件", len(elements))
        return elements

    def parse_from_string(self, xml_string: str) -> List[UIElement]:
        """从 XML 字符串解析"""
        logger.info("从字符串解析 UI Dump")
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error("XML 字符串解析失败: %s", e)
            return []

        elements: List[UIElement] = []
        self._traverse(root, elements)
        logger.info("Dump 解析完成, 提取 %d 个控件", len(elements))
        return elements

    def parse_tree(self, dump_path: str) -> Dict[str, Any]:
        """
        解析 dump XML 为原始树结构。
        - 保留 XML 节点原始属性字段
        - 仅新增 node_id 和 children
        """
        logger.info("解析 UI Dump 原始树: %s", dump_path)
        try:
            tree = ET.parse(dump_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error("XML 解析失败: %s", e)
            return {}

        self._raw_node_counter = 0
        raw_tree = self._build_raw_tree_node(root)
        logger.info("Dump 原始树解析完成, 节点数=%d", self._raw_node_counter)
        return raw_tree

    def parse_tree_from_string(self, xml_string: str) -> Dict[str, Any]:
        """从 XML 字符串解析原始树结构。"""
        logger.info("从字符串解析 UI Dump 原始树")
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error("XML 字符串解析失败: %s", e)
            return {}

        self._raw_node_counter = 0
        raw_tree = self._build_raw_tree_node(root)
        logger.info("Dump 原始树解析完成, 节点数=%d", self._raw_node_counter)
        return raw_tree

    def _traverse(self, node: ET.Element, elements: List[UIElement]):
        """递归遍历 XML 节点树"""
        element = self._parse_node(node)
        if element is not None:
            # 过滤不可见/面积为零的节点
            if element.area > 0 and element.enabled:
                elements.append(element)

        for child in node:
            self._traverse(child, elements)

    def _parse_node(self, node: ET.Element) -> Optional[UIElement]:
        """解析单个 XML 节点为 UIElement"""
        attrib = node.attrib
        if not attrib:
            return None

        bounds_str = attrib.get("bounds", "")
        bounds = self._parse_bounds(bounds_str)
        if bounds is None:
            return None

        return UIElement(
            resource_id=attrib.get("resource-id", ""),
            class_name=attrib.get("class", ""),
            text=attrib.get("text", ""),
            content_desc=attrib.get("content-desc", ""),
            bounds=bounds,
            clickable=attrib.get("clickable", "false") == "true",
            scrollable=attrib.get("scrollable", "false") == "true",
            enabled=attrib.get("enabled", "true") == "true",
            focusable=attrib.get("focusable", "false") == "true",
            editable=attrib.get("editable", "false") == "true",
            focused=attrib.get("focused", "false") == "true",
            checkable=attrib.get("checkable", "false") == "true",
            checked=attrib.get("checked", "false") == "true",
            selected=attrib.get("selected", "false") == "true",
            package=attrib.get("package", ""),
            index=int(attrib.get("index", "0")),
        )

    def _parse_bounds(self, bounds_str: str) -> Optional[tuple]:
        """解析 bounds 字符串 '[x1,y1][x2,y2]' 为元组"""
        match = self.BOUNDS_RE.match(bounds_str)
        if not match:
            return None
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )

    def _build_raw_tree_node(self, node: ET.Element) -> Dict[str, Any]:
        """构建保留原始字段的树节点，仅增加 node_id 和 children。"""
        self._raw_node_counter += 1
        current_id = self._raw_node_counter

        payload: Dict[str, Any] = dict(node.attrib)
        payload["node_id"] = current_id
        payload["children"] = [self._build_raw_tree_node(child) for child in list(node)]
        return payload
