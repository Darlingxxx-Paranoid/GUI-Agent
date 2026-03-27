"""
感知层总控模块
协调 CV 检测（UIED）与 Dump 解析，构建统一的 UIState
"""
import json
import os
import logging
from typing import Optional

from Perception.uied.detect import WidgetDetector
from Perception.dump_parser import DumpParser
from Perception.context_builder import ContextBuilder, UIState

logger = logging.getLogger(__name__)


class PerceptionManager:
    """
    多模态环境感知层总控
    负责获取并清洗当前 GUI 环境的客观事实
    1. 调用 WidgetDetector 做 CV 检测（OCR + 组件检测 + 融合）
    2. 调用 DumpParser 解析 Android UI Dump
    3. 调用 ContextBuilder 融合两路数据，输出精简 UIState
    """

    def __init__(self, cv_output_dir: str, resize_height: int = 800):
        self.widget_detector = WidgetDetector()
        self.widget_detector.resize_length = resize_height
        self.widget_detector.output_root = cv_output_dir

        self.dump_parser = DumpParser()
        self.context_builder = ContextBuilder()

        self.cv_output_dir = cv_output_dir
        os.makedirs(cv_output_dir, exist_ok=True)
        self.context_output_dir = os.path.join(os.path.dirname(cv_output_dir), "context")
        os.makedirs(self.context_output_dir, exist_ok=True)

        logger.info("PerceptionManager 初始化完成, CV 输出目录: %s", cv_output_dir)

    def perceive(
        self,
        screenshot_path: str,
        dump_path: Optional[str] = None,
        screen_size: tuple = (1080, 1920),
        activity_name: str = "",
        package_name: str = "",
        keyboard_visible: bool = False,
    ) -> UIState:
        """
        执行一次完整的环境感知
        :param screenshot_path: 截图文件路径
        :param dump_path: UI Dump XML 文件路径（可选）
        :param screen_size: 屏幕尺寸 (width, height)
        :param activity_name: 当前 Activity 名
        :param package_name: 当前包名
        :param keyboard_visible: 当前软键盘是否可见
        :return: 构建的 UIState
        """
        logger.info("开始环境感知: screenshot=%s, dump=%s", screenshot_path, dump_path)

        # ---- Dump 解析 ----
        dump_elements = []
        if dump_path and os.path.exists(dump_path):
            try:
                dump_elements = self.dump_parser.parse(dump_path)
                logger.info("Dump 解析完成: 解析到 %d 个控件", len(dump_elements))
            except Exception as e:
                logger.error("Dump 解析失败: %s, 将仅使用 CV 数据", e)
        else:
            logger.warning("未提供 Dump 文件或文件不存在, 将仅使用 CV 数据")

        # ---- CV 检测 ----
        cv_elements = []
        resize_ratio = 1.0
        run_cv = self._should_run_cv(dump_elements)
        if run_cv:
            try:
                img_res_path, resize_ratio, cv_compos = self.widget_detector.detect(
                    img_path=screenshot_path, debug=False
                )
                cv_elements = cv_compos
                logger.info("CV 检测完成: 检测到 %d 个元素, resize_ratio=%.3f", len(cv_elements), resize_ratio)
            except Exception as e:
                logger.error("CV 检测失败: %s, 将仅使用 Dump 数据", e)
        else:
            logger.info("Dump 信息已足够丰富，跳过 CV 检测以降低时延")

        # ---- 融合构建 UIState ----
        ui_state = self.context_builder.build(
            cv_elements=cv_elements,
            dump_elements=dump_elements,
            screenshot_path=screenshot_path,
            screen_size=screen_size,
            resize_ratio=resize_ratio,
            activity_name=activity_name,
            package_name=package_name,
            keyboard_visible=keyboard_visible,
        )

        self._save_context(ui_state, dump_path=dump_path)
        logger.info("环境感知完成: 最终控件数=%d", len(ui_state.widgets))
        return ui_state

    def _should_run_cv(self, dump_elements: list) -> bool:
        """
        在 Dump 语义信息充分时跳过 CV，减少 OCR/UIED 计算开销。
        """
        if not dump_elements:
            return True

        interactive = 0
        text_nodes = 0
        total_nodes = len(dump_elements)
        for elem in dump_elements:
            if getattr(elem, "is_interactive", False) or getattr(elem, "is_editable", False):
                interactive += 1
            if getattr(elem, "text", "") or getattr(elem, "content_desc", ""):
                text_nodes += 1

        # 经验阈值：节点规模足够且具备交互/文本语义时，Dump 已可支撑规划与评估
        if total_nodes >= 40 and (interactive >= 10 or text_nodes >= 8):
            return False
        if interactive >= 18 and text_nodes >= 6:
            return False
        return True

    def _save_context(self, ui_state: UIState, dump_path: Optional[str] = None) -> None:
        screenshot_basename = os.path.basename(ui_state.screenshot_path or "")
        stem, _ = os.path.splitext(screenshot_basename)
        if not stem:
            stem = "context"

        out_path = os.path.join(self.context_output_dir, f"{stem}.json")
        payload = {
            "activity_name": ui_state.activity_name,
            "package_name": ui_state.package_name,
            "screen_width": ui_state.screen_width,
            "screen_height": ui_state.screen_height,
            "keyboard_visible": ui_state.keyboard_visible,
            "editable_widgets_count": sum(1 for w in ui_state.widgets if getattr(w, "editable", False)),
            "focused_widgets_count": sum(1 for w in ui_state.widgets if getattr(w, "focused", False)),
            "screenshot_path": ui_state.screenshot_path,
            "dump_path": dump_path or "",
            "widgets_count": len(ui_state.widgets),
            "widgets": [
                {**w.to_dict(), "cv_confidence": getattr(w, "cv_confidence", 0.0)}
                for w in ui_state.widgets
            ],
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("已保存合并控件列表: %s", out_path)
        except Exception as e:
            logger.warning("保存合并控件列表失败: %s", e)
