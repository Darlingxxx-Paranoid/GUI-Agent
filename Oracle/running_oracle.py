"""Running-Oracle: detect runtime anomalies without deciding task success."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


@dataclass
class RunningOracleResult:
    phase: str
    screenshot_path: str
    checked: bool
    has_runtime_exception: bool
    is_black_screen: bool
    is_white_screen: bool
    mean_luma: float | None
    variance: float | None
    variance_threshold: float
    message: str = ""
    tags: list[str] = field(default_factory=list)


class RunningOracle:
    """Runtime anomaly checker. Signals are advisory only."""

    def __init__(
        self,
        screen_variance_threshold: float = 10.0,
        black_mean_threshold: float = 10.0,
        white_mean_threshold: float = 245.0,
    ):
        self.screen_variance_threshold = float(screen_variance_threshold)
        self.black_mean_threshold = float(black_mean_threshold)
        self.white_mean_threshold = float(white_mean_threshold)
        logger.info(
            "RunningOracle 初始化完成: var<=%.2f, black<=%.2f, white>=%.2f",
            self.screen_variance_threshold,
            self.black_mean_threshold,
            self.white_mean_threshold,
        )

    def reset(self) -> None:
        """Placeholder for future stateful checks."""
        return

    def check(self, screenshot_path: str, phase: str = "runtime") -> RunningOracleResult:
        path = str(screenshot_path or "").strip()
        base = RunningOracleResult(
            phase=str(phase or "runtime"),
            screenshot_path=path,
            checked=False,
            has_runtime_exception=False,
            is_black_screen=False,
            is_white_screen=False,
            mean_luma=None,
            variance=None,
            variance_threshold=self.screen_variance_threshold,
            message="",
            tags=[],
        )

        if not path:
            base.message = "skip_empty_screenshot_path"
            return base

        if not os.path.exists(path):
            base.message = "skip_missing_screenshot"
            return base

        if cv2 is None:
            base.message = "skip_opencv_unavailable"
            logger.warning("RunningOracle 跳过检查: OpenCV 不可用")
            return base

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            base.message = "skip_image_decode_failed"
            logger.warning("RunningOracle 跳过检查: 截图解码失败 %s", path)
            return base

        mean_luma = float(image.mean())
        variance = float(image.var())
        flat_screen = variance <= self.screen_variance_threshold
        is_black = flat_screen and mean_luma <= self.black_mean_threshold
        is_white = flat_screen and mean_luma >= self.white_mean_threshold

        tags: list[str] = []
        if is_black:
            tags.append("black_screen")
        if is_white:
            tags.append("white_screen")

        has_runtime_exception = bool(tags)
        message = "runtime_screen_anomaly" if has_runtime_exception else "runtime_screen_ok"

        result = RunningOracleResult(
            phase=str(phase or "runtime"),
            screenshot_path=path,
            checked=True,
            has_runtime_exception=has_runtime_exception,
            is_black_screen=is_black,
            is_white_screen=is_white,
            mean_luma=mean_luma,
            variance=variance,
            variance_threshold=self.screen_variance_threshold,
            message=message,
            tags=tags,
        )

        if has_runtime_exception:
            logger.warning(
                "RunningOracle 检测到运行时异常: %s (mean=%.2f, var=%.2f, path=%s)",
                ",".join(tags),
                mean_luma,
                variance,
                path,
            )
        else:
            logger.debug(
                "RunningOracle 检查通过: mean=%.2f, var=%.2f, path=%s",
                mean_luma,
                variance,
                path,
            )

        return result
