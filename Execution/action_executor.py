"""
ADB 指令执行模块
将坐标和动作类型转化为物理 ADB 指令执行
"""
import os
import subprocess
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    ADB 命令执行器
    封装所有与 Android 设备的物理交互
    """

    def __init__(self, serial: str = "", screenshot_dir: str = "data/screenshots", dump_dir: str = "data/dumps"):
        """
        :param serial: ADB 设备序列号，空则使用默认设备
        :param screenshot_dir: 截图保存目录
        :param dump_dir: dump 文件保存目录
        """
        self.serial = serial
        self.screenshot_dir = screenshot_dir
        self.dump_dir = dump_dir

        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(dump_dir, exist_ok=True)

        logger.info("ActionExecutor 初始化完成, serial='%s'", serial or "default")

    def _adb_cmd(self, *args) -> subprocess.CompletedProcess:
        """构造并执行 ADB 命令"""
        cmd = ["adb"]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)

        logger.debug("执行 ADB 命令: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("ADB 命令返回非零: %s, stderr: %s", result.returncode, result.stderr.strip())
            return result
        except subprocess.TimeoutExpired:
            logger.error("ADB 命令超时: %s", " ".join(cmd))
            raise
        except FileNotFoundError:
            logger.error("未找到 adb 命令, 请确保 adb 在 PATH 中")
            raise

    def tap(self, x: int, y: int):
        """点击指定坐标"""
        logger.info("执行点击: (%d, %d)", x, y)
        self._adb_cmd("shell", "input", "tap", str(x), str(y))

    def long_press(self, x: int, y: int, duration_ms: int = 1000):
        """长按指定坐标"""
        logger.info("执行长按: (%d, %d), 时长=%dms", x, y, duration_ms)
        self._adb_cmd("shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
        """滑动"""
        logger.info("执行滑动: (%d,%d) -> (%d,%d), 时长=%dms", x1, y1, x2, y2, duration_ms)
        self._adb_cmd(
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        )

    def input_text(self, text: str):
        """输入文本（需要先聚焦输入框）"""
        logger.info("执行文本输入: '%s'", text[:50])
        # 对中文等非ASCII字符使用 am broadcast
        if any(ord(c) > 127 for c in text):
            self._adb_cmd(
                "shell", "am", "broadcast",
                "-a", "ADB_INPUT_TEXT",
                "--es", "msg", text,
            )
        else:
            # 使用 input text 命令（仅支持ASCII）
            # 转义空格
            escaped = text.replace(" ", "%s")
            self._adb_cmd("shell", "input", "text", escaped)

    def back(self):
        """按返回键"""
        logger.info("执行返回键")
        self._adb_cmd("shell", "input", "keyevent", "KEYCODE_BACK")

    def home(self):
        """按 Home 键"""
        logger.info("执行 Home 键")
        self._adb_cmd("shell", "input", "keyevent", "KEYCODE_HOME")

    def enter(self):
        """按回车键"""
        logger.info("执行回车键")
        self._adb_cmd("shell", "input", "keyevent", "KEYCODE_ENTER")

    def screenshot(self, filename: Optional[str] = None) -> str:
        """
        截取当前屏幕
        :param filename: 文件名，默认使用时间戳
        :return: 截图保存的本地路径
        """
        if filename is None:
            filename = f"screenshot_{int(time.time() * 1000)}.png"

        device_path = f"/sdcard/{filename}"
        local_path = os.path.join(self.screenshot_dir, filename)

        logger.info("截屏: %s", local_path)
        self._adb_cmd("shell", "screencap", "-p", device_path)
        self._adb_cmd("pull", device_path, local_path)
        self._adb_cmd("shell", "rm", device_path)

        return local_path

    def dump_ui(self, filename: Optional[str] = None) -> str:
        """
        导出当前 UI 结构
        :param filename: 文件名，默认使用时间戳
        :return: dump 保存的本地路径
        """
        if filename is None:
            filename = f"dump_{int(time.time() * 1000)}.xml"

        device_path = "/sdcard/window_dump.xml"
        local_path = os.path.join(self.dump_dir, filename)

        logger.info("Dump UI: %s", local_path)
        self._adb_cmd("shell", "uiautomator", "dump", device_path)
        self._adb_cmd("pull", device_path, local_path)

        return local_path

    def get_current_activity(self) -> str:
        """获取当前前台 Activity 名"""
        result = self._adb_cmd(
            "shell", "dumpsys", "activity", "activities",
        )
        # 解析 mResumedActivity 或 mFocusedActivity
        for line in result.stdout.split("\n"):
            if "mResumedActivity" in line or "mFocusedActivity" in line:
                parts = line.strip().split()
                for part in parts:
                    if "/" in part and "." in part:
                        logger.debug("当前 Activity: %s", part)
                        return part
        return ""

    def get_current_package(self) -> str:
        """获取当前前台应用包名"""
        result = self._adb_cmd(
            "shell", "dumpsys", "window", "windows",
        )
        for line in result.stdout.split("\n"):
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                parts = line.strip().split()
                for part in parts:
                    if "/" in part:
                        return part.split("/")[0]
        return ""

    def get_screen_size(self) -> tuple:
        """获取设备屏幕分辨率"""
        result = self._adb_cmd("shell", "wm", "size")
        # 输出格式: Physical size: 1080x1920
        for line in result.stdout.split("\n"):
            if "Physical size" in line:
                size_str = line.split(":")[-1].strip()
                w, h = size_str.split("x")
                size = (int(w), int(h))
                logger.info("设备屏幕分辨率: %s", size)
                return size
        logger.warning("无法获取屏幕分辨率, 使用默认 1080x1920")
        return (1080, 1920)
