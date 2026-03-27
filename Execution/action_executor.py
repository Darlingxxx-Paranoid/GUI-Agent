"""
ADB 指令执行模块
将坐标和动作类型转化为物理 ADB 指令执行
"""
import base64
import os
import re
import shutil
import subprocess
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    ADB 命令执行器
    封装所有与 Android 设备的物理交互
    """

    _COMMON_ADB_PATHS = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    ]
    _ADB_ENV_VARS = ("ADB", "ANDROID_SDK_ROOT", "ANDROID_HOME")
    _COMPONENT_RE = re.compile(r"([A-Za-z0-9._$]+/[A-Za-z0-9._$]+)")
    _ASCII_ESCAPE_MAP = {
        " ": "%s",
        "&": r"\&",
        "<": r"\<",
        ">": r"\>",
        "|": r"\|",
        ";": r"\;",
        "(": r"\(",
        ")": r"\)",
        "[": r"\[",
        "]": r"\]",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "*": r"\*",
        "?": r"\?",
        "#": r"\#",
        "!": r"\!",
        "~": r"\~",
        "`": r"\`",
        '"': r"\"",
        "'": r"\'",
        "\\": r"\\",
    }

    def __init__(
        self,
        serial: str = "",
        screenshot_dir: str = "data/screenshots",
        dump_dir: str = "data/dumps",
        adb_path: str = "",
    ):
        """
        :param serial: ADB 设备序列号，空则使用默认设备
        :param screenshot_dir: 截图保存目录
        :param dump_dir: dump 文件保存目录
        """
        self.serial = serial
        self.screenshot_dir = screenshot_dir
        self.dump_dir = dump_dir
        self.adb_path = self._resolve_adb_path(adb_path)

        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(dump_dir, exist_ok=True)

        logger.info(
            "ActionExecutor 初始化完成, serial='%s', adb='%s'",
            serial or "default",
            self.adb_path,
        )

    def _resolve_adb_path(self, explicit_path: str = "") -> str:
        """解析 adb 可执行文件路径，优先显式路径，再尝试常见 SDK 位置。"""
        candidates: List[str] = []

        if explicit_path:
            candidates.append(os.path.expanduser(explicit_path))

        which_adb = shutil.which("adb")
        if which_adb:
            candidates.append(which_adb)

        for env_var in self._ADB_ENV_VARS:
            value = os.environ.get(env_var, "").strip()
            if not value:
                continue
            expanded = os.path.expanduser(value)
            if os.path.basename(expanded) == "adb":
                candidates.append(expanded)
            else:
                candidates.append(os.path.join(expanded, "platform-tools", "adb"))

        candidates.extend(self._COMMON_ADB_PATHS)

        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        logger.warning("未自动发现 adb, 将回退为依赖 PATH 中的 'adb'")
        return "adb"

    def _adb_cmd(self, *args) -> subprocess.CompletedProcess:
        """构造并执行 ADB 命令"""
        cmd = [self.adb_path]
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
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not normalized:
            logger.debug("输入文本为空，跳过执行")
            return

        lines = normalized.split("\n")
        for idx, line in enumerate(lines):
            if line:
                if any(ord(c) > 127 for c in line):
                    if not (self._broadcast_text(line) or self._broadcast_text_base64(line)):
                        logger.warning("非 ASCII 文本广播输入失败: %s", line[:60])
                else:
                    self._input_ascii_chunks(line)

            if idx < len(lines) - 1:
                self.enter()
                time.sleep(0.1)

    def _broadcast_text(self, text: str) -> bool:
        """尝试使用 ADB Keyboard 风格广播输入文本。"""
        result = self._adb_cmd(
            "shell",
            "am",
            "broadcast",
            "-a",
            "ADB_INPUT_TEXT",
            "--es",
            "msg",
            text,
        )
        stdout = (result.stdout or "").strip()
        if result.returncode == 0 and "Broadcast completed" in stdout:
            return True
        logger.debug("广播输入不可用: stdout=%s stderr=%s", stdout, (result.stderr or "").strip())
        return False

    def _broadcast_text_base64(self, text: str) -> bool:
        """兼容 ADBKeyBoard 的 base64 输入广播通道。"""
        payload = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
        result = self._adb_cmd(
            "shell",
            "am",
            "broadcast",
            "-a",
            "ADB_INPUT_B64",
            "--es",
            "msg",
            payload,
        )
        stdout = (result.stdout or "").strip()
        if result.returncode == 0 and "Broadcast completed" in stdout:
            return True
        logger.debug("Base64 广播输入不可用: stdout=%s stderr=%s", stdout, (result.stderr or "").strip())
        return False

    def _input_ascii_chunks(self, text: str, chunk_size: int = 48):
        """将 ASCII 文本拆成多段输入，兼容长文本与换行场景。"""
        for chunk in self._chunk_ascii_text(text, chunk_size=chunk_size):
            escaped = self._escape_ascii_text(chunk)
            if not escaped:
                continue
            self._adb_cmd("shell", "input", "text", escaped)
            time.sleep(0.08)

    def _chunk_ascii_text(self, text: str, chunk_size: int = 48) -> List[str]:
        """优先在空白处切分，避免长文本单条 input text 失败。"""
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        cursor = 0
        text_len = len(text)

        while cursor < text_len:
            end = min(text_len, cursor + chunk_size)
            split_at = end

            if end < text_len:
                soft_split = text.rfind(" ", cursor + max(1, chunk_size // 2), end)
                if soft_split > cursor:
                    split_at = soft_split + 1

            if split_at <= cursor:
                split_at = end

            chunk = text[cursor:split_at]
            if chunk:
                chunks.append(chunk)
            cursor = split_at

        return chunks or [text]

    def _escape_ascii_text(self, text: str) -> str:
        """转义 `adb shell input text` 对 shell/空格敏感的字符。"""
        pieces: List[str] = []
        for char in text:
            pieces.append(self._ASCII_ESCAPE_MAP.get(char, char))
        return "".join(pieces)

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
        lines = (result.stdout or "").splitlines()
        priorities = (
            "ResumedActivity:",
            "topResumedActivity=",
            "mResumedActivity",
            "mFocusedActivity",
            "mFocusedApp=",
        )
        for marker in priorities:
            for line in lines:
                if marker in line:
                    component = self._extract_component(line)
                    if component:
                        logger.debug("当前 Activity: %s", component)
                        return component
        return ""

    def get_current_package(self) -> str:
        """获取当前前台应用包名"""
        result = self._adb_cmd(
            "shell", "dumpsys", "window", "windows",
        )
        lines = (result.stdout or "").splitlines()
        for marker in ("mCurrentFocus", "mFocusedApp", "mFocusedWindow"):
            for line in lines:
                if marker in line:
                    component = self._extract_component(line)
                    if component:
                        return component.split("/")[0]

        activity = self.get_current_activity()
        if activity:
            return activity.split("/")[0]
        return ""

    def get_keyboard_visible(self) -> bool:
        """读取输入法状态，判断软键盘是否可见。"""
        result = self._adb_cmd("shell", "dumpsys", "input_method")
        output = result.stdout or ""

        for key in ("mInputShown", "mIsInputViewShown", "isInputShown", "imeVisible"):
            match = re.search(rf"{key}\s*=\s*(true|false)", output, flags=re.IGNORECASE)
            if match:
                visible = match.group(1).lower() == "true"
                logger.debug("软键盘状态(%s): %s", key, visible)
                return visible

        match = re.search(r"mImeWindowVis\s*=\s*(\d+)", output)
        if match:
            vis_flags = int(match.group(1))
            visible = bool(vis_flags & 0x2)
            logger.debug("软键盘状态(mImeWindowVis=%d): %s", vis_flags, visible)
            return visible

        logger.debug("未从 dumpsys input_method 解析到软键盘状态，默认 False")
        return False

    def _extract_component(self, text: str) -> str:
        """从 dumpsys 输出片段中提取 `package/activity` 组件名。"""
        match = self._COMPONENT_RE.search(text)
        if not match:
            return ""
        return match.group(1)

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
