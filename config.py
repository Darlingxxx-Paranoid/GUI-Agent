"""
全局配置模块
包含 ADB、LLM、路径、Agent 运行参数等所有配置项
"""
import os


class AgentConfig:
    """GUI Agent 全局配置"""

    def __init__(self):
        # ========================
        # ADB 配置
        # ========================
        self.adb_serial: str = ""  # 设备序列号，空则使用默认设备
        self.screenshot_dir: str = os.path.join(os.path.dirname(__file__), "data", "screenshots")
        self.dump_dir: str = os.path.join(os.path.dirname(__file__), "data", "dumps")

        # ========================
        # LLM 配置 (OpenAI 兼容接口)
        # ========================
        self.llm_api_base: str = "https://api.poixe.com/v1"
        self.llm_api_key: str = "sk-rAR4iHHBCwD1T3FI6VJtakvMFkaaIf8GsMmH4j5wPCIdDFiw"
        self.llm_model: str = "gpt-5.2"
        self.llm_temperature: float = 0
        self.llm_max_tokens: int = 5000

        # ========================
        # CV/UIED 配置
        # ========================
        self.cv_output_dir: str = os.path.join(os.path.dirname(__file__), "data", "cv_output")
        self.bbox_screenshot_dir: str = os.path.join(os.path.dirname(__file__), "data", "bbox_screenshots")
        self.cv_resize_height: int = 800

        # ========================
        # Agent 运行参数
        # ========================
        self.max_steps: int = 30  # 单个 Task 最大步数
        self.max_retries_per_subgoal: int = 3  # 单个子目标最大重试次数
        self.screen_variance_threshold: float = 10.0  # 白屏/黑屏方差阈值

    def ensure_dirs(self):
        """确保所有工作目录存在"""
        dirs = [
            self.screenshot_dir,
            self.dump_dir,
            self.cv_output_dir,
            self.bbox_screenshot_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
