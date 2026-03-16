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
        self.llm_api_key: str = "sk-kbkg2ZXBcai8A0lq36mYN5qolckALs73nGz6u7xYoi2SOAEF"
        self.llm_model: str = "gpt-5.2"
        self.llm_temperature: float = 0
        self.llm_max_tokens: int = 5000
        self.llm_timeout: int = 60  # 秒

        # ========================
        # CV/UIED 配置
        # ========================
        self.cv_output_dir: str = os.path.join(os.path.dirname(__file__), "data", "cv_output")
        self.cv_resize_height: int = 800

        # ========================
        # 记忆系统配置
        # ========================
        self.memory_dir: str = os.path.join(os.path.dirname(__file__), "data", "memory")
        self.long_term_memory_file: str = os.path.join(
            os.path.dirname(__file__), "data", "memory", "long_term.json"
        )
        self.experience_similarity_threshold: float = 0.75  # 经验匹配相似度阈值

        # ========================
        # Agent 运行参数
        # ========================
        self.max_steps: int = 30  # 单个 Task 最大步数
        self.max_retries_per_subgoal: int = 3  # 单个子目标最大重试次数
        self.dead_loop_threshold: int = 3  # 死循环检测: 连续相同动作次数
        self.dead_end_threshold: int = 3  # 死胡同检测: 连续失败次数
        self.screen_variance_threshold: float = 10.0  # 白屏/黑屏方差阈值

        # ========================
        # 安全拦截关键词
        # ========================
        self.high_risk_keywords: list = [
            "支付", "付款", "购买", "删除", "卸载", "清除数据",
            "授权", "权限", "登录", "密码", "转账", "确认支付",
            "pay", "delete", "uninstall", "authorize", "purchase",
        ]

    def ensure_dirs(self):
        """确保所有工作目录存在"""
        dirs = [
            self.screenshot_dir,
            self.dump_dir,
            self.cv_output_dir,
            self.memory_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
