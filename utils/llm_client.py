"""
LLM 客户端模块
封装与大语言模型的 API 交互，使用 OpenAI 兼容接口
"""
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai 包未安装, LLM 功能将不可用; 请运行 pip install openai")


class LLMClient:
    """
    LLM 调用客户端
    使用 OpenAI 兼容接口（支持 OpenAI / Azure / 本地部署的兼容 API）
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if OpenAI is None:
            logger.error("openai 包未安装, 无法初始化 LLMClient")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=timeout,
            )

        logger.info("LLMClient 初始化完成: model=%s, api_base=%s", model, api_base)

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        发送单轮对话请求
        :param prompt: 用户 prompt
        :param system_prompt: 系统 prompt（可选）
        :return: LLM 响应文本
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化, 请安装 openai 包并正确配置 API")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug("发送 LLM 请求: model=%s, prompt_len=%d", self.model, len(prompt))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            logger.debug("LLM 响应: len=%d", len(content) if content else 0)
            return content or ""
        except Exception as e:
            logger.error("LLM 请求失败: %s", e)
            raise
