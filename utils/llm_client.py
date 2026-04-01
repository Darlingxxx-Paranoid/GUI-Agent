"""
LLM 客户端模块
封装与大语言模型的 API 交互，使用 OpenAI 兼容接口
"""
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai 包未安装, LLM 功能将不可用; 请运行 pip install openai")

from utils.audit_recorder import AuditRecorder


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
        self.default_timeout = max(1, int(timeout or 60))
        self._deadline_ts: Optional[float] = None
        self.audit = AuditRecorder(component="llm_client")

        if OpenAI is None:
            logger.error("openai 包未安装, 无法初始化 LLMClient")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=self.default_timeout,
                # 严格受任务预算约束，避免 SDK 内部重试把短超时放大为长阻塞。
                max_retries=0,
            )

        logger.info("LLMClient 初始化完成: model=%s, api_base=%s", model, api_base)

    def set_deadline(self, deadline_ts: Optional[float]) -> None:
        """
        设置当前任务级截止时间戳（秒）。
        传入 None 表示清除限制。
        """
        self._deadline_ts = float(deadline_ts) if deadline_ts else None

    def remaining_seconds(self) -> Optional[int]:
        """
        返回当前任务级时限的剩余秒数。
        若未设置任务时限，则返回 None。
        """
        if self._deadline_ts is None:
            return None
        try:
            remaining = int(self._deadline_ts - time.time())
        except Exception:
            return None
        return max(0, remaining)

    def _resolve_timeout(self, timeout: Optional[float]) -> int:
        timeout_sec = self.default_timeout
        if timeout is not None:
            try:
                timeout_sec = max(1, int(float(timeout)))
            except Exception:
                timeout_sec = self.default_timeout

        if self._deadline_ts is None:
            return timeout_sec

        remaining = int(self._deadline_ts - time.time())
        if remaining <= 1:
            raise TimeoutError("任务时限已到，取消 LLM 调用")
        return max(1, min(timeout_sec, remaining))

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        audit_meta: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        发送单轮对话请求
        :param prompt: 用户 prompt
        :param system_prompt: 系统 prompt（可选）
        :param timeout: 本次请求超时秒数（可选）
        :return: LLM 响应文本
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化, 请安装 openai 包并正确配置 API")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_timeout = self._resolve_timeout(timeout)
        logger.debug(
            "发送 LLM 请求: model=%s, prompt_len=%d, timeout=%ds",
            self.model,
            len(prompt),
            request_timeout,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                timeout=request_timeout,
            )
            content = response.choices[0].message.content
            logger.debug("LLM 响应: len=%d", len(content) if content else 0)
            self._save_audit_record(
                stage=str((audit_meta or {}).get("stage") or "chat"),
                payload={
                    "module": str((audit_meta or {}).get("module") or "llm"),
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout_sec": request_timeout,
                    "prompt": prompt,
                    "system_prompt": system_prompt or "",
                    "response_text": content or "",
                    "error": "",
                },
            )
            return content or ""
        except Exception as e:
            logger.error("LLM 请求失败: %s", e)
            self._save_audit_record(
                stage=str((audit_meta or {}).get("stage") or "chat_error"),
                payload={
                    "module": str((audit_meta or {}).get("module") or "llm"),
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout_sec": request_timeout,
                    "prompt": prompt,
                    "system_prompt": system_prompt or "",
                    "response_text": "",
                    "error": str(e),
                },
            )
            raise

    def _save_audit_record(self, stage: str, payload: dict[str, Any]) -> None:
        module = str(payload.get("module") or "llm")
        try:
            self.audit.record(
                module=module,
                stage=stage,
                event="llm_exchange",
                payload=payload,
            )
        except Exception as exc:
            logger.warning("写入 LLM 审计记录失败: %s", exc)
