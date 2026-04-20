"""
LLM 客户端模块
封装与大语言模型的 API 交互，使用 OpenAI 兼容接口
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai 包未安装, LLM 功能将不可用; 请运行 pip install openai")

from utils.audit_recorder import AuditRecorder


@dataclass
class LLMRequest:
    """结构化多模态请求。"""

    system: str = ""
    user: str = ""
    images: list[str] = field(default_factory=list)
    response_format: Any = None
    audit_meta: Optional[dict[str, Any]] = None


class LLMStructuredOutputError(RuntimeError):
    """Structured output parse/validation failed but raw model output is available."""

    def __init__(
        self,
        message: str,
        raw_text: str = "",
        raw_payload: Any = None,
        parse_error: Exception | None = None,
        validation_error: Exception | None = None,
    ):
        super().__init__(message)
        self.raw_text = str(raw_text or "")
        self.raw_payload = raw_payload
        self.parse_error = parse_error
        self.validation_error = validation_error


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
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.audit = AuditRecorder(component="llm_client")

        if OpenAI is None:
            logger.error("openai 包未安装, 无法初始化 LLMClient")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                # 禁用 SDK 自动重试，避免重复请求带来的额外延迟。
                max_retries=0,
            )

        logger.info("LLMClient 初始化完成: model=%s, api_base=%s", model, api_base)

    def chat(
        self,
        prompt: str | LLMRequest,
        system_prompt: Optional[str] = None,
        audit_meta: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        统一对话接口：
        - chat(prompt="...", system_prompt="...")
        - chat(LLMRequest(...))
        :param prompt: 用户 prompt 或 LLMRequest
        :param system_prompt: 系统 prompt（可选）
        :return: 文本响应或结构化解析对象
        """
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化, 请安装 openai 包并正确配置 API")
        if isinstance(prompt, LLMRequest):
            request = prompt
        else:
            request = LLMRequest(
                system=str(system_prompt or ""),
                user=str(prompt or ""),
                images=[],
                response_format=None,
                audit_meta=audit_meta,
            )

        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": str(request.system)})

        if request.images:
            user_parts: list[dict[str, Any]] = [{"type": "text", "text": str(request.user or "")}]
            for image_path in request.images:
                image_url = self._local_image_to_data_url(image_path)
                user_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": str(request.user or "")})

        logger.info(
            "发送 LLM 请求: model=%s, user_len=%d, images=%d, structured=%s",
            self.model,
            len(request.user or ""),
            len(request.images),
            bool(request.response_format),
        )

        meta = request.audit_meta or {}
        try:
            if request.response_format is not None:
                try:
                    response = self.client.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=request.response_format,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                    )
                    message = response.choices[0].message
                    parsed = getattr(message, "parsed", None)
                    if parsed is None:
                        refusal = getattr(message, "refusal", None)
                        if refusal:
                            raise RuntimeError(f"模型拒绝响应: {refusal}")
                        raise RuntimeError("模型返回无法解析为 response_format 的内容")
                    self._save_audit_record(
                        artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                        step=(meta or {}).get("step"),
                        stage=str((meta or {}).get("stage") or "chat_structured"),
                        llm_response=self._serialize_parsed(parsed),
                        error="",
                    )
                    return parsed
                except Exception as parse_exc:
                    logger.warning(
                        "结构化解析失败，回退抓取原始 JSON 输出: %s",
                        parse_exc,
                    )
                    raw_content = ""
                    raw_payload: Any = None
                    try:
                        raw_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=self.temperature,
                            max_completion_tokens=self.max_tokens,
                        )
                        raw_content = str(raw_response.choices[0].message.content or "")
                        raw_payload = json.loads(raw_content)
                    except Exception as fallback_exc:
                        self._save_audit_record(
                            artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                            step=(meta or {}).get("step"),
                            stage=str((meta or {}).get("stage") or "chat_structured_error"),
                            llm_response=raw_content,
                            error=f"parse_error={parse_exc}; fallback_error={fallback_exc}",
                        )
                        raise

                    validation_exc: Exception | None = None
                    try:
                        if hasattr(request.response_format, "model_validate"):
                            parsed = request.response_format.model_validate(raw_payload)
                        else:
                            parsed = raw_payload
                        self._save_audit_record(
                            artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                            step=(meta or {}).get("step"),
                            stage=str((meta or {}).get("stage") or "chat_structured_fallback"),
                            llm_response=self._serialize_parsed(raw_payload),
                            error=f"parse_error={parse_exc}",
                        )
                        return parsed
                    except Exception as val_exc:
                        validation_exc = val_exc
                        self._save_audit_record(
                            artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                            step=(meta or {}).get("step"),
                            stage=str((meta or {}).get("stage") or "chat_structured_validation_error"),
                            llm_response=self._serialize_parsed(raw_payload),
                            error=f"parse_error={parse_exc}; validation_error={val_exc}",
                        )
                        raise LLMStructuredOutputError(
                            message=f"结构化输出校验失败: {val_exc}",
                            raw_text=raw_content,
                            raw_payload=raw_payload,
                            parse_error=parse_exc,
                            validation_error=validation_exc,
                        ) from val_exc

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            self._save_audit_record(
                artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                step=(meta or {}).get("step"),
                stage=str((meta or {}).get("stage") or "chat"),
                llm_response=content or "",
                error="",
            )
            return content or ""
        except Exception as e:
            logger.error("LLM 请求失败: %s", e)
            stage = "chat_structured_error" if request.response_format is not None else "chat_error"
            self._save_audit_record(
                artifact_kind=str((meta or {}).get("artifact_kind") or ""),
                step=(meta or {}).get("step"),
                stage=str((meta or {}).get("stage") or stage),
                llm_response="",
                error=str(e),
            )
            raise

    def _local_image_to_data_url(self, image_path: str) -> str:
        text = str(image_path or "").strip()
        if not text:
            raise ValueError("截图路径不能为空")

        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"截图文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"截图路径不是文件: {path}")

        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _serialize_parsed(self, value: Any) -> str:
        if hasattr(value, "model_dump"):
            try:
                return json.dumps(value.model_dump(), ensure_ascii=False)
            except Exception:
                pass
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _save_audit_record(
        self,
        artifact_kind: str,
        step: Any,
        stage: str,
        llm_response: str,
        error: str,
    ) -> None:
        if not artifact_kind:
            return
        try:
            step_num = int(step)
        except Exception:
            return
        if step_num <= 0:
            return
        try:
            self.audit.record_step(
                artifact_kind=artifact_kind,
                step=step_num,
                payload={
                    "stage": stage,
                    "llm_response": llm_response,
                    "error": error,
                },
                llm=True,
                append=True,
            )
        except Exception as exc:
            logger.warning("写入 LLM 审计记录失败: %s", exc)
