"""
ReAct 动态规划器
评估当前UI状态与目标差距，生成单个子目标
支持长期记忆经验复用
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from Perception.context_builder import UIState
from Memory.memory_manager import MemoryManager
from prompt.planner_prompt import PLANNER_PROMPT

logger = logging.getLogger(__name__)


# ========================
# 数据类定义
# ========================


@dataclass
class SubGoal:
    """单个子目标"""
    description: str                           # 子目标描述
    target_widget_text: str = ""               # 目标控件文本特征
    target_widget_id: Optional[int] = None     # 目标控件 ID
    action_type: str = ""                      # 预期动作类型 (tap/swipe/input/back)
    input_text: str = ""                       # 输入文本（action_type=input 时）
    acceptance_criteria: str = ""              # 验收标准描述
    expected_transition: str = ""              # 跳变预期类型 (partial_refresh/new_page/external_app/dialog)
    from_experience: bool = False              # 是否来自长期记忆复用


@dataclass
class PlanResult:
    """规划结果"""
    subgoal: SubGoal
    is_task_complete: bool = False             # LLM 判断任务是否已完成
    reasoning: str = ""                        # 推理过程


class Planner:
    """
    ReAct 动态规划器
    - 优先检索长期记忆中的经验
    - 无命中时通过 LLM 生成单个子目标
    - 子目标包含验收标准和跳变预期
    """

    def __init__(self, llm_client, memory_manager: MemoryManager):
        """
        :param llm_client: LLM 调用客户端 (需有 chat 方法)
        :param memory_manager: 记忆管理器
        """
        self.llm = llm_client
        self.memory = memory_manager
        # 规划调用预算：避免单步多次长调用耗尽任务总时限。
        self._plan_primary_timeout_sec = 18
        self._plan_retry_timeout_sec = 10
        self._plan_retry_min_remaining_sec = 14
        self._plan_min_remaining_sec = 5
        logger.info("Planner 初始化完成")

    def plan(self, task: str, ui_state: UIState) -> PlanResult:
        """
        生成下一个子目标
        :param task: 最终任务描述
        :param ui_state: 当前 UI 状态
        :return: PlanResult
        """
        logger.info("开始规划, 任务: '%s'", task[:80])

        # 第一步：尝试从长期记忆检索经验
        experience = self.memory.search_experience(task)
        if experience:
            logger.info("命中长期记忆经验, 将复用历史动作链路")
            replay_plan = self._plan_from_experience(experience)
            if replay_plan is not None:
                return replay_plan

        # 第二步：通过 LLM 动态规划
        return self._plan_with_llm(task, ui_state)

    def _plan_from_experience(self, experience) -> Optional[PlanResult]:
        """从历史经验中复用动作"""
        if not experience.action_sequence:
            logger.warning("经验记录中动作序列为空, 回退到 LLM 规划")
            return None

        progress_idx = self._experience_progress_index()
        if progress_idx >= len(experience.action_sequence):
            logger.info(
                "经验动作已回放完毕(progress=%d, total=%d), 回退 LLM 动态规划",
                progress_idx,
                len(experience.action_sequence),
            )
            return None

        action = experience.action_sequence[progress_idx]
        target_anchor = action.get("target_anchor", {}) or {}
        target_widget_text = (
            action.get("target_widget_text", "")
            or target_anchor.get("widget_text", "")
            or target_anchor.get("content_desc", "")
            or target_anchor.get("resource_id", "")
            or action.get("target_content_desc", "")
            or action.get("target_resource_id", "")
        )
        subgoal = SubGoal(
            description=action.get("description", "来自经验的动作"),
            target_widget_text=target_widget_text,
            target_widget_id=action.get("target_widget_id", action.get("widget_id")),
            action_type=action.get("action_type", "tap"),
            input_text=action.get("input_text", ""),
            acceptance_criteria=action.get("acceptance_criteria", ""),
            expected_transition=action.get("expected_transition", "partial_refresh"),
            from_experience=True,
        )
        logger.info(
            "从经验复用子目标(progress=%d/%d): '%s'",
            progress_idx + 1,
            len(experience.action_sequence),
            subgoal.description,
        )

        return PlanResult(
            subgoal=subgoal,
            reasoning=f"复用长期记忆中的成功经验, progress={progress_idx + 1}",
        )

    def _experience_progress_index(self) -> int:
        """根据当前任务内已成功复用步数计算经验回放位置。"""
        progress = 0
        for step in self.memory.short_term.history:
            result = (step.get("result") or "").lower()
            if step.get("from_experience") and result.startswith("success"):
                progress += 1
        return progress

    def _plan_with_llm(self, task: str, ui_state: UIState) -> PlanResult:
        """通过 LLM 生成子目标"""
        history_text = self.memory.short_term.get_context_summary()
        ui_text = ui_state.to_prompt_text()
        remaining = self._remaining_task_seconds()
        if remaining is not None and remaining < self._plan_min_remaining_sec:
            raise TimeoutError(
                f"规划阶段剩余预算不足: remaining={remaining}s < {self._plan_min_remaining_sec}s"
            )

        prompt = PLANNER_PROMPT.format(
            task=task,
            ui_state=ui_text,
            history=history_text or "暂无历史记录",
        )

        logger.debug("发送规划请求到 LLM")
        try:
            primary_timeout = self._select_llm_timeout(
                default_timeout=self._plan_primary_timeout_sec,
                reserve_seconds=8,
            )
            response = self.llm.chat(prompt, timeout=primary_timeout)
            result = self._parse_plan_response(response)
            used_retry = False
            if self._is_redundant_subgoal(result.subgoal):
                if self._can_afford_retry():
                    logger.info("检测到重复子目标，触发一次去重重规划")
                    dedupe_prompt = self._build_dedupe_prompt(prompt)
                    retry_timeout = self._select_llm_timeout(
                        default_timeout=self._plan_retry_timeout_sec,
                        reserve_seconds=6,
                    )
                    retry_response = self.llm.chat(dedupe_prompt, timeout=retry_timeout)
                    retry_result = self._parse_plan_response(retry_response)
                    if not self._is_redundant_subgoal(retry_result.subgoal):
                        result = retry_result
                    else:
                        logger.info("去重重规划仍重复，保留原始规划结果")
                    used_retry = True
                else:
                    logger.info("检测到重复子目标，但剩余时限不足，跳过去重重规划")
            misaligned, reason = self._is_subgoal_misaligned(task, result.subgoal)
            if misaligned and not used_retry:
                if self._can_afford_retry():
                    logger.info("检测到任务主题偏移，触发一次对齐重规划: %s", reason)
                    align_prompt = self._build_alignment_prompt(prompt, task, reason)
                    align_timeout = self._select_llm_timeout(
                        default_timeout=self._plan_retry_timeout_sec,
                        reserve_seconds=6,
                    )
                    align_response = self.llm.chat(align_prompt, timeout=align_timeout)
                    align_result = self._parse_plan_response(align_response)
                    misaligned_retry, _ = self._is_subgoal_misaligned(task, align_result.subgoal)
                    if not misaligned_retry:
                        result = align_result
                    else:
                        logger.info("主题对齐重规划仍偏移，保留原始规划结果")
                else:
                    logger.info("检测到任务主题偏移，但剩余时限不足，使用任务保护子目标: %s", reason)
                    guard_plan = self._build_topic_guard_plan(task=task, reason=reason)
                    if guard_plan is not None:
                        return guard_plan
            logger.info(
                "LLM 规划完成: subgoal='%s', action=%s, complete=%s",
                result.subgoal.description, result.subgoal.action_type, result.is_task_complete,
            )
            return result
        except Exception as e:
            logger.error("LLM 规划失败: %s", e)
            return PlanResult(
            subgoal=SubGoal(description="LLM规划失败，需要重试"),
            reasoning=f"LLM调用异常: {e}",
        )

    def _remaining_task_seconds(self) -> Optional[int]:
        getter = getattr(self.llm, "remaining_seconds", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:
            return None
        if value is None:
            return None
        try:
            remaining = int(value)
        except Exception:
            return None
        return max(0, remaining)

    def _select_llm_timeout(self, default_timeout: int, reserve_seconds: int = 6) -> int:
        timeout = max(1, int(default_timeout or 1))
        remaining = self._remaining_task_seconds()
        if remaining is None:
            return timeout
        budget = max(1, remaining - max(0, int(reserve_seconds or 0)))
        return max(1, min(timeout, budget))

    def _can_afford_retry(self) -> bool:
        remaining = self._remaining_task_seconds()
        if remaining is None:
            return True
        return remaining >= self._plan_retry_min_remaining_sec

    def _build_topic_guard_plan(self, task: str, reason: str) -> Optional[PlanResult]:
        if not str(task or "").strip():
            return None

        return PlanResult(
            subgoal=SubGoal(
                description="Use Back once to recover from off-topic navigation.",
                action_type="back",
                acceptance_criteria="Return to previous relevant screen.",
                expected_transition="new_page",
            ),
            reasoning=f"task_guard({reason})",
        )

    def _build_dedupe_prompt(self, base_prompt: str) -> str:
        recent_success = self._recent_success_subgoals(limit=3)
        recent_failure = self._recent_failure_subgoals(limit=3)
        return (
            f"{base_prompt}\n\n"
            "Additional hard constraints:\n"
            f"- Recent successful subgoals: {json.dumps(recent_success, ensure_ascii=False)}\n"
            f"- Recent failed subgoals: {json.dumps(recent_failure, ensure_ascii=False)}\n"
            "- The next subgoal MUST be different from recent successful subgoals.\n"
            "- Do not repeat the same field input or same Navigate up/Back style action unless the previous attempt clearly failed.\n"
            "- Prefer the next unmet requirement toward final task completion.\n"
        )

    def _recent_success_subgoals(self, limit: int = 3) -> List[str]:
        values: List[str] = []
        history = list(getattr(self.memory.short_term, "history", []) or [])
        for step in reversed(history):
            result = str(step.get("result") or "").lower()
            if not result.startswith("success"):
                continue
            subgoal = str(step.get("subgoal") or "").strip()
            if not subgoal:
                continue
            values.append(subgoal)
            if len(values) >= limit:
                break
        return values

    def _recent_failure_subgoals(self, limit: int = 3) -> List[str]:
        values: List[str] = []
        history = list(getattr(self.memory.short_term, "history", []) or [])
        for step in reversed(history):
            result = str(step.get("result") or "").lower()
            if not result.startswith("failed"):
                continue
            subgoal = str(step.get("subgoal") or "").strip()
            if not subgoal:
                continue
            values.append(subgoal)
            if len(values) >= limit:
                break
        return values

    def _normalize_subgoal(self, text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return ""
        value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value[:160]

    def _is_redundant_subgoal(self, subgoal: SubGoal) -> bool:
        candidate_text = self._normalize_subgoal(subgoal.description)
        candidate_action = str(subgoal.action_type or "").strip().lower()
        if not candidate_text:
            return False

        history = list(getattr(self.memory.short_term, "history", []) or [])
        success_steps = [
            step for step in reversed(history)
            if str(step.get("result") or "").lower().startswith("success")
        ][:3]
        if not success_steps:
            return False

        for step in success_steps:
            previous_text = self._normalize_subgoal(step.get("subgoal") or "")
            previous_action = str((step.get("action") or {}).get("action_type") or "").strip().lower()
            if not previous_text:
                continue
            same_action = not candidate_action or not previous_action or candidate_action == previous_action
            text_overlap = (
                candidate_text == previous_text
                or candidate_text in previous_text
                or previous_text in candidate_text
            )
            if same_action and text_overlap:
                return True
        return False

    def _build_alignment_prompt(self, base_prompt: str, task: str, reason: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "Additional hard constraints:\n"
            f"- Final task: {task}\n"
            f"- Previous candidate was rejected for topic drift: {reason}\n"
            "- The next subgoal MUST stay strictly on the final task topic.\n"
            "- Do not touch unrelated features.\n"
        )

    def _is_subgoal_misaligned(self, task: str, subgoal: SubGoal) -> tuple:
        task_text = self._normalize_subgoal(task)
        subgoal_text = self._normalize_subgoal(subgoal.description)
        if not task_text or not subgoal_text:
            return False, ""

        task_keywords = self._extract_keywords(task_text)
        subgoal_keywords = self._extract_keywords(subgoal_text)
        if len(task_keywords) < 2 or len(subgoal_keywords) < 2:
            return False, ""

        overlap = task_keywords & subgoal_keywords
        if overlap:
            return False, ""

        # 当关键词完全无交集时，认为可能偏题，触发一次对齐重规划。
        if len(task_keywords) >= 3:
            return True, "topic_keywords_no_overlap"
        return False, ""

    def _extract_keywords(self, text: str) -> set:
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(text or "").lower())
        stopwords = {
            "the",
            "a",
            "an",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "for",
            "with",
            "at",
            "from",
            "by",
            "is",
            "are",
            "be",
            "do",
            "does",
            "did",
            "tap",
            "click",
            "press",
            "open",
            "go",
            "enter",
            "into",
            "page",
            "screen",
            "应用",
            "页面",
            "点击",
            "打开",
            "进入",
            "返回",
            "然后",
            "继续",
        }
        return {token for token in tokens if len(token) > 1 and token not in stopwords}

    def _parse_plan_response(self, response: str) -> PlanResult:
        """解析 LLM 返回的 JSON 规划结果"""
        # 提取 JSON 块
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.warning("JSON 解析失败: %s, 原始响应: %s", e, response[:200])
            return PlanResult(
                subgoal=SubGoal(description=response[:100]),
                reasoning="JSON解析失败，使用原始响应",
            )
        if not isinstance(data, dict):
            logger.warning("LLM 规划返回非对象 JSON，回退默认子目标: %r", data)
            return PlanResult(
                subgoal=SubGoal(description="LLM规划返回无效结构，需要重试"),
                reasoning="JSON结构无效",
            )

        sg_data = data.get("subgoal", {})
        if not isinstance(sg_data, dict):
            sg_data = {}
        subgoal = SubGoal(
            description=sg_data.get("description", ""),
            target_widget_text=sg_data.get("target_widget_text", ""),
            target_widget_id=sg_data.get("target_widget_id"),
            action_type=sg_data.get("action_type", "tap"),
            input_text=sg_data.get("input_text", ""),
            acceptance_criteria=sg_data.get("acceptance_criteria", ""),
            expected_transition=sg_data.get("expected_transition", "partial_refresh"),
        )
        subgoal.action_type = self._normalize_action_type_from_text(subgoal)

        return PlanResult(
            subgoal=subgoal,
            is_task_complete=data.get("is_task_complete", False),
            reasoning=data.get("reasoning", ""),
        )

    def _normalize_action_type_from_text(self, subgoal: SubGoal) -> str:
        raw_action = str(getattr(subgoal, "action_type", "") or "").strip().lower()
        if raw_action in {"longpress", "long-press", "press_hold", "press-and-hold"}:
            return "long_press"
        if raw_action in {"scrollup", "scroll-up"}:
            return "scroll_up"
        if raw_action in {"scrolldown", "scroll-down"}:
            return "scroll_down"
        if raw_action not in {"tap", "input", "swipe", "scroll", "scroll_up", "scroll_down", "back", "enter", "long_press"}:
            raw_action = "tap"

        text = " ".join(
            [
                getattr(subgoal, "description", "") or "",
                getattr(subgoal, "acceptance_criteria", "") or "",
                getattr(subgoal, "target_widget_text", "") or "",
            ]
        ).lower()
        if raw_action == "tap":
            if self._infer_long_press_action(text):
                return "long_press"
            motion_action = self._infer_motion_action(text)
            if motion_action:
                return motion_action
        return raw_action

    def _infer_long_press_action(self, text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        markers = (
            "long press",
            "long-press",
            "press and hold",
            "press & hold",
            "touch and hold",
            "hold down",
            "长按",
            "按住",
        )
        return any(marker in value for marker in markers)

    def _infer_motion_action(self, text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return ""
        if "swipe" in value or "下拉" in value:
            return "swipe"
        if (
            "scroll down" in value
            or "向下滚" in value
            or "下滑" in value
            or "往下滑" in value
            or "往下滚" in value
        ):
            return "scroll_down"
        if (
            "scroll up" in value
            or "向上滚" in value
            or "上滑" in value
            or "往上滑" in value
            or "往上滚" in value
        ):
            return "scroll_up"
        if "scroll" in value:
            return "scroll"
        return ""
