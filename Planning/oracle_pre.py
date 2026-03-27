"""
事前 Oracle 模块
为子目标生成更通用的状态型验收标准和跳变预期
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from Perception.context_builder import UIState
from Planning.planner import SubGoal
from prompt.oracle_pre_prompt import ORACLE_PRE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PreConstraints:
    """事前约束数据（通用状态语义版）"""

    # 页面状态语义
    source_state_type: str = ""   # list/detail/form/dialog/search/menu/tab/selection/unknown
    target_state_type: str = ""   # list/detail/form/dialog/search/menu/tab/selection/unknown

    # 跳变预期
    transition_type: str = "partial_refresh"   # partial_refresh/new_page/dialog/external_app/none
    allowed_transition_types: List[str] = field(default_factory=list)
    transition_description: str = ""

    # 环境边界
    must_stay_in_app: bool = False
    expected_package: str = ""
    forbidden_packages: List[str] = field(default_factory=list)
    expected_activity: str = ""

    # UI变化要求
    require_ui_change: bool = True
    trigger_may_disappear: bool = True

    # 强约束：必须出现/消失的控件特征
    widget_should_exist: List[str] = field(default_factory=list)
    widget_should_vanish: List[str] = field(default_factory=list)

    # 弱证据：辅助命中，不是必须全部满足
    supporting_texts: List[str] = field(default_factory=list)
    supporting_widgets: List[str] = field(default_factory=list)
    min_support_score: float = 1.0

    # 高阶语义约束（由 LLM 在事后评估时判断）
    semantic_criteria: str = ""


class OraclePre:
    """
    事前 Oracle
    在动作执行前，为子目标生成可验证的通用约束条件
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("OraclePre 初始化完成")

    def generate_constraints(self, subgoal: SubGoal, ui_state: UIState) -> PreConstraints:
        """
        为子目标生成事前约束
        :param subgoal: 当前子目标
        :param ui_state: 当前 UI 状态
        :return: 事前约束数据
        """
        logger.info("生成事前约束: subgoal='%s'", subgoal.description)

        # 如果子目标自带约束信息（来自经验），直接使用简化版本
        if subgoal.from_experience and subgoal.acceptance_criteria:
            logger.info("使用经验中的约束信息")
            return PreConstraints(
                transition_type=subgoal.expected_transition or "partial_refresh",
                allowed_transition_types=[subgoal.expected_transition] if subgoal.expected_transition else ["partial_refresh"],
                semantic_criteria=subgoal.acceptance_criteria,
            )

        fast_constraints = self._build_fast_constraints(subgoal, ui_state)
        if fast_constraints is not None:
            logger.info("使用快速约束路径: action=%s", subgoal.action_type)
            return fast_constraints

        return self._generate_with_llm(subgoal, ui_state)

    def _build_fast_constraints(self, subgoal: SubGoal, ui_state: UIState) -> Optional[PreConstraints]:
        """
        通用快速路径：当 Planner 已提供可用语义时，直接构建约束，减少额外 LLM 调用。
        """
        action_type = (subgoal.action_type or "").strip().lower()
        if action_type not in {"tap", "input", "swipe", "scroll_up", "scroll_down", "back", "enter", "long_press"}:
            return None

        semantic = (subgoal.acceptance_criteria or subgoal.description or "").strip()
        if not semantic:
            return None

        transition = (subgoal.expected_transition or "").strip() or self._default_transition(action_type)
        target_state = "form" if action_type == "input" else "unknown"
        source_state = self._infer_source_state(ui_state)
        must_stay_in_app = transition != "external_app"
        expected_package = ui_state.package_name if must_stay_in_app else ""

        supporting_widgets: List[str] = []
        widget_hint = (subgoal.target_widget_text or "").strip()
        if widget_hint:
            supporting_widgets.append(widget_hint[:30])

        return PreConstraints(
            source_state_type=source_state,
            target_state_type=target_state,
            transition_type=transition,
            allowed_transition_types=[transition],
            transition_description=f"fast_path_by_action:{action_type}",
            must_stay_in_app=must_stay_in_app,
            expected_package=expected_package,
            forbidden_packages=[],
            expected_activity="",
            require_ui_change=action_type not in {"noop"},
            trigger_may_disappear=True,
            widget_should_exist=[],
            widget_should_vanish=[],
            supporting_texts=self._extract_supporting_texts(subgoal.input_text),
            supporting_widgets=supporting_widgets,
            min_support_score=0.5,
            semantic_criteria=semantic,
        )

    def _default_transition(self, action_type: str) -> str:
        if action_type == "back":
            return "new_page"
        if action_type in {"swipe", "scroll_up", "scroll_down"}:
            return "partial_refresh"
        if action_type == "input":
            return "partial_refresh"
        return "partial_refresh"

    def _infer_source_state(self, ui_state: UIState) -> str:
        editable_count = sum(1 for w in ui_state.widgets if getattr(w, "editable", False))
        scrollable_count = sum(1 for w in ui_state.widgets if getattr(w, "scrollable", False))
        if editable_count >= 1:
            return "form"
        if scrollable_count >= 1:
            return "list"
        return "unknown"

    def _extract_supporting_texts(self, input_text: str) -> List[str]:
        text = (input_text or "").strip()
        if not text:
            return []

        candidates: List[str] = []
        lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
        for line in lines[:2]:
            candidates.append(line[:30])

        for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            candidates.append(email)

        deduped: List[str] = []
        seen = set()
        for value in candidates:
            key = value.lower()
            if not value or key in seen:
                continue
            deduped.append(value)
            seen.add(key)
        return deduped[:3]

    def _generate_with_llm(self, subgoal: SubGoal, ui_state: UIState) -> PreConstraints:
        """通过 LLM 生成约束"""
        prompt = ORACLE_PRE_PROMPT.format(
            description=subgoal.description,
            action_type=subgoal.action_type,
            target_widget=subgoal.target_widget_text or "无",
            input_text=subgoal.input_text or "无",
            ui_summary=ui_state.to_prompt_text()[:1800],
        )

        try:
            response = self.llm.chat(prompt)
            constraints = self._parse_response(response)

            # 合法化 allowed_transition_types
            if not constraints.allowed_transition_types:
                constraints.allowed_transition_types = [constraints.transition_type]

            logger.info(
                "事前约束生成完成: target_state=%s, transition=%s",
                constraints.target_state_type,
                constraints.transition_type,
            )
            return constraints

        except Exception as e:
            logger.error("事前约束生成失败: %s, 使用默认约束", e)
            return PreConstraints(
                transition_type=subgoal.expected_transition or "partial_refresh",
                allowed_transition_types=[subgoal.expected_transition] if subgoal.expected_transition else ["partial_refresh"],
                target_state_type="unknown",
                semantic_criteria=subgoal.acceptance_criteria or subgoal.description,
            )

    def _parse_response(self, response: str) -> PreConstraints:
        """解析 LLM 返回的约束 JSON"""
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.warning("约束 JSON 解析失败: %s", e)
            return PreConstraints(
                target_state_type="unknown",
                semantic_criteria=response[:300]
            )

        constraints = PreConstraints(
            source_state_type=data.get("source_state_type", ""),
            target_state_type=data.get("target_state_type", "unknown"),

            transition_type=data.get("transition_type", "partial_refresh"),
            allowed_transition_types=data.get("allowed_transition_types", []),
            transition_description=data.get("transition_description", ""),

            must_stay_in_app=data.get("must_stay_in_app", False),
            expected_package=data.get("expected_package", ""),
            forbidden_packages=data.get("forbidden_packages", []),
            expected_activity=data.get("expected_activity", ""),

            require_ui_change=data.get("require_ui_change", True),
            trigger_may_disappear=data.get("trigger_may_disappear", True),

            widget_should_exist=data.get("widget_should_exist", []),
            widget_should_vanish=data.get("widget_should_vanish", []),

            supporting_texts=data.get("supporting_texts", []),
            supporting_widgets=data.get("supporting_widgets", []),
            min_support_score=float(data.get("min_support_score", 1.0)),

            semantic_criteria=data.get("semantic_criteria", ""),
        )
        return constraints
