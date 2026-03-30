"""
事前 Oracle 模块
为子目标生成更通用的状态型验收标准和跳变预期
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from Perception.context_builder import UIState
from Planning.planner import SubGoal
from prompt.oracle_pre_prompt import ORACLE_PRE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PreConstraints:
    """事前约束数据（状态变化证据版）"""

    action_anchor: Dict[str, Any] = field(default_factory=dict)
    success_evidence_plan: Dict[str, Any] = field(default_factory=dict)
    boundary_constraints: Dict[str, Any] = field(default_factory=dict)
    semantic_goal: str = ""


class OraclePre:
    """
    事前 Oracle
    在动作执行前，为子目标生成可验证的通用约束条件
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        logger.info("OraclePre 初始化完成")

    def generate_constraints(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        task_hint: str = "",
    ) -> PreConstraints:
        """
        为子目标生成事前约束
        :param subgoal: 当前子目标
        :param ui_state: 当前 UI 状态
        :param task_hint: 任务级语义提示（用于跨步骤保持目标包约束）
        :return: 事前约束数据
        """
        logger.info("生成事前约束: subgoal='%s'", subgoal.description)

        semantic_goal = (subgoal.acceptance_criteria or subgoal.description or "").strip()
        if not semantic_goal:
            semantic_goal = subgoal.description.strip()

        fast_constraints = self._build_fast_constraints(
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task_hint,
        )
        if fast_constraints is not None:
            logger.info("使用快速约束路径: action=%s", subgoal.action_type)
            if semantic_goal and not fast_constraints.semantic_goal:
                fast_constraints.semantic_goal = semantic_goal
            return self._stabilize_boundary_constraints(
                constraints=fast_constraints,
                subgoal=subgoal,
                ui_state=ui_state,
                task_hint=task_hint,
            )

        llm_constraints = self._generate_with_llm(
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task_hint,
        )
        return self._stabilize_boundary_constraints(
            constraints=llm_constraints,
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task_hint,
        )

    def _build_fast_constraints(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        task_hint: str = "",
    ) -> Optional[PreConstraints]:
        """
        通用快速路径：当 Planner 已提供可用语义时，直接构建约束，减少额外 LLM 调用。
        """
        action_type = (subgoal.action_type or "").strip().lower()
        if action_type not in {"tap", "input", "swipe", "scroll", "scroll_up", "scroll_down", "back", "enter", "long_press"}:
            return None

        semantic_goal = (subgoal.acceptance_criteria or subgoal.description or "").strip()
        if not semantic_goal:
            return None

        boundary_defaults = self._infer_boundary_defaults(
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task_hint,
        )
        must_stay_in_app = boundary_defaults["must_stay_in_app"]
        expected_package = boundary_defaults["expected_package"]
        package_mismatch_severity = boundary_defaults["package_mismatch_severity"]
        related_package_tokens = boundary_defaults["related_package_tokens"]
        current_pkg = str(getattr(ui_state, "package_name", "") or "").strip()

        action_anchor = self._build_action_anchor(subgoal, ui_state)
        evidence_plan = self._build_fast_evidence_plan(
            action_type=action_type,
            must_stay_in_app=must_stay_in_app,
            input_text=subgoal.input_text or "",
            semantic_goal=semantic_goal,
            target_widget_text=subgoal.target_widget_text or "",
            expected_package=expected_package,
            current_package=current_pkg,
        )
        boundary_constraints = {
            "must_stay_in_app": must_stay_in_app,
            "expected_package": expected_package,
            "forbidden_packages": [],
            "expected_activity_contains": "",
            "forbidden_ui_risks": [],
            "package_mismatch_severity": package_mismatch_severity,
            "related_package_tokens": related_package_tokens,
        }

        return PreConstraints(
            action_anchor=action_anchor,
            success_evidence_plan=evidence_plan,
            boundary_constraints=boundary_constraints,
            semantic_goal=semantic_goal,
        )

    def _build_fast_evidence_plan(
        self,
        action_type: str,
        must_stay_in_app: bool,
        input_text: str = "",
        semantic_goal: str = "",
        target_widget_text: str = "",
        expected_package: str = "",
        current_package: str = "",
    ) -> Dict[str, Any]:
        def _signal(
            type_: str,
            operator: str,
            scope: str,
            target: Optional[Dict[str, Any]] = None,
            value: str = "",
            weight: float = 1.0,
            optional: bool = False,
        ) -> Dict[str, Any]:
            return {
                "type": type_,
                "target": target or {},
                "operator": operator,
                "value": value,
                "scope": scope,
                "weight": float(weight),
                "optional": bool(optional),
            }

        required: List[Dict[str, Any]] = []
        supporting: List[Dict[str, Any]] = []
        counter: List[Dict[str, Any]] = []
        semantic_text = " ".join([semantic_goal or "", target_widget_text or ""]).lower()
        expected_pkg = str(expected_package or "").strip().lower()
        current_pkg = str(current_package or "").strip().lower()
        context_switch_expected = bool(expected_pkg and (not current_pkg or current_pkg != expected_pkg))
        focus_intent_markers = (
            "focus",
            "cursor",
            "input field",
            "search box",
            "search bar",
            "keyboard",
            "聚焦",
            "输入框",
            "搜索框",
            "光标",
            "键盘",
        )
        focus_intent = any(marker in semantic_text for marker in focus_intent_markers)

        if must_stay_in_app:
            counter.append(_signal(type_="package_changed", operator="changed", scope="global", weight=1.0))

        if action_type in {"scroll_up", "scroll_down", "swipe", "scroll"}:
            expected_scope = "local"
            expected_nature = ["visibility", "content"]
            required = [
                _signal(type_="region_changed", operator="changed", scope="local", weight=1.0),
                _signal(type_="overlay_changed", operator="changed", scope="global", weight=0.3, optional=True),
            ]
        elif action_type == "back":
            expected_scope = "global"
            expected_nature = ["navigation"]
            required = [
                _signal(type_="activity_changed", operator="changed", scope="global", weight=1.0),
            ]
        elif action_type == "input":
            expected_scope = "local"
            expected_nature = ["content", "visibility"]
            input_hint = str(input_text or "").strip()[:40]
            required = [
                _signal(type_="region_changed", operator="changed", scope="local", weight=1.0),
            ]
            if input_hint:
                required.append(
                    _signal(
                        type_="text_appeared",
                        operator="contains",
                        scope="global",
                        target={"text": input_hint},
                        value=input_hint,
                        weight=0.9,
                        optional=False,
                    )
                )
            supporting = [
                _signal(type_="focus_changed", operator="changed", scope="global", target={"class": "EditText"}, weight=0.6, optional=True),
                _signal(type_="keyboard_changed", operator="changed", scope="global", weight=0.5, optional=True),
            ]
        else:
            expected_scope = "local"
            expected_nature = ["focus", "visibility"] if focus_intent else ["content"]
            if focus_intent:
                required = [
                    _signal(type_="focus_changed", operator="changed", scope="global", weight=1.0, optional=False),
                ]
                supporting = [
                    _signal(type_="keyboard_changed", operator="changed", scope="global", weight=0.7, optional=True),
                    _signal(type_="region_changed", operator="changed", scope="local", weight=0.5, optional=True),
                ]
            else:
                required = [
                    _signal(type_="region_changed", operator="changed", scope="local", weight=1.0, optional=False),
                    _signal(type_="activity_changed", operator="changed", scope="global", weight=1.0, optional=True),
                    _signal(type_="overlay_changed", operator="changed", scope="global", weight=0.7, optional=True),
                ]
                text_hint = str(target_widget_text or "").strip()[:40]
                if text_hint:
                    required.append(
                        _signal(
                            type_="text_disappeared",
                            operator="contains",
                            scope="global",
                            target={"text": text_hint},
                            value=text_hint,
                            weight=0.8,
                            optional=True,
                        )
                    )
                supporting = [
                    _signal(type_="focus_changed", operator="changed", scope="global", weight=0.4, optional=True),
                    _signal(type_="keyboard_changed", operator="changed", scope="global", weight=0.3, optional=True),
                ]

        if context_switch_expected:
            # 预期上下文切换时，强制把“包名变化”纳入必备证据，避免仅靠局部 UI 变化误判成功。
            expected_scope = "global"
            expected_nature = list(dict.fromkeys([*expected_nature, "navigation", "app_switch"]))
            required.insert(0, _signal(type_="package_changed", operator="changed", scope="global", weight=1.2, optional=False))
            supporting.insert(0, _signal(type_="activity_changed", operator="changed", scope="global", weight=0.7, optional=True))
            supporting.append(_signal(type_="overlay_changed", operator="changed", scope="global", weight=0.4, optional=True))

        return {
            "expected_change_scope": expected_scope,
            "expected_change_nature": expected_nature,
            "required_signals_any_of": required[:4],
            "supporting_signals_any_of": supporting[:6],
            "counter_signals_any_of": counter[:6],
        }

    def _infer_boundary_defaults(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        task_hint: str = "",
    ) -> Dict[str, Any]:
        """
        边界默认策略（通用版）:
        - external_app/new_page/dialog：默认允许跨包
        - 其他场景：默认留在当前包
        """
        expected_transition = (subgoal.expected_transition or "").strip().lower()
        current_pkg = str(getattr(ui_state, "package_name", "") or "").strip()

        # new_page/dialog 常意味着上下文切换，默认允许跨包，避免把“有效跳转”误拦截。
        allow_cross_package = expected_transition in {"external_app", "new_page", "dialog"}
        must_stay_in_app = not allow_cross_package
        expected_package = current_pkg if (must_stay_in_app and current_pkg) else ""
        package_mismatch_severity = self._infer_package_mismatch_severity(
            subgoal=subgoal,
            must_stay_in_app=must_stay_in_app,
        )
        related_package_tokens = self._extract_package_tokens(expected_package or current_pkg)

        return {
            "must_stay_in_app": bool(must_stay_in_app),
            "expected_package": expected_package,
            "package_mismatch_severity": package_mismatch_severity,
            "related_package_tokens": related_package_tokens,
        }

    def _infer_package_mismatch_severity(
        self,
        subgoal: SubGoal,
        must_stay_in_app: bool,
    ) -> str:
        """
        包名漂移默认按 soft 处理，仅在显式“必须留在同一应用”语义时升级为 hard。
        """
        if not must_stay_in_app:
            return "soft"

        text = " ".join(
            [
                subgoal.description or "",
                subgoal.acceptance_criteria or "",
            ]
        ).lower()
        strict_markers = (
            "stay in app",
            "same app",
            "within app",
            "without leaving",
            "当前应用",
            "同一应用",
            "本应用",
            "不要离开",
            "不离开",
        )
        if any(marker in text for marker in strict_markers):
            return "hard"
        return "soft"

    def _extract_package_tokens(self, package_name: str) -> List[str]:
        """
        提取包名中可表示“同产品族”的 token，用于 related package 判定。
        """
        pkg = str(package_name or "").strip().lower()
        if not pkg:
            return []
        ignored = {
            "com",
            "org",
            "net",
            "android",
            "google",
            "apps",
            "app",
            "launcher",
            "activity",
        }
        tokens = []
        normalized = pkg.replace("-", ".").replace("_", ".")
        for token in normalized.split("."):
            for t in self._expand_package_token(token):
                if not t or t in ignored or len(t) <= 2:
                    continue
                if t not in tokens:
                    tokens.append(t)
        return tokens[:8]

    def _expand_package_token(self, token: str) -> List[str]:
        raw = str(token or "").strip().lower()
        if not raw:
            return []
        variants: List[str] = [raw]

        without_tail_digits = re.sub(r"\d+$", "", raw)
        if without_tail_digits and without_tail_digits != raw:
            variants.append(without_tail_digits)

        for prefix in ("google", "android"):
            if raw.startswith(prefix) and len(raw) > len(prefix) + 2:
                variants.append(raw[len(prefix):])

        alpha_parts = re.findall(r"[a-z]+", raw)
        if len(alpha_parts) >= 2:
            merged_alpha = "".join(alpha_parts)
            if merged_alpha:
                variants.append(merged_alpha)
            for part in alpha_parts:
                if len(part) > 2:
                    variants.append(part)

        deduped: List[str] = []
        for v in variants:
            value = str(v or "").strip().lower()
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _stabilize_boundary_constraints(
        self,
        constraints: PreConstraints,
        subgoal: SubGoal,
        ui_state: UIState,
        task_hint: str = "",
    ) -> PreConstraints:
        """
        统一边界约束的默认策略，避免模型输出或快速路径产生过度约束。
        """
        boundary = constraints.boundary_constraints or {}
        if not isinstance(boundary, dict):
            boundary = {}

        defaults = self._infer_boundary_defaults(
            subgoal=subgoal,
            ui_state=ui_state,
            task_hint=task_hint,
        )
        current_pkg = str(getattr(ui_state, "package_name", "") or "").strip()

        must_stay = defaults["must_stay_in_app"]
        if "must_stay_in_app" in boundary:
            try:
                must_stay = bool(boundary.get("must_stay_in_app"))
            except Exception:
                must_stay = defaults["must_stay_in_app"]

        # 当默认允许跨包时，不将 must_stay 覆盖为更严格策略。
        if not defaults["must_stay_in_app"]:
            must_stay = False

        expected_pkg = str(boundary.get("expected_package") or defaults.get("expected_package") or "").strip()
        if must_stay:
            if not expected_pkg:
                expected_pkg = current_pkg
        else:
            # 允许跨包时，保持模型或默认提供的 expected_package（若有）。
            expected_pkg = str(expected_pkg or "").strip()

        forbidden_packages = boundary.get("forbidden_packages") or []
        if not isinstance(forbidden_packages, list):
            forbidden_packages = []

        forbidden_ui_risks = boundary.get("forbidden_ui_risks") or []
        if not isinstance(forbidden_ui_risks, list):
            forbidden_ui_risks = []

        mismatch_severity = str(boundary.get("package_mismatch_severity") or defaults["package_mismatch_severity"]).strip().lower()
        if mismatch_severity not in {"soft", "hard"}:
            mismatch_severity = defaults["package_mismatch_severity"]

        related_tokens_raw = boundary.get("related_package_tokens")
        if not isinstance(related_tokens_raw, list):
            related_tokens_raw = defaults.get("related_package_tokens") or []
        related_tokens = []
        for token in related_tokens_raw:
            value = str(token or "").strip().lower()
            if value and value not in related_tokens:
                related_tokens.append(value)
        if not related_tokens:
            related_tokens = self._extract_package_tokens(expected_pkg or current_pkg)

        constraints.boundary_constraints = {
            "must_stay_in_app": bool(must_stay),
            "expected_package": expected_pkg,
            "forbidden_packages": [str(p).strip() for p in forbidden_packages if str(p).strip()],
            "expected_activity_contains": str(boundary.get("expected_activity_contains") or "").strip(),
            "forbidden_ui_risks": [str(v).strip() for v in forbidden_ui_risks if str(v).strip()],
            "package_mismatch_severity": mismatch_severity,
            "related_package_tokens": related_tokens[:8],
        }
        return constraints

    def _generate_with_llm(
        self,
        subgoal: SubGoal,
        ui_state: UIState,
        task_hint: str = "",
    ) -> PreConstraints:
        """通过 LLM 生成约束"""
        try:
            prompt = ORACLE_PRE_PROMPT.format(
                description=subgoal.description,
                action_type=subgoal.action_type,
                target_widget=subgoal.target_widget_text or "无",
                input_text=subgoal.input_text or "无",
                ui_summary=ui_state.to_prompt_text()[:1800],
            )
        except KeyError as e:
            logger.error("OraclePre prompt 模板缺少占位符: %s，回退快速约束", e)
            boundary_defaults = self._infer_boundary_defaults(
                subgoal=subgoal,
                ui_state=ui_state,
                task_hint=task_hint,
            )
            must_stay_in_app = boundary_defaults["must_stay_in_app"]
            current_pkg = str(getattr(ui_state, "package_name", "") or "").strip()
            return PreConstraints(
                action_anchor=self._build_action_anchor(subgoal, ui_state),
                success_evidence_plan=self._build_fast_evidence_plan(
                    action_type=(subgoal.action_type or "").strip().lower(),
                    must_stay_in_app=must_stay_in_app,
                    input_text=subgoal.input_text or "",
                    semantic_goal=(subgoal.acceptance_criteria or subgoal.description or "").strip(),
                    target_widget_text=subgoal.target_widget_text or "",
                    expected_package=boundary_defaults["expected_package"],
                    current_package=current_pkg,
                ),
                boundary_constraints={
                    "must_stay_in_app": must_stay_in_app,
                    "expected_package": boundary_defaults["expected_package"],
                    "forbidden_packages": [],
                    "expected_activity_contains": "",
                    "forbidden_ui_risks": [],
                    "package_mismatch_severity": boundary_defaults["package_mismatch_severity"],
                    "related_package_tokens": boundary_defaults["related_package_tokens"],
                },
                semantic_goal=(subgoal.acceptance_criteria or subgoal.description or "").strip(),
            )

        try:
            response = self.llm.chat(prompt)
            constraints = self._parse_response(response)
            logger.info("事前约束生成完成")
            return constraints

        except Exception as e:
            logger.error("事前约束生成失败: %s, 使用默认约束", e)
            boundary_defaults = self._infer_boundary_defaults(
                subgoal=subgoal,
                ui_state=ui_state,
                task_hint=task_hint,
            )
            must_stay_in_app = boundary_defaults["must_stay_in_app"]
            current_pkg = str(getattr(ui_state, "package_name", "") or "").strip()
            return PreConstraints(
                action_anchor=self._build_action_anchor(subgoal, ui_state),
                success_evidence_plan=self._build_fast_evidence_plan(
                    action_type=(subgoal.action_type or "").strip().lower(),
                    must_stay_in_app=must_stay_in_app,
                    input_text=subgoal.input_text or "",
                    semantic_goal=(subgoal.acceptance_criteria or subgoal.description or "").strip(),
                    target_widget_text=subgoal.target_widget_text or "",
                    expected_package=boundary_defaults["expected_package"],
                    current_package=current_pkg,
                ),
                boundary_constraints={
                    "must_stay_in_app": must_stay_in_app,
                    "expected_package": boundary_defaults["expected_package"],
                    "forbidden_packages": [],
                    "expected_activity_contains": "",
                    "forbidden_ui_risks": [],
                    "package_mismatch_severity": boundary_defaults["package_mismatch_severity"],
                    "related_package_tokens": boundary_defaults["related_package_tokens"],
                },
                semantic_goal=(subgoal.acceptance_criteria or subgoal.description or "").strip(),
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
            return self._default_constraints_from_text(response[:300])

        if not isinstance(data, dict):
            return self._default_constraints_from_text(str(response)[:300])

        return self._normalize_constraints(data)

    def _default_constraints_from_text(self, semantic_goal: str) -> PreConstraints:
        return PreConstraints(
            action_anchor={},
            success_evidence_plan={
                "expected_change_scope": "local",
                "expected_change_nature": ["content"],
                "required_signals_any_of": [],
                "supporting_signals_any_of": [],
                "counter_signals_any_of": [],
            },
            boundary_constraints={
                "must_stay_in_app": False,
                "expected_package": "",
                "forbidden_packages": [],
                "expected_activity_contains": "",
                "forbidden_ui_risks": [],
                "package_mismatch_severity": "soft",
                "related_package_tokens": [],
            },
            semantic_goal=str(semantic_goal or "").strip(),
        )

    def _normalize_constraints(self, data: Dict[str, Any]) -> PreConstraints:
        allowed_types = {
            "widget_appeared",
            "widget_disappeared",
            "text_appeared",
            "text_disappeared",
            "focus_changed",
            "keyboard_changed",
            "package_changed",
            "activity_changed",
            "region_changed",
            "overlay_changed",
            "risk_ui_detected",
        }
        allowed_ops = {
            "exists",
            "not_exists",
            "contains",
            "changed",
            "increased",
            "decreased",
            "appeared",
            "disappeared",
        }
        allowed_scopes = {"anchor", "local", "global"}

        action_anchor = data.get("action_anchor") or {}
        if not isinstance(action_anchor, dict):
            action_anchor = {}

        success_plan = data.get("success_evidence_plan") or {}
        if not isinstance(success_plan, dict):
            success_plan = {}

        boundary = data.get("boundary_constraints") or {}
        if not isinstance(boundary, dict):
            boundary = {}

        scope = str(success_plan.get("expected_change_scope") or "local").strip().lower()
        if scope not in allowed_scopes:
            scope = "local"

        nature = success_plan.get("expected_change_nature")
        if not isinstance(nature, list):
            nature = [str(nature)] if nature else []
        nature = [str(v).strip().lower() for v in nature if str(v).strip()]
        if not nature:
            nature = ["content"]

        def _norm_signal(item: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(item, dict):
                return None
            type_ = str(item.get("type") or "").strip()
            operator = str(item.get("operator") or "").strip()
            scope_ = str(item.get("scope") or "").strip().lower()
            if type_ not in allowed_types:
                return None
            if operator not in allowed_ops:
                return None
            if scope_ not in allowed_scopes:
                scope_ = "global"

            target = item.get("target") or {}
            if not isinstance(target, dict):
                target = {}

            value = str(item.get("value") or "")
            weight_raw = item.get("weight", 1.0)
            try:
                weight = float(weight_raw)
            except Exception:
                weight = 1.0
            if weight < 0.0:
                weight = 0.0
            if weight > 2.0:
                weight = 2.0

            optional = bool(item.get("optional", False))

            return {
                "type": type_,
                "target": target,
                "operator": operator,
                "value": value,
                "scope": scope_,
                "weight": weight,
                "optional": optional,
            }

        def _norm_list(key: str) -> List[Dict[str, Any]]:
            raw = success_plan.get(key) or []
            if not isinstance(raw, list):
                return []
            out = []
            for it in raw:
                normed = _norm_signal(it)
                if normed is not None:
                    out.append(normed)
            return out

        required = _norm_list("required_signals_any_of")
        supporting = _norm_list("supporting_signals_any_of")
        counter = _norm_list("counter_signals_any_of")

        semantic_goal = str(data.get("semantic_goal") or "").strip()

        normalized_plan = {
            "expected_change_scope": scope,
            "expected_change_nature": nature,
            "required_signals_any_of": required,
            "supporting_signals_any_of": supporting,
            "counter_signals_any_of": counter,
        }

        must_stay_in_app = bool(boundary.get("must_stay_in_app", False))
        expected_package = str(boundary.get("expected_package") or "").strip()
        forbidden_packages = boundary.get("forbidden_packages") or []
        if not isinstance(forbidden_packages, list):
            forbidden_packages = []
        forbidden_packages = [str(p) for p in forbidden_packages if str(p).strip()]

        expected_activity_contains = str(boundary.get("expected_activity_contains") or "").strip()
        forbidden_ui_risks = boundary.get("forbidden_ui_risks") or []
        if not isinstance(forbidden_ui_risks, list):
            forbidden_ui_risks = []
        forbidden_ui_risks = [str(v).strip() for v in forbidden_ui_risks if str(v).strip()]
        package_mismatch_severity = str(boundary.get("package_mismatch_severity") or "soft").strip().lower()
        if package_mismatch_severity not in {"soft", "hard"}:
            package_mismatch_severity = "soft"
        related_package_tokens = boundary.get("related_package_tokens") or []
        if not isinstance(related_package_tokens, list):
            related_package_tokens = []
        related_package_tokens = [str(v).strip().lower() for v in related_package_tokens if str(v).strip()]
        if not related_package_tokens:
            related_package_tokens = self._extract_package_tokens(expected_package)

        normalized_boundary = {
            "must_stay_in_app": must_stay_in_app,
            "expected_package": expected_package,
            "forbidden_packages": forbidden_packages,
            "expected_activity_contains": expected_activity_contains,
            "forbidden_ui_risks": forbidden_ui_risks,
            "package_mismatch_severity": package_mismatch_severity,
            "related_package_tokens": related_package_tokens[:8],
        }

        return PreConstraints(
            action_anchor=action_anchor,
            success_evidence_plan=normalized_plan,
            boundary_constraints=normalized_boundary,
            semantic_goal=semantic_goal,
        )

    def _build_action_anchor(self, subgoal: SubGoal, ui_state: UIState) -> Dict[str, Any]:
        widget = None
        if subgoal.target_widget_id is not None:
            widget = ui_state.find_widget_by_id(int(subgoal.target_widget_id))
        if widget is None and subgoal.target_widget_text:
            widget = ui_state.find_widget_by_text(subgoal.target_widget_text)

        features: Dict[str, Any] = {
            "text": (subgoal.target_widget_text or "")[:40],
            "class": "",
            "resource_id": "",
            "content_desc": "",
            "clickable": False,
        }
        if widget is not None:
            features.update(
                {
                    "text": (widget.text or "")[:60],
                    "class": (widget.class_name or "")[:80],
                    "resource_id": (widget.resource_id or "")[:120],
                    "content_desc": (widget.content_desc or "")[:60],
                    "clickable": bool(getattr(widget, "clickable", False)),
                }
            )

        return {
            "target_widget_id": int(subgoal.target_widget_id) if subgoal.target_widget_id is not None else -1,
            "target_widget_features": features,
            "target_bounds_before": list(getattr(widget, "bounds", (0, 0, 0, 0)) or (0, 0, 0, 0)) if widget is not None else [0, 0, 0, 0],
            "target_center_before": list(getattr(widget, "center", (0, 0)) or (0, 0)) if widget is not None else [0, 0],
        }
