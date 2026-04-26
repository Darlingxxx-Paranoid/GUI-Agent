"""Pre-Oracle: semantic transition prediction + rule-based assertion mapping."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Planning.planner import PlanResult
from prompt.oracle_pre_prompt import ORACLE_PRE_SYSTEM_PROMPT, ORACLE_PRE_USER_PROMPT
from utils.audit_recorder import AuditRecorder
from utils.llm_client import LLMRequest

logger = logging.getLogger(__name__)

ContextMode = Literal["local", "global"]
TransitionType = Literal[
    "NavigationTransition",
    "NodeAppearance",
    "AttributeModification",
    "ContentUpdate",
    "ContainerExpansion",
]
AssertionCategory = Literal["widget", "activity", "package"]
AssertionRelation = Literal["exact_match", "contains", "is_true", "is_false"]


class SemanticHint(BaseModel):
    """High-level semantic hint from first-stage LLM prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=1)
    value: str = Field(default="")


class SemanticTransitionContract(BaseModel):
    """First-stage output: semantic transition prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    context_mode: ContextMode
    transition_type: TransitionType
    success_definition: str = Field(min_length=1)
    semantic_hints: list[SemanticHint] = Field(default_factory=list)


class UIAssertionTarget(BaseModel):
    """Resolved target used by assertion contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    node_id: Optional[int] = None
    resource_id: str = Field(default="")
    field: str = Field(default="")
    text: str = Field(default="")
    class_name: str = Field(default="")


class UIAssertion(BaseModel):
    """Second-stage assertion item."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: AssertionCategory
    target: Optional[UIAssertionTarget] = None
    relation: AssertionRelation
    content: str = Field(default="")

    @model_validator(mode="after")
    def validate_shape(self) -> "UIAssertion":
        if self.category == "widget":
            if self.target is None:
                raise ValueError("widget assertion requires target")
            if str(self.target.field or "").strip() == "":
                raise ValueError("widget assertion target.field must be non-empty")
            if (
                self.target.node_id is None
                and str(self.target.resource_id or "").strip() == ""
                and str(self.target.text or "").strip() == ""
                and str(self.target.class_name or "").strip() == ""
            ):
                raise ValueError("widget assertion target must contain locator info")
        else:
            if self.target is not None:
                raise ValueError("activity/package assertion target must be null")
            if self.relation not in {"exact_match", "contains"}:
                raise ValueError("activity/package assertion only supports exact_match|contains")

        if self.relation in {"exact_match", "contains"} and str(self.content or "").strip() == "":
            raise ValueError("content must be non-empty for exact_match/contains")
        return self


class UIAssertionContract(BaseModel):
    """Second-stage output: XML-checkable assertions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    context_mode: ContextMode
    transition_type: TransitionType
    success_definition: str = Field(min_length=1)
    assertions: list[UIAssertion] = Field(min_length=1)


class PreOracleOutput(BaseModel):
    """Unified output of Pre-Oracle two-stage pipeline."""

    model_config = ConfigDict(extra="forbid")

    semantic_contract: SemanticTransitionContract
    assertion_contract: UIAssertionContract


class OraclePre:
    """Generate semantic transition then map to rule-based UI assertions."""

    _LOCAL_ACTIONS = {"tap", "input", "long_press"}
    _BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.audit = AuditRecorder(component="oracle_pre")
        self._project_root = Path(__file__).resolve().parents[1]
        logger.info("OraclePre 初始化完成 (semantic -> assertion two-stage)")

    def generate_contract(
        self,
        plan: PlanResult,
        dump_tree: dict,
        screenshot_path: str,
        anchor_result: Any = None,
        widgets: list[dict[str, Any]] | None = None,
        step: int | None = None,
    ) -> PreOracleOutput:
        if self.llm is None:
            logger.error("Pre-Oracle 生成失败: llm_client 未初始化")
            raise RuntimeError("OraclePre 需要 llm_client 才能生成契约")
        if not isinstance(dump_tree, dict):
            logger.error("Pre-Oracle 生成失败: dump_tree 类型非法(type=%s)", type(dump_tree).__name__)
            raise ValueError("dump_tree 必须是 dict")

        node_index, rid_index = self._index_dump_tree(dump_tree)
        if not node_index:
            logger.error("Pre-Oracle 生成失败: dump_tree 中未找到 node_id")
            raise ValueError("dump_tree 中未找到任何 node_id")

        preferred_mode = self._preferred_context_mode(action_type=str(plan.action_type or ""))
        context_image_path, actual_mode, fallback_reason = self._select_context_image(
            screenshot_path=screenshot_path,
            preferred_mode=preferred_mode,
            anchor_result=anchor_result,
            widgets=list(widgets or []),
            step=step,
        )
        if fallback_reason:
            logger.info("Pre-Oracle 局部上下文降级为全局: %s", fallback_reason)

        plan_payload = plan.model_dump()
        plan_payload.pop("reasoning", None)
        context_payload = self._build_context_payload(
            dump_tree=dump_tree,
            actual_mode=actual_mode,
            fallback_reason=fallback_reason,
            anchor_result=anchor_result,
            widgets=list(widgets or []),
            context_image_path=context_image_path,
        )
        user_prompt = ORACLE_PRE_USER_PROMPT.format(
            plan_json=json.dumps(plan_payload, ensure_ascii=False),
            context_json=json.dumps(context_payload, ensure_ascii=False),
        )
        request = LLMRequest(
            system=ORACLE_PRE_SYSTEM_PROMPT,
            user=user_prompt,
            images=[context_image_path],
            response_format=SemanticTransitionContract,
            audit_meta={
                "artifact_kind": "SemanticTransitionContract",
                "step": step,
                "stage": "pre_oracle_semantic",
            },
        )

        logger.info(
            "开始生成 SemanticTransitionContract: action=%s, context=%s, dump_nodes=%d",
            plan.action_type,
            actual_mode,
            len(node_index),
        )
        try:
            parsed = self.llm.chat(request)
            semantic_contract = (
                parsed
                if isinstance(parsed, SemanticTransitionContract)
                else SemanticTransitionContract.model_validate(parsed)
            )
            if semantic_contract.context_mode != actual_mode:
                logger.warning(
                    "SemanticTransitionContract context_mode=%s 与实际上下文=%s 不一致，已强制覆盖",
                    semantic_contract.context_mode,
                    actual_mode,
                )
                semantic_contract = semantic_contract.model_copy(update={"context_mode": actual_mode})

            assertion_contract = self._map_semantic_to_assertions(
                semantic_contract=semantic_contract,
                plan=plan,
                node_index=node_index,
                rid_index=rid_index,
                dump_tree=dump_tree,
                anchor_result=anchor_result,
                widgets=list(widgets or []),
            )
            output = PreOracleOutput(
                semantic_contract=semantic_contract,
                assertion_contract=assertion_contract,
            )
            self._record_semantic_artifact(step=step, payload=semantic_contract)
            self._record_assertion_artifact(step=step, payload=assertion_contract)
            logger.info(
                "Pre-Oracle 生成完成: transition=%s, assertions=%d",
                output.semantic_contract.transition_type,
                len(output.assertion_contract.assertions),
            )
            return output
        except Exception as exc:
            self._record_pre_error(step=step, error=str(exc))
            logger.error("Pre-Oracle 生成失败: %s", exc)
            raise

    def _preferred_context_mode(self, action_type: str) -> ContextMode:
        token = str(action_type or "").strip().lower()
        return "local" if token in self._LOCAL_ACTIONS else "global"

    def _select_context_image(
        self,
        screenshot_path: str,
        preferred_mode: ContextMode,
        anchor_result: Any,
        widgets: list[dict[str, Any]],
        step: int | None,
    ) -> tuple[str, ContextMode, str]:
        path = str(screenshot_path or "").strip()
        if preferred_mode != "local":
            return path, "global", ""

        anchor_widget = self._resolve_anchor_widget(anchor_result=anchor_result, widgets=widgets)
        if anchor_widget is None:
            return path, "global", "anchor_widget_not_found"
        bounds = self._widget_bounds(anchor_widget)
        if bounds is None:
            return path, "global", "anchor_bounds_invalid"

        try:
            local_path = self._crop_local_context_image(
                screenshot_path=path,
                bounds=bounds,
                step=step,
            )
            return local_path, "local", ""
        except Exception as exc:
            logger.warning("生成局部截图失败，降级全局: %s", exc)
            return path, "global", f"local_crop_failed:{exc}"

    def _build_context_payload(
        self,
        dump_tree: dict,
        actual_mode: ContextMode,
        fallback_reason: str,
        anchor_result: Any,
        widgets: list[dict[str, Any]],
        context_image_path: str,
    ) -> dict[str, Any]:
        activity, package = self._extract_app_context(dump_tree)
        anchor_widget = self._resolve_anchor_widget(anchor_result=anchor_result, widgets=widgets)
        payload: dict[str, Any] = {
            "context_mode": actual_mode,
            "context_image_path": str(context_image_path or ""),
            "fallback_reason": str(fallback_reason or ""),
            "current_package": package,
            "current_activity": activity,
        }
        if anchor_widget is not None:
            try:
                widget_id = int(anchor_widget.get("widget_id", -1))
            except Exception:
                widget_id = -1
            payload["anchor_widget"] = {
                "widget_id": widget_id,
                "text": str(anchor_widget.get("text") or ""),
                "class": str(anchor_widget.get("class") or ""),
                "bounds": self._widget_bounds(anchor_widget),
            }
        return payload

    def _map_semantic_to_assertions(
        self,
        semantic_contract: SemanticTransitionContract,
        plan: PlanResult,
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        dump_tree: dict,
        anchor_result: Any,
        widgets: list[dict[str, Any]],
    ) -> UIAssertionContract:
        hint_map = self._build_hint_map(semantic_contract.semantic_hints)
        anchor_node_id = self._resolve_anchor_node_id(
            anchor_result=anchor_result,
            widgets=widgets,
            node_index=node_index,
        )
        _, package_before = self._extract_app_context(dump_tree)

        selectors = {
            "resource_id": self._first_hint(hint_map, "target_resource_id"),
            "node_id": self._parse_int_hint(self._first_hint(hint_map, "target_node_id")),
            "text": self._first_hint(hint_map, "target_text"),
            "class_name": self._first_hint(hint_map, "target_class"),
            "field": self._first_hint(hint_map, "target_field"),
            "expected_text": self._first_hint(hint_map, "expected_text"),
            "expected_bool": self._first_hint(hint_map, "expected_bool"),
        }

        transition = semantic_contract.transition_type
        assertions: list[UIAssertion] = []
        if transition == "NavigationTransition":
            assertions.extend(
                self._build_navigation_assertions(
                    hint_map=hint_map,
                    plan=plan,
                    package_before=package_before,
                )
            )
        elif transition == "NodeAppearance":
            assertions.append(
                self._build_node_appearance_assertion(
                    selectors=selectors,
                    node_index=node_index,
                    rid_index=rid_index,
                    anchor_node_id=anchor_node_id,
                )
            )
        elif transition == "AttributeModification":
            assertions.append(
                self._build_attribute_modification_assertion(
                    selectors=selectors,
                    node_index=node_index,
                    rid_index=rid_index,
                    anchor_node_id=anchor_node_id,
                )
            )
        elif transition == "ContentUpdate":
            assertions.append(
                self._build_content_update_assertion(
                    selectors=selectors,
                    plan=plan,
                    node_index=node_index,
                    rid_index=rid_index,
                    anchor_node_id=anchor_node_id,
                )
            )
        elif transition == "ContainerExpansion":
            assertions.append(
                self._build_container_expansion_assertion(
                    selectors=selectors,
                    node_index=node_index,
                    rid_index=rid_index,
                    anchor_node_id=anchor_node_id,
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported transition_type={transition}")

        self._validate_transition_assertions(transition=transition, assertions=assertions)
        self._validate_widget_targets(
            assertions=assertions,
            node_index=node_index,
            rid_index=rid_index,
        )
        return UIAssertionContract(
            context_mode=semantic_contract.context_mode,
            transition_type=semantic_contract.transition_type,
            success_definition=semantic_contract.success_definition,
            assertions=assertions,
        )

    def _build_navigation_assertions(
        self,
        hint_map: dict[str, list[str]],
        plan: PlanResult,
        package_before: str,
    ) -> list[UIAssertion]:
        assertions: list[UIAssertion] = []
        expected_package = (
            self._first_hint(hint_map, "target_package")
            or str(getattr(plan, "launch_package", "") or "").strip()
            or str(package_before or "").strip()
        )
        if expected_package:
            assertions.append(
                UIAssertion(
                    category="package",
                    target=None,
                    relation="exact_match",
                    content=expected_package,
                )
            )

        for value in self._all_hints(hint_map, "target_activity_exact"):
            if value:
                assertions.append(
                    UIAssertion(
                        category="activity",
                        target=None,
                        relation="exact_match",
                        content=value,
                    )
                )
        for value in self._all_hints(hint_map, "target_activity_contains"):
            if value:
                assertions.append(
                    UIAssertion(
                        category="activity",
                        target=None,
                        relation="contains",
                        content=value,
                    )
                )

        launch_activity = str(getattr(plan, "launch_activity", "") or "").strip()
        if not any(item.category == "activity" for item in assertions) and launch_activity:
            assertions.append(
                UIAssertion(
                    category="activity",
                    target=None,
                    relation="contains",
                    content=launch_activity,
                )
            )
        if not assertions:
            raise ValueError("NavigationTransition 必须生成至少一条 package/activity 断言")
        return self._deduplicate_assertions(assertions)

    def _build_node_appearance_assertion(
        self,
        selectors: dict[str, Any],
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        anchor_node_id: int | None,
    ) -> UIAssertion:
        target, node = self._resolve_assertion_target(
            selectors=selectors,
            node_index=node_index,
            rid_index=rid_index,
            anchor_node_id=anchor_node_id,
            allow_anchor_fallback=True,
        )
        expected_text = str(selectors.get("expected_text") or "").strip()
        if target.resource_id and "resource-id" in node:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": "resource-id"}),
                relation="exact_match",
                content=target.resource_id,
            )
        if expected_text and "text" in node:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": "text"}),
                relation="contains",
                content=expected_text,
            )
        if target.text and "text" in node:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": "text"}),
                relation="contains",
                content=target.text,
            )
        if target.class_name and "class" in node:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": "class"}),
                relation="contains",
                content=target.class_name,
            )
        field = self._pick_existing_field(node=node, candidates=["enabled", "selected", "focused", "checked"])
        if not field:
            raise ValueError("NodeAppearance 未找到可用的 widget 字段")
        return UIAssertion(
            category="widget",
            target=target.model_copy(update={"field": field}),
            relation="is_true",
            content="true",
        )

    def _build_attribute_modification_assertion(
        self,
        selectors: dict[str, Any],
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        anchor_node_id: int | None,
    ) -> UIAssertion:
        target, node = self._resolve_assertion_target(
            selectors=selectors,
            node_index=node_index,
            rid_index=rid_index,
            anchor_node_id=anchor_node_id,
            allow_anchor_fallback=True,
        )
        desired_field = str(selectors.get("field") or "").strip()
        if desired_field not in node:
            desired_field = self._pick_existing_field(node=node, candidates=["focused", "selected", "checked", "enabled"])
        if not desired_field:
            raise ValueError("AttributeModification 未找到可用字段")

        raw_expected_bool = str(selectors.get("expected_bool") or "").strip().lower()
        relation: AssertionRelation = "is_true"
        content = "true"
        if raw_expected_bool in {"false", "0", "no", "off"}:
            relation = "is_false"
            content = "false"
        return UIAssertion(
            category="widget",
            target=target.model_copy(update={"field": desired_field}),
            relation=relation,
            content=content,
        )

    def _build_content_update_assertion(
        self,
        selectors: dict[str, Any],
        plan: PlanResult,
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        anchor_node_id: int | None,
    ) -> UIAssertion:
        target, node = self._resolve_assertion_target(
            selectors=selectors,
            node_index=node_index,
            rid_index=rid_index,
            anchor_node_id=anchor_node_id,
            allow_anchor_fallback=True,
        )
        expected_text_hint = str(selectors.get("expected_text") or "").strip()
        desired_field = str(selectors.get("field") or "").strip()
        if desired_field not in node:
            desired_field = ""
        if not desired_field:
            node_text = str(node.get("text") or "").strip()
            node_content_desc = str(node.get("content-desc") or "").strip()
            if expected_text_hint and "text" in node:
                desired_field = "text"
            elif node_content_desc and "content-desc" in node:
                desired_field = "content-desc"
            elif node_text and "text" in node:
                desired_field = "text"
            elif "text" in node:
                desired_field = "text"
            else:
                desired_field = self._pick_existing_field(node=node, candidates=["content-desc", "class"])
        if not desired_field:
            raise ValueError("ContentUpdate 未找到可用字段")

        expected_text = (
            expected_text_hint
            or str(getattr(plan, "input_description", "") or "").strip()
            or str(target.text or "").strip()
        )
        if not expected_text:
            expected_text = str(node.get(desired_field, "") or "").strip()
        if expected_text:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": desired_field}),
                relation="contains",
                content=expected_text,
            )

        raw_expected_bool = str(selectors.get("expected_bool") or "").strip().lower()
        if raw_expected_bool:
            bool_field = self._pick_existing_field(
                node=node,
                candidates=["enabled", "clickable", "focused", "selected", "checked"],
            )
            if bool_field:
                relation: AssertionRelation = "is_true"
                content = "true"
                if any(token in raw_expected_bool for token in ["false", "0", "no", "off"]):
                    relation = "is_false"
                    content = "false"
                return UIAssertion(
                    category="widget",
                    target=target.model_copy(update={"field": bool_field}),
                    relation=relation,
                    content=content,
                )

        fallback_class = str(target.class_name or node.get("class") or "").strip()
        if fallback_class and "class" in node:
            return UIAssertion(
                category="widget",
                target=target.model_copy(update={"field": "class"}),
                relation="contains",
                content=fallback_class,
            )
        raise ValueError("ContentUpdate 缺少 expected_text")

    def _build_container_expansion_assertion(
        self,
        selectors: dict[str, Any],
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        anchor_node_id: int | None,
    ) -> UIAssertion:
        target, node = self._resolve_assertion_target(
            selectors=selectors,
            node_index=node_index,
            rid_index=rid_index,
            anchor_node_id=anchor_node_id,
            allow_anchor_fallback=True,
        )
        desired_field = str(selectors.get("field") or "").strip()
        if desired_field not in node:
            desired_field = self._pick_existing_field(
                node=node,
                candidates=["expanded", "scrollable", "selected", "focused", "enabled"],
            )
        if not desired_field:
            raise ValueError("ContainerExpansion 未找到可用字段")
        return UIAssertion(
            category="widget",
            target=target.model_copy(update={"field": desired_field}),
            relation="is_true",
            content="true",
        )

    def _resolve_assertion_target(
        self,
        selectors: dict[str, Any],
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        anchor_node_id: int | None,
        allow_anchor_fallback: bool,
    ) -> tuple[UIAssertionTarget, dict]:
        selector_resource_id = str(selectors.get("resource_id") or "").strip()
        selector_node_id = selectors.get("node_id")
        selector_text = str(selectors.get("text") or "").strip()
        selector_class = str(selectors.get("class_name") or "").strip()

        selected_node_id: int | None = None
        resource_match_error: str = ""
        if selector_resource_id:
            selected_node_id, match_score = self._find_node_by_resource_id_with_score(
                selector_resource_id=selector_resource_id,
                node_index=node_index,
                rid_index=rid_index,
                text=selector_text,
                class_name=selector_class,
                anchor_node_id=anchor_node_id,
            )
            if selected_node_id is None:
                resource_match_error = f"target_resource_id='{selector_resource_id}' 无法解析到 node"
            else:
                selected_node = node_index.get(int(selected_node_id)) or {}
                logger.debug(
                    "selector resource_id 命中: query=%r node_id=%s score=%.2f text=%r class=%r rid=%r",
                    selector_resource_id,
                    selected_node_id,
                    float(match_score),
                    str(selected_node.get("text", "") or "")[:80],
                    str(selected_node.get("class", "") or "")[:80],
                    str(selected_node.get("resource-id", "") or ""),
                )

        if selected_node_id is None and selector_node_id is not None:
            node_id = int(selector_node_id)
            if node_id not in node_index:
                raise ValueError(f"target_node_id={node_id} 不存在")
            selected_node_id = node_id

        if selected_node_id is None and (selector_text or selector_class):
            selected_node_id, match_score = self._find_node_by_text_class_with_score(
                node_index=node_index,
                text=selector_text,
                class_name=selector_class,
            )
            if selected_node_id is None:
                if not (allow_anchor_fallback and anchor_node_id is not None):
                    if selector_resource_id and resource_match_error:
                        raise ValueError(resource_match_error)
                    raise ValueError("target_text/target_class 无法解析到 node")
            else:
                selected_node = node_index.get(int(selected_node_id)) or {}
                logger.debug(
                    "selector text/class 命中: node_id=%s score=%.2f text=%r class=%r rid=%r",
                    selected_node_id,
                    float(match_score),
                    str(selected_node.get("text", "") or "")[:80],
                    str(selected_node.get("class", "") or "")[:80],
                    str(selected_node.get("resource-id", "") or ""),
                )

        if selected_node_id is None and allow_anchor_fallback and anchor_node_id is not None:
            if selector_resource_id and resource_match_error:
                logger.warning(
                    "target_resource_id=%r 无法命中，回退 anchor_node_id=%s",
                    selector_resource_id,
                    anchor_node_id,
                )
            selected_node_id = int(anchor_node_id)

        if selected_node_id is None:
            if resource_match_error:
                raise ValueError(resource_match_error)
            raise ValueError("未找到可用 target 选择器，且无法回退 anchor node")

        node = node_index.get(int(selected_node_id))
        if node is None:
            raise ValueError(f"解析 target 失败，node_id={selected_node_id} 不存在")

        target = UIAssertionTarget(
            node_id=int(selected_node_id),
            resource_id=str(node.get("resource-id", "") or ""),
            field=str(selectors.get("field") or ""),
            text=str(node.get("text", "") or ""),
            class_name=str(node.get("class", "") or ""),
        )
        return target, node

    def _find_node_by_resource_id_with_score(
        self,
        selector_resource_id: str,
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
        text: str,
        class_name: str,
        anchor_node_id: int | None,
    ) -> tuple[int | None, float]:
        query_norm = self._normalize_resource_id(selector_resource_id)
        if not query_norm:
            return None, -1.0

        text_norm = str(text or "").strip().lower()
        class_norm = str(class_name or "").strip().lower()

        best_node_id: int | None = None
        best_score = -1.0
        best_rank_tuple: tuple[float, float, float, float, float, float, float] | None = None

        for rid, raw_node_ids in rid_index.items():
            rid_raw = str(rid or "").strip()
            if not rid_raw:
                continue
            rid_score = self._resource_id_match_score(query=query_norm, candidate=rid_raw)
            if rid_score <= 0.0:
                continue

            for raw_node_id in raw_node_ids:
                try:
                    node_id_int = int(raw_node_id)
                except Exception:
                    continue
                node = node_index.get(node_id_int)
                if not isinstance(node, dict):
                    continue

                node_text = str(node.get("text", "") or "").strip().lower()
                node_class = str(node.get("class", "") or "").strip().lower()
                text_score = (
                    self._selector_value_match_score(query=text_norm, candidate=node_text) if text_norm else 0.0
                )
                class_score = (
                    self._selector_value_match_score(query=class_norm, candidate=node_class) if class_norm else 0.0
                )
                is_anchor = 1.0 if anchor_node_id is not None and node_id_int == int(anchor_node_id) else 0.0
                interactive_count = float(
                    sum(
                        1
                        for key in ["clickable", "focusable", "checkable", "scrollable"]
                        if self._node_attr_is_true(node=node, key=key)
                    )
                )

                score = (rid_score * 3.0) + (text_score * 1.5) + class_score + (is_anchor * 1.2)
                rank_tuple = (
                    float(score),
                    float(rid_score),
                    float(text_score),
                    float(class_score),
                    is_anchor,
                    interactive_count,
                    float(-node_id_int),
                )
                if best_rank_tuple is None or rank_tuple > best_rank_tuple:
                    best_node_id = node_id_int
                    best_score = float(score)
                    best_rank_tuple = rank_tuple

        return best_node_id, float(best_score)

    def _find_node_by_text_class(
        self,
        node_index: Dict[int, dict],
        text: str,
        class_name: str,
    ) -> int | None:
        best_id, _ = self._find_node_by_text_class_with_score(
            node_index=node_index,
            text=text,
            class_name=class_name,
        )
        return best_id

    def _find_node_by_text_class_with_score(
        self,
        node_index: Dict[int, dict],
        text: str,
        class_name: str,
    ) -> tuple[int | None, float]:
        text_norm = str(text or "").strip().lower()
        class_norm = str(class_name or "").strip().lower()
        if not text_norm and not class_norm:
            return None, -1.0

        best_id: int | None = None
        best_score = -1.0
        best_rank_tuple: tuple[float, float, float, float] | None = None

        for node_id, node in node_index.items():
            node_text = str(node.get("text", "") or "").strip().lower()
            node_class = str(node.get("class", "") or "").strip().lower()

            text_score = 0.0
            class_score = 0.0

            if text_norm and class_norm:
                text_score = self._selector_value_match_score(query=text_norm, candidate=node_text)
                class_score = self._selector_value_match_score(query=class_norm, candidate=node_class)
                if text_score <= 0.0 or class_score <= 0.0:
                    continue
                score = (text_score * 2.0) + class_score
            elif text_norm:
                text_score = self._selector_value_match_score(query=text_norm, candidate=node_text)
                if text_score <= 0.0:
                    continue
                score = text_score * 2.0
            elif class_norm:
                class_score = self._selector_value_match_score(query=class_norm, candidate=node_class)
                if class_score <= 0.0:
                    continue
                score = class_score
            else:
                continue

            node_rid = str(node.get("resource-id", "") or "").strip()
            has_rid = 1.0 if node_rid else 0.0
            interactive_count = float(
                sum(
                    1
                    for key in ["clickable", "focusable", "checkable", "scrollable"]
                    if self._node_attr_is_true(node=node, key=key)
                )
            )
            try:
                node_id_int = int(node_id)
            except Exception:
                continue

            rank_tuple = (
                float(score),
                has_rid,
                interactive_count,
                float(-node_id_int),
            )

            if best_rank_tuple is None or rank_tuple > best_rank_tuple:
                best_score = score
                best_id = node_id_int
                best_rank_tuple = rank_tuple
        return best_id, float(best_score)

    def _resource_id_match_score(self, query: str, candidate: str) -> float:
        query_norm = self._normalize_resource_id(query)
        candidate_norm = self._normalize_resource_id(candidate)
        if not query_norm or not candidate_norm:
            return 0.0
        if query_norm == candidate_norm:
            return 8.0

        query_leaf = self._resource_id_leaf(query_norm)
        candidate_leaf = self._resource_id_leaf(candidate_norm)
        score = 0.0

        if query_leaf and candidate_leaf:
            if query_leaf == candidate_leaf:
                score = max(score, 6.0)
            if len(query_leaf) >= 3 and query_leaf in candidate_leaf:
                score = max(score, 4.0)
            if len(candidate_leaf) >= 3 and candidate_leaf in query_leaf:
                score = max(score, 2.0)

            query_tokens = self._identifier_tokens(query_leaf)
            candidate_tokens = self._identifier_tokens(candidate_leaf)
            if query_tokens and candidate_tokens:
                overlap = query_tokens & candidate_tokens
                if overlap:
                    coverage = float(len(overlap)) / float(len(query_tokens))
                    score = max(score, 1.5 + coverage)

        if len(query_norm) >= 3 and query_norm in candidate_norm:
            score = max(score, 5.0)
        if len(candidate_norm) >= 3 and candidate_norm in query_norm:
            score = max(score, 2.0)
        return score

    def _normalize_resource_id(self, value: str) -> str:
        token = str(value or "").strip().lower()
        if not token:
            return ""
        return re.sub(r"\s+", "_", token)

    def _resource_id_leaf(self, value: str) -> str:
        token = self._normalize_resource_id(value)
        if not token:
            return ""
        if ":" in token:
            token = token.split(":", 1)[1]
        if "/" in token:
            token = token.rsplit("/", 1)[-1]
        return token

    def _identifier_tokens(self, value: str) -> set[str]:
        token = str(value or "").strip().lower()
        if not token:
            return set()
        return {part for part in re.split(r"[^a-z0-9]+", token) if len(part) >= 2}

    def _selector_value_match_score(self, query: str, candidate: str) -> float:
        query_norm = str(query or "").strip().lower()
        candidate_norm = str(candidate or "").strip().lower()
        if not query_norm or not candidate_norm:
            return 0.0
        if query_norm == candidate_norm:
            return 5.0
        if query_norm in candidate_norm:
            return 3.0
        if len(query_norm) >= 3 and len(candidate_norm) >= 3 and candidate_norm in query_norm:
            return 1.5
        return 0.0

    def _node_attr_is_true(self, node: dict[str, Any], key: str) -> bool:
        value = node.get(str(key or ""))
        if isinstance(value, bool):
            return value
        token = str(value or "").strip().lower()
        return token in {"true", "1", "yes", "on"}

    def _validate_transition_assertions(
        self,
        transition: TransitionType,
        assertions: list[UIAssertion],
    ) -> None:
        if not assertions:
            raise ValueError("assertions 不能为空")
        if transition == "NavigationTransition":
            if not any(item.category in {"package", "activity"} for item in assertions):
                raise ValueError("NavigationTransition 需要 package/activity 断言")
            return
        if not any(item.category == "widget" for item in assertions):
            raise ValueError(f"{transition} 需要至少一个 widget 断言")

    def _validate_widget_targets(
        self,
        assertions: list[UIAssertion],
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
    ) -> None:
        for idx, assertion in enumerate(assertions, start=1):
            if assertion.category != "widget":
                continue
            target = assertion.target
            if target is None:
                raise ValueError(f"Assertion[{idx}] widget target 不能为空")

            node_id = int(target.node_id) if target.node_id is not None else None
            if node_id is None or node_id not in node_index:
                raise ValueError(f"Assertion[{idx}] node_id 无效: {target.node_id}")
            node = node_index[node_id]

            field = str(target.field or "").strip()
            if field not in node:
                raise ValueError(f"Assertion[{idx}] field='{field}' 不存在于 node_id={node_id}")

            expected_rid = str(target.resource_id or "")
            node_rid = str(node.get("resource-id", "") or "")
            if expected_rid and expected_rid != node_rid:
                raise ValueError(
                    f"Assertion[{idx}] resource_id 不匹配(node_id={node_id}, target={expected_rid}, dump={node_rid})"
                )
            if expected_rid:
                rid_node_ids = rid_index.get(expected_rid, [])
                if node_id not in rid_node_ids:
                    raise ValueError(f"Assertion[{idx}] node_id={node_id} 未通过 resource_id 索引校验")
        logger.info("UIAssertion widget target 校验通过")

    def _index_dump_tree(self, dump_tree: dict) -> tuple[Dict[int, dict], Dict[str, list[int]]]:
        index: Dict[int, dict] = {}
        rid_index: Dict[str, list[int]] = {}

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return

            raw_node_id = node.get("node_id")
            current_id: int | None = None
            if isinstance(raw_node_id, int):
                current_id = raw_node_id
                index[current_id] = node
            elif raw_node_id is not None:
                try:
                    current_id = int(raw_node_id)
                    index[current_id] = node
                except Exception:
                    current_id = None

            if current_id is not None:
                rid = str(node.get("resource-id", "") or "")
                rid_index.setdefault(rid, []).append(current_id)

            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child)

        walk(dump_tree)
        return index, rid_index

    def _extract_app_context(self, dump_tree: dict) -> tuple[str, str]:
        keys_activity = ("activity", "activity_name", "current_activity")
        keys_package = ("package", "package_name", "current_package")
        activity = self._find_first_string(dump_tree, keys_activity)
        package = self._find_first_string(dump_tree, keys_package)
        return activity, package

    def _find_first_string(self, node: Any, keys: tuple[str, ...]) -> str:
        if isinstance(node, dict):
            for key in keys:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in node.values():
                found = self._find_first_string(value, keys)
                if found:
                    return found
            return ""
        if isinstance(node, list):
            for child in node:
                found = self._find_first_string(child, keys)
                if found:
                    return found
        return ""

    def _resolve_anchor_node_id(
        self,
        anchor_result: Any,
        widgets: list[dict[str, Any]],
        node_index: Dict[int, dict],
    ) -> int | None:
        anchor_widget = self._resolve_anchor_widget(anchor_result=anchor_result, widgets=widgets)
        if anchor_widget is None:
            return None
        widget_bounds = self._widget_bounds(anchor_widget)
        widget_text = str(anchor_widget.get("text") or "").strip().lower()
        widget_class = str(anchor_widget.get("class") or "").strip().lower()

        best_node_id: int | None = None
        best_score = -1.0
        best_rank_tuple: tuple[float, float, float, float, float, float, float] | None = None
        for node_id, node in node_index.items():
            node_bounds = self._parse_node_bounds(node)
            iou = self._bbox_iou(widget_bounds, node_bounds) if (widget_bounds and node_bounds) else 0.0

            node_text = str(node.get("text") or "").strip().lower()
            node_class = str(node.get("class") or "").strip().lower()
            score = iou * 5.0
            if widget_text and node_text and (widget_text in node_text or node_text in widget_text):
                score += 3.0
            if widget_class and node_class and (widget_class in node_class or node_class in widget_class):
                score += 2.0

            try:
                node_id_int = int(node_id)
            except Exception:
                continue
            interactive_count = float(
                sum(
                    1
                    for key in ["clickable", "focusable", "checkable", "scrollable"]
                    if self._node_attr_is_true(node=node, key=key)
                )
            )
            has_rid = 1.0 if str(node.get("resource-id") or "").strip() else 0.0
            clickable = 1.0 if self._node_attr_is_true(node=node, key="clickable") else 0.0
            area = 0.0
            if node_bounds is not None:
                area = float(max(0, (node_bounds[2] - node_bounds[0]) * (node_bounds[3] - node_bounds[1])))

            rank_tuple = (
                float(score),
                float(iou),
                interactive_count,
                clickable,
                has_rid,
                float(-area),
                float(-node_id_int),
            )
            if best_rank_tuple is None or rank_tuple > best_rank_tuple:
                best_score = score
                best_node_id = node_id_int
                best_rank_tuple = rank_tuple
        if best_node_id is None or best_score <= 0.0:
            return None
        return best_node_id

    def _resolve_anchor_widget(
        self,
        anchor_result: Any,
        widgets: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            target_widget_id = int(getattr(anchor_result, "target_widget_id", -1))
        except Exception:
            target_widget_id = -1
        if target_widget_id < 0:
            return None
        for widget in widgets:
            try:
                if int(widget.get("widget_id")) == target_widget_id:
                    return widget
            except Exception:
                continue
        bounds_raw = list(getattr(anchor_result, "target_widget_bounds", []) or [])
        center_raw = list(getattr(anchor_result, "target_widget_center", []) or [])
        if len(bounds_raw) != 4 and len(center_raw) != 2:
            return None
        try:
            bounds = (
                [int(bounds_raw[0]), int(bounds_raw[1]), int(bounds_raw[2]), int(bounds_raw[3])]
                if len(bounds_raw) == 4
                else []
            )
            center = (
                [int(center_raw[0]), int(center_raw[1])]
                if len(center_raw) == 2
                else []
            )
        except Exception:
            return None

        payload: dict[str, Any] = {
            "widget_id": int(target_widget_id),
            "class": str(getattr(anchor_result, "target_widget_class", "") or ""),
            "text": str(getattr(anchor_result, "target_widget_text", "") or ""),
            "resource_id": str(getattr(anchor_result, "target_widget_resource_id", "") or ""),
            "content_desc": str(getattr(anchor_result, "target_widget_content_desc", "") or ""),
            "hint": str(getattr(anchor_result, "target_widget_hint", "") or ""),
            "source": str(getattr(anchor_result, "target_widget_source", "") or ""),
        }
        if len(bounds) == 4:
            payload["bounds"] = bounds
            payload["width"] = max(1, int(bounds[2] - bounds[0]))
            payload["height"] = max(1, int(bounds[3] - bounds[1]))
        if len(center) == 2:
            payload["center"] = center
        elif len(bounds) == 4:
            payload["center"] = [int((bounds[0] + bounds[2]) // 2), int((bounds[1] + bounds[3]) // 2)]
        return payload

    def _widget_bounds(self, widget: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
        if not isinstance(widget, dict):
            return None
        bounds = widget.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return None
        try:
            x1, y1, x2, y2 = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _parse_node_bounds(self, node: dict[str, Any]) -> tuple[int, int, int, int] | None:
        raw = str(node.get("bounds") or "").strip()
        match = self._BOUNDS_RE.match(raw)
        if not match:
            return None
        try:
            x1 = int(match.group(1))
            y1 = int(match.group(2))
            x2 = int(match.group(3))
            y2 = int(match.group(4))
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _bbox_iou(
        self,
        a: tuple[int, int, int, int] | None,
        b: tuple[int, int, int, int] | None,
    ) -> float:
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float((ax2 - ax1) * (ay2 - ay1))
        area_b = float((bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _crop_local_context_image(
        self,
        screenshot_path: str,
        bounds: tuple[int, int, int, int],
        step: int | None,
    ) -> str:
        if cv2 is None:
            raise RuntimeError("opencv unavailable")
        path = str(screenshot_path or "").strip()
        if not path:
            raise ValueError("screenshot_path is empty")
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"failed to read screenshot: {path}")
        img_h, img_w = image.shape[:2]

        x1, y1, x2, y2 = bounds
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        min_size = 320
        pad_x = max(int(round(0.25 * width)), int(round(max(0, min_size - width) / 2)))
        pad_y = max(int(round(0.25 * height)), int(round(max(0, min_size - height) / 2)))

        crop = self._fit_crop_bounds(
            x1=x1 - pad_x,
            y1=y1 - pad_y,
            x2=x2 + pad_x,
            y2=y2 + pad_y,
            img_w=img_w,
            img_h=img_h,
            min_w=min_size,
            min_h=min_size,
        )
        cx1, cy1, cx2, cy2 = crop
        crop_img = image[cy1:cy2, cx1:cx2]
        if crop_img.size == 0:
            raise RuntimeError("empty crop image")

        out_dir = self._project_root / "data" / "pre_oracle_context"
        out_dir.mkdir(parents=True, exist_ok=True)
        if step is None:
            filename = "local_context.png"
        else:
            filename = f"step_{int(step)}_local.png"
        out_path = out_dir / filename
        ok = cv2.imwrite(str(out_path), crop_img)
        if not ok:
            raise RuntimeError("cv2.imwrite returned false")
        return str(out_path)

    def _fit_crop_bounds(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_w: int,
        img_h: int,
        min_w: int,
        min_h: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(int(img_w), int(x2))
        y2 = min(int(img_h), int(y2))

        if x2 <= x1:
            x2 = min(img_w, x1 + 1)
        if y2 <= y1:
            y2 = min(img_h, y1 + 1)

        def expand_axis(
            low: int,
            high: int,
            max_size: int,
            min_size: int,
        ) -> tuple[int, int]:
            current = high - low
            if current >= min_size:
                return low, high
            need = min_size - current
            left = need // 2
            right = need - left

            low = max(0, low - left)
            high = min(max_size, high + right)
            current = high - low
            if current >= min_size:
                return low, high

            remaining = min_size - current
            extra_left = min(remaining, low)
            low -= extra_left
            remaining -= extra_left
            if remaining > 0:
                extra_right = min(remaining, max_size - high)
                high += extra_right
            return low, high

        x1, x2 = expand_axis(x1, x2, img_w, min_w)
        y1, y2 = expand_axis(y1, y2, img_h, min_h)
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def _build_hint_map(self, hints: list[SemanticHint]) -> dict[str, list[str]]:
        payload: dict[str, list[str]] = {}
        for hint in list(hints or []):
            key = str(hint.key or "").strip().lower()
            if not key:
                continue
            value = str(hint.value or "").strip()
            payload.setdefault(key, []).append(value)
        return payload

    def _first_hint(self, hint_map: dict[str, list[str]], key: str) -> str:
        values = hint_map.get(str(key or "").strip().lower(), [])
        for value in values:
            if str(value or "").strip():
                return str(value).strip()
        return ""

    def _all_hints(self, hint_map: dict[str, list[str]], key: str) -> list[str]:
        values = hint_map.get(str(key or "").strip().lower(), [])
        return [str(item).strip() for item in values if str(item or "").strip()]

    def _parse_int_hint(self, value: str) -> int | None:
        token = str(value or "").strip()
        if not token:
            return None
        try:
            return int(token)
        except Exception:
            return None

    def _pick_existing_field(self, node: dict[str, Any], candidates: list[str]) -> str:
        for item in candidates:
            key = str(item or "").strip()
            if key and key in node:
                return key
        return ""

    def _deduplicate_assertions(self, assertions: list[UIAssertion]) -> list[UIAssertion]:
        seen: set[tuple[str, str, str, str, str]] = set()
        output: list[UIAssertion] = []
        for item in assertions:
            target = item.target
            node_id = str(target.node_id) if target is not None else ""
            key = (
                str(item.category),
                node_id,
                str(target.field if target is not None else ""),
                str(item.relation),
                str(item.content),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    def _record_semantic_artifact(
        self,
        step: int | None,
        payload: SemanticTransitionContract,
    ) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="SemanticTransitionContract",
                step=int(step),
                payload=payload,
            )
        except Exception as exc:
            logger.error("写入 SemanticTransitionContract 审计记录失败: %s", exc)

    def _record_assertion_artifact(
        self,
        step: int | None,
        payload: UIAssertionContract,
    ) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="UIAssertionContract",
                step=int(step),
                payload=payload,
            )
        except Exception as exc:
            logger.error("写入 UIAssertionContract 审计记录失败: %s", exc)

    def _record_pre_error(self, step: int | None, error: str) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="PreOracleError",
                step=int(step),
                payload={"error": str(error or "")},
                llm=False,
                append=True,
            )
        except Exception as exc:
            logger.error("写入 PreOracleError 审计记录失败: %s", exc)
