"""Post-Oracle: XML verification + LLM secondary verification."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from Oracle.pre_oracle import SemanticTransitionContract, UIAssertion, UIAssertionContract, UIAssertionTarget
from prompt.post_oracle_prompt import ORACLE_POST_SYSTEM_PROMPT, ORACLE_POST_USER_PROMPT
from utils.audit_recorder import AuditRecorder
from utils.llm_client import LLMRequest

logger = logging.getLogger(__name__)

PostOracleDecision = Literal[
    "semantic_success",
    "semantic_fail_ui_changed",
    "semantic_fail_ui_unchanged",
]
PostOracleEvidenceSource = Literal[
    "xml_assertions",
    "llm_secondary",
    "llm_secondary_retry_fallback",
]


class AssertionSummary(BaseModel):
    """Summary of XML assertion evaluation."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    failed: int = Field(ge=0)


class PostOracleLLMRecheck(BaseModel):
    """Structured LLM secondary verification output."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    semantic_success: bool
    ui_changed: bool
    rationale: str = Field(min_length=1)
    attempts: int = Field(default=1, ge=1)


class PostOracleResult(BaseModel):
    """Post-Oracle output contract."""

    model_config = ConfigDict(extra="forbid")

    decision: PostOracleDecision
    needs_back: bool
    reason: str = Field(default="")
    evidence_source: PostOracleEvidenceSource
    assertion_summary: AssertionSummary
    failed_assertions: list[str] = Field(default_factory=list)
    llm_recheck: Optional[PostOracleLLMRecheck] = None


class PostOracle:
    """Evaluate assertion evidence and fallback to LLM secondary verification."""

    def __init__(self, llm_client=None, max_llm_retries: int = 1):
        self.llm = llm_client
        self.max_llm_retries = max(0, int(max_llm_retries))
        self.audit = AuditRecorder(component="post_oracle")
        logger.info("PostOracle 初始化完成 (xml + llm-secondary)")

    def evaluate(
        self,
        before_dump_tree: dict,
        after_dump_tree: dict,
        action: dict[str, Any] | None,
        semantic_contract: SemanticTransitionContract | dict[str, Any] | None,
        assertion_contract: UIAssertionContract | Sequence[UIAssertion] | Sequence[dict[str, Any]],
        before_screenshot_path: str = "",
        after_screenshot_path: str = "",
        step: int | None = None,
    ) -> PostOracleResult:
        if not isinstance(after_dump_tree, dict):
            logger.error("Post-Oracle 评估失败: after_dump_tree 类型非法(type=%s)", type(after_dump_tree).__name__)
            raise ValueError("after_dump_tree 必须是 dict")
        if not isinstance(before_dump_tree, dict):
            before_dump_tree = {}

        assertion_items, assertion_mode, mapping_reason = self._normalize_assertions(assertion_contract)
        node_index = self._index_dump_tree(after_dump_tree)
        activity_after, package_after = self._extract_app_context(after_dump_tree)

        if assertion_mode == "semantic_only":
            logger.info("Post-Oracle 跳过 XML 断言(semantic_only): reason=%s", mapping_reason or "n/a")
            summary = AssertionSummary(total=0, failed=0)
            fallback_failed_items: list[dict[str, Any]] = []
            if mapping_reason:
                fallback_failed_items.append(
                    {
                        "index": 0,
                        "message": f"semantic_only_mapping: {mapping_reason}",
                        "assertion": {},
                    }
                )
            llm_recheck = self._secondary_verify_with_llm(
                before_screenshot_path=before_screenshot_path,
                after_screenshot_path=after_screenshot_path,
                action=action or {},
                semantic_contract=semantic_contract,
                assertion_items=assertion_items,
                failed_items=fallback_failed_items,
                before_dump_tree=before_dump_tree,
                after_dump_tree=after_dump_tree,
                step=step,
            )
            if llm_recheck is None:
                result = PostOracleResult(
                    decision="semantic_fail_ui_unchanged",
                    needs_back=False,
                    reason="llm_secondary_failed_after_retry",
                    evidence_source="llm_secondary_retry_fallback",
                    assertion_summary=summary,
                    failed_assertions=[],
                    llm_recheck=None,
                )
                self._record_post_oracle_artifact(step=step, result=result)
                logger.warning("Post-Oracle semantic_only 二次验证失败(重试后降级): decision=%s", result.decision)
                return result

            if llm_recheck.semantic_success:
                decision: PostOracleDecision = "semantic_success"
                needs_back = False
            elif llm_recheck.ui_changed:
                decision = "semantic_fail_ui_changed"
                needs_back = True
            else:
                decision = "semantic_fail_ui_unchanged"
                needs_back = False

            result = PostOracleResult(
                decision=decision,
                needs_back=needs_back,
                reason=llm_recheck.rationale,
                evidence_source="llm_secondary",
                assertion_summary=summary,
                failed_assertions=[],
                llm_recheck=llm_recheck,
            )
            self._record_post_oracle_artifact(step=step, result=result)
            logger.info(
                "Post-Oracle semantic_only 二次验证完成: decision=%s, needs_back=%s, attempts=%d",
                result.decision,
                result.needs_back,
                int(llm_recheck.attempts),
            )
            return result

        failed_messages: list[str] = []
        failed_items: list[dict[str, Any]] = []
        for idx, assertion in enumerate(assertion_items, start=1):
            ok, reason = self._check_assertion(
                assertion=assertion,
                node_index=node_index,
                activity=activity_after,
                package=package_after,
            )
            if not ok:
                message = f"Assertion[{idx}] {reason}"
                failed_messages.append(message)
                failed_items.append(
                    {
                        "index": idx,
                        "message": message,
                        "assertion": assertion.model_dump(),
                    }
                )

        summary = AssertionSummary(total=len(assertion_items), failed=len(failed_messages))
        if not failed_messages:
            result = PostOracleResult(
                decision="semantic_success",
                needs_back=False,
                reason="xml_assertions_passed",
                evidence_source="xml_assertions",
                assertion_summary=summary,
                failed_assertions=[],
                llm_recheck=None,
            )
            self._record_post_oracle_artifact(step=step, result=result)
            logger.info("Post-Oracle 评估通过(XML): assertions=%d", summary.total)
            return result

        logger.info(
            "Post-Oracle XML 断言未完全通过: assertions=%d, failed=%d，触发 LLM 二次验证",
            summary.total,
            summary.failed,
        )
        for message in failed_messages:
            logger.info("Post-Oracle 失败详情: %s", message)

        llm_recheck = self._secondary_verify_with_llm(
            before_screenshot_path=before_screenshot_path,
            after_screenshot_path=after_screenshot_path,
            action=action or {},
            semantic_contract=semantic_contract,
            assertion_items=assertion_items,
            failed_items=failed_items,
            before_dump_tree=before_dump_tree,
            after_dump_tree=after_dump_tree,
            step=step,
        )

        if llm_recheck is None:
            result = PostOracleResult(
                decision="semantic_fail_ui_unchanged",
                needs_back=False,
                reason="llm_secondary_failed_after_retry",
                evidence_source="llm_secondary_retry_fallback",
                assertion_summary=summary,
                failed_assertions=failed_messages,
                llm_recheck=None,
            )
            self._record_post_oracle_artifact(step=step, result=result)
            logger.warning(
                "Post-Oracle 二次验证失败(重试后降级): decision=%s",
                result.decision,
            )
            return result

        if llm_recheck.semantic_success:
            decision: PostOracleDecision = "semantic_success"
            needs_back = False
        elif llm_recheck.ui_changed:
            decision = "semantic_fail_ui_changed"
            needs_back = True
        else:
            decision = "semantic_fail_ui_unchanged"
            needs_back = False

        result = PostOracleResult(
            decision=decision,
            needs_back=needs_back,
            reason=llm_recheck.rationale,
            evidence_source="llm_secondary",
            assertion_summary=summary,
            failed_assertions=failed_messages,
            llm_recheck=llm_recheck,
        )
        self._record_post_oracle_artifact(step=step, result=result)
        logger.info(
            "Post-Oracle 二次验证完成: decision=%s, needs_back=%s, attempts=%d",
            result.decision,
            result.needs_back,
            int(llm_recheck.attempts),
        )
        return result

    def to_output(self, result: PostOracleResult) -> Dict[str, Any]:
        return dict(result.model_dump(exclude_none=True))

    def _secondary_verify_with_llm(
        self,
        before_screenshot_path: str,
        after_screenshot_path: str,
        action: dict[str, Any],
        semantic_contract: SemanticTransitionContract | dict[str, Any] | None,
        assertion_items: Sequence[UIAssertion],
        failed_items: list[dict[str, Any]],
        before_dump_tree: dict[str, Any],
        after_dump_tree: dict[str, Any],
        step: int | None,
    ) -> PostOracleLLMRecheck | None:
        if self.llm is None:
            logger.warning("Post-Oracle 跳过 LLM 二次验证: llm_client 未初始化")
            return None

        before_path = str(before_screenshot_path or "").strip()
        after_path = str(after_screenshot_path or "").strip()
        if not before_path or not after_path:
            logger.warning("Post-Oracle 跳过 LLM 二次验证: before/after 截图路径为空")
            return None
        if not os.path.exists(before_path) or not os.path.exists(after_path):
            logger.warning("Post-Oracle 跳过 LLM 二次验证: before/after 截图不存在")
            return None

        semantic_payload = self._to_jsonable(semantic_contract)
        assertion_payload = {
            "assertions": [item.model_dump() for item in assertion_items],
        }
        app_context_payload = self._build_app_context_payload(
            before_dump_tree=before_dump_tree,
            after_dump_tree=after_dump_tree,
        )

        user_prompt = ORACLE_POST_USER_PROMPT.format(
            action_json=json.dumps(self._to_jsonable(action), ensure_ascii=False),
            semantic_json=json.dumps(semantic_payload, ensure_ascii=False),
            assertion_json=json.dumps(assertion_payload, ensure_ascii=False),
            failed_assertions_json=json.dumps(failed_items, ensure_ascii=False),
            app_context_json=json.dumps(app_context_payload, ensure_ascii=False),
        )

        total_attempts = self.max_llm_retries + 1
        for attempt in range(1, total_attempts + 1):
            request = LLMRequest(
                system=ORACLE_POST_SYSTEM_PROMPT,
                user=user_prompt,
                images=[before_path, after_path],
                response_format=PostOracleLLMRecheck,
                audit_meta={
                    "artifact_kind": "PostOracleLLMRecheck",
                    "step": step,
                    "stage": f"post_oracle_secondary_attempt_{attempt}",
                },
            )
            try:
                parsed = self.llm.chat(request)
                verified = (
                    parsed if isinstance(parsed, PostOracleLLMRecheck) else PostOracleLLMRecheck.model_validate(parsed)
                )
                return verified.model_copy(update={"attempts": int(attempt)})
            except Exception as exc:
                logger.warning(
                    "Post-Oracle LLM 二次验证失败(attempt=%d/%d): %s",
                    attempt,
                    total_attempts,
                    exc,
                )
                continue

        return None

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            payload: dict[str, Any] = {}
            for key, item in value.items():
                payload[str(key)] = self._to_jsonable(item)
            return payload
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        return value

    def _build_app_context_payload(
        self,
        before_dump_tree: dict[str, Any],
        after_dump_tree: dict[str, Any],
    ) -> dict[str, Any]:
        before_activity, before_package = self._extract_app_context(before_dump_tree)
        after_activity, after_package = self._extract_app_context(after_dump_tree)
        return {
            "before": {
                "package": before_package,
                "activity": before_activity,
            },
            "after": {
                "package": after_package,
                "activity": after_activity,
            },
        }

    def _normalize_assertions(
        self,
        assertion_contract: UIAssertionContract | Sequence[UIAssertion] | Sequence[dict[str, Any]],
    ) -> tuple[list[UIAssertion], str, str]:
        assertion_mode = "xml_assertions"
        mapping_reason = ""
        if isinstance(assertion_contract, UIAssertionContract):
            items = list(assertion_contract.assertions or [])
            assertion_mode = str(assertion_contract.assertion_mode or "xml_assertions")
            mapping_reason = str(assertion_contract.mapping_reason or "")
        elif isinstance(assertion_contract, Sequence) and not isinstance(assertion_contract, (str, bytes)):
            items = []
            for value in assertion_contract:
                if isinstance(value, UIAssertion):
                    items.append(value)
                else:
                    items.append(UIAssertion.model_validate(value))
        else:
            raise ValueError("assertion_contract 必须是 UIAssertionContract 或 UIAssertion 列表")

        if assertion_mode == "semantic_only":
            return items, assertion_mode, mapping_reason
        if not items:
            raise ValueError("assertions 不能为空")
        return items, assertion_mode, mapping_reason

    def _index_dump_tree(self, dump_tree: dict) -> Dict[int, dict]:
        index: Dict[int, dict] = {}

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                raw_node_id = node.get("node_id")
                if isinstance(raw_node_id, int):
                    index[raw_node_id] = node
                elif raw_node_id is not None:
                    try:
                        index[int(raw_node_id)] = node
                    except Exception:
                        pass
                children = node.get("children")
                if isinstance(children, list):
                    for child in children:
                        walk(child)
                return

            if isinstance(node, list):
                for child in node:
                    walk(child)

        walk(dump_tree)
        return index

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

    def _check_assertion(
        self,
        assertion: UIAssertion,
        node_index: Dict[int, dict],
        activity: str,
        package: str,
    ) -> tuple[bool, str]:
        category = str(assertion.category or "").strip().lower()
        relation = str(assertion.relation or "").strip().lower()
        expected_content = str(assertion.content or "")

        if category == "widget":
            target = assertion.target
            if target is None:
                return False, "widget 目标为空"

            node_id, node = self._locate_widget_node(target=target, node_index=node_index)
            if node is None or node_id is None:
                return False, "widget 目标定位失败"

            expected_rid = str(target.resource_id or "")
            if expected_rid:
                node_rid = str(node.get("resource-id", "") or "")
                if expected_rid != node_rid:
                    rematched_id, rematched_node = self._relocate_by_expected_resource_id(
                        expected_resource_id=expected_rid,
                        target=target,
                        node_index=node_index,
                    )
                    if rematched_node is None or rematched_id is None:
                        return False, (
                            f"resource_id 不匹配(node_id={node_id}, expected='{expected_rid}', actual='{node_rid}')"
                        )
                    node_id, node = rematched_id, rematched_node

            field = str(target.field or "")
            if field not in node:
                return False, f"field='{field}' 不存在于 node_id={node_id}"

            actual_value = node.get(field)
            passed = self._match_relation(
                relation=relation,
                actual_value=actual_value,
                expected_content=expected_content,
            )
            if passed:
                return True, "ok"
            return False, (
                f"widget 字段不满足关系(field='{field}', relation='{relation}', "
                f"expected='{expected_content}', actual='{self._to_text(actual_value)}')"
            )

        if category == "activity":
            passed = self._match_relation(
                relation=relation,
                actual_value=activity,
                expected_content=expected_content,
            )
            if passed:
                return True, "ok"
            return False, (
                f"activity 不满足关系(relation='{relation}', expected='{expected_content}', actual='{activity}')"
            )

        if category == "package":
            passed = self._match_relation(
                relation=relation,
                actual_value=package,
                expected_content=expected_content,
            )
            if passed:
                return True, "ok"
            return False, (
                f"package 不满足关系(relation='{relation}', expected='{expected_content}', actual='{package}')"
            )

        return False, f"不支持的 category='{category}'"

    def _locate_widget_node(
        self,
        target: UIAssertionTarget,
        node_index: Dict[int, dict],
    ) -> tuple[int | None, dict[str, Any] | None]:
        if target.node_id is not None:
            node_id = int(target.node_id)
            node = node_index.get(node_id)
            if node is not None:
                return node_id, node

        resource_id = str(target.resource_id or "").strip()
        text = str(target.text or "").strip().lower()
        class_name = str(target.class_name or "").strip().lower()
        best_id: int | None = None
        best_score = -1
        for node_id, node in node_index.items():
            score = 0
            node_rid = str(node.get("resource-id", "") or "").strip()
            node_text = str(node.get("text", "") or "").strip().lower()
            node_class = str(node.get("class", "") or "").strip().lower()

            if resource_id:
                if node_rid != resource_id:
                    continue
                score += 3
            if text:
                if not (text in node_text or node_text in text):
                    continue
                score += 2
            if class_name:
                if not (class_name in node_class or node_class in class_name):
                    continue
                score += 1
            if score > best_score:
                best_score = score
                best_id = int(node_id)
        if best_id is None:
            return None, None
        return best_id, node_index.get(best_id)

    def _relocate_by_expected_resource_id(
        self,
        expected_resource_id: str,
        target: UIAssertionTarget,
        node_index: Dict[int, dict],
    ) -> tuple[int | None, dict[str, Any] | None]:
        expected = str(expected_resource_id or "").strip()
        if not expected:
            return None, None
        rematch_target = UIAssertionTarget(
            node_id=None,
            resource_id=expected,
            field=str(target.field or ""),
            text=str(target.text or ""),
            class_name=str(target.class_name or ""),
        )
        rematched_id, rematched_node = self._locate_widget_node(target=rematch_target, node_index=node_index)
        if rematched_node is not None and rematched_id is not None:
            return rematched_id, rematched_node

        fallback_target = UIAssertionTarget(
            node_id=None,
            resource_id=expected,
            field=str(target.field or ""),
            text="",
            class_name="",
        )
        return self._locate_widget_node(target=fallback_target, node_index=node_index)

    def _match_relation(self, relation: str, actual_value: Any, expected_content: str) -> bool:
        if relation == "exact_match":
            return self._exact_match(actual_value=actual_value, expected_content=expected_content)
        if relation == "contains":
            return expected_content in self._to_text(actual_value)
        if relation == "is_true":
            return self._to_bool(actual_value)
        if relation == "is_false":
            return not self._to_bool(actual_value)
        return False

    def _exact_match(self, actual_value: Any, expected_content: str) -> bool:
        if isinstance(actual_value, bool):
            expected_bool = self._parse_bool(expected_content)
            if expected_bool is None:
                return self._to_text(actual_value) == self._to_text(expected_content)
            return actual_value is expected_bool

        if isinstance(actual_value, (int, float)):
            return self._to_text(actual_value) == self._to_text(expected_content)

        return self._to_text(actual_value) == self._to_text(expected_content)

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    def _to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return False

        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "on", "checked", "enabled", "focused", "selected"}:
            return True
        if token in {"false", "0", "no", "n", "off", ""}:
            return False
        return bool(token)

    def _parse_bool(self, value: Any) -> bool | None:
        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off"}:
            return False
        return None

    def _record_post_oracle_artifact(
        self,
        step: int | None,
        result: PostOracleResult,
    ) -> None:
        if step is None:
            return
        try:
            self.audit.record_step_phase(
                artifact_kind="PostOracle",
                step=int(step),
                phase="post",
                payload=result.model_dump(exclude_none=True),
            )
        except Exception as exc:
            logger.error("写入 PostOracle 审计记录失败: %s", exc)
