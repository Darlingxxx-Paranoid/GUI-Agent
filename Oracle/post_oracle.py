"""Post-Oracle: evaluate after-action dump tree against UI assertions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from Oracle.pre_oracle import UIAssertion, UIAssertionContract, UIAssertionTarget
from utils.audit_recorder import AuditRecorder

logger = logging.getLogger(__name__)


class PostOracleResult(BaseModel):
    """Post-Oracle output contract."""

    model_config = ConfigDict(extra="forbid")

    is_goal_complete: bool
    action_history: Optional[List[Any]] = None


class PostOracle:
    """Evaluate assertion evidence from after-action dump tree."""

    def __init__(self):
        self.audit = AuditRecorder(component="post_oracle")
        logger.info("PostOracle 初始化完成")

    def evaluate(
        self,
        dump_tree: dict,
        assertions: UIAssertionContract | Sequence[UIAssertion] | Sequence[dict[str, Any]],
        action_history: Sequence[Any] | None = None,
        step: int | None = None,
    ) -> PostOracleResult:
        if not isinstance(dump_tree, dict):
            logger.error("Post-Oracle 评估失败: dump_tree 类型非法(type=%s)", type(dump_tree).__name__)
            raise ValueError("dump_tree 必须是 dict")

        assertion_items = self._normalize_assertions(assertions)
        node_index = self._index_dump_tree(dump_tree)
        activity, package = self._extract_app_context(dump_tree)

        failed_messages: list[str] = []
        for idx, assertion in enumerate(assertion_items, start=1):
            ok, reason = self._check_assertion(
                assertion=assertion,
                node_index=node_index,
                activity=activity,
                package=package,
            )
            if not ok:
                failed_messages.append(f"Assertion[{idx}] {reason}")

        is_goal_complete = len(failed_messages) == 0
        result = PostOracleResult(
            is_goal_complete=is_goal_complete,
            action_history=(list(action_history or []) if not is_goal_complete else None),
        )
        self._record_post_oracle_artifact(
            step=step,
            result=result,
            failure_messages=failed_messages,
            assertion_count=len(assertion_items),
        )

        if is_goal_complete:
            logger.info("Post-Oracle 评估通过: assertions=%d", len(assertion_items))
        else:
            logger.info(
                "Post-Oracle 评估失败: assertions=%d, failed=%d",
                len(assertion_items),
                len(failed_messages),
            )
            for item in failed_messages:
                logger.info("Post-Oracle 失败详情: %s", item)

        return result

    def to_output(self, result: PostOracleResult) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"is_goal_complete": bool(result.is_goal_complete)}
        if not result.is_goal_complete:
            payload["action_history"] = list(result.action_history or [])
        return payload

    def _normalize_assertions(
        self,
        assertions: UIAssertionContract | Sequence[UIAssertion] | Sequence[dict[str, Any]],
    ) -> list[UIAssertion]:
        if isinstance(assertions, UIAssertionContract):
            items = list(assertions.assertions or [])
        elif isinstance(assertions, Sequence) and not isinstance(assertions, (str, bytes)):
            items = []
            for value in assertions:
                if isinstance(value, UIAssertion):
                    items.append(value)
                else:
                    items.append(UIAssertion.model_validate(value))
        else:
            raise ValueError("assertions 必须是 UIAssertionContract 或 UIAssertion 列表")

        if not items:
            raise ValueError("assertions 不能为空")
        return items

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
                    return False, (
                        f"resource_id 不匹配(node_id={node_id}, expected='{expected_rid}', actual='{node_rid}')"
                    )

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
        failure_messages: list[str],
        assertion_count: int,
    ) -> None:
        if step is None:
            return
        try:
            payload: Dict[str, Any] = {
                "is_goal_complete": bool(result.is_goal_complete),
                "assertion_count": int(assertion_count),
                "failed_assertion_count": int(len(failure_messages)),
            }
            if failure_messages:
                payload["failure_messages"] = list(failure_messages)
            if not result.is_goal_complete:
                payload["action_history"] = list(result.action_history or [])
            self.audit.record_step(
                artifact_kind="PostOracle",
                step=int(step),
                payload=payload,
            )
        except Exception as exc:
            logger.error("写入 PostOracle 审计记录失败: %s", exc)
