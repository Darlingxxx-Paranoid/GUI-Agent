"""Post-Oracle: evaluate after-action dump tree against expectations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from Oracle.pre_oracle import ContractExpectation, StepContract
from utils.audit_recorder import AuditRecorder

logger = logging.getLogger(__name__)


class PostOracleResult(BaseModel):
    """Post-Oracle output contract."""

    model_config = ConfigDict(extra="forbid")

    is_goal_complete: bool
    action_history: Optional[List[Any]] = None


class PostOracle:
    """Evaluate expectation evidence from after-action dump tree."""

    def __init__(self):
        self.audit = AuditRecorder(component="post_oracle")
        logger.info("PostOracle 初始化完成")

    def evaluate(
        self,
        dump_tree: dict,
        expectations: StepContract | Sequence[ContractExpectation] | Sequence[dict[str, Any]],
        action_history: Sequence[Any] | None = None,
        step: int | None = None,
    ) -> PostOracleResult:
        if not isinstance(dump_tree, dict):
            logger.error("Post-Oracle 评估失败: dump_tree 类型非法(type=%s)", type(dump_tree).__name__)
            raise ValueError("dump_tree 必须是 dict")

        expectation_items = self._normalize_expectations(expectations)
        node_index = self._index_dump_tree(dump_tree)
        activity, package = self._extract_app_context(dump_tree)

        failed_messages: list[str] = []
        for idx, expectation in enumerate(expectation_items, start=1):
            ok, reason = self._check_expectation(
                expectation=expectation,
                node_index=node_index,
                activity=activity,
                package=package,
            )
            if not ok:
                failed_messages.append(f"Expectation[{idx}] {reason}")

        is_goal_complete = len(failed_messages) == 0
        result = PostOracleResult(
            is_goal_complete=is_goal_complete,
            action_history=(list(action_history or []) if not is_goal_complete else None),
        )
        self._record_post_oracle_artifact(
            step=step,
            result=result,
            failure_messages=failed_messages,
            expectation_count=len(expectation_items),
        )

        if is_goal_complete:
            logger.info("Post-Oracle 评估通过: expectations=%d", len(expectation_items))
        else:
            logger.info(
                "Post-Oracle 评估失败: expectations=%d, failed=%d",
                len(expectation_items),
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

    def _normalize_expectations(
        self,
        expectations: StepContract | Sequence[ContractExpectation] | Sequence[dict[str, Any]],
    ) -> list[ContractExpectation]:
        if isinstance(expectations, StepContract):
            items = list(expectations.Expectations or [])
        elif isinstance(expectations, Sequence) and not isinstance(expectations, (str, bytes)):
            items = []
            for value in expectations:
                if isinstance(value, ContractExpectation):
                    items.append(value)
                else:
                    items.append(ContractExpectation.model_validate(value))
        else:
            raise ValueError("expectations 必须是 StepContract 或 ContractExpectation 列表")

        if not items:
            raise ValueError("expectations 不能为空")
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

    def _check_expectation(
        self,
        expectation: ContractExpectation,
        node_index: Dict[int, dict],
        activity: str,
        package: str,
    ) -> tuple[bool, str]:
        category = str(expectation.Target_category or "").strip().lower()
        relation = str(expectation.Relation or "").strip().lower()
        expected_content = str(expectation.content or "")

        if category == "widget":
            target = expectation.Target
            if target is None:
                return False, "widget 目标为空"

            node_id = int(target.node_id)
            node = node_index.get(node_id)
            if node is None:
                return False, f"引用 node_id={node_id} 不存在"

            expected_rid = str(target.resource_id or "")
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

        return False, f"不支持的 Target_category='{category}'"

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
        expectation_count: int,
    ) -> None:
        if step is None:
            return
        try:
            payload: Dict[str, Any] = {
                "is_goal_complete": bool(result.is_goal_complete),
                "expectation_count": int(expectation_count),
                "failed_expectation_count": int(len(failure_messages)),
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
