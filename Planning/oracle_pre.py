"""Pre-Oracle: build StepContract from PlanResult and raw dump tree."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Planning.planner import PlanResult
from prompt.oracle_pre_prompt import ORACLE_PRE_SYSTEM_PROMPT, ORACLE_PRE_USER_PROMPT
from utils.audit_recorder import AuditRecorder
from utils.llm_client import LLMRequest

logger = logging.getLogger(__name__)

TargetCategory = Literal["widget", "activity", "package"]
RelationType = Literal["exact_match", "contains","is_true","is_false"]
DumpField = Literal[
    "text",
    "class",
    "content-desc",
    "checked",
    "enabled",
    "focused",
    "selected"
]


class ExpectationTarget(BaseModel):
    """Widget expectation target reference."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    node_id: int
    resource_id: str = Field(default="")
    field: DumpField


class ContractExpectation(BaseModel):
    """Expectation contract item."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    Target_category: TargetCategory
    Target: Optional[ExpectationTarget] = None
    Relation: RelationType
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_shape(self) -> "ContractExpectation":
        if self.Target_category == "widget":
            if self.Target is None:
                logger.error("StepContract校验失败: widget expectation 缺少 Target")
                raise ValueError("Target must be provided when Target_category='widget'.")
            return self

        if self.Target is not None:
            logger.error(
                "StepContract校验失败: 非widget expectation 不允许 Target(category=%s)",
                self.Target_category,
            )
            raise ValueError("Target must be null when Target_category is activity/package.")
        return self


class StepContract(BaseModel):
    """Pre-Oracle structured output."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    success_definition: str = Field(min_length=1)
    Expectations: list[ContractExpectation] = Field(min_length=1)


class OraclePre:
    """Generate StepContract via LLM from plan + raw dump tree."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.audit = AuditRecorder(component="oracle_pre")
        logger.info("OraclePre 初始化完成 (raw dump + structured output)")

    def generate_contract(
        self,
        plan: PlanResult,
        dump_tree: dict,
        step: int | None = None,
    ) -> StepContract:
        if self.llm is None:
            logger.error("StepContract 生成失败: llm_client 未初始化")
            raise RuntimeError("OraclePre 需要 llm_client 才能生成 StepContract")
        if not isinstance(dump_tree, dict):
            logger.error("StepContract 生成失败: dump_tree 类型非法(type=%s)", type(dump_tree).__name__)
            raise ValueError("dump_tree 必须是 dict")

        node_index, rid_index = self._index_dump_tree(dump_tree)
        if not node_index:
            logger.error("StepContract 生成失败: dump_tree 中未找到 node_id")
            raise ValueError("dump_tree 中未找到任何 node_id")

        plan_payload = plan.model_dump()
        plan_payload.pop("reasoning", None)
        logger.info("Pre-Oracle 输入已移除 PlanResult.reasoning 字段")

        user_prompt = ORACLE_PRE_USER_PROMPT.format(
            plan_json=json.dumps(plan_payload, ensure_ascii=False),
            dump_tree_json=json.dumps(dump_tree, ensure_ascii=False),
        )
        request = LLMRequest(
            system=ORACLE_PRE_SYSTEM_PROMPT,
            user=user_prompt,
            response_format=StepContract,
            audit_meta={
                "artifact_kind": "StepContract",
                "step": step,
                "stage": "generate_contract",
            },
        )

        logger.info(
            "开始生成 StepContract: plan_action=%s, dump_nodes=%d",
            plan.action_type,
            len(node_index),
        )
        try:
            parsed = self.llm.chat(request)
            contract = parsed if isinstance(parsed, StepContract) else StepContract.model_validate(parsed)
            self._validate_widget_targets(
                contract=contract,
                node_index=node_index,
                rid_index=rid_index,
            )
            self._record_oracle_pre_artifact(step=step, payload=contract)
            logger.info(
                "StepContract 生成完成: expectations=%d, success_definition=%s",
                len(contract.Expectations),
                contract.success_definition,
            )
            return contract
        except Exception as exc:
            self._record_oracle_pre_error(step=step, error=str(exc))
            logger.error("StepContract 生成失败: %s", exc)
            raise

    def _index_dump_tree(self, dump_tree: dict) -> tuple[Dict[int, dict], Dict[str, list[int]]]:
        index: Dict[int, dict] = {}
        rid_index: Dict[str, list[int]] = {}

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return

            raw_node_id = node.get("node_id")
            if isinstance(raw_node_id, int):
                index[raw_node_id] = node
            elif raw_node_id is not None:
                try:
                    idx = int(raw_node_id)
                    index[idx] = node
                except Exception:
                    pass

            if isinstance(raw_node_id, (int, str)):
                try:
                    current_id = int(raw_node_id)
                except Exception:
                    current_id = None
                if current_id is not None:
                    rid = str(node.get("resource-id", "") or "")
                    if rid not in rid_index:
                        rid_index[rid] = []
                    rid_index[rid].append(current_id)

            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child)

        walk(dump_tree)
        return index, rid_index

    def _validate_widget_targets(
        self,
        contract: StepContract,
        node_index: Dict[int, dict],
        rid_index: Dict[str, list[int]],
    ) -> None:
        for idx, expectation in enumerate(contract.Expectations, start=1):
            if expectation.Target_category != "widget":
                continue

            target = expectation.Target
            if target is None:
                logger.error("StepContract校验失败: Expectation[%d] widget target 为空", idx)
                raise ValueError(f"Expectation[{idx}] widget target 不能为空")

            node = node_index.get(int(target.node_id))
            if node is None:
                logger.error(
                    "StepContract校验失败: Expectation[%d] 引用不存在 node_id=%s",
                    idx,
                    target.node_id,
                )
                raise ValueError(
                    f"Expectation[{idx}] 引用了不存在的 node_id={target.node_id}"
                )

            field = str(target.field or "").strip()
            if field not in node:
                logger.error(
                    "StepContract校验失败: Expectation[%d] field=%s 不存在于 node_id=%s",
                    idx,
                    field,
                    target.node_id,
                )
                raise ValueError(
                    f"Expectation[{idx}] field='{field}' 不存在于 node_id={target.node_id} 原始字段中"
                )

            node_rid = str(node.get("resource-id", "") or "")
            target_rid = str(target.resource_id or "")
            if target_rid != node_rid:
                logger.error(
                    "StepContract校验失败: Expectation[%d] resource_id 不匹配(node_id=%s, target=%s, dump=%s)",
                    idx,
                    target.node_id,
                    target_rid,
                    node_rid,
                )
                raise ValueError(
                    f"Expectation[{idx}] resource_id 与 node_id={target.node_id} 的 dump 值不一致"
                )

            rid_node_ids = rid_index.get(target_rid, [])
            if int(target.node_id) not in rid_node_ids:
                logger.error(
                    "StepContract校验失败: Expectation[%d] node_id=%s 不在 resource_id=%s 索引内",
                    idx,
                    target.node_id,
                    target_rid,
                )
                raise ValueError(
                    f"Expectation[{idx}] node_id={target.node_id} 未通过 resource_id 索引校验"
                )
        logger.info("StepContract widget target 校验通过")

    def _record_oracle_pre_artifact(self, step: int | None, payload: StepContract) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="StepContract",
                step=int(step),
                payload=payload,
            )
        except Exception as exc:
            logger.error("写入 StepContract 审计记录失败: %s", exc)

    def _record_oracle_pre_error(self, step: int | None, error: str) -> None:
        if step is None:
            return
        try:
            self.audit.record_step(
                artifact_kind="StepContract",
                step=int(step),
                payload={
                    "stage": "generate_contract_error",
                    "error": str(error or ""),
                },
                llm=True,
                append=True,
            )
        except Exception as exc:
            logger.error("写入 StepContract 错误审计记录失败: %s", exc)
