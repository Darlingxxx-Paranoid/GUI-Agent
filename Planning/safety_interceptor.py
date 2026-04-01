"""Human-in-the-loop safety interception for high-risk intents."""

from __future__ import annotations

import logging
from typing import Iterable, List

from Oracle.contracts import StepContract
from Planning.planner import PlanResult

logger = logging.getLogger(__name__)


class SafetyInterceptor:
    """Block risky operations until user confirmation."""

    def __init__(self, high_risk_keywords: List[str]):
        self.keywords = [str(kw or "").lower() for kw in high_risk_keywords if str(kw or "").strip()]
        logger.info("SafetyInterceptor 初始化完成, 高风险关键词: %d 个", len(self.keywords))

    def check(self, plan: PlanResult, contract: StepContract | None = None) -> bool:
        risk_matches = self._detect_risk(plan=plan, contract=contract)
        if not risk_matches:
            logger.debug("安全检查通过: '%s'", plan.goal.summary)
            return True

        logger.warning(
            "⚠️ 检测到高风险操作: goal='%s', action=%s, 匹配关键词=%s",
            plan.goal.summary,
            plan.requested_action_type,
            risk_matches,
        )
        return self._request_human_confirmation(plan=plan, risk_keywords=risk_matches)

    def _detect_risk(self, plan: PlanResult, contract: StepContract | None = None) -> List[str]:
        texts: list[str] = [
            str(plan.goal.summary or ""),
            str(plan.goal.success_definition or ""),
            str(plan.input_text or ""),
            str(plan.reasoning or ""),
        ]

        target = plan.target or (contract.target if contract else None)
        if target is not None:
            for selector in target.selectors:
                texts.append(str(selector.value or ""))

        if contract is not None:
            for policy in contract.policies:
                texts.extend(self._flatten_values(policy.extra.values()))

        check_text = " ".join(texts).lower()
        matches: List[str] = []
        for keyword in self.keywords:
            if keyword and keyword in check_text and keyword not in matches:
                matches.append(keyword)
        return matches

    def _flatten_values(self, values: Iterable[object]) -> List[str]:
        out: list[str] = []
        for value in values:
            if isinstance(value, (list, tuple, set)):
                out.extend(self._flatten_values(value))
            elif isinstance(value, dict):
                out.extend(self._flatten_values(value.values()))
            else:
                out.append(str(value or ""))
        return out

    def _request_human_confirmation(self, plan: PlanResult, risk_keywords: List[str]) -> bool:
        print("\n" + "=" * 60)
        print("安全拦截 - 检测到高风险操作")
        print("=" * 60)
        print(f"  子目标: {plan.goal.summary}")
        print(f"  动作类型: {plan.requested_action_type}")
        print(f"  输入文本: {plan.input_text}")
        print(f"  风险关键词: {', '.join(risk_keywords)}")
        print("=" * 60)

        while True:
            try:
                response = input("是否允许执行该操作？(y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                logger.warning("用户中断输入, 默认拒绝高风险操作")
                return False

            if response in {"y", "yes", "是"}:
                logger.info("用户确认执行高风险操作: '%s'", plan.goal.summary)
                return True
            if response in {"n", "no", "否"}:
                logger.info("用户拒绝执行高风险操作: '%s'", plan.goal.summary)
                return False
            print("请输入 y 或 n")
