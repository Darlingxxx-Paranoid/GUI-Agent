"""Replanning decisions based on StepEvaluation (V3.1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from Memory.memory_manager import MemoryManager
from Oracle.contracts import StepEvaluation, to_plain_dict
from Perception.context_builder import UIState

logger = logging.getLogger(__name__)


@dataclass
class ReplanDecision:
    action: str  # retry/back/replan/abort
    reason: str = ""
    back_steps: int = 1


class Replanner:
    def __init__(self, llm_client, memory_manager: MemoryManager, dead_end_threshold: int = 3):
        self.llm = llm_client
        self.memory = memory_manager
        self.dead_end_threshold = max(2, int(dead_end_threshold or 3))
        self.consecutive_failures = 0
        logger.info("Replanner 初始化完成 (V3.1), dead_end_threshold=%d", self.dead_end_threshold)

    def handle_failure(
        self,
        subgoal_description: str,
        evaluation: StepEvaluation,
        ui_state: UIState,
    ) -> ReplanDecision:
        self.consecutive_failures += 1

        decision = str(getattr(evaluation, "decision", "") or "")
        recommended = (
            str(getattr(getattr(evaluation, "recommended_action", None), "kind", "") or "").strip().lower()
        )
        reason = self._format_reason(evaluation)

        self.memory.short_term.add_failure(f"[{subgoal_description}] {reason}")

        if self.consecutive_failures >= self.dead_end_threshold:
            self.consecutive_failures = 0
            return ReplanDecision(
                action="back",
                reason=f"连续失败达到阈值，回退恢复上下文: {reason}",
                back_steps=1,
            )

        if recommended == "backtrack":
            return ReplanDecision(action="back", reason=f"follow_evaluation: {reason}", back_steps=1)
        if recommended == "replan":
            return ReplanDecision(action="replan", reason=f"follow_evaluation: {reason}")
        if recommended == "abort":
            return ReplanDecision(action="abort", reason=f"follow_evaluation: {reason}")
        if recommended in {"retry", "observe"}:
            return ReplanDecision(action="retry", reason=f"follow_evaluation: {reason}")

        if decision == "uncertain":
            return ReplanDecision(action="retry", reason=f"oracle_uncertain: {reason}")
        if decision == "fail":
            return ReplanDecision(action="replan", reason=f"oracle_fail: {reason}")

        return ReplanDecision(action="retry", reason=f"fallback_retry: {reason}")

    def handle_success(self, task_description: str):
        self.consecutive_failures = 0

    def save_task_experience(self, task_description: str, success: bool):
        history = self.memory.short_term.history
        triplets = []
        for step in history:
            if str(step.get("result") or "") != "success":
                continue
            contract = step.get("contract") or {}
            action = step.get("action") or {}
            evaluation = step.get("evaluation") or {}
            if not isinstance(contract, dict) or not isinstance(action, dict) or not isinstance(evaluation, dict):
                continue
            triplets.append(
                {
                    "step": int(step.get("step", len(triplets) + 1)),
                    "goal": step.get("goal") or (contract.get("goal") if isinstance(contract, dict) else {}),
                    "contract": contract,
                    "action": action,
                    "evaluation": evaluation,
                }
            )

        if success and triplets:
            self.memory.save_experience(
                task_description=task_description,
                step_triplets=triplets,
                success=True,
                metadata={
                    "source": "replanner.save_task_experience.v3",
                    "step_triplets_count": len(triplets),
                },
            )
            logger.info("任务经验已沉淀(v3): %d steps", len(triplets))
        elif not success:
            logger.info("任务失败，不沉淀经验")

    def reset(self):
        self.consecutive_failures = 0

    def _format_reason(self, evaluation: StepEvaluation) -> str:
        reason_parts = []
        if evaluation.assessments:
            top = evaluation.assessments[-1]
            reason_parts.append(str(top.message or "").strip())
        reason_parts.append(f"decision={evaluation.decision}")
        rec = getattr(evaluation, "recommended_action", None)
        if rec is not None:
            reason_parts.append(f"recommended={rec.kind}")
        return "; ".join(part for part in reason_parts if part)
