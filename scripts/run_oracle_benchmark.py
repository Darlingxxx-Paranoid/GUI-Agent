#!/usr/bin/env python3
"""
Run a small AndroidWorld-inspired benchmark suite for Oracle iteration.

Goals:
- Use real device execution (no dry-run by default).
- Isolate Oracle behavior from long-term memory replay.
- Save per-task logs and summary JSON for regression comparison.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import AgentConfig
from main import clean_data_directory, setup_logging
from agent_loop import AgentLoop


LOGGER = logging.getLogger(__name__)


DEFAULT_TASKS: List[dict] = []


PATTERN_RUNTIME_HARD = re.compile(r"runtime_hard", re.IGNORECASE)
PATTERN_HARD_BOUNDARY = re.compile(
    r"must_stay_in_app violated\(hard|expected_package mismatch\(hard|forbidden_package opened|forbidden_ui_risks hit",
    re.IGNORECASE,
)
PATTERN_SHORTCUT_FOCUS = re.compile(r"基于焦点证据直接通过")
PATTERN_UNCERTAIN_RETRY = re.compile(r"评估结果不确定")
PATTERN_TIMEOUT = re.compile(r"任务超时", re.IGNORECASE)


@dataclass
class TaskRunResult:
    task_id: str
    task: str
    success: bool
    duration_sec: float
    log_file: str
    runtime_hard_count: int
    hard_boundary_count: int
    focus_shortcut_count: int
    uncertain_observe_count: int
    error: str = ""
    failure_summary: str = ""


def _home(adb_path: str, serial: str) -> None:
    cmd = [adb_path]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["shell", "input", "keyevent", "KEYCODE_HOME"])
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    time.sleep(1.2)


def _count_patterns(log_path: str) -> dict:
    if not os.path.isfile(log_path):
        return {
            "runtime_hard_count": 0,
            "hard_boundary_count": 0,
            "focus_shortcut_count": 0,
            "uncertain_observe_count": 0,
        }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return {
        "runtime_hard_count": len(PATTERN_RUNTIME_HARD.findall(content)),
        "hard_boundary_count": len(PATTERN_HARD_BOUNDARY.findall(content)),
        "focus_shortcut_count": len(PATTERN_SHORTCUT_FOCUS.findall(content)),
        "uncertain_observe_count": len(PATTERN_UNCERTAIN_RETRY.findall(content)),
    }


def _summarize_failure(log_path: str, success: bool, error: str = "") -> str:
    if success:
        return ""
    if error:
        return f"exception: {error[:240]}"
    if not os.path.isfile(log_path):
        return "missing_log_file"

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    markers = [
        "任务超时",
        "✗ 子目标失败",
        "事中 Oracle 拒绝执行",
        "LLM 失败分析",
        "❌ 任务未完成",
    ]
    for marker in markers:
        for raw in reversed(lines):
            if marker not in raw:
                continue
            text = raw.strip()
            text = re.sub(
                r"^\d{4}-\d{2}-\d{2} .*?\|\s+\w+\s+\|\s+[^|]+\|\s+",
                "",
                text,
            )
            if " | " in text:
                text = text.split(" | ")[0].strip()
            return text[:280]

    for raw in reversed(lines):
        text = raw.strip()
        if text:
            return text[-280:]
    return "unknown_failure"


def run_single_task(
    task_id: str,
    task_text: str,
    serial: str,
    max_steps: int,
    run_dir: str,
    adb_path: str,
    dry_run: bool = False,
    disable_cv: bool = True,
) -> TaskRunResult:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:80] or "task"
    log_file = os.path.join(run_dir, f"{safe_name}.log")

    # Reset to launcher before each task to reduce cross-task context coupling.
    _home(adb_path=adb_path, serial=serial)
    setup_logging(log_level="INFO", log_file=log_file)

    start = time.time()
    error = ""
    success = False

    try:
        config = AgentConfig()
        config.adb_serial = serial
        config.max_steps = max_steps
        # Isolate Oracle behavior from experience replay and safety pauses.
        config.experience_similarity_threshold = 1.1
        config.high_risk_keywords = []

        agent = AgentLoop(config)
        # Keep memory out of this benchmark run.
        agent.replanner.save_task_experience = lambda *_args, **_kwargs: None
        if disable_cv:
            # Oracle 隔离模式：Dump 充分时关闭 CV；若 Dump 缺失则自动回退 CV，避免“0 观测”假失败。
            agent.perception._should_run_cv = lambda dump_elements: not bool(dump_elements)
        success = bool(agent.run(task=task_text, dry_run=dry_run))
    except Exception as exc:  # pragma: no cover - runtime-only safeguard
        error = str(exc)
        LOGGER.exception("Task failed with exception: %s", task_id)

    elapsed = time.time() - start
    counts = _count_patterns(log_file)
    failure_summary = _summarize_failure(log_file, success, error)

    return TaskRunResult(
        task_id=task_id,
        task=task_text,
        success=success,
        duration_sec=round(elapsed, 2),
        log_file=log_file,
        runtime_hard_count=counts["runtime_hard_count"],
        hard_boundary_count=counts["hard_boundary_count"],
        focus_shortcut_count=counts["focus_shortcut_count"],
        uncertain_observe_count=counts["uncertain_observe_count"],
        error=error,
        failure_summary=failure_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Oracle benchmark suite.")
    parser.add_argument("--serial", type=str, default="emulator-5554")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--label", type=str, default="oracle-bench")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-cv", action="store_true", default=False)
    parser.add_argument("--enable-cv", action="store_true")
    parser.add_argument(
        "--tasks-json",
        type=str,
        default="",
        help="Optional JSON file, format: [{'id':'...','task':'...'}, ...]",
    )
    parser.add_argument(
        "--adb-path",
        type=str,
        default=os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    )
    return parser.parse_args()


def _load_tasks(path: str) -> List[dict]:
    if not path:
        if DEFAULT_TASKS:
            return list(DEFAULT_TASKS)
        raise ValueError("No built-in benchmark tasks. Please provide --tasks-json.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tasks-json must be a JSON list")
    out = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("id") or f"task_{idx+1}")
        task = str(item.get("task") or "").strip()
        if not task:
            continue
        out.append({"id": task_id, "task": task})
    if not out:
        raise ValueError("No valid task entries in tasks-json")
    return out


def main() -> int:
    args = parse_args()
    clean_data_directory(os.path.join(REPO_ROOT, "data"))
    tasks = _load_tasks(args.tasks_json)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("data", "benchmarks", f"{args.label}-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    results: List[TaskRunResult] = []
    disable_cv = bool(args.disable_cv and (not args.enable_cv))
    for item in tasks:
        result = run_single_task(
            task_id=item["id"],
            task_text=item["task"],
            serial=args.serial,
            max_steps=args.max_steps,
            run_dir=run_dir,
            adb_path=args.adb_path,
            dry_run=args.dry_run,
            disable_cv=disable_cv,
        )
        results.append(result)

    success_count = sum(1 for r in results if r.success)
    total = len(results)
    aggregate = {
        "run_dir": run_dir,
        "label": args.label,
        "serial": args.serial,
        "max_steps": args.max_steps,
        "dry_run": bool(args.dry_run),
        "disable_cv": disable_cv,
        "success_count": success_count,
        "total_count": total,
        "success_rate": round((success_count / total) if total else 0.0, 3),
        "runtime_hard_total": sum(r.runtime_hard_count for r in results),
        "hard_boundary_total": sum(r.hard_boundary_count for r in results),
        "focus_shortcut_total": sum(r.focus_shortcut_count for r in results),
        "uncertain_observe_total": sum(r.uncertain_observe_count for r in results),
        "timeout_skipped_count": sum(
            1 for r in results if bool(PATTERN_TIMEOUT.search(r.failure_summary or ""))
        ),
        "results": [asdict(r) for r in results],
    }

    out_path = os.path.join(run_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if success_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
