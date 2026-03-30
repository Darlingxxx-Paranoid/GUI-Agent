# Oracle 迭代总结（更新至 2026-03-29）

## 目标与执行原则
- 目标：持续测试 `main`，定位 Oracle/动作设计问题，最小改动修复，反复回归。
- 原则：保持 `pre oracle / running oracle / post oracle` 三段结构，不改 memory 机制主逻辑。
- 任务策略：难例不长时间卡住；触发超时后直接跳过并记录失败原因（符合“跳过+总结失败原因”）。

---

## 本轮主要修改（按模块）

### 1) `Execution/action_executor.py`
- 修复 `uiautomator dump` 陈旧文件污染问题：
  - dump 前清理本地/远端旧文件。
  - 识别 `could not get idle state` 时直接返回空 dump。
  - pull 前校验远端文件存在，pull 后校验本地文件存在与最小大小。
- 新增 `launch_app(package)`：
  - 优先 `monkey -p <pkg> -c LAUNCHER 1`。
  - 失败时回退 `resolve-activity + am start -n`。

### 2) `config.py` / `main.py` / `agent_loop.py`
- 增加任务级超时：
  - `config.max_task_seconds`（默认 0 表示不限时）。
  - CLI 参数 `--max-task-seconds`。
  - `AgentLoop.run` 内按 wall-clock 超时中断并记录 `任务超时...跳过该任务`。
- `agent_loop._execute_action` 新增 `launch_app` 动作分支。
- 运行期回退增强：引入自适应 back（首次无状态变化则补一次）。

### 3) `scripts/run_oracle_benchmark.py`
- 扩展 benchmark 统计与失败可读性：
  - `TaskRunResult.failure_summary`。
  - `_summarize_failure(...)` 从日志尾部抽取关键失败原因。
  - 聚合指标 `timeout_skipped_count`。
  - 支持 `--max-task-seconds` 传入 agent。

### 4) `Execution/action_mapper.py`
- 新增确定性映射，减少 CV-only 场景误点：
  - Clock 底部 tab 定向映射（Timer/Stopwatch/Alarm...）。
  - Timer 键盘定向映射（数字键、00、回删；clear/reset 语义触发 long-press backspace）。
  - Launcher 搜索栏定向映射（顶部中心点，避免误点左上角图标）。
  - Launcher 打开 App 定向映射（按文本匹配图标；找不到时回退 `launch_app(package)`）。
- 滑动兜底增强：
  - `swipe/scroll` 在“从底部上滑/打开 app drawer”等语义下强制走默认屏幕手势。

### 5) `Planning/oracle_pre.py`
- 边界约束增强：
  - 根据子目标语义推断目标 package（打开某 app 场景）。
  - 打开 app 场景下允许跨包但保留 `expected_package` 校验（避免“点错 app 也判成功”）。
- 快路径支持 `action=scroll`，避免无谓走 LLM prompt 分支。
- 增加 prompt format 防护：模板占位缺失时回退快速约束，不再直接异常中断。

### 6) `Evaluation/evaluator.py`（前一轮已落地并沿用）
- 移除焦点捷径直通 success。
- 弱视觉变化需有动作归因，避免“无实质变化也 success”。

---

## 关键回归结果

### A. 稳定期（修改后第一轮）
- `oracle-iter2-general-20260329-133444`
- 结果：`4/5` 成功，`timeout_skipped_count=1`
- 失败：仅 `ClockTimerEntry_00_00_10` 超时跳过。

### B. 引入新映射后的一次退化
- `oracle-iter3-general-20260329-140942`
- 结果：`2/5` 成功，`timeout_skipped_count=3`
- 新增失败：`OpenAppTaskEval_Chrome`、`OpenAppTaskEval_Settings` 也超时。
- 主要现象：
  - launcher/app drawer 搜索栏与图标定位抖动。
  - 子目标在“回到主屏/再搜索”路径中循环，触发超时。

### C. 开应用专项回归（用于定位）
- `oracle-openapps-v1-20260329-143517`：`0/3`
- `oracle-chrome-only-v2-20260329-150226`：`0/1`（出现单任务时长远超设置值）。

### D. Timer 专项回归
- `oracle-debug-timer-*` 三轮均未完成（均超时跳过）。
- 已确认改进点：Timer 键盘点击与 clear/reset 映射生效；
- 仍失败主因：起始脏状态 + OCR易错 + 评估归因不充分，导致步骤推进效率低。

---

## 当前已确认的问题与失败原因

### 1) 任务超时未严格受控（高优先级）
- 现象：`--max-task-seconds` 设为 180，但单任务实际跑到 3660s。
- 推断原因：超时判断在循环边界，但单步内存在长阻塞调用（LLM/CV/OCR），导致超时不能及时打断。
- 影响：难例会拖长总回归时长，违反“及时跳过”目标。

### 2) launcher 场景动作不稳定（中高优先级）
- 现象：
  - 搜索栏点击偶发点进 Clock。
  - App icon 识别不稳时产生 replan/back 循环。
- 已做修复：定向映射 + launch_app 回退。
- 仍需验证：在长回归中是否显著提升成功率。

### 3) Timer 难例仍不稳定（中优先级）
- 现象：`ClockTimerEntry_00_00_10` 持续超时跳过。
- 主要原因：
  - 初始 timer 值脏（非 00:00:00）。
  - OCR 对 `07/77` 等读值噪声大。
  - 评估对“数字键动作→目标值变化”的归因证据不足。

---

## 本轮结论（实话实说）
- 已完成：
  - dump 陈旧文件问题修复并验证有效。
  - 超时跳过机制与失败原因沉淀机制已打通。
  - 若干动作层确定性映射已接入（Clock tab、Timer keypad、Launcher search/app）。
- 未完成：
  - “general 任务稳定”目标尚未达标（出现从 `4/5` 退化到 `2/5` 的阶段）。
- 当前状态：
  - 系统具备“遇难例不死磕、自动跳过并给出失败原因”的能力；
  - 但仍需进一步收敛 launcher 打开 App 流程与超时打断机制，才能恢复/超过 `iter2` 稳定性。

---

## 本轮关键文件
- `Execution/action_executor.py`
- `Execution/action_mapper.py`
- `Planning/oracle_pre.py`
- `agent_loop.py`
- `config.py`
- `main.py`
- `scripts/run_oracle_benchmark.py`
- `data/benchmarks/oracle-iter2-general-20260329-133444/summary.json`
- `data/benchmarks/oracle-iter3-general-20260329-140942/summary.json`
- `data/benchmarks/oracle-openapps-v1-20260329-143517/summary.json`
- `data/benchmarks/oracle-chrome-only-v2-20260329-150226/summary.json`
- `data/benchmarks/oracle-debug-timer-20260329-135008/summary.json`
- `data/benchmarks/oracle-debug-timer-v2-20260329-135755/summary.json`
- `data/benchmarks/oracle-debug-timer-v3-20260329-140424/summary.json`
