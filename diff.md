# GUI-Agent 最近 3 个提交改动总结

## 1) `c96b7818658a7a6cc44aa190a2e0ef2a88b0a412`（`fix: adb`）
- 作者：jiangqinchen
- 时间：2026-03-27 17:19:05 +0800
- 规模：`176 files changed, 32382 insertions(+), 3624 deletions(-)`（大量 `data/*` 样本与日志）

### 核心改动
- `Execution/action_executor.py`
  - 增强 ADB 可执行路径解析（显式路径/环境变量/常见 SDK 路径）。
  - 文本输入改为更鲁棒方案：ASCII 分块输入、非 ASCII 广播/base64 广播、支持多行输入。
  - 增加软键盘可见性探测 `get_keyboard_visible()`，并增强当前包名/Activity 解析。
- `Perception/*` + `Perception/dump_parser.py`
  - `UIState` 新增 `keyboard_visible`。
  - 解析并保留更多控件属性：`editable/focused/checkable/...`。
  - dump 语义足够时支持跳过 CV，降低时延。
- `Evaluation/evaluator.py`
  - 评估接入 `action` 上下文。
  - 增加输入动作效果验证、焦点验证、目标区域匹配等逻辑。
  - “是否有有效 UI 变化”判断纳入键盘/焦点变化。
- `agent_loop.py`
  - 评估调用传入当前 `action`。
  - 输入动作前增加“目标是否已就绪”判断，避免重复预点击。
  - 记录更完整 step 元数据（包名、activity、keyboard、acceptance 等）。
- `Planning/planner.py` + `Evaluation/replanner.py` + `Memory/*`
  - 经验回放支持进度索引，避免每次都从经验第一步开始。
  - 经验沉淀从“原始动作日志”升级为“语义步骤”并带 schema/version 元数据。

---

## 2) `5a623689dc2725938753c903c36c301dac3e28e1`（`收敛结论`）
- 作者：jiangqinchen
- 时间：2026-03-30 21:09:16 +0800
- 规模：`198 files changed, 17118 insertions(+), 28120 deletions(-)`
  - 代码层（排除 `data/*` 与 benchmark 产物）约：`23 files, +5755 / -573`

### 核心改动
- Oracle 契约重构（Pre/Running/Post）
  - `Planning/oracle_pre.py` + `prompt/oracle_pre_prompt.py`
  - 约束结构由旧字段收敛为：`action_anchor`、`success_evidence_plan`、`boundary_constraints`、`semantic_goal`。
  - 增加包边界强弱（`package_mismatch_severity`）及相关 token 机制。
- Post Oracle（`Evaluation/evaluator.py`）大幅增强
  - machine-first：先提取 `delta_facts`，再匹配 success evidence signals。
  - 输出 `decision=success|fail|uncertain`、`confidence`、`suggested_next_action`。
  - 仅在必要时触发语义补判；加入剩余时限预算控制。
- Running Oracle（`Execution/oracle_runtime.py`）增强
  - 包漂移判断更细（same/related/unrelated），减少“一刀切”硬失败。
- 动作层增强
  - `Execution/action_mapper.py` 新增大量确定性映射与归一化（launcher/search/open app、timer keypad、clock tab、settings toggle、long_press/scroll 等）。
  - `Execution/action_executor.py` 增加 `launch_app()`、ADB 调用超时控制、dump 稳定化处理。
- 任务时限/基准流程增强
  - `agent_loop.py`、`main.py`、`config.py`、`utils/llm_client.py`
  - 加入任务级 deadline 与 LLM 动态 timeout；限制重试拖慢。
  - `scripts/run_oracle_benchmark.py` 重写，增加清理、失败摘要、timeout 跳过统计。
- 文档与沉淀
  - 新增 `.trae/skills/gui-agent-oracles/SKILL.md`。
  - `README.md`、`summary.md`、`summary_3.md`、`报告.md` 等同步更新。

---

## 3) `667ad9c9f8f2dd8ef5a9121808c7a431db77d935`（`移除所有specified设置`）
- 作者：jiangqinchen
- 时间：2026-03-30 21:35:25 +0800
- 规模：`9 files changed, 180 insertions(+), 1261 deletions(-)`

### 核心改动（对上一个提交做“去特化/去指定化”）
- `Execution/action_mapper.py`
  - 删除大量场景特化映射（launcher/search/open app、clock/timer、settings 指定逻辑）。
  - 保留更通用的动作映射与默认滑动策略。
- `Planning/oracle_pre.py`
  - 删除 launcher 场景识别、目标包名推断、alias 包映射等“指定化”逻辑。
  - 边界策略改为更通用默认规则（以 transition 类型控制是否允许跨包）。
- `Planning/planner.py`
  - 删除“打开 App 已完成”客观判定逻辑与部分任务特化 guard。
  - 主题偏离检测改为通用关键词重叠机制。
- `Evaluation/evaluator.py`
  - 删除控制态切换（pause/resume）的专门成功捷径逻辑。
- `Execution/action_executor.py` + `agent_loop.py`
  - 删除 `launch_app` 执行链路。
- `scripts/run_oracle_benchmark.py`
  - 清空内置默认任务，强制通过 `--tasks-json` 提供任务。
- 其他
  - `prompt/oracle_pre_prompt.py` 示例字段改为更中性。
  - `.gitignore` 调整：`data/`、新增 `.trae/` 忽略。

---

## 总体脉络（一句话）
- `c96b781`：先把 ADB/输入/感知与经验回放基础打稳。
- `5a62368`：把 Oracle 评估框架全面收敛到 evidence-driven。
- `667ad9c`：再把上一轮中“过于指定化”的策略回退到通用实现。
