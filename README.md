# GUI Agent 🤖

基于 Oracle 反馈驱动的 Android GUI 自动化代理。系统采用 ReAct 单步闭环，在每一步执行中完成：感知 -> 规划 -> 执行 -> 评估 -> 重规划。

## 1. 环境准备

1. Python：建议 `3.10+`（项目当前也可在 `3.9` 运行）
2. 虚拟环境：建议使用项目内 `myvenv`
3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. ADB 连接：确保可通过 `adb devices` 看到设备（真机或模拟器）

## 2. 启动方式

基础运行：

```bash
python main.py --task "打开设置，连接WiFi"
```

指定设备与最大步数：

```bash
python main.py --task "打开设置，连接WiFi" --serial emulator-5554 --max-steps 50
```

Dry-run（不下发真实 ADB 动作）：

```bash
python main.py --task "测试任务" --dry-run --log-level DEBUG
```

参数：

* `--task/-t`：任务描述（必填）
* `--serial/-s`：设备序列号
* `--dry-run`：流程联调
* `--log-level`：`DEBUG|INFO|WARNING|ERROR`
* `--log-file`：日志输出路径
* `--max-steps`：单任务最大步数

## 3. 整体架构

入口：`main.py` -> `agent_loop.py:AgentLoop`

单步链路：

1. `Perception`：解析当前 UI（dump + 可选 CV/OCR）
2. `Planner`：生成下一步 `SubGoal`
3. `Pre Oracle`：生成 `PreConstraints`
4. `SafetyInterceptor`：高风险拦截
5. `ActionMapper`：`SubGoal -> Action`
6. `Running Oracle`：执行前/后检查
7. `ActionExecutor`：ADB 执行动作
8. `Evaluator (Post Oracle)`：变化证据评估与语义兜底
9. `Replanner`：失败恢复（back/replan/abort）
10. `Memory`：任务成功后沉淀经验

## 4. 模块职责

### 4.1 Perception

关键文件：

* `Perception/perception_manager.py`
* `Perception/dump_parser.py`
* `Perception/context_builder.py`

职责：

* 构建统一 `UIState`（widgets、activity/package、keyboard 等）
* 产出 `data/context/*.json` 供复盘

### 4.2 Planning

关键文件：

* `Planning/planner.py`
* `Planning/oracle_pre.py`

职责：

* 按当前状态生成单步 `SubGoal`
* 生成结构化 `PreConstraints` 供运行时与后评估复用

### 4.3 Execution

关键文件：

* `Execution/action_mapper.py`
* `Execution/action_executor.py`
* `Execution/oracle_runtime.py`

职责：

* 子目标动作化、ADB 执行
* 事中防死循环和边界守门

### 4.4 Evaluation + Replan

关键文件：

* `Evaluation/evaluator.py`
* `Evaluation/replanner.py`

职责：

* 先机器化证据匹配，再在不确定时调用 LLM
* 失败时决定 back/replan/abort

### 4.5 Memory

关键文件：

* `Memory/memory_manager.py`
* `Memory/experience_store.py`

职责：

* 保存短期轨迹与长期成功经验

## 5. Oracle 设计（当前实现）

系统保持三段 Oracle：

1. **Pre Oracle**：定义变化验证计划（`action_anchor/success_evidence_plan/boundary_constraints/semantic_goal`）
2. **Running Oracle**：执行前后做硬边界守门与软告警
3. **Post Oracle**：基于 `delta_facts` 做证据匹配和决策（`success/fail/uncertain`）

核心原则：

* Oracle 关注“变化是否发生”，而不是“场景属于哪一类”。

## 6. 本次关键更新（2026-03）

### [本次改动] Oracle 主链路

* 边界判定从“包名变化即硬失败”升级为“包关系 + 严重度”联合判定
* 新增 `package_mismatch_severity` 与 `related_package_tokens`
* 相关包漂移默认 soft，不强制回退

### [本次改动] Post Oracle 机器决策

* 引入 `delta_facts` + `signal match` 的机器化评估层
* 仅在 `uncertain` 时才走 LLM 语义补充
* 支持 `observe_again`（自动再观测一次）

### [本次改动] 证据稳定性

* 快路径 `region_changed` 从锚点依赖转为 local/global 变化优先
* 降低锚点不可用导致的误判

### [本次改动] 执行层稳定性

* 修复无目标 `swipe` 退化为 `tap`
* 输入/聚焦场景加入可编辑控件重定向
* 增加 adaptive back（首次 back 无变化则自动补一次）

## 7. 回归结果（本次迭代）

通用任务矩阵（设置/时钟/Chrome）回归：4/4 成功。

观测指标：

* `runtime_hard = 0`
* `hard_boundary = 0`
* `semantic_fails = 0`

说明：完整说明见 `报告.md`。

## 8. 注意事项

* 经验复用机制当前仍可能在跨场景下产生链路污染（本次未重构）
* 若遇 API 400，可检查 `utils/llm_client.py` 的 token 参数与模型兼容性
