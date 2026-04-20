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

## 3. 当前框架（参考 `method.md`）

入口：`main.py` -> `agent_loop.py:AgentLoop`

单步闭环：

1. `Perception`
2. `Planner (LLM)`
3. `Pre-Oracle (LLM)`
4. `Execution`
5. `Running-Oracle`
6. `Post-Oracle`

## 4. 模块映射

* `Perception`：
  * `Perception/dump_parser.py`（Dump 解析）
  * `Perception/uied_controls.py`（OCR+CV 可见控件提取）
* `Planner`：
  * `Planning/planner.py`
* `Pre-Oracle`：
  * `Oracle/pre_oracle.py`
* `Execution`：
  * `Execution/action_executor.py`
* `Running-Oracle`：
  * `Oracle/running_oracle.py`
* `Post-Oracle`：
  * `Oracle/post_oracle.py`

## 5. 关键输出

每一步会在 `data/` 下输出以下结构化结果：

* `Observation`
* `PlanResult`
* `AnchorResult`
* `StepContract`
* `ResolvedAction`
* `RunningOracle`
* `PostOracle`
* `StepResult`

## 6. 注意事项

* 当前仓库仅保留主流程必需模块，历史 benchmark/log 样例已移除。
* 若遇 API 参数错误，可优先检查 `utils/llm_client.py` 与 `config.py` 的模型配置。
