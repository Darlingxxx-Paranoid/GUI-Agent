# GUI Agent 🤖

基于 Oracle 反馈驱动的智能 GUI 代理。该代理通过 ReAct（Reason + Act）架构，结合 LLM 视觉分析、事前/事中约束检查以及多层级的重规划策略，实现对 Android 物理机或模拟器的自动化自然语言指令控制。

## 一、环境准备

1. **Python 环境**: 建议使用 `Python 3.10+`
2. **虚拟环境**: 项目下建议使用虚拟环境（如当前已存在的 `myvenv`）
3. **依赖安装**:
   ```bash
   pip install -r requirements.txt
   ```
   *注意：必须安装并配置 `openai` 库以及相应的 API Key 才能进行规划操作。*
4. **ADB 连接**: 确保 `adb` 已经安装在系统环境变量中，并且设备已通过 USB 调试或模拟器桥接连接。可以通过 `adb devices` 验证。

## 二、配置说明

在项目根目录下，通常存在一个配置项源（例如环境变量或配置文件），运行前请确保相关配置（尤其是 `llm_api_base`、`llm_api_key` 和 `llm_model`）已正确设置。

## 三、启动方式与命令行参数

项目的核心入口是 `main.py`。你可以通过命令行快速下发自然语言任务，并控制代理的运行细节。

### 常用运行示例

- **执行基础任务** (默认连接第一台 ADB 设备)
  ```bash
  python main.py --task "打开微信，搜索张三并发送你好"
  ```

- **指定设备和最大步数**
  ```bash
  python main.py --task "打开设置，连接WiFi" --serial emulator-5554 --max-steps 50
  ```

- **安全模拟验证 (Dry Run 模式开发调试必现)**
  *不实际向手机发送任何物理 ADB 点击指令，系统会基于空界面进行流程测试和 LLM API 连通性测试。*
  ```bash
  python main.py --task "测试任务" --dry-run --log-level DEBUG
  ```

### 全部参数说明

| 参数简写 | 参数全称 | 必填 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `-t` | `--task` | **是** | 无 | **任务描述**，需要 Agent 执行的自然语言指令。 |
| `-s` | `--serial` | 否 | `""` (空) | **ADB 设备序列号**。当有多台设备连接时使用，若为空则默认使用第一台设备。 |
| | `--dry-run` | 否 | `False` | **干跑模式**。开启后不实际执行 ADB 命令，仅验证推理规划流程。 |
| | `--log-level` | 否 | `INFO` | **日志输出级别**。可选值：`DEBUG`, `INFO`, `WARNING`, `ERROR`。 |
| | `--log-file` | 否 | 日志目录 | **日志输出文件路径**。默认会写入到 `./data/logs/agent.log`，通过此参数可重定向到其他文件。 |
| | `--max-steps`| 否 | `30` | **单个任务最大执行步数**。限制防止无限死循环，超过此步数未成功则任务中止。 |

## 四、核心流程简介

运行 `main.py` 后，流程将进入 `AgentLoop` (定义于 `agent_loop.py`)：
1. **Perception**: 利用 ADB Dump 和本地视觉/文本引擎，解析当前屏幕。
2. **Planning**: LLM 根据屏幕内容，输出子目标与需要进行的动作。
3. **Oracle Constraints**: 对输出动作生成约束（预期页面跳变、文本出现）。
4. **Safety Interceptor**: 拦截删除、支付等高风险子目标，拉起人工审批确认交互。
5. **Execution**: 解析物理坐标并发送物理指令（`tap`, `swipe`, `input`, 等）。
6. **Evaluation & Replan**: 比对操作发生后的屏幕状态，如果不符预期且多次死循环则启动回溯或重规划。

---
*提示：如遇到 API 请求 400 错误，请检查 `utils/llm_client.py` 中的 `max_completion_tokens` 与您当前选择的大模型版本是否兼容。*
