---
name: "gui-agent-oracles"
description: "总结 GUI-Agent 的 Pre/Running/Post Oracle：实现位置、调用链、数据结构字段与含义。用户询问 oracle 设计/调试失败原因/需要快速定位证据链时调用。"
---

# GUI-Agent Oracles（Pre / Running / Post）速查

用于在 GUI-Agent 项目中快速解释三段 Oracle 的实现、数据结构字段、以及它们分别提供的能力（能发现什么、会阻断什么、会产出什么证据）。

## 何时调用

- 用户问“pre-oracle / running-oracle / post-oracle 是怎么实现的、字段含义是什么、失败原因怎么解释”。
- 需要定位一次 step 的失败到底是：边界违规、死循环、白屏黑屏、证据不足、还是语义不确定。
- 需要把 oracle 输出（constraints / runtime_check / eval_result / delta_facts）讲清楚并给出排障路径。

## 总体调用链（一步 step 的顺序）

- Pre-Oracle：规划出 subgoal 后生成 `PreConstraints`
  - 入口：[agent_loop.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/agent_loop.py#L217-L244)
- Running-Oracle：
  - 执行前：`pre_execution_check(action)` 死循环熔断
  - 执行后：`post_execution_check(...)` 白屏/黑屏 + 边界/跳变守门
  - 入口：[agent_loop.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/agent_loop.py#L258-L372)
- Post-Oracle（Evaluator）：机器证据评估，uncertain 才 LLM 兜底；支持二次观测
  - 入口：[agent_loop.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/agent_loop.py#L373-L421)

## Pre-Oracle（事前 Oracle）

**实现位置**
- `Planning/oracle_pre.py`：`OraclePre` + `PreConstraints`
  - 定义：[oracle_pre.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Planning/oracle_pre.py#L18-L25)
  - 入口：[oracle_pre.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Planning/oracle_pre.py#L38-L83)
- Prompt schema：`prompt/oracle_pre_prompt.py`
  - 信号原语与 JSON 契约：[oracle_pre_prompt.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/prompt/oracle_pre_prompt.py)

**能力（它能“提前定义什么”）**
- 把“成功”定义为“可验证的状态变化证据计划”，而不是页面语义分类或严格文本匹配。
- 给出跨包/风险 UI 的边界策略，供 Running 与 Post 复用。
- 为后续差分提供锚点（action_anchor），让“anchor/local/global”变化判断可落地。


## Running-Oracle（事中 Oracle / Runtime Oracle）

**实现位置**
- `Execution/oracle_runtime.py`：`OracleRuntime`
  - 文件：[oracle_runtime.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Execution/oracle_runtime.py)

**能力（它能“立即阻断什么”）**
- 执行前：检测“连续相同动作”死循环，直接拒绝执行并触发重规划
  - 入口：[oracle_runtime.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Execution/oracle_runtime.py#L39-L58)
- 执行后：检测白屏/黑屏（截图 HSV 方差极低），并做边界/跳变硬守门
  - 白屏/黑屏：[oracle_runtime.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Execution/oracle_runtime.py#L83-L90)
  - 边界/跳变：[oracle_runtime.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Execution/oracle_runtime.py#L193-L286)


## Post-Oracle（事后 Oracle / Evaluator）

**实现位置**
- `Evaluation/evaluator.py`：`Evaluator` + `EvalResult`
  - EvalResult 定义：[evaluator.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Evaluation/evaluator.py#L21-L39)
  - 主流程：[evaluator.py](file:///Users/bytedance/Desktop/Darlingxxx/GUI-Agent/Evaluation/evaluator.py#L56-L168)

**能力（它能“解释为什么成功/失败”）**
- machine-first：先做硬边界检查，再提取 UI 变化事实（delta_facts），然后按 Pre 的 signals 做机器匹配打分。
- uncertain 才 LLM：只有机器评估输出 uncertain 时，才用 semantic_goal + delta_facts 做语义确认。
- 可二次观测：uncertain 时在 loop 里 sleep 再抓 after2 状态重新评估。
