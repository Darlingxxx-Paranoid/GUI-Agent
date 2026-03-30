# Oracle 迭代收敛总结（2026-03-30）

## 1) 硬约束（按你的要求收敛）
- Oracle 核心不是“识别场景”，而是**验证变化**。
- 架构保持 `pre oracle / running oracle / post oracle` 三段，不拆不并。
- 不改经验沉淀（memory）主逻辑。
- 感知策略收敛为：**默认先用 DUMP**；仅当 DUMP 证据不足时，**CV+OCR 联合触发**（OCR 不单独跑）。

---

## 2) 本轮已收敛并保留的改动

### A. 感知层（Perception）
- `Perception/perception_manager.py`
- 默认 `dump-first`，并将 OCR 改为与 CV 同步触发：
  - `run_cv == True` 时才执行 `CV+OCR`。
  - DUMP 充足时直接跳过 `CV+OCR`，降低时延。
- 增加中等规模 DUMP 的“可用阈值”判断，减少不必要视觉补感知。

### B. 执行层稳定性（Execution）
- `Execution/action_executor.py`
- `uiautomator dump` 稳定性增强：
  - dump 前后清理陈旧文件；
  - 超时/失败重试 + 失败熔断冷却；
  - normal 不稳时进入 compressed 优先窗口，降低每步双超时成本。

### C. 规划/评估预算治理（Planning / Evaluation / LLM）
- `Planning/planner.py`
  - 低剩余预算时不再继续发规划请求，避免临界时长再被 LLM 调用拖穿。
- `Evaluation/evaluator.py`
  - 低剩余预算时跳过语义补判，直接走机器证据回退，避免临界超时。
- `utils/llm_client.py`
  - 关闭 SDK 内部重试（`max_retries=0`），避免“配置 1s 超时，实际阻塞多秒”。

### D. 动作语义一致性修复
- `Planning/planner.py` + `Execution/action_mapper.py`
- 修复 `long-press` 被错当 `tap` 的问题：
  - 规划动作归一化支持 `long_press`；
  - 映射层/执行层完整走 `long_press` 通道。

---

## 3) 实验结果（已完成）
> 说明：部分结果来自 `benchmark_history/*.json`，部分来自当轮控制台完整输出（因 benchmark 脚本会清理 `data/`）。

### 已完成且可核对的 run
1. `oracle-v3-general-core-iter3`：`1/3`（timeout 2）
2. `oracle-v3-general-core-iter6`：`2/3`（timeout 1）
3. `oracle-v3-general-core-iter7`：`2/3`（timeout 1）
4. `oracle-v3-general-stable-iter4`：`2/3`（timeout 1）
5. `oracle-v3-stopwatch-iter2`：`0/1`（timeout 1）
6. `oracle-v3-clipboard-single-iter8`：`0/1`（timeout 1）

### 主要失败模式（不是 Oracle 三段式结构错误）
- `uiautomator dump` 抖动导致观测阶段时延高；
- 长链任务（尤其 Clipboard、WiFi+Calendar）在 240s 预算内容易被时延放大后超时；
- Calendar 任务中出现 `com.google.android.gms` 登录引导页，导致额外步骤与预算挤压。

---

## 4) 明确不纳入最终框架的方向
- 不采用“某类任务/场景特判即成功”的路径。
- 不把 Oracle 退化为“识别某页面/某文案就通过”。
- 任何判断都应回到**动作后可观测变化证据**（package/activity/anchor/toggle/focus/text delta 等）。

---

## 5) 最终框架（收敛版）

### Pre Oracle
- 只做“可验证变化”目标约束与边界约束。
- 不做场景识别式成功捷径。

### Running Oracle
- 只做执行期风险/越界/死循环拦截。
- 不对场景语义做完成判定。

### Post Oracle
- 机器证据优先，语义补判次之。
- 预算不足时优先保留机器证据结论，不让语义调用拖垮任务时限。

---

## 6) 收敛结论
- 框架方向已按你的要求收敛：Oracle 维持“**验证变化**”主线，且三段式结构保持完整。
- 当前不稳定项主要是执行/观测时延与任务链路长度，而非“Oracle 场景识别能力”不足。
