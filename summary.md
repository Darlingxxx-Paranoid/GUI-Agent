# 一、问题起点

最初目标是设计一个 **可靠的 GUI Agent 方法框架**，核心思想是：

**利用 Oracle 验证 GUI 操作是否真正实现语义目标。**

基本动机：

传统 GUI Agent 存在几个问题：

1. Agent 难以判断一步操作是否成功  
2. GUI 中的成功反馈（toast / popup）是短暂的  
3. UI 动态变化导致重复操作  
4. Agent 难以判断任务是否已经完成  

因此希望构建：

**Oracle‑feedback driven GUI agent**

即：

```
Action → Oracle Verification → Next Decision
```

---

# 二、方法设计的逐步演化

整个方法经历了几个阶段的重构。

## 阶段1：最初 pipeline

最初结构：

```
Perception
→ Planner
→ Pre‑Oracle
→ Execution
→ Running Oracle
→ Post‑Oracle
→ Progress Context
```

当时有几个问题：

- Planner直接输出 widget id
- Pre‑Oracle完全依赖XML
- 是否需要ActionMapper
- Pre‑Oracle上下文如何获取

---

## 阶段2：引入 Action Mapper

我们讨论了是否需要：

```
Planner → Action Mapper → Execution
```

原因：

Planner负责：

```
semantic decision
```

ActionMapper负责：

```
UI grounding
```

但后来参考 **ScenGen论文**后发现：

他们把这两个阶段合并为 **Decider**：

```
Logical Decision
+
Widget Grounding
```

于是最终决定：

**目标控件锚定在 Planner 内完成。**

---

## 阶段3：Pre‑Oracle重新设计

最初 Pre‑Oracle 的问题：

- 截图还是 XML
- 全局还是局部上下文
- 直接生成断言显得方法薄

经过多次讨论，形成以下设计：

Pre‑Oracle 三阶段生成机制：

```
Action Scope Modeling
Semantic Outcome Inference
Structural Predicate Compilation
```

也就是：

```
goal
↓
semantic outcome
↓
state transition
↓
XML predicates
```

这是 **Semantic → Structural 的转换机制**。

---

# 三、提出 Oracle Transition Framework

为了让方法更“厚”，提出统一概念：

**Oracle Transition Framework**

核心思想：

```
Action success
=
Expected UI State Transition
```

定义 UI 状态转移类型：

```
NavigationTransition
NodeAppearance
AttributeModification
ContentUpdate
ContainerExpansion
```

Pre‑Oracle:

```
预测 state transition
```

Post‑Oracle:

```
验证 state transition
```

---

# 四、当前最终方法框架

经过多轮讨论，得到当前方法结构：

```
Perception
   ↓
Planner
   ├ Logical Decision
   └ Widget Grounding
   ↓
Pre‑Oracle
   ├ Context Modeling
   ├ Semantic Outcome Inference
   └ Transition Compilation
   ↓
Execution
   ↓
Running Oracle
   ↓
Post‑Oracle
   ├ Structural Verification
   └ Semantic Re‑Evaluation
   ↓
Progress Context / Experience Context
```

---

# 五、各模块最终定义

## 1 Perception

构建 GUI 的双重表示：

```
visual semantics
+
structural state
```

输出：

```
Screenshot
Visible Widgets List
XML UI tree
```

---

## 2 Planner

Planner负责：

```
下一步动作语义推理
+
目标控件锚定
```

输入：

```
Screenshot
Progress Context
Experience Context
```

输出：

```
Action intent
Action type
Target widget description
Target widget id
```

---

## 3 Pre‑Oracle

作用：

**预测动作成功后的 UI 状态转移。**

流程：

```
1 Action Scope Modeling
2 Semantic Outcome Inference
3 Structural Predicate Compilation
```

最终生成：

```
StepContract
```

包含：

```
transition type
XML predicates
```

---

## 4 Execution

执行 Planner 决策：

```
tap
input
swipe
back
launch_app
```

---

## 5 Running Oracle

检测执行环境异常：

```
black screen
white screen
UI freeze
dump failure
unexpected app switch
```

出现异常直接触发 replanning。

---

## 6 Post‑Oracle

验证 Pre‑Oracle 生成的断言。

两层验证：

### Structural Verification

基于 XML：

```
转移是否发生
```

### Semantic Re‑Evaluation

LLM 判断是否：

```
oracle错误
或实际成功
```

---

## 7 Progress Context

记录：

```
已经成功完成的语义步骤
```

作用：

- 防止重复操作  
- 判断任务完成  
- 提供任务进度

---

## 8 Experience Context

记录：

```
失败操作
```

作用：

避免重复错误决策。

---

# 六、核心创新点

最终形成三个核心创新点：

### 1 Oracle‑driven action verification

通过 Oracle 明确验证 GUI 操作是否成功。

---

### 2 Semantic‑to‑structural oracle generation

将语义操作目标转换为可验证的 UI 状态转移。

---

### 3 Oracle‑guided planning feedback

使用 Oracle 结果驱动下一步决策。

---

# 七、目前仍然较弱的部分

当前方法仍有两个较薄的点：

### 1 Pre‑Oracle transition model

需要明确 transition 类型集合，而不是简单规则。

---

### 2 Running Oracle

目前检测机制偏简单，需要强化：

```
UI freeze detection
navigation anomalies
```

---

# 八、整体方法逻辑

完整闭环：

```
Observe
↓
Plan semantic action
↓
Predict UI transition
↓
Execute
↓
Verify transition
↓
Update progress
↓
Next action
```

---

# 九、整体评价

当前方法已经具备：

- 清晰 pipeline  
- oracle verification  
- self‑correction  
- memory feedback  

属于一个比较完整的 **GUI agent framework**。

大致研究强度：

```
6.5 / 10  (当前)
8 / 10    (加强 transition model 后)
```

---

如果愿意，我可以在下一步帮你把整个框架 **整理成一页完整 Method Overview（论文图1级别）**，结构会特别清晰。