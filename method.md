## Perception
### OCR+CV
- 输入：截图
- 输出：OCR结果+Visible widgets List

### Dump解析
- 输入：dump文件
- 输出：完整Dump树

## Planner（LLM）
### 1. 规划阶段
#### 输入
- Task: 任务描述
- Screenshot: 截图
- Experience context（最近3步没有通过验证的动作语义信息，没有则为空）
- Progress Context

#### 输出
PlanResult:
- `goal`: 动作意图
- `action_type`: 动作类型
- `target_description`: 目标描述
- `input_description`: 输入动作文本（仅`action_type=input`时非空）
- `is_task_complete`：是否任务已完成  
- `reasoning`：规划理由

### 2. 锚定阶段
* 先通过target_description配合OCR/UIED进行匹配
* 否则LLM输入Screenshot+ Visible widgets List进行匹配
* 若OCR/UIED+LLM均未锚定到有效控件：本步失败并触发重规划，不执行危险兜底点击

## Pre-Oracle（LLM）
| 职责：定义成功的证据
### 1. 上下文信息选择
| 根据动作预测类型选择，尽量减少噪声
* 局部上下文：适用于控件中心的操作（如点击、输入），通过截取目标控件周围区域获得语义上下文
* 全局上下文：适用于界面导航类操作（如启动应用、滑动或返回），使用完整界面截图提供语义信息

### 2. 在获得视觉上下文后，Pre‑Oracle 利用多模态 LLM 根据当前动作语义推断动作成功后的界面语义变化
### 3. 通过一组规则将语义变化映射为UI状态转移模式，并最终生成基于XML界面树的验证断言
### 输出（两层）
- `SemanticTransitionContract`
  - `context_mode`: `local|global`
  - `transition_type`: UI语义转移类型
  - `success_definition`: 成功语义描述
  - `semantic_hints`: 高层语义提示
- `UIAssertionContract`
  - `context_mode`: `local|global`
  - `transition_type`: UI语义转移类型
  - `success_definition`: 成功语义描述
  - `assertions`: 基于XML可执行断言
#### UI转移类型
```
NavigationTransition
NodeAppearance
AttributeModification
ContentUpdate
ContainerExpansion
```

## Execution

## Running-Oracle
| 职责：检测运行时异常，不参与任务成功判定

检测黑屏 / 白屏

## Post-Oracle
| 职责：评估变化证据，判断任务成功

### 1.基于XML树进行UI状态断言的验证
### 2.如果没有完全通过，使用LLM进行再次验证
#### 上下文信息：
* 输入前后截图
* 执行动作
* 未通过的UI状态断言（原始语义和XML树断言）

#### 判定
* 动作语义成功（即断言预测或验证错误）：将当前动作加入Progress Context，进入下一步Plan
* 动作语义失败且UI有变化：回退，动作加入Experience context ，重新Plan
* 动作语义失败且UI无变化：动作加入Experience context ，重新Plan


 

## Global-Oracle（不单独作为一个模块，全局性的一个判断，通过Planner的Progress Context实现）
| 职责：全局评估任务成功
- progress_context: 任务执行进度
