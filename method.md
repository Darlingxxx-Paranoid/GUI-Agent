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
- Experience context（之前n步没有通过验证的动作的语义信息，没有则为空）
- Progress Context

#### 输出
PlanResult:
- `goal`: 动作意图
- `action_type`: 动作类型
- `target_description`: 目标描述
- `is_task_complete`：是否任务已完成  
- `reasoning`：规划理由

### 2. 锚定阶段
* 先通过target_description配合OCR进行
* 否则LLM输入Screenshot+ Visible widgets List进行匹配

## Pre-Oracle（LLM）
| 职责：定义成功的证据
### 输入
- PlanResult: Planner 输出，移除`reasoning`字段
- 完整Dump树

### 输出
`StepContract`
- `success_definition`：成功的自然语言描述
- `Expectations`: 成功的期望结果
  - `Target_category`: 目标类型（包括`widget`、`activity`、`package`）(后期考虑加入OCR)
  - `Target`: 目标(如果是`widget`，则为控件ID+控件指定属性字段(必须基于完整Dump树)；如果是`activity`和`package`，则为空)
    - `node_id`: 控件ID
    - `resource_id`: 控件资源ID
    - `field`: 控件属性字段（text, class, content-desc, checked, enabled, focused, selected），用来和目标内容进行关系运算
  - `Relation`: 关系类型（包括`exact_match`、`contains`）
  - `content`: 目标内容(如果是`exact_match`，则为固定值；如果是`contains`，则为包含关系的字符串)

## Execution

## Running-Oracle
| 职责：检测运行时异常，不参与任务成功判定

检测黑屏 / 白屏

## Post-Oracle
| 职责：评估变化证据，判断任务成功

### 输入
- 完整Dump树（动作执行后）
- `Expectations`

### 输出
- `is_goal_complete`：是否目标成功
- `action_history`：动作执行历史（失败才返回）
 

## Global-Oracle（不单独作为一个模块，全局性的一个判断，通过Planner的Progress Context实现）
| 职责：全局评估任务成功
- progress_context: 任务执行进度
