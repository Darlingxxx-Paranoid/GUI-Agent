## Planner（LLM）
### 输入
- Task: 任务描述
- Screenshot: 截图

### 输出
PlanResult:
- `goal`: 目标描述
- `action_type`: 动作类型
- `input_text`: 输入类动作文本
- `is_task_complete`：是否任务已完成  
- `reasoning`：规划理由

## Pre-Oracle（LLM）
### 输入
- PlanResult: Planner 输出，移除`reasoning`字段
- 完整Dump树

### 输出
`StepContract`
- `success_definition`：成功的自然语言描述
- `Expectations`: 成功的期望结果
  - `Target_category`: 目标类型（包括`widget`、`activity`、`package`）
  - `Target`: 目标(如果是`widget`，则为控件ID+控件指定属性字段(必须基于完整Dump树)；如果是`activity`和`package`，则为空)
    - `node_id`: 控件ID
    - `resource_id`: 控件资源ID
    - `field`: 控件属性字段（text, class, content-desc, checked, enabled, focused, selected）
  - `Relation`: 关系类型（包括`exact_match`、`contains`）
  - `content`: 目标内容(如果是`exact_match`，则为固定值；如果是`contains`，则为包含关系的字符串)