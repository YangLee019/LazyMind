# ALFWorld 评测

面向现有 LazyMind `handle_chat` 服务入口的 ALFWorld 基准最小适配器。

此处将 ALFWorld 用作交互式环境，而非静态 QA 数据。评测时会调用 `algorithm/lazymind/chat/service/chat_service.py::handle_chat`，并临时注册一个 `alfworld` 工具组。Agent 会收到初始观测，只能通过调用 `alfworld_step(action)` 来推进任务。

## 安装

```bash
pip install alfworld pyyaml
alfworld-download
```

如需完整的 ALFWorld 视觉模式支持，请安装上游包及其可选依赖。本模块优先面向文本环境，因为这是最小且可用的基准路径。

## 配置

使用上游 ALFWorld 的 `configs/base_config.yaml`。加载器会读取：

```yaml
env:
  type: AlfredTWEnv
```

并通过以下方式初始化环境：

```python
from alfworld.agents.environment import get_environment

env = get_environment(config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)
```

默认 split 为 `eval_out_of_distribution`。

从 `LazyRAG` 根目录直接运行脚本时，需要显式设置：

```bash
cd /path/to/LazyRAG
export ALFWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"
```

否则脚本内的兼容导入路径可能找不到 `algorithm/lazymind`，报 `ModuleNotFoundError: No module named 'lazymind'`。

## 演示

```bash
cd /path/to/LazyRAG
export ALFWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/alfworld_eval/run_demo.py /path/to/alfworld/configs/base_config.yaml \
  --split train \
  --model-config "$PWD/algorithm/lazymind/common/runtime_models.inner.yaml"
```

该命令会运行一个任务，并打印每个任务的结果及汇总指标。运行过程中会直接写入 `conversations` 和 `chat_histories` 表。
如需用 demo 临时多跑几个任务，可以显式传 `--num-tasks N`；正式批量评测建议使用 `run_eval.py`。

`run_demo.py` 顶部预留了调试用户参数：

```python
CREATE_USER_ID = "alfworld_eval"
CREATE_USER_NAME = "ALFWorld Eval"
```

按需改成你的用户即可。数据库连接沿用 LazyMind 配置，优先读取 `LAZYMIND_CORE_DATABASE_URL`，其次读取 `LAZYMIND_DATABASE_URL`。

如果需要指定 `handle_chat` 的 `model_config`，可以传 JSON 字符串或 JSON/YAML 文件路径：

```bash
cd /path/to/LazyRAG
export ALFWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/alfworld_eval/run_demo.py /path/to/alfworld/configs/base_config.yaml \
  --model-config "$PWD/algorithm/lazymind/common/runtime_models.inner.yaml"
```

## 批量评测

```bash
cd /path/to/LazyRAG
export ALFWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/alfworld_eval/run_eval.py /path/to/alfworld/configs/base_config.yaml \
  --split eval_out_of_distribution \
  --num-tasks 100 \
  --seed 42 \
  --max-steps 50 \
  --max-agent-retries 50
```

同样支持透传 `model_config`：

```bash
cd /path/to/LazyRAG
export ALFWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/alfworld_eval/run_eval.py /path/to/alfworld/configs/base_config.yaml \
  --split train \
  --num-tasks 100 \
  --seed 42 \
  --max-steps 50 \
  --max-agent-retries 50 \
  --model-config '{"llm": {"model": "your-model"}}'
```

`--split` 支持 `train`、`eval_in_distribution`、`eval_out_of_distribution`。`--seed` 会在取前 `--num-tasks` 个任务前固定打乱任务顺序。`--max-steps` 可外部覆盖，但 ALFWorld 环境单任务上限为 50，传入值必须在 1 到 50 之间。`--max-agent-retries` 会覆盖 LazyMind ReAct 工具调用循环次数，默认同样为 50。

输出摘要包含以下字段：

- `total_tasks`
- `success_count`
- `success_rate`
- `avg_steps`
- `avg_success_steps`
- `non_empty_trajectory_count`
- `empty_trajectory_count`
- `skill_usage_count`
- `skill_usage_denominator`
- `skill_usage_rate`
- `skill_usage_by_name`
- `max_step_failure_rate`
- `error_count`

每个任务的结果格式如下：

```json
{
  "task_id": 0,
  "success": true,
  "steps": 18,
  "final_reward": 1.0,
  "done": true,
  "error": null
}
```

单个任务的异常会记录在该任务的 `error` 字段中，不会中断整个基准测试。

## 工具注册方式

`handle_chat_runner.py` 会在评测期间临时修改 `chat_service.DEFAULT_TOOLS`：

```python
from lazymind.chat.service import chat_service
from lazymind.chat.service.component import ToolGroupConfig

chat_service.DEFAULT_TOOLS.append(
    ToolGroupConfig(
        name="alfworld",
        label="ALFWorld",
        description="Interact with one ALFWorld benchmark environment.",
        instance=tool,
    )
)
```

注册只在 benchmark 运行上下文内生效，结束后会恢复原始工具列表。调用 `handle_chat` 时会传入 `available_tools=["alfworld"]`、`use_memory=False`、`available_skills=[]`，确保被测 Agent 只看到 ALFWorld 工具。
