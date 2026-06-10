# AppWorld 评测

面向现有 LazyMind `handle_chat` 服务入口的 AppWorld 基准最小适配器。

此处将 AppWorld 用作交互式代码执行环境，而非静态 QA 数据。评测时会先准备一个 AppWorld task session，然后调用 `algorithm/lazymind/chat/service/chat_service.py::handle_chat`。Agent 通过 `appworld_execute(code)` 在持久 Python 环境中执行代码，并可用 `appworld_task_info`、`appworld_api_docs`、`appworld_status` 辅助完成任务。

成功与否由 AppWorld 的 `evaluate` 结果决定；入库逻辑沿用 `conversations` 和 `chat_histories`。

## 安装

AppWorld 需要 Python 3.11+。如果直接使用上游 AppWorld 包，可按官方流程安装并下载数据：

```bash
pip install appworld
appworld install
appworld download data --root /path/to/appworld/root
```

如果使用本地源码仓库：

```bash
cd /path/to/appworld
pip install -e .
appworld install --repo
appworld download data --root /path/to/appworld/root
```

`appworld download data` 会在 AppWorld root 下生成 `data/datasets`、`data/tasks`、`data/base_dbs` 等目录。后续运行时通过 `--data-root` 或 `LAZYMIND_APPWORLD_DATA_ROOT` 指向这个 AppWorld root。

如需检查 AppWorld 自身安装是否可用：

```bash
appworld verify tasks --root /path/to/appworld/root
```

## 配置

本适配器需要先启动 AppWorld environment 和 APIs 两个服务。建议分别在两个终端中运行：

```bash
# terminal 1
appworld serve environment --port 18000 --root /path/to/appworld/root
```

```bash
# terminal 2
appworld serve apis --port 19000 --root /path/to/appworld/root
```

启动后确认 `appworld_env.sh` 中的 `LAZYMIND_APPWORLD_ENVIRONMENT_URL` 和 `LAZYMIND_APPWORLD_APIS_URL` 指向这两个服务。评测脚本不会自动拉起 AppWorld 服务。

常用配置项如下：

```bash
cd /path/to/LazyRAG
source tests/algorithm/review/skill_review/appworld_eval/appworld_env.sh
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"
```

`appworld_env.sh` 中包含以下环境变量，可按机器环境修改：

```bash
LAZYMIND_APPWORLD_DATA_ROOT=/path/to/appworld/root
LAZYMIND_APPWORLD_REPO_ROOT=/path/to/appworld
LAZYMIND_APPWORLD_ENVIRONMENT_URL=http://127.0.0.1:18000
LAZYMIND_APPWORLD_APIS_URL=http://127.0.0.1:19000
LAZYMIND_CORE_DATABASE_URL=postgresql+psycopg://root:123456@localhost:5432/core
APPWORLD_EVAL_EXTRA_PYTHONPATH=
```

也可以不使用 shell 文件，直接在 `run_eval.py` 中传 `--data-root`、`--repo-root`、`--environment-url`、`--apis-url`。

从 `LazyRAG` 根目录直接运行脚本时，需要显式设置：

```bash
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"
```

否则脚本内的兼容导入路径可能找不到 `algorithm/lazymind`，报 `ModuleNotFoundError: No module named 'lazymind'`。

默认 dataset 为 `dev`，支持 `train`、`dev`、`test_normal`、`test_challenge`。
任务 ID 来自：

```text
{data_root}/data/datasets/{dataset}.txt
```

## 演示

```bash
cd /path/to/LazyRAG
source tests/algorithm/review/skill_review/appworld_eval/appworld_env.sh
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/appworld_eval/run_demo.py \
  --model-config "$PWD/algorithm/lazymind/common/runtime_models.inner.yaml"
```

该命令会从 `dev` split 取 1 个任务，并打印汇总指标。运行过程中会直接写入 `conversations` 和 `chat_histories` 表。
如果当前 LazyMind 使用 `LAZYMIND_MODEL_CONFIG_PATH=dynamic`，必须通过 `--model-config` 或 `APPWORLD_EVAL_MODEL_CONFIG` 提供 `llm` 配置，否则 Agent 会在第一次调用模型时报 `No source is configured for dynamic LLM source.`。

`model_config.yaml` 示例：

```yaml
llm:
  source: openai
  model: your-model
  base_url: http://your-llm-endpoint/v1/
  api_key: ${YOUR_API_KEY}
```

`run_demo.py` 顶部预留了调试用户参数：

```python
CREATE_USER_ID = "appworld_demo"
CREATE_USER_NAME = "AppWorld Demo"
```

按需改成你的用户即可。数据库连接沿用 LazyMind 配置，优先读取 `LAZYMIND_CORE_DATABASE_URL`，其次读取 `LAZYMIND_DATABASE_URL`。

如果你的 LazyMind/LazyRAG 源码不在当前 `LazyRAG/algorithm` 下，可以在 `appworld_env.sh` 中设置 `APPWORLD_EVAL_EXTRA_PYTHONPATH`，作用等同于旧 demo 里手动追加 `sys.path`，但不会把某个人机器上的路径写死进 Python 文件。

## 批量评测

```bash
cd /path/to/LazyRAG
source tests/algorithm/review/skill_review/appworld_eval/appworld_env.sh
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/appworld_eval/run_eval.py \
  --dataset dev \
  --episodes 50 \
  --seed 42 \
  --max-steps 50
```

也可以指定具体任务：

```bash
cd /path/to/LazyRAG
source tests/algorithm/review/skill_review/appworld_eval/appworld_env.sh
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/appworld_eval/run_eval.py \
  --task-id 0a1b2c3d \
  --max-steps 200
```

同样支持透传 `model_config`：

```bash
cd /path/to/LazyRAG
source tests/algorithm/review/skill_review/appworld_eval/appworld_env.sh
export APPWORLD_EVAL_EXTRA_PYTHONPATH="$PWD/algorithm:$PWD/algorithm/lazyllm"

python tests/algorithm/review/skill_review/appworld_eval/run_eval.py \
  --dataset dev \
  --episodes 10 \
  --model-config "$PWD/algorithm/lazymind/common/runtime_models.inner.yaml"
```

常用参数：

- `--data-root`: AppWorld root，即包含 `data/` 的目录
- `--repo-root`: AppWorld 源码仓库根目录
- `--environment-url`: AppWorld environment 服务地址
- `--apis-url`: AppWorld APIs 服务地址
- `--seed`: 从 dataset 中随机抽取任务时使用的随机种子；不传则按文件顺序取前 N 个
- `--model-config`: JSON 字符串或 JSON/YAML 文件，透传给 `handle_chat(model_config=...)`
- `--no-persist-history`: 只跑评测，不写 `chat_histories`
- `--print-results`: 打印完整结果，否则只打印 metrics

输出摘要包含：

- `total_tasks`
- `success_count`
- `success_rate`
- `completed_count`
- `completed_rate`
- `avg_steps`
- `avg_success_steps`
- `max_step_failure_rate`
- `error_count`
- `total_tests`
- `total_passes`
- `test_pass_rate`

单个任务结果示例：

```json
{
  "episode_index": 1,
  "task_id": "example_task_id",
  "success": true,
  "steps": 12,
  "completed": true,
  "evaluation": {
    "success": true,
    "num_tests": 3,
    "passes": ["..."]
  },
  "task_status": {
    "session_id": "...",
    "task_id": "example_task_id",
    "initialized": true,
    "completed": true,
    "interaction_count": 12
  },
  "error": null
}
```

单个任务的异常会记录在该任务的 `error` 字段中，不会中断整个基准测试。

## 工具注册方式

`handle_chat_runner.py` 会在评测期间确保 AppWorld 工具组可用，如果环境中没有该工具组，则临时注册：

```python
from lazymind.chat.service import chat_service
from lazymind.chat.service.component import ToolGroupConfig
from lazyllm.tools.agent.toolsManager import ToolGroup

chat_service.DEFAULT_TOOLS.append(
    ToolGroupConfig(
        name="appworld_eval",
        label="AppWorld",
        description="Run AppWorld benchmark environment tools.",
        instance=ToolGroup(
            tools=[
                tool.appworld_execute,
                tool.appworld_task_info,
                tool.appworld_api_docs,
                tool.appworld_status,
            ],
            name="appworld_eval",
            desc="Run AppWorld benchmark environment tools.",
            lazy=False,
            prefix=False,
        ),
    )
)
```

注册只在 benchmark 运行上下文内生效，结束后会恢复原始工具列表。调用 `handle_chat` 时会传入 `available_tools=["appworld_eval"]`、`use_memory=False`。内部 `ToolGroup(prefix=False)` 会把工具暴露为 `appworld_execute`、`appworld_task_info`、`appworld_api_docs`、`appworld_status`，不会生成 `AppWorldTool_AppWorldTool_...` 这类前缀名。
