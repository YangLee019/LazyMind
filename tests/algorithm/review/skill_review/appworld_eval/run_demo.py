from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_repo_imports() -> None:
    current = Path(__file__).resolve()

    package_parent = next(
        p for p in [current.parent, *current.parents]
        if p.name == 'LazyRAG' or p.name == 'LazyMind'
    )
    lazyrag_root = package_parent
    extra_pythonpath = os.getenv('APPWORLD_EVAL_EXTRA_PYTHONPATH', '')
    paths = [
        *(Path(path).expanduser() for path in extra_pythonpath.split(':') if path.strip()),
        package_parent,
        lazyrag_root / 'algorithm' / 'lazyllm',
        lazyrag_root / 'algorithm',
        lazyrag_root / 'tests' / 'algorithm' / 'review' / 'skill_review'
    ]
    for path in reversed(paths):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_bootstrap_repo_imports()

from appworld_eval.appworld_tool import AppWorldTool
from appworld_eval.env_loader import plan_task_ids


CREATE_USER_ID = 'appworld-5'
CREATE_USER_NAME = 'eval'


def _load_model_config(raw: str | None) -> dict | None:
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        text = candidate.read_text(encoding='utf-8')
        if candidate.suffix.lower() in {'.yaml', '.yml'}:
            import yaml

            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
    else:
        value = json.loads(raw)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('model_config must be a JSON/YAML object')
    return value

model_config = {
    "llm": {
        "source": "openai",
        "model": "minimax-m27",
        "base_url": "http://106.75.235.251:9000/v1/",
        "api_key": "sk-maas-GDZmEQsilc4uGXXTaWnIHmET9V0eenZ8F6eWk3LaPzE"
    },
    "embed_main": {
        "source": "openai",
        "model": "lazyllm",
        "base_url": "http://localhost:9800/v1/embeddings",
        "skip_auth": True
    }
}


def _require_env(name: str) -> str:
    value = str(os.getenv(name, '') or '').strip()
    if not value:
        raise SystemExit(f'{name} is required. Source appworld_eval/appworld_env.sh first.')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one AppWorld demo task.')
    parser.add_argument(
        '--model-config',
        default=os.getenv('APPWORLD_EVAL_MODEL_CONFIG', ''),
        help='JSON string or JSON/YAML file path passed through to handle_chat(model_config=...).',
    )
    args = parser.parse_args()

    data_root = _require_env('LAZYMIND_APPWORLD_DATA_ROOT')
    task_ids = plan_task_ids(data_root=data_root, dataset='dev', episodes=1)
    tool = AppWorldTool(
        repo_root=_require_env('LAZYMIND_APPWORLD_REPO_ROOT'),
        data_root=data_root,
        environment_url=_require_env('LAZYMIND_APPWORLD_ENVIRONMENT_URL'),
        apis_url=_require_env('LAZYMIND_APPWORLD_APIS_URL'),
        experiment_name=os.getenv('LAZYMIND_APPWORLD_EXPERIMENT_NAME', 'lazyrag'),
    )
    from appworld_eval.handle_chat_runner import run_appworld_eval_with_handle_chat_sync

    summary = run_appworld_eval_with_handle_chat_sync(
        tool=tool,
        task_ids=task_ids,
        max_steps=200,
        create_user_id=CREATE_USER_ID,
        create_user_name=CREATE_USER_NAME,
        model_config=model_config,
        persist_history=True,
    )
    print(json.dumps(summary['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

