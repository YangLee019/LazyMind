from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_repo_imports() -> None:
    package_parent = Path(__file__).resolve().parent.parent
    lazyrag_root = package_parent / 'LazyRAG'
    extra_pythonpath = os.getenv('ALFWORLD_EVAL_EXTRA_PYTHONPATH', '')
    paths = [
        *(Path(path).expanduser() for path in extra_pythonpath.split(':') if path.strip()),
        package_parent,
        lazyrag_root / 'algorithm' / 'lazyllm',
        lazyrag_root / 'algorithm',
    ]
    for path in reversed(paths):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_bootstrap_repo_imports()

from alfworld_eval.alfworld_tool import ALFWorldTool
from alfworld_eval.env_loader import init_alfworld_env
from alfworld_eval.handle_chat_runner import run_alfworld_eval_with_handle_chat_sync


CREATE_USER_ID = '49437df8-55ed-4533-8afe-7efe4950a3b0'
CREATE_USER_NAME = 'eval'
os.environ.setdefault('LAZYMIND_CORE_DATABASE_URL', 'postgresql+psycopg://root:123456@localhost:5432/core')
os.environ.setdefault('LAZYLLM_MINIMAX_API_KEY', 'sk-maas-GDZmEQsilc4uGXXTaWnIHmET9V0eenZ8F6eWk3LaPzE')


DEFAULT_MODEL_CONFIG = {
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


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _normalize_model_config(value: dict) -> dict:
    normalized = {}
    for role, raw_cfg in value.items():
        cfg = raw_cfg[0] if isinstance(raw_cfg, list) and raw_cfg else raw_cfg
        if not isinstance(cfg, dict):
            continue
        item = _expand_env(dict(cfg))
        if 'model' not in item and 'name' in item:
            item['model'] = item.pop('name')
        if 'base_url' not in item and 'url' in item:
            item['base_url'] = item.pop('url')
        normalized[role] = item
    return normalized


def _load_model_config(raw: str | None) -> dict:
    if not raw:
        return DEFAULT_MODEL_CONFIG
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
    if not isinstance(value, dict):
        raise ValueError('model_config must be a JSON/YAML object')
    return _normalize_model_config(value)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one ALFWorld demo.')
    parser.add_argument('config_path', help='Path to ALFWorld configs/base_config.yaml')
    parser.add_argument('--split', default='train')
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument(
        '--model-config',
        help='JSON string or JSON/YAML file path passed through to handle_chat(model_config=...).',
    )
    args = parser.parse_args()

    env = init_alfworld_env(args.config_path, split=args.split, batch_size=1)
    tool = ALFWorldTool(env)
    summary = run_alfworld_eval_with_handle_chat_sync(
        tool=tool,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        create_user_id=CREATE_USER_ID,
        create_user_name=CREATE_USER_NAME,
        model_config=_load_model_config(args.model_config),
        persist_history=True,
    )
    print(json.dumps(summary['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
