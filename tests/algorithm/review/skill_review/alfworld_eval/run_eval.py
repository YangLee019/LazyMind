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
from alfworld_eval.env_loader import SUPPORTED_SPLITS, init_alfworld_env


CREATE_USER_ID = '49437df8-55ed-4533-8afe-7efe4950a3b0'
CREATE_USER_NAME = 'eval'
MAX_ALFWORLD_STEPS = 50


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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ALFWorld benchmark tasks.')
    parser.add_argument('config_path', help='Path to ALFWorld configs/base_config.yaml')
    parser.add_argument('--split', choices=SUPPORTED_SPLITS, default='eval_out_of_distribution')
    parser.add_argument('--num-tasks', type=int, default=100)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=MAX_ALFWORLD_STEPS,
        help=f'Max ALFWorld environment steps per task. Must be between 1 and {MAX_ALFWORLD_STEPS}.',
    )
    parser.add_argument('--seed', type=int, help='Random seed for shuffling ALFWorld tasks before taking num-tasks.')
    parser.add_argument('--max-agent-retries', type=int, default=MAX_ALFWORLD_STEPS)
    parser.add_argument(
        '--model-config',
        help='JSON string or JSON/YAML file path passed through to handle_chat(model_config=...).',
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_steps > MAX_ALFWORLD_STEPS:
        parser.error(f'--max-steps must be between 1 and {MAX_ALFWORLD_STEPS}.')

    env = init_alfworld_env(args.config_path, split=args.split, batch_size=1, seed=args.seed)
    tool = ALFWorldTool(env)
    os.environ['LAZYMIND_MAX_RETRIES'] = str(args.max_agent_retries)
    from alfworld_eval.handle_chat_runner import run_alfworld_eval_with_handle_chat_sync

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
