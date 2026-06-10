from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_repo_imports() -> None:
    package_parent = Path(__file__).resolve().parent.parent
    lazyrag_root = package_parent / 'LazyRAG'
    extra_pythonpath = os.getenv('APPWORLD_EVAL_EXTRA_PYTHONPATH', '')
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

from appworld_eval.appworld_tool import AppWorldTool
from appworld_eval.env_loader import SUPPORTED_DATASETS, plan_task_ids


CREATE_USER_ID = '2f019fad-c981-46f1-9b5e-f7a1515a4340'
CREATE_USER_NAME = 'train'


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


def _setting(value: str | None) -> str:
    return str(value or '').strip()


def _require_setting(parser: argparse.ArgumentParser, value: str, flag: str, env_name: str) -> str:
    cleaned = _setting(value)
    if not cleaned:
        parser.error(
            f'{flag} is required. Pass {flag}, set {env_name}, '
            'or source appworld_eval/appworld_env.sh first.'
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description='Run AppWorld benchmark tasks.')
    parser.add_argument('--data-root', default=os.getenv('LAZYMIND_APPWORLD_DATA_ROOT', ''))
    parser.add_argument('--repo-root', default=os.getenv('LAZYMIND_APPWORLD_REPO_ROOT', ''))
    parser.add_argument('--dataset', choices=SUPPORTED_DATASETS, default='dev')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, help='Random seed for sampling task ids from the dataset.')
    parser.add_argument('--task-id', action='append', default=[], help='Explicit AppWorld task id. Repeatable.')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--max-agent-retries', type=int, default=50)
    parser.add_argument(
        '--environment-url',
        default=os.getenv('LAZYMIND_APPWORLD_ENVIRONMENT_URL', ''),
    )
    parser.add_argument(
        '--apis-url',
        default=os.getenv('LAZYMIND_APPWORLD_APIS_URL', ''),
    )
    parser.add_argument(
        '--experiment-name',
        default=os.getenv('LAZYMIND_APPWORLD_EXPERIMENT_NAME', 'lazyrag'),
    )
    parser.add_argument('--user-id', default=CREATE_USER_ID)
    parser.add_argument('--user-name', default=CREATE_USER_NAME)
    parser.add_argument(
        '--model-config',
        help='JSON string or JSON/YAML file path passed through to handle_chat(model_config=...).',
    )
    parser.add_argument('--no-persist-history', action='store_true')
    parser.add_argument('--print-results', action='store_true')
    args = parser.parse_args()
    data_root = _require_setting(parser, args.data_root, '--data-root', 'LAZYMIND_APPWORLD_DATA_ROOT')
    repo_root = _require_setting(parser, args.repo_root, '--repo-root', 'LAZYMIND_APPWORLD_REPO_ROOT')
    environment_url = _require_setting(
        parser,
        args.environment_url,
        '--environment-url',
        'LAZYMIND_APPWORLD_ENVIRONMENT_URL',
    )
    apis_url = _require_setting(parser, args.apis_url, '--apis-url', 'LAZYMIND_APPWORLD_APIS_URL')

    task_ids = plan_task_ids(
        data_root=data_root,
        dataset=args.dataset,
        episodes=args.episodes,
        task_ids=args.task_id,
        seed=args.seed,
    )
    tool = AppWorldTool(
        repo_root=repo_root,
        data_root=data_root,
        environment_url=environment_url,
        apis_url=apis_url,
        experiment_name=args.experiment_name,
    )
    os.environ['LAZYMIND_MAX_RETRIES'] = str(args.max_agent_retries)
    from appworld_eval.handle_chat_runner import run_appworld_eval_with_handle_chat_sync

    summary = run_appworld_eval_with_handle_chat_sync(
        tool=tool,
        task_ids=task_ids,
        max_steps=args.max_steps,
        create_user_id=args.user_id,
        create_user_name=args.user_name,
        model_config=_load_model_config(args.model_config),
        persist_history=not args.no_persist_history,
    )
    print(json.dumps(summary if args.print_results else summary['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
