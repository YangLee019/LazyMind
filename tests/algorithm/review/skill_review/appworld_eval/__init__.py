"""Minimal AppWorld benchmark integration for the LazyMind agent runtime."""

from __future__ import annotations

from typing import Any


__all__ = [
    'APPWORLD_SYSTEM_PROMPT',
    'APPWORLD_TOOL_NAMES',
    'AppWorldTool',
    'SUPPORTED_DATASETS',
    'compute_metrics',
    'insert_chat_history_row',
    'load_task_ids',
    'plan_task_ids',
    'register_appworld_tool',
    'run_appworld_eval',
    'run_appworld_eval_with_handle_chat',
    'run_appworld_eval_with_handle_chat_sync',
]


def __getattr__(name: str) -> Any:
    if name in {'APPWORLD_TOOL_NAMES', 'AppWorldTool'}:
        from .appworld_tool import APPWORLD_TOOL_NAMES, AppWorldTool

        return {'APPWORLD_TOOL_NAMES': APPWORLD_TOOL_NAMES, 'AppWorldTool': AppWorldTool}[name]
    if name in {'SUPPORTED_DATASETS', 'load_task_ids', 'plan_task_ids'}:
        from .env_loader import SUPPORTED_DATASETS, load_task_ids, plan_task_ids

        return {
            'SUPPORTED_DATASETS': SUPPORTED_DATASETS,
            'load_task_ids': load_task_ids,
            'plan_task_ids': plan_task_ids,
        }[name]
    if name in {
        'register_appworld_tool',
        'run_appworld_eval_with_handle_chat',
        'run_appworld_eval_with_handle_chat_sync',
    }:
        from .handle_chat_runner import (
            register_appworld_tool,
            run_appworld_eval_with_handle_chat,
            run_appworld_eval_with_handle_chat_sync,
        )

        return {
            'register_appworld_tool': register_appworld_tool,
            'run_appworld_eval_with_handle_chat': run_appworld_eval_with_handle_chat,
            'run_appworld_eval_with_handle_chat_sync': run_appworld_eval_with_handle_chat_sync,
        }[name]
    if name == 'insert_chat_history_row':
        from .history_db import insert_chat_history_row

        return insert_chat_history_row
    if name == 'compute_metrics':
        from .metrics import compute_metrics

        return compute_metrics
    if name == 'APPWORLD_SYSTEM_PROMPT':
        from .prompt import APPWORLD_SYSTEM_PROMPT

        return APPWORLD_SYSTEM_PROMPT
    if name == 'run_appworld_eval':
        from .runner import run_appworld_eval

        return run_appworld_eval
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
