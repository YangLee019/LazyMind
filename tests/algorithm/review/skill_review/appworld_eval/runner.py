from __future__ import annotations

from typing import Any

from .handle_chat_runner import run_appworld_eval_with_handle_chat_sync


def run_appworld_eval(
    tool: Any,
    task_ids: list[str],
    max_steps: int = 200,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run AppWorld tasks and return per-task results plus summary metrics."""
    return run_appworld_eval_with_handle_chat_sync(
        tool=tool,
        task_ids=task_ids,
        max_steps=max_steps,
        **kwargs,
    )
