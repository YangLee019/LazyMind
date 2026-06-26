from __future__ import annotations

import inspect
from typing import Any, Callable

from .metrics import MAX_STEP_ERROR, compute_metrics, extract_gamefile, extract_won, infer_task_type_from_gamefile
from .prompt import ALFWORLD_SYSTEM_PROMPT


AgentFactory = Callable[..., Any]


def run_alfworld_eval(
    agent: Any,
    tool: Any,
    num_tasks: int = 100,
    max_steps: int = 50,
) -> dict[str, Any]:
    """Run ALFWorld tasks and return per-task results plus summary metrics.

    ``agent`` can be an already constructed agent or a factory. The runner
    supports common agent methods such as ``run``, ``invoke``, ``chat``,
    ``__call__``, and LazyLLM-style direct calls. If an agent returns an action
    string or ``{"action": "..."}``, the runner calls ``tool.step`` itself.
    If the agent performs tool calls internally, the runner only observes
    ``tool.done`` and ``tool.step_count``.
    """
    if num_tasks < 1:
        raise ValueError('num_tasks must be >= 1')
    if max_steps < 1:
        raise ValueError('max_steps must be >= 1')

    results: list[dict[str, Any]] = []
    for task_id in range(num_tasks):
        task_agent = _build_or_reuse_agent(agent, tool, max_steps)
        result = _run_single_task(task_agent, tool, task_id, max_steps)
        results.append(result)

    return {
        'results': results,
        'metrics': compute_metrics(results),
    }


def _run_single_task(agent: Any, tool: Any, task_id: int, max_steps: int) -> dict[str, Any]:
    final_reward = 0.0
    try:
        _reset_agent_context(agent)
        _inject_system_prompt(agent, ALFWORLD_SYSTEM_PROMPT)
        initial = tool.reset()
        observation = str(initial.get('observation') or '')

        while not bool(getattr(tool, 'done', False)):
            if int(getattr(tool, 'step_count', 0)) >= max_steps:
                return _task_result(task_id, False, tool, final_reward, MAX_STEP_ERROR)

            response = _call_agent(agent, observation)
            action = _extract_action(response)
            if action:
                step_result = tool.step(action)
                observation = str(step_result.get('observation') or '')
                final_reward = float(step_result.get('reward') or 0.0)
            else:
                status = tool.status()
                observation = str(status.get('observation') or observation)

            if int(getattr(tool, 'step_count', 0)) >= max_steps and not bool(getattr(tool, 'done', False)):
                return _task_result(task_id, False, tool, final_reward, MAX_STEP_ERROR)

        final_reward = float(getattr(tool, 'reward', final_reward) or 0.0)
        return _task_result(task_id, _infer_success(tool, final_reward), tool, final_reward, None)
    except Exception as exc:
        return _task_result(task_id, False, tool, final_reward, str(exc))


def _build_or_reuse_agent(agent: Any, tool: Any, max_steps: int) -> Any:
    if not inspect.isfunction(agent) and not inspect.isclass(agent):
        return agent

    signature = inspect.signature(agent)
    kwargs = {}
    for name in signature.parameters:
        if name == 'tool':
            kwargs[name] = tool
        elif name == 'tools':
            kwargs[name] = [tool]
        elif name in {'system_prompt', 'prompt'}:
            kwargs[name] = ALFWORLD_SYSTEM_PROMPT
        elif name == 'max_steps':
            kwargs[name] = max_steps
    return agent(**kwargs)


def _reset_agent_context(agent: Any) -> None:
    for name in ('reset_context', 'reset_history', 'clear_history', 'reset'):
        method = getattr(agent, name, None)
        if callable(method):
            method()
            return


def _inject_system_prompt(agent: Any, prompt: str) -> None:
    for name in ('set_system_prompt', 'set_prompt'):
        method = getattr(agent, name, None)
        if callable(method):
            method(prompt)
            return
    for attr in ('system_prompt', 'prompt'):
        if hasattr(agent, attr):
            try:
                setattr(agent, attr, prompt)
                return
            except Exception:
                return


def _call_agent(agent: Any, observation: str) -> Any:
    for name in ('run', 'invoke', 'chat'):
        method = getattr(agent, name, None)
        if callable(method):
            return method(observation)
    if callable(agent):
        return agent(observation)
    raise TypeError('agent must be callable or expose run/invoke/chat')


def _extract_action(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        for key in ('action', 'tool_input', 'command'):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(response, str):
        text = response.strip()
        if not text:
            return None
        if text.startswith('alfworld_step('):
            return text.removeprefix('alfworld_step(').rstrip(')').strip(' "\'')
        return text.splitlines()[0].strip()
    return None


def _task_result(
    task_id: int,
    success: bool,
    tool: Any,
    final_reward: float,
    error: str | None,
) -> dict[str, Any]:
    won = extract_won(getattr(tool, 'info', {}))
    gamefile = extract_gamefile(getattr(tool, 'info', {})) or str(getattr(tool, 'gamefile', '') or '')
    return {
        'task_id': task_id,
        'success': bool(success),
        'steps': int(getattr(tool, 'step_count', 0) or 0),
        'final_reward': final_reward,
        'won': won,
        'gamefile': gamefile,
        'task_type': infer_task_type_from_gamefile(gamefile),
        'done': bool(getattr(tool, 'done', False)),
        'error': error,
    }


def _infer_success(tool: Any, final_reward: float) -> bool:
    won = extract_won(getattr(tool, 'info', {}))
    if won is not None:
        return won
    if final_reward > 0:
        return True
    return bool(getattr(tool, 'done', False))
