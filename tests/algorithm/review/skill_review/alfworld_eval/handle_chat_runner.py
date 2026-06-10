from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from lazyllm import LOG

from .history_db import insert_chat_history_row
from .metrics import MAX_STEP_ERROR, compute_metrics
from .prompt import ALFWORLD_SYSTEM_PROMPT


def _build_flat_alfworld_tool_group(tool: Any) -> Any:
    """Build a LazyLLM tool group whose callable names stay unprefixed."""
    from lazyllm.tools.agent.toolsManager import ToolGroup

    return ToolGroup(
        tools=[
            tool.alfworld_step,
            tool.alfworld_status,
        ],
        name='alfworld',
        desc='Interact with one ALFWorld benchmark environment.',
        lazy=False,
        prefix=False,
    )


def _disabled_tool_names(active_tool_name: str) -> list[str]:
    from lazymind.chat.service import chat_service

    return [
        str(getattr(cfg, 'name', ''))
        for cfg in chat_service.DEFAULT_TOOLS
        if str(getattr(cfg, 'name', '')) != active_tool_name
    ]


def _adapt_handle_chat_kwargs(handle_chat: Any, kwargs: dict[str, Any], active_tool_name: str) -> dict[str, Any]:
    signature = inspect.signature(handle_chat)
    if 'disabled_tools' in signature.parameters:
        kwargs = dict(kwargs)
        kwargs['disabled_tools'] = _disabled_tool_names(active_tool_name)
    if not any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return kwargs


@contextmanager
def register_alfworld_tool(tool: Any):
    """Temporarily register ALFWorld as a handle_chat tool group."""
    from lazymind.chat.service import chat_service
    from lazymind.chat.service.component import ToolGroupConfig

    original_tools = list(chat_service.DEFAULT_TOOLS)
    original_build_system_prompt = chat_service.build_system_prompt

    def build_alfworld_system_prompt(active_groups: set[str], **kwargs: Any) -> str:
        prompt = original_build_system_prompt(active_groups, **kwargs)
        if 'alfworld' not in active_groups:
            return prompt
        return f'{prompt}\n\n## ALFWorld Benchmark Rules\n{ALFWORLD_SYSTEM_PROMPT}'

    chat_service.DEFAULT_TOOLS.append(
        ToolGroupConfig(
            name='alfworld',
            label='ALFWorld',
            description='Interact with one ALFWorld benchmark environment.',
            instance=_build_flat_alfworld_tool_group(tool),
        )
    )
    chat_service.build_system_prompt = build_alfworld_system_prompt
    try:
        yield
    finally:
        chat_service.DEFAULT_TOOLS[:] = original_tools
        chat_service.build_system_prompt = original_build_system_prompt


async def run_alfworld_eval_with_handle_chat(
    tool: Any,
    num_tasks: int = 100,
    max_steps: int = 50,
    *,
    session_prefix: str = 'alfworld-eval',
    create_user_id: str = 'alfworld_eval',
    create_user_name: str = '',
    model_config: dict[str, Any] | None = None,
    tool_config: dict[str, str] | None = None,
    persist_history: bool = True,
) -> dict[str, Any]:
    """Run ALFWorld tasks through chat_service.handle_chat."""
    if num_tasks < 1:
        raise ValueError('num_tasks must be >= 1')
    if max_steps < 1:
        raise ValueError('max_steps must be >= 1')

    results: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    with register_alfworld_tool(tool):
        for task_id in range(num_tasks):
            result, history_row = await _run_single_handle_chat_task(
                tool=tool,
                task_id=task_id,
                max_steps=max_steps,
                session_id=str(uuid.uuid4()),
                session_prefix=session_prefix,
                create_user_id=create_user_id,
                model_config=model_config,
                tool_config=tool_config,
            )
            results.append(result)
            history_rows.append(history_row)
            if persist_history:
                insert_chat_history_row(
                    history_row,
                    create_user_id=create_user_id,
                    create_user_name=create_user_name or create_user_id,
                )
                LOG.info(f'[ALFWorldEval] inserted chat history row successfully')

    summary = {
        'results': results,
        'metrics': compute_metrics(results),
        'chat_histories': history_rows,
    }
    return summary


def run_alfworld_eval_with_handle_chat_sync(
    tool: Any,
    num_tasks: int = 100,
    max_steps: int = 50,
    **kwargs: Any,
) -> dict[str, Any]:
    return asyncio.run(
        run_alfworld_eval_with_handle_chat(
            tool=tool,
            num_tasks=num_tasks,
            max_steps=max_steps,
            **kwargs,
        )
    )


async def _run_single_handle_chat_task(
    *,
    tool: Any,
    task_id: int,
    max_steps: int,
    session_id: str,
    session_prefix: str,
    create_user_id: str,
    model_config: dict[str, Any] | None,
    tool_config: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_payloads: list[dict[str, Any]] = []
    query = ''
    session_started_at = _utc_now_iso()
    try:
        tool.max_steps = max_steps
        initial = tool.reset()
        print(f'[ALFWorldEval] initial: {initial}')
        query = _build_task_query(str(initial.get('observation') or ''), max_steps)

        from lazymind.chat.service.chat_service import handle_chat

        handle_chat_kwargs = {
            'query': query,
            'history': [],
            'session_id': session_id,
            'filters': {},
            'files': [],
            'debug': False,
            'reasoning': False,
            'databases': [],
            'dataset': None,
            'priority': None,
            'available_tools': ['alfworld'],
            'available_skills': [],
            'memory': None,
            'user_preference': None,
            'use_memory': False,
            'trace': False,
            'environment_context': None,
            'user_id': create_user_id,
            'model_config': model_config,
            'tool_config': tool_config,
        }
        handle_chat_kwargs = _adapt_handle_chat_kwargs(handle_chat, handle_chat_kwargs, 'alfworld')
        response = await handle_chat(**handle_chat_kwargs)
        final_payloads = [payload async for payload in _iter_sse_payloads(response)]

        error = _extract_error(final_payloads)
        if error is None and not bool(getattr(tool, 'done', False)) and int(getattr(tool, 'step_count', 0) or 0) >= max_steps:
            error = MAX_STEP_ERROR

        result = _task_result(task_id, tool, error)
        return result, _build_chat_history_row(
            task_id=task_id,
            session_id=session_id,
            query=query,
            payloads=final_payloads,
            result=result,
            tool=tool,
            create_time=session_started_at,
            session_prefix=session_prefix,
        )
    except Exception as exc:
        error = str(exc)
        if 'max_steps exceeded' in error:
            error = MAX_STEP_ERROR
        result = _task_result(task_id, tool, error)
        return result, _build_chat_history_row(
            task_id=task_id,
            session_id=session_id,
            query=query,
            payloads=final_payloads,
            result=result,
            tool=tool,
            create_time=session_started_at,
            session_prefix=session_prefix,
        )
    finally:
        tool.max_steps = None


def _build_task_query(observation: str, max_steps: int) -> str:
    return (
        f'Initial observation:\n{observation}\n\n'
        'Call alfworld_step(action) until done=true, then stop.'
        'After each call to `alfworld_step`, you will observe a change in the `observation` and determine the next action based on the returned `admissible_actions`.'
        # 'Begin the task now. You can call `alfworld_step("")` to get the current admissible actions.'
    )


async def _iter_sse_payloads(response: Any) -> AsyncIterator[dict[str, Any]]:
    if isinstance(response, dict):
        yield response
        return
    body_iterator = getattr(response, 'body_iterator', None)
    if body_iterator is None:
        return
    async for chunk in body_iterator:
        text = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
        for block in text.strip().split('\n\n'):
            line = block.strip()
            if not line:
                continue
            if line.startswith('data:'):
                line = line.removeprefix('data:').strip()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _extract_error(payloads: list[dict[str, Any]]) -> str | None:
    for payload in reversed(payloads):
        if payload.get('code') == 500:
            return str(payload.get('msg') or 'handle_chat failed')
    return None


def _build_chat_history_row(
    *,
    task_id: int,
    session_id: str,
    query: str,
    payloads: list[dict[str, Any]],
    result: dict[str, Any],
    tool: Any,
    create_time: str,
    session_prefix: str,
) -> dict[str, Any]:
    text, reasoning_content, sources = _collect_stream_parts(payloads)
    update_time = _utc_now_iso()
    ext = {
        'benchmark': 'alfworld',
        'task_id': task_id,
        'session_id': session_id,
        'session_prefix': session_prefix,
        'success': result.get('success'),
        'steps': result.get('steps'),
        'final_reward': result.get('final_reward'),
        'done': result.get('done'),
        'error': result.get('error'),
        'tool_events': getattr(tool, 'tool_events', []),
        'sse_payloads': payloads,
    }
    ext['reasoning_content'] = reasoning_content

    return {
        'id': f'h_alfworld_{uuid.uuid4().hex[:24]}',
        'seq': task_id + 1,
        'conversation_id': session_id,
        'raw_content': query,
        'retrieval_result': {'sources': sources or None},
        'content': query,
        'result': _format_backend_result(text),
        'feed_back': 0,
        'reason': '',
        'expected_answer': '',
        'ext': ext,
        'version': '2.3',
        'create_time': create_time,
        'update_time': update_time,
    }


def _collect_stream_parts(payloads: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    think_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for payload in payloads:
        data = payload.get('data')
        if not isinstance(data, dict):
            continue
        if data.get('text'):
            text_parts.append(str(data.get('text')))
        if data.get('think'):
            think_parts.append(str(data.get('think')))
        if isinstance(data.get('sources'), list) and data.get('sources'):
            sources.extend(data['sources'])
    return ''.join(text_parts), ''.join(think_parts), sources


def _format_backend_result(text: str) -> str:
    """Match the backend chat_histories.result shape used by core service.

    Backend rows store the assistant/tool transcript as a plain text blob. In
    existing dumps, non-empty results are written with a leading blank line
    before the assistant content, while reasoning lives in ext.reasoning_content.
    """
    normalized = str(text or '')
    if not normalized:
        return ''
    return normalized if normalized.startswith('\n\n') else f'\n\n{normalized.lstrip()}'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_result(task_id: int, tool: Any, error: str | None) -> dict[str, Any]:
    final_reward = float(getattr(tool, 'reward', 0.0) or 0.0)
    return {
        'task_id': task_id,
        'success': error is None and final_reward > 0,
        'steps': int(getattr(tool, 'step_count', 0) or 0),
        'final_reward': final_reward,
        'done': bool(getattr(tool, 'done', False)),
        'error': error,
    }
