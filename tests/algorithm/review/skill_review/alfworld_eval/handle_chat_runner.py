from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from lazyllm import LOG
from sqlalchemy import text

from .history_db import ensure_conversation_row, insert_chat_history_row
from .metrics import MAX_STEP_ERROR, compute_metrics, extract_gamefile, extract_won, infer_task_type_from_gamefile
from .prompt import ALFWORLD_SYSTEM_PROMPT
try:
    from ..skill_usage import skill_usage_counts
except ImportError:
    from skill_usage import skill_usage_counts


def _load_available_skill_names(create_user_id: str) -> list[str]:
    user_id = str(create_user_id or '').strip()
    if not user_id:
        return []

    try:
        from lazymind.review.skill_review.db import _get_app_conn

        with _get_app_conn().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT category, skill_name
                    FROM skill_resources
                    WHERE owner_user_id = :user_id
                      AND node_type = 'parent'
                      AND is_enabled = TRUE
                    ORDER BY category ASC, skill_name ASC, updated_at DESC
                    """
                ),
                {'user_id': user_id},
            ).mappings().all()
    except Exception as exc:
        LOG.warning(f'[ALFWorldEval] failed to load skills for user_id={user_id!r}: {exc}')
        return []

    skill_names: list[str] = []
    first_category_by_name: dict[str, str] = {}
    duplicate_categories: dict[str, set[str]] = {}
    for row in rows:
        skill_name = str(row.get('skill_name') or '').strip()
        category = str(row.get('category') or '').strip()
        if not skill_name:
            continue
        previous_category = first_category_by_name.get(skill_name)
        if previous_category is None:
            first_category_by_name[skill_name] = category
            skill_names.append(skill_name)
            continue
        if previous_category != category:
            duplicate_categories.setdefault(skill_name, {previous_category}).add(category)

    if duplicate_categories:
        collisions = ', '.join(
            f'{name}({", ".join(sorted(categories))})'
            for name, categories in sorted(duplicate_categories.items())
        )
        LOG.warning(
            f'[ALFWorldEval] duplicate skill names detected for user_id={user_id!r}; '
            f'using first match per plain name: {collisions}'
        )

    LOG.info(
        f'[ALFWorldEval] loaded {len(skill_names)} available skills for user_id={user_id!r}: '
        f'{skill_names}'
    )
    return skill_names


def _export_skills_to_local_dir(create_user_id: str) -> str | None:
    user_id = str(create_user_id or '').strip()
    if not user_id:
        return None

    try:
        from lazymind.review.skill_review.db import _get_app_conn

        with _get_app_conn().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT category, skill_name, node_type, parent_skill_name,
                           relative_path, content, file_ext, is_enabled
                    FROM skill_resources
                    WHERE owner_user_id = :user_id
                      AND is_enabled = TRUE
                    ORDER BY category ASC, skill_name ASC, node_type ASC, relative_path ASC
                    """
                ),
                {'user_id': user_id},
            ).mappings().all()
    except Exception as exc:
        LOG.warning(f'[ALFWorldEval] failed to export local skills for user_id={user_id!r}: {exc}')
        return None

    if not rows:
        return None

    root = Path(tempfile.mkdtemp(prefix='alfworld_eval_skills_'))
    parents_written = 0
    child_written = 0
    for row in rows:
        node_type = str(row.get('node_type') or '').strip()
        category = str(row.get('category') or '').strip() or 'uncategorized'
        skill_name = str(row.get('skill_name') or '').strip()
        if not skill_name:
            continue
        skill_dir = root / category / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        if node_type == 'parent':
            skill_md = skill_dir / 'SKILL.md'
            skill_md.write_text(str(row.get('content') or ''), encoding='utf-8')
            parents_written += 1
            continue

        rel_path = str(row.get('relative_path') or '').strip().replace('\\', '/')
        prefix = f'{category}/{str(row.get("parent_skill_name") or skill_name).strip()}/'
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]
        rel_path = rel_path.lstrip('/')
        if not rel_path:
            ext = str(row.get('file_ext') or 'md').strip().lstrip('.') or 'md'
            rel_path = f'{skill_name}.{ext}'
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(row.get('content') or ''), encoding='utf-8')
        child_written += 1

    if parents_written == 0:
        shutil.rmtree(root, ignore_errors=True)
        return None

    print(
        f'[ALFWorldEval] exported local skills for user_id={user_id!r} '
        f'root={str(root)!r} parents={parents_written} children={child_written}',
        flush=True,
    )
    return str(root)


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
    from lazymind.config import config as _cfg

    if num_tasks < 1:
        raise ValueError('num_tasks must be >= 1')
    if max_steps < 1:
        raise ValueError('max_steps must be >= 1')

    results: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    available_skills = _load_available_skill_names(create_user_id)
    local_skill_dir = _export_skills_to_local_dir(create_user_id) if available_skills else None
    print(
        f'[ALFWorldEval] create_user_id={create_user_id!r} '
        f'loaded_skills={len(available_skills)} {available_skills} '
        f'skill_fs_url={local_skill_dir or _cfg["skill_fs_url"]!r}',
        flush=True,
    )
    if not available_skills:
        LOG.warning(
            f'[ALFWorldEval] no enabled skills found for user_id={create_user_id!r}; '
            'skill tools will not be enabled for this run.'
        )
    try:
        with _cfg.temp('skill_fs_url', local_skill_dir or _cfg['skill_fs_url']):
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
                        available_skills=available_skills,
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
    finally:
        if local_skill_dir:
            shutil.rmtree(local_skill_dir, ignore_errors=True)

    summary = {
        'results': results,
        'metrics': compute_metrics(results, skill_names=available_skills),
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
    available_skills: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_payloads: list[dict[str, Any]] = []
    query = ''
    session_started_at = _utc_now_iso()
    try:
        tool.max_steps = max_steps
        initial = tool.reset()
        print(f'[ALFWorldEval] initial: {initial}')
        query = _build_task_query(str(initial.get('observation') or ''), max_steps)
        ensure_conversation_row(
            session_id,
            create_user_id=create_user_id,
            create_user_name=create_user_id,
            display_name=f'ALFWorld task {task_id + 1}',
            created_at=session_started_at,
            updated_at=session_started_at,
        )

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
            'available_skills': available_skills,
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

        usage_counts = skill_usage_counts(final_payloads, available_skills)
        result = _task_result(
            task_id,
            tool,
            error,
            used_skill=any(usage_counts.values()),
            skill_usage=usage_counts,
        )
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
        usage_counts = skill_usage_counts(final_payloads, available_skills)
        result = _task_result(
            task_id,
            tool,
            error,
            used_skill=any(usage_counts.values()),
            skill_usage=usage_counts,
        )
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


def _task_result(
    task_id: int,
    tool: Any,
    error: str | None,
    *,
    used_skill: bool,
    skill_usage: dict[str, int],
) -> dict[str, Any]:
    final_reward = float(getattr(tool, 'reward', 0.0) or 0.0)
    won = extract_won(getattr(tool, 'info', {}))
    gamefile = extract_gamefile(getattr(tool, 'info', {})) or str(getattr(tool, 'gamefile', '') or '')
    success = error is None and (
        won if won is not None else final_reward > 0 or bool(getattr(tool, 'done', False))
    )
    return {
        'task_id': task_id,
        'success': success,
        'steps': int(getattr(tool, 'step_count', 0) or 0),
        'final_reward': final_reward,
        'won': won,
        'gamefile': gamefile,
        'task_type': infer_task_type_from_gamefile(gamefile),
        'done': bool(getattr(tool, 'done', False)),
        'used_skill': used_skill,
        'skill_usage': skill_usage,
        'error': error,
    }
