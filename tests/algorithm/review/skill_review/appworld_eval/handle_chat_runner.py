from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from sqlalchemy import text

try:
    from lazyllm import LOG
except ModuleNotFoundError:
    LOG = logging.getLogger(__name__)

from .appworld_tool import APPWORLD_TOOL_NAMES
from .history_db import ensure_conversation_row, insert_chat_history_row
from .metrics import MAX_STEP_ERROR, compute_metrics
from .prompt import APPWORLD_SYSTEM_PROMPT
try:
    from ..skill_usage import used_get_skill
except ImportError:
    from skill_usage import used_get_skill


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
        LOG.warning(f'[AppWorldEval] failed to load skills for user_id={user_id!r}: {exc}')
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
            f'[AppWorldEval] duplicate skill names detected for user_id={user_id!r}; '
            f'using first match per plain name: {collisions}'
        )

    LOG.info(
        f'[AppWorldEval] loaded {len(skill_names)} available skills for user_id={user_id!r}: '
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
        LOG.warning(f'[AppWorldEval] failed to export local skills for user_id={user_id!r}: {exc}')
        return None

    if not rows:
        return None

    root = Path(tempfile.mkdtemp(prefix='appworld_eval_skills_'))
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
            (skill_dir / 'SKILL.md').write_text(str(row.get('content') or ''), encoding='utf-8')
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
        f'[AppWorldEval] exported local skills for user_id={user_id!r} '
        f'root={str(root)!r} parents={parents_written} children={child_written}',
        flush=True,
    )
    return str(root)


def _inspect_exported_skill_names(local_skill_dir: str | None, available_skills: list[str]) -> list[str]:
    if not local_skill_dir:
        return []
    try:
        from lazyllm.tools.agent.skill_manager import SkillManager

        manager = SkillManager(dir=local_skill_dir, skills=available_skills)
        manager._load_skills_index()
        return sorted(str(name) for name in manager._skills_index)
    except Exception as exc:  # noqa: BLE001
        LOG.warning(f'[AppWorldEval] failed to inspect exported skills: {exc}')
        return []


def _build_flat_appworld_tool_group(tool: Any) -> Any:
    """Build a LazyLLM tool group whose callable names stay unprefixed."""
    from lazyllm.tools.agent.toolsManager import ToolGroup

    return ToolGroup(
        tools=[
            tool.appworld_execute,
            tool.appworld_task_info,
            tool.appworld_api_docs,
            tool.appworld_status,
        ],
        name='appworld_eval',
        desc='Run AppWorld benchmark environment tools.',
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
def register_appworld_tool(tool: Any):
    """Ensure AppWorld tools are visible to handle_chat during this run."""
    from lazymind.chat.service import chat_service
    from lazymind.chat.service.component import ToolGroupConfig

    original_tools = list(chat_service.DEFAULT_TOOLS)
    original_build_system_prompt = chat_service.build_system_prompt
    original_check_sensitive_content = chat_service.check_sensitive_content

    def build_appworld_system_prompt(active_groups: set[str], **kwargs: Any) -> str:
        prompt = original_build_system_prompt(active_groups, **kwargs)
        if 'appworld_eval' not in active_groups:
            return prompt
        if '## AppWorld Task Mode' in prompt or '## AppWorld Benchmark Rules' in prompt:
            return prompt
        return f'{prompt}\n\n## AppWorld Benchmark Rules\n{APPWORLD_SYSTEM_PROMPT}'

    def tool_group_names(cfg: Any) -> set[str]:
        aliases = getattr(cfg, 'aliases', ()) or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        return {str(getattr(cfg, 'name', '')), *(str(alias) for alias in aliases)}

    has_appworld_group = any(
        not tool_group_names(cfg).isdisjoint({'appworld_eval', *APPWORLD_TOOL_NAMES})
        for cfg in chat_service.DEFAULT_TOOLS
    )
    if not has_appworld_group:
        tool_group_kwargs = {
            'name': 'appworld_eval',
            'label': 'AppWorld',
            'description': 'Run AppWorld benchmark environment tools.',
            'instance': _build_flat_appworld_tool_group(tool),
        }
        if 'aliases' in inspect.signature(ToolGroupConfig).parameters:
            tool_group_kwargs['aliases'] = tuple(APPWORLD_TOOL_NAMES)
        chat_service.DEFAULT_TOOLS.append(ToolGroupConfig(**tool_group_kwargs))
    chat_service.build_system_prompt = build_appworld_system_prompt
    chat_service.check_sensitive_content = lambda query: None
    try:
        yield
    finally:
        chat_service.DEFAULT_TOOLS[:] = original_tools
        chat_service.build_system_prompt = original_build_system_prompt
        chat_service.check_sensitive_content = original_check_sensitive_content


async def run_appworld_eval_with_handle_chat(
    tool: Any,
    task_ids: list[str],
    max_steps: int = 200,
    *,
    session_prefix: str = 'appworld-eval',
    create_user_id: str = 'appworld_eval',
    create_user_name: str = '',
    model_config: dict[str, Any] | None = None,
    tool_config: dict[str, str] | None = None,
    persist_history: bool = True,
) -> dict[str, Any]:
    """Run AppWorld tasks through chat_service.handle_chat."""
    from lazymind.config import config as _cfg

    planned_task_ids = [str(task_id).strip() for task_id in task_ids if str(task_id).strip()]
    if not planned_task_ids:
        raise ValueError('task_ids must contain at least one AppWorld task id')
    if max_steps < 1:
        raise ValueError('max_steps must be >= 1')

    results: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    available_skills = _load_available_skill_names(create_user_id)
    local_skill_dir = _export_skills_to_local_dir(create_user_id) if available_skills else None
    indexed_skills = _inspect_exported_skill_names(local_skill_dir, available_skills)
    print(
        f'[AppWorldEval] create_user_id={create_user_id!r} '
        f'loaded_skills={len(available_skills)} {available_skills} '
        f'indexed_skills={len(indexed_skills)} {indexed_skills} '
        f'skill_fs_url={local_skill_dir or _cfg["skill_fs_url"]!r}',
        flush=True,
    )
    if not available_skills:
        LOG.warning(
            f'[AppWorldEval] no enabled skills found for user_id={create_user_id!r}; '
            'skill tools will not be enabled for this run.'
        )
    try:
        with _cfg.temp('skill_fs_url', local_skill_dir or _cfg['skill_fs_url']):
            with register_appworld_tool(tool):
                for episode_index, task_id in enumerate(planned_task_ids, start=1):
                    session_id = str(uuid.uuid4())
                    try:
                        result, history_row = await _run_single_handle_chat_task(
                            tool=tool,
                            episode_index=episode_index,
                            task_id=task_id,
                            max_steps=max_steps,
                            session_id=session_id,
                            session_prefix=session_prefix,
                            create_user_id=create_user_id,
                            model_config=model_config,
                            tool_config=tool_config,
                            available_skills=available_skills,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.exception(f'[AppWorldEval] failed episode={episode_index} task_id={task_id}')
                        result, history_row = _build_unhandled_error_task(
                            episode_index=episode_index,
                            task_id=task_id,
                            session_id=session_id,
                            session_prefix=session_prefix,
                            error=str(exc),
                        )
                    results.append(result)
                    history_rows.append(history_row)
                    if persist_history:
                        try:
                            insert_chat_history_row(
                                history_row,
                                create_user_id=create_user_id,
                                create_user_name=create_user_name or create_user_id,
                            )
                            LOG.info('[AppWorldEval] inserted chat history row successfully')
                        except Exception:  # noqa: BLE001
                            LOG.exception(
                                f'[AppWorldEval] failed to persist chat history '
                                f'episode={episode_index} task_id={task_id}; continuing'
                            )
    finally:
        if local_skill_dir:
            shutil.rmtree(local_skill_dir, ignore_errors=True)

    return {
        'results': results,
        'metrics': compute_metrics(results),
        'chat_histories': history_rows,
    }


def run_appworld_eval_with_handle_chat_sync(
    tool: Any,
    task_ids: list[str],
    max_steps: int = 200,
    **kwargs: Any,
) -> dict[str, Any]:
    return asyncio.run(
        run_appworld_eval_with_handle_chat(
            tool=tool,
            task_ids=task_ids,
            max_steps=max_steps,
            **kwargs,
        )
    )


async def _run_single_handle_chat_task(
    *,
    tool: Any,
    episode_index: int,
    task_id: str,
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
    prepare_data: dict[str, Any] = {}
    task_status: dict[str, Any] = {}
    evaluation_payload: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    error: str | None = None
    session_started_at = _utc_now_iso()
    original_max_interactions = getattr(tool, 'max_interactions', None)
    try:
        if original_max_interactions is None:
            tool.max_interactions = max_steps

        prepare_data = tool.prepare(session_id=session_id, task_id=task_id)
        task_info = prepare_data.get('task_info') if isinstance(prepare_data.get('task_info'), dict) else {}
        query = _build_task_query(prepare_data)
        print(f'[AppWorldEval] prepared task={task_id}: {query[:200]}')
        ensure_conversation_row(
            session_id,
            create_user_id=create_user_id,
            create_user_name=create_user_id,
            display_name=f'AppWorld task {task_id}',
            benchmark='appworld',
            task_id=task_id,
            created_at=session_started_at,
            updated_at=session_started_at,
        )

        environment_context = tool.environment_context(task_id=task_id, task_info=task_info)

        response = await _call_handle_chat(
            query=query,
            session_id=session_id,
            create_user_id=create_user_id,
            environment_context=environment_context,
            model_config=model_config,
            tool_config=tool_config,
            available_skills=available_skills,
        )
        stream_error: Exception | None = None
        try:
            async for payload in _iter_payloads(response):
                final_payloads.append(payload)
        except Exception as exc:  # noqa: BLE001
            stream_error = exc
        service_tool_call_turns = _extract_service_tool_call_turns(final_payloads)
        runtime = _extract_runtime(final_payloads)
        error = _extract_error(final_payloads)
        if error is None and stream_error is not None:
            error = str(stream_error)

        task_status = _safe_call_dict(lambda: tool.status(session_id))
        if not runtime:
            runtime = _extract_status_trace(task_status)
        evaluation_payload = _safe_call_dict(lambda: tool.evaluate(session_id))
        if error is None:
            error = _extract_control_error(task_status, evaluation_payload)

        steps = _extract_step_count(task_status, runtime)
        completed = _extract_completed(task_status, runtime)
        if error is None and not completed and steps >= max_steps:
            error = MAX_STEP_ERROR

        result = _task_result(
            episode_index=episode_index,
            task_id=task_id,
            task_status=task_status,
            evaluation_payload=evaluation_payload,
            runtime=runtime,
            error=error,
            service_tool_call_turns=service_tool_call_turns,
            used_skill=used_get_skill(final_payloads),
        )
        return result, _build_chat_history_row(
            episode_index=episode_index,
            task_id=task_id,
            session_id=session_id,
            query=query,
            payloads=final_payloads,
            result=result,
            prepare_data=prepare_data,
            task_status=task_status,
            evaluation_payload=evaluation_payload,
            runtime=runtime,
            create_time=session_started_at,
            session_prefix=session_prefix,
        )
    except Exception as exc:
        error = str(exc)
        if 'max_interactions' in error or 'max_steps' in error:
            error = MAX_STEP_ERROR
        result = _task_result(
            episode_index=episode_index,
            task_id=task_id,
            task_status=task_status,
            evaluation_payload=evaluation_payload,
            runtime=runtime,
            error=error,
            service_tool_call_turns=None,
            used_skill=used_get_skill(final_payloads),
        )
        return result, _build_chat_history_row(
            episode_index=episode_index,
            task_id=task_id,
            session_id=session_id,
            query=query,
            payloads=final_payloads,
            result=result,
            prepare_data=prepare_data,
            task_status=task_status,
            evaluation_payload=evaluation_payload,
            runtime=runtime,
            create_time=session_started_at,
            session_prefix=session_prefix,
        )
    finally:
        tool.max_interactions = original_max_interactions
        try:
            tool.cleanup(session_id)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f'[AppWorldEval] cleanup skipped for session={session_id}: {exc}')
        _teardown_lazyllm_session(session_id)


async def _call_handle_chat(
    *,
    query: str,
    session_id: str,
    create_user_id: str,
    environment_context: dict[str, Any],
    model_config: dict[str, Any] | None,
    tool_config: dict[str, Any] | None,
    available_skills: list[str],
) -> Any:
    from lazymind.chat.service.chat_service import handle_chat

    kwargs: dict[str, Any] = {
        'query': query,
        'history': [],
        'session_id': session_id,
        'filters': {},
        'files': [],
        'debug': False,
        'reasoning': True,
        'databases': [],
        'dataset': None,
        'priority': None,
        'available_tools': ['appworld_eval'],
        'available_skills': available_skills,
        'memory': None,
        'user_preference': None,
        'use_memory': False,
        'trace': False,
        'environment_context': environment_context,
        'user_id': create_user_id,
        'model_config': model_config,
        'tool_config': tool_config,
    }
    kwargs = _adapt_handle_chat_kwargs(handle_chat, kwargs, 'appworld_eval')
    return await handle_chat(**kwargs)


def _teardown_lazyllm_session(session_id: str) -> None:
    try:
        import lazyllm

        lazyllm.globals._init_sid(sid=session_id)
        lazyllm.locals._init_sid(sid=session_id)
        lazyllm.globals.clear()
        lazyllm.locals.clear()
    except Exception:  # noqa: BLE001
        return


def _build_task_query(prepare_data: dict[str, Any]) -> str:
    task_info = prepare_data.get('task_info') if isinstance(prepare_data.get('task_info'), dict) else {}
    instruction = str(task_info.get('instruction') or prepare_data.get('prepared_query') or '').strip()
    if instruction:
        return instruction
    return 'Use appworld_task_info() to inspect the prepared AppWorld task, then complete it.'


def _build_unhandled_error_task(
    *,
    episode_index: int,
    task_id: str,
    session_id: str,
    session_prefix: str,
    error: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = f'AppWorld task {task_id} failed before the task instruction could be recorded.'
    result = {
        'episode_index': episode_index,
        'task_id': task_id,
        'success': False,
        'steps': 0,
        'tool_call_rounds': 0,
        'handle_chat_tool_call_turns': None,
        'used_skill': False,
        'completed': False,
        'evaluation': {},
        'task_status': {},
        'error': error,
    }
    history_row = _build_chat_history_row(
        episode_index=episode_index,
        task_id=task_id,
        session_id=session_id,
        query=query,
        payloads=[],
        result=result,
        prepare_data={},
        task_status={},
        evaluation_payload={},
        runtime={},
        create_time=_utc_now_iso(),
        session_prefix=session_prefix,
    )
    return result, history_row


async def _iter_payloads(response: Any) -> AsyncIterator[dict[str, Any]]:
    if isinstance(response, dict):
        yield response
        return
    body_iterator = getattr(response, 'body_iterator', None)
    if body_iterator is None:
        return
    buffer = ''
    async for chunk in body_iterator:
        text = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
        buffer += text
        while '\n\n' in buffer:
            block, buffer = buffer.split('\n\n', 1)
            payload = _parse_sse_payload_block(block)
            if payload is not None:
                yield payload
    payload = _parse_sse_payload_block(buffer)
    if payload is not None:
        yield payload


def _parse_sse_payload_block(block: str) -> dict[str, Any] | None:
    data_lines: list[str] = []
    raw_json_lines: list[str] = []
    for raw_line in str(block or '').splitlines():
        line = raw_line.strip()
        if not line or line.startswith(':'):
            continue
        if line.startswith('data:'):
            data_lines.append(line.removeprefix('data:').strip())
        else:
            raw_json_lines.append(line)
    if data_lines:
        line = '\n'.join(data_lines).strip()
    elif raw_json_lines:
        line = '\n'.join(raw_json_lines).strip()
    else:
        return None
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        LOG.warning(f'[AppWorldEval] dropped malformed SSE block: {line[:500]!r}')
        return None
    return payload if isinstance(payload, dict) else None


def _extract_error(payloads: list[dict[str, Any]]) -> str | None:
    for payload in reversed(payloads):
        if payload.get('code') == 500:
            return str(payload.get('msg') or 'handle_chat failed')
    return None


def _extract_service_tool_call_turns(payloads: list[dict[str, Any]]) -> int | None:
    for payload in reversed(payloads):
        data = payload.get('data')
        if isinstance(data, dict) and 'tool_call_turns' in data:
            try:
                return int(data.get('tool_call_turns') or 0)
            except (TypeError, ValueError):
                return None
    return None


def _extract_runtime(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    for payload in reversed(payloads):
        data = payload.get('data')
        if not isinstance(data, dict):
            continue
        runtime = data.get('runtime')
        if isinstance(runtime, dict):
            return runtime
    return {}


def _extract_status_trace(task_status: dict[str, Any]) -> dict[str, Any]:
    trace = task_status.get('trace')
    return trace if isinstance(trace, dict) else {}


def _safe_call_dict(callback: Any) -> dict[str, Any]:
    try:
        value = callback()
    except Exception as exc:  # noqa: BLE001
        return {'error': str(exc)}
    return value if isinstance(value, dict) else {}


def _extract_control_error(*payloads: dict[str, Any]) -> str | None:
    for payload in payloads:
        value = payload.get('error')
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_step_count(task_status: dict[str, Any], runtime: dict[str, Any]) -> int:
    for value in (
        task_status.get('interaction_count'),
        _nested_get(runtime, 'environment_final', 'status', 'interaction_count'),
    ):
        if isinstance(value, (int, float)):
            return int(value)
    tool_trace = runtime.get('tool_trace')
    if isinstance(tool_trace, list):
        return sum(
            1
            for item in tool_trace
            if isinstance(item, dict) and item.get('tool_name') == 'appworld_execute'
        )
    environment_trace = runtime.get('environment_trace')
    if isinstance(environment_trace, list):
        return len(environment_trace)
    return 0


def _extract_completed(task_status: dict[str, Any], runtime: dict[str, Any]) -> bool:
    for value in (
        task_status.get('completed'),
        _nested_get(runtime, 'environment_final', 'status', 'completed'),
    ):
        if isinstance(value, bool):
            return value
    return False


def _extract_evaluation(evaluation_payload: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    evaluation = evaluation_payload.get('evaluation')
    if isinstance(evaluation, dict):
        return evaluation
    evaluation = _nested_get(runtime, 'environment_final', 'evaluation')
    return evaluation if isinstance(evaluation, dict) else {}


def _nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_adjusted_success_info(
    *,
    success: bool,
    completed: bool,
    error: str | None,
    evaluation: dict[str, Any],
) -> tuple[bool, str | None]:
    if success:
        return True, None
    if error is not None or not completed or not isinstance(evaluation, dict):
        return False, None

    failures = evaluation.get('failures')
    passes = evaluation.get('passes')
    num_tests_raw = evaluation.get('num_tests')
    if not isinstance(failures, list) or len(failures) != 1:
        return False, None
    failure = failures[0] if isinstance(failures[0], dict) else {}
    requirement = str(failure.get('requirement') or '').strip().lower()
    trace = str(failure.get('trace') or '')
    if requirement != 'assert answers match.':
        return False, None
    if "'null'" not in trace and '==\nnull' not in trace.lower():
        return False, None
    if not isinstance(passes, list):
        return False, None
    try:
        num_tests = int(num_tests_raw or 0)
    except (TypeError, ValueError):
        return False, None
    if num_tests <= 0 or len(passes) != max(num_tests - 1, 0):
        return False, None
    return True, 'answer_should_be_null'


def _build_chat_history_row(
    *,
    episode_index: int,
    task_id: str,
    session_id: str,
    query: str,
    payloads: list[dict[str, Any]],
    result: dict[str, Any],
    prepare_data: dict[str, Any],
    task_status: dict[str, Any],
    evaluation_payload: dict[str, Any],
    runtime: dict[str, Any],
    create_time: str,
    session_prefix: str,
) -> dict[str, Any]:
    text, reasoning_content, sources = _collect_stream_parts(payloads)
    if not text and result.get('error'):
        text = f"[AppWorldEval] task failed: {result.get('error')}"
    update_time = _utc_now_iso()
    ext = {
        'benchmark': 'appworld',
        'episode_index': episode_index,
        'task_id': task_id,
        'session_id': session_id,
        'session_prefix': session_prefix,
        'success': result.get('success'),
        'adjusted_success': result.get('adjusted_success'),
        'adjusted_success_reason': result.get('adjusted_success_reason'),
        'steps': result.get('steps'),
        'tool_call_rounds': result.get('tool_call_rounds'),
        'handle_chat_tool_call_turns': result.get('handle_chat_tool_call_turns'),
        'completed': result.get('completed'),
        'error': result.get('error'),
        'prepare': prepare_data,
        'task_status': task_status,
        'evaluation': evaluation_payload,
        'runtime': runtime,
        'sse_payloads': payloads,
    }
    ext['reasoning_content'] = reasoning_content

    return {
        'id': f'h_appworld_{uuid.uuid4().hex[:24]}',
        'seq': episode_index,
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
    normalized = str(text or '')
    if not normalized:
        return ''
    return normalized if normalized.startswith('\n\n') else f'\n\n{normalized.lstrip()}'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_result(
    *,
    episode_index: int,
    task_id: str,
    task_status: dict[str, Any],
    evaluation_payload: dict[str, Any],
    runtime: dict[str, Any],
    error: str | None,
    service_tool_call_turns: int | None,
    used_skill: bool,
) -> dict[str, Any]:
    evaluation = _extract_evaluation(evaluation_payload, runtime)
    steps = _extract_step_count(task_status, runtime)
    completed = _extract_completed(task_status, runtime)
    success = error is None and bool(evaluation.get('success') is True)
    adjusted_success, adjusted_success_reason = _extract_adjusted_success_info(
        success=success,
        completed=completed,
        error=error,
        evaluation=evaluation,
    )
    return {
        'episode_index': episode_index,
        'task_id': task_id,
        'success': success,
        'adjusted_success': adjusted_success,
        'adjusted_success_reason': adjusted_success_reason,
        'steps': steps,
        'tool_call_rounds': steps,
        'handle_chat_tool_call_turns': service_tool_call_turns,
        'used_skill': used_skill,
        'completed': completed,
        'evaluation': evaluation,
        'task_status': {key: value for key, value in task_status.items() if key != 'trace'},
        'error': error,
    }
