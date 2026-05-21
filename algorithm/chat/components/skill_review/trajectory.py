from __future__ import annotations

import re
from typing import Any, Iterable

from chat.components.skill_review.schemas import SessionData, SessionMessage, Trajectory, TrajectoryStep

_TOOL_ROLE_NAMES = {'tool', 'function', 'tool_call'}
_FINAL_ANSWER_LIMIT = 4000


def build_trajectory(
    session: SessionData,
    *,
    min_user_turns: int,
    min_tool_turns: int,
) -> Trajectory:
    steps: list[TrajectoryStep] = []
    called_tools: list[str] = []
    called_skills: list[str] = []
    tool_call_skill_by_id: dict[str, str] = {}
    final_answer: str | None = None

    for index, message in enumerate(session.messages, start=1):
        role = _normalize_role(message.role)
        if role == 'user':
            user_text = _shorten(message.content, 2000)
            steps.append(
                TrajectoryStep(
                    step_index=len(steps) + 1,
                    role='user',
                    kind='user_message',
                    action=user_text,
                    state=user_text,
                    user_message=user_text,
                    message_index=index,
                    raw=message.raw,
                )
            )
            continue

        if role == 'tool':
            tool_name = message.tool_name or _extract_tool_name(message)
            skill_name = message.skill_name or _extract_skill_name(message)
            if not skill_name:
                skill_name = _infer_skill_name_from_tool_result(tool_name, message, tool_call_skill_by_id)
            if tool_name:
                called_tools.append(tool_name)
            if skill_name:
                called_skills.append(skill_name)
            output_text = _shorten(message.content, 2000)
            steps.append(
                TrajectoryStep(
                    step_index=len(steps) + 1,
                    role='tool',
                    kind='tool_result',
                    action=output_text,
                    state=output_text,
                    tool_name=tool_name,
                    skill_name=skill_name,
                    message_index=index,
                    tool_output=_extract_tool_output(message),
                    raw=message.raw,
                )
            )
            continue

        assistant_reasoning = _shorten(_extract_reasoning(message), 4000)
        assistant_text = _shorten(_extract_assistant_text(message), 4000)
        tool_calls = _extract_tool_calls(message)
        tool_name = message.tool_name or _extract_tool_name(message)
        skill_name = message.skill_name or _extract_skill_name(message)
        if not skill_name:
            skill_name = _infer_skill_name_from_tool_calls(tool_calls)
        if tool_name:
            called_tools.append(tool_name)
        if skill_name:
            called_skills.append(skill_name)

        if assistant_reasoning or assistant_text:
            steps.append(
                TrajectoryStep(
                    step_index=len(steps) + 1,
                    role='assistant',
                    kind='assistant_message',
                    action=assistant_text or assistant_reasoning or '',
                    state=assistant_reasoning or assistant_text or '',
                    tool_name=tool_name,
                    skill_name=skill_name,
                    message_index=index,
                    reasoning=assistant_reasoning or None,
                    result=assistant_text or None,
                    tool_input=tool_calls[0]['arguments'] if tool_calls else None,
                    raw=message.raw,
                )
            )

        for sub_index, tool_call in enumerate(tool_calls, start=1):
            call_name = str(tool_call.get('name') or '').strip() or None
            call_id = str(tool_call.get('id') or '').strip() or None
            if call_name:
                called_tools.append(call_name)
            if call_id and skill_name:
                tool_call_skill_by_id[call_id] = skill_name
            steps.append(
                TrajectoryStep(
                    step_index=len(steps) + 1,
                    role='assistant',
                    kind='tool_call',
                    action=_shorten(call_name or '', 2000),
                    state='',
                    tool_name=call_name,
                    skill_name=skill_name,
                    message_index=index,
                    sub_index=sub_index,
                    tool_input=tool_call.get('arguments'),
                    raw=tool_call,
                )
            )

        if _is_final_answer_candidate(session.messages, index, message, tool_calls):
            final_answer = _shorten(_extract_assistant_text(message), _FINAL_ANSWER_LIMIT) or final_answer
            for step in reversed(steps):
                if step.message_index == index and step.role == 'assistant':
                    step.is_final = True
                    break

    user_turns = sum(1 for step in steps if step.role == 'user')
    tool_turns = sum(
        1 for step in steps
        if step.kind in {'tool_call', 'tool_result'} or step.role in _TOOL_ROLE_NAMES
    )
    qualified = user_turns >= min_user_turns and tool_turns >= min_tool_turns
    skip_reason = None
    if not qualified:
        skip_reason = (
            f'trigger threshold not met: user_turns={user_turns}, '
            f'tool_turns={tool_turns}, min_user_turns={min_user_turns}, '
            f'min_tool_turns={min_tool_turns}'
        )

    return Trajectory(
        session_id=session.session_id,
        user_turns=user_turns,
        tool_turns=tool_turns,
        called_tools=_unique(called_tools),
        called_skills=_unique(called_skills),
        steps=steps,
        final_answer=final_answer,
        qualified=qualified,
        skip_reason=skip_reason,
    )


def _normalize_role(role: str) -> str:
    lowered = str(role or '').strip().lower()
    if lowered in {'human', 'customer'}:
        return 'user'
    if lowered in {'ai', 'agent', 'bot'}:
        return 'assistant'
    if 'tool' in lowered or 'function' in lowered:
        return 'tool'
    return lowered or 'unknown'


def _extract_tool_name(message: SessionMessage) -> str | None:
    raw = message.raw if isinstance(message.raw, dict) else {}
    name = raw.get('tool_name') or raw.get('name')
    if isinstance(name, str) and name.strip():
        return name.strip()

    tool_calls = _extract_tool_calls(message)
    if tool_calls:
        call_name = tool_calls[0].get('name')
        if isinstance(call_name, str) and call_name.strip():
            return call_name.strip()
    return None


def _extract_skill_name(message: SessionMessage) -> str | None:
    raw = message.raw if isinstance(message.raw, dict) else {}
    for key in ('skill_name', 'skill', 'called_skill'):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_reasoning(message: SessionMessage) -> str:
    raw = message.raw if isinstance(message.raw, dict) else {}
    reasoning = raw.get('reasoning_content') or raw.get('reasoning') or raw.get('think') or ''
    if reasoning:
        return str(reasoning)
    return ''


def _extract_assistant_text(message: SessionMessage) -> str:
    raw = message.raw if isinstance(message.raw, dict) else {}
    content = raw.get('content')
    if content is None:
        return str(message.content or '')
    return str(content)


def _extract_tool_output(message: SessionMessage) -> Any:
    raw = message.raw if isinstance(message.raw, dict) else {}
    if 'result' in raw and raw.get('result') is not None:
        return raw.get('result')
    if 'content' in raw and raw.get('content') is not None:
        return _safe_json_loads(raw.get('content'))
    return _safe_json_loads(message.content)


def _infer_skill_name_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    for tool_call in tool_calls:
        call_name = str(tool_call.get('name') or '').strip()
        arguments = tool_call.get('arguments')
        if not isinstance(arguments, dict):
            continue
        if call_name in {'skill_manage', 'get_skill', 'run_script'}:
            name = arguments.get('name')
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _infer_skill_name_from_tool_result(
    tool_name: str | None,
    message: SessionMessage,
    tool_call_skill_by_id: dict[str, str],
) -> str | None:
    raw = message.raw if isinstance(message.raw, dict) else {}
    tool_call_id = raw.get('tool_call_id')
    if isinstance(tool_call_id, str):
        mapped = tool_call_skill_by_id.get(tool_call_id)
        if mapped:
            return mapped
    if tool_name in {'skill_manage', 'get_skill', 'run_script'}:
        content = raw.get('content')
        if isinstance(content, str) and content.strip():
            parsed = _safe_json_loads(content)
            if isinstance(parsed, dict):
                result = parsed.get('result')
                if isinstance(result, dict):
                    name = result.get('name')
                    if isinstance(name, str) and name.strip():
                        return name.strip()
    return None


def _extract_tool_calls(message: SessionMessage) -> list[dict[str, Any]]:
    raw = message.raw if isinstance(message.raw, dict) else {}
    tool_calls = raw.get('tool_calls')
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else {}
        arguments = tool_call.get('arguments')
        if arguments is None and isinstance(function, dict):
            arguments = function.get('arguments')
        normalized.append({
            'id': tool_call.get('id'),
            'name': tool_call.get('name') or function.get('name'),
            'arguments': arguments if isinstance(arguments, dict) else _safe_json_loads(arguments),
        })
    return normalized


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        import json

        return json.loads(value)
    except Exception:
        return value


def _is_final_answer_candidate(
    messages: list[SessionMessage],
    index: int,
    message: SessionMessage,
    tool_calls: list[dict[str, Any]],
) -> bool:
    raw = message.raw if isinstance(message.raw, dict) else {}
    if raw.get('tool_calls'):
        return False
    text = str(message.content or '').strip()
    if not text:
        return False
    if message.role not in {'assistant', 'ai', 'agent', 'bot'}:
        return False
    if index != len(messages):
        return False
    return True


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        trimmed = str(value or '').strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        result.append(trimmed)
    return result


def _shorten(text: str, limit: int) -> str:
    text = str(text or '').strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + '...'
