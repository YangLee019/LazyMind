from __future__ import annotations

import re
from typing import Any, Iterable

from chat.components.skill_review.schemas import SessionData, SessionMessage, Trajectory, TrajectoryStep

_TOOL_ROLE_NAMES = {'tool', 'function', 'tool_call'}
_PREFERRED_RESULT_KEYS = (
    'success',
    'status',
    'error',
    'message',
    'name',
    'title',
    'path',
    'id',
    'type',
    'count',
    'total',
    'query',
    'location',
    'condition',
)


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

    for index, message in enumerate(session.messages, start=1):
        role = _normalize_role(message.role)
        if role == 'user':
            user_text = _compress_user_message(message.content)
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
            tool_output = _compress_tool_output(_extract_tool_output(message))
            output_text = _summarize_tool_result(tool_name, tool_output)
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
                    tool_output=tool_output,
                    raw=message.raw,
                )
            )
            continue

        assistant_reasoning = _compress_reasoning(_extract_reasoning(message))
        assistant_text = _compress_assistant_text(_extract_assistant_text(message))
        tool_calls = _extract_tool_calls(message)
        primary_tool_input = _compress_tool_input(tool_calls[0]['arguments']) if tool_calls else None
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
                    action=_summarize_assistant_action(assistant_text, assistant_reasoning, tool_calls),
                    state=_summarize_assistant_state(assistant_text, assistant_reasoning, tool_calls),
                    tool_name=tool_name,
                    skill_name=skill_name,
                    message_index=index,
                    reasoning=assistant_reasoning or None,
                    result=assistant_text or None,
                    tool_input=primary_tool_input,
                    raw=message.raw,
                )
            )

        for sub_index, tool_call in enumerate(tool_calls, start=1):
            call_name = str(tool_call.get('name') or '').strip() or None
            call_id = str(tool_call.get('id') or '').strip() or None
            compact_input = _compress_tool_input(tool_call.get('arguments'))
            if call_name:
                called_tools.append(call_name)
            if call_id and skill_name:
                tool_call_skill_by_id[call_id] = skill_name
            steps.append(
                TrajectoryStep(
                    step_index=len(steps) + 1,
                    role='assistant',
                    kind='tool_call',
                    action=_summarize_tool_call(call_name, compact_input),
                    state='',
                    tool_name=call_name,
                    skill_name=skill_name,
                    message_index=index,
                    sub_index=sub_index,
                    tool_input=compact_input,
                    raw=tool_call,
                )
            )

    _assign_task_segments(steps)
    _mark_task_ends(steps)

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


def _compress_user_message(text: str) -> str:
    return _normalize_text(text, 600)


def _compress_assistant_text(text: str) -> str:
    return _normalize_text(text, 800)


def _compress_reasoning(text: str) -> str:
    return _normalize_text(text, 500)


def _compress_tool_input(value: Any) -> Any:
    return _compress_payload(value, depth=0, max_items=4)


def _compress_tool_output(value: Any) -> Any:
    return _compress_payload(value, depth=0, max_items=4)


def _compress_payload(value: Any, *, depth: int, max_items: int) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if parsed is not value:
            return _compress_payload(parsed, depth=depth, max_items=max_items)
        return _normalize_text(value, 500)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        items = [(str(key), sub_value) for key, sub_value in value.items() if _has_signal(sub_value)]
        if not items:
            return {}
        prioritized = sorted(
            items,
            key=lambda item: (
                0 if item[0] in _PREFERRED_RESULT_KEYS else 1,
                item[0],
            ),
        )
        compressed: dict[str, Any] = {}
        for key, sub_value in prioritized[:max_items]:
            compressed[key] = _compress_payload(sub_value, depth=depth + 1, max_items=3 if depth >= 1 else 4)
        omitted = len(prioritized) - len(compressed)
        if omitted > 0:
            compressed['_truncated'] = omitted
        return compressed
    if isinstance(value, list):
        if not value:
            return []
        sample = [
            _compress_payload(item, depth=depth + 1, max_items=3)
            for item in value[:max_items]
        ]
        if len(value) <= max_items and all(not isinstance(item, (dict, list)) for item in value):
            return sample
        if len(value) <= max_items and depth >= 2:
            return sample
        return {
            'count': len(value),
            'sample': sample,
        }
    return _normalize_text(str(value), 500)


def _summarize_assistant_action(
    assistant_text: str,
    assistant_reasoning: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    if assistant_text:
        return assistant_text
    if tool_calls:
        return _summarize_tool_call(
            str(tool_calls[0].get('name') or '').strip() or None,
            _compress_tool_input(tool_calls[0].get('arguments')),
        )
    return assistant_reasoning


def _summarize_assistant_state(
    assistant_text: str,
    assistant_reasoning: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    if assistant_reasoning:
        return assistant_reasoning
    if tool_calls:
        return _normalize_text(
            f"Preparing tool call: {_summarize_tool_call(str(tool_calls[0].get('name') or '').strip() or None, _compress_tool_input(tool_calls[0].get('arguments')))}",
            400,
        )
    return assistant_text


def _summarize_tool_call(tool_name: str | None, tool_input: Any) -> str:
    name = tool_name or 'tool'
    compact_input = _summarize_value(tool_input, 220)
    if compact_input:
        return _normalize_text(f"Call {name} with {compact_input}", 260)
    return f"Call {name}"


def _summarize_tool_result(tool_name: str | None, tool_output: Any) -> str:
    name = tool_name or 'tool'
    summary = _summarize_value(tool_output, 260)
    if summary:
        return _normalize_text(f"{name} returned {summary}", 320)
    return f"{name} returned a result"


def _summarize_value(value: Any, limit: int = 260) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if key == '_truncated':
                parts.append(f"{item} more fields")
                continue
            if isinstance(item, dict) and 'count' in item and 'sample' in item:
                parts.append(f"{key}({item['count']} items)")
                continue
            if isinstance(item, list):
                rendered = ', '.join(_normalize_text(str(part), 60) for part in item[:3])
                if rendered:
                    parts.append(f"{key}=[{rendered}]")
                continue
            rendered = _normalize_text(str(item), 80)
            if rendered:
                parts.append(f"{key}={rendered}")
        return _shorten(', '.join(parts), limit)
    if isinstance(value, list):
        rendered = ', '.join(_normalize_text(str(item), 60) for item in value[:3])
        return _shorten(rendered, limit)
    return _normalize_text(str(value), limit)


def _normalize_text(text: str, limit: int) -> str:
    value = str(text or '').strip()
    if not value:
        return ''
    value = re.sub(r'```(?:json|text|markdown)?\s*', '', value)
    value = value.replace('```', ' ')
    value = re.sub(r'https?://\S+', '[url]', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return _shorten(value, limit)


def _has_signal(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _assign_task_segments(steps: list[TrajectoryStep]) -> None:
    segment_id = 0
    for step in steps:
        if step.kind == 'user_message' and step.role == 'user':
            segment_id += 1
        elif segment_id == 0:
            segment_id = 1
        step.task_segment_id = segment_id


def _mark_task_ends(steps: list[TrajectoryStep]) -> None:
    last_assistant_step_by_segment: dict[int, TrajectoryStep] = {}
    for step in steps:
        if not _is_task_end_candidate(step):
            continue
        last_assistant_step_by_segment[step.task_segment_id] = step

    for step in last_assistant_step_by_segment.values():
        step.is_task_end = True


def _is_task_end_candidate(step: TrajectoryStep) -> bool:
    if step.kind == 'assistant_message' and step.role == 'assistant':
        raw = step.raw if isinstance(step.raw, dict) else {}
        if raw.get('tool_calls'):
            return False
        return bool(str(step.result or '').strip())
    if step.kind == 'tool_result' and step.role == 'tool':
        return bool(str(step.action or '').strip())
    return False
