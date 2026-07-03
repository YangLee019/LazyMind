from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_SKILL_TOOL_NAMES = {'get_skill', 'read_reference', 'run_script'}
_SKILL_NAME_ARGUMENTS = ('name', 'skill_name', 'skill')


def used_get_skill(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        if _payload_contains_skill_tool(payload):
            return True
    return False


def skill_usage_counts(
    payloads: list[dict[str, Any]],
    skill_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, int]:
    """Count skill tool calls by injected skill name."""
    constrained_to_injected = skill_names is not None
    skill_name_map = _build_skill_name_map(skill_names)
    counts = {name: 0 for name in _ordered_skill_names(skill_names)}
    if constrained_to_injected and not skill_name_map:
        return counts
    for raw_name in _iter_skill_tool_call_names(payloads):
        skill_name = _match_injected_skill_name(raw_name, skill_name_map)
        if not skill_name:
            continue
        counts[skill_name] = counts.get(skill_name, 0) + 1
    if constrained_to_injected:
        return counts
    return {name: count for name, count in counts.items() if count > 0}


def has_non_empty_trajectory(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    for key in ('steps', 'tool_call_rounds', 'handle_chat_tool_call_turns'):
        if _safe_non_negative_int(result.get(key)) > 0:
            return True
    return bool(result.get('used_skill'))


def aggregate_skill_usage_counts(
    results: list[dict[str, Any]],
    skill_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, int]:
    constrained_to_injected = skill_names is not None
    counts = {name: 0 for name in _ordered_skill_names(skill_names)}
    allowed = _build_skill_name_map(skill_names)
    if constrained_to_injected and not allowed:
        return counts
    for result in results:
        if not isinstance(result, dict):
            continue
        usage = result.get('skill_usage')
        if not isinstance(usage, dict):
            continue
        for raw_name, raw_count in usage.items():
            skill_name = _match_injected_skill_name(str(raw_name or ''), allowed)
            if not skill_name:
                continue
            counts[skill_name] = counts.get(skill_name, 0) + _safe_non_negative_int(raw_count)
    if constrained_to_injected:
        return counts
    return counts


def _payload_contains_skill_tool(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    if _matches_tool_name(payload.get('name')) or _matches_tool_name(payload.get('tool_name')):
        return True

    data = payload.get('data')
    if isinstance(data, dict):
        if _matches_tool_name(data.get('name')) or _matches_tool_name(data.get('tool_name')):
            return True
        if _structured_tool_calls_contain_skill(data.get('tool_calls')):
            return True
        if _structured_tool_results_contain_skill(data.get('tool_results')):
            return True
        if _text_contains_skill_tool_call(str(data.get('text') or '')):
            return True

    return False


def _iter_skill_tool_call_names(payloads: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        names.extend(_extract_skill_names_from_tool_call(payload))
        tool_calls = payload.get('tool_calls')
        if isinstance(tool_calls, list):
            names.extend(_structured_tool_call_names(tool_calls))

        data = payload.get('data')
        if not isinstance(data, dict):
            continue
        names.extend(_extract_skill_names_from_tool_call(data))
        tool_calls = data.get('tool_calls')
        if isinstance(tool_calls, list):
            names.extend(_structured_tool_call_names(tool_calls))
        names.extend(_text_skill_tool_call_names(str(data.get('text') or '')))
    return names


def _extract_skill_names_from_tool_call(payload: dict[str, Any]) -> list[str]:
    if not (
        _matches_tool_name(payload.get('name'))
        or _matches_tool_name(payload.get('tool_name'))
    ):
        return []
    arguments = _extract_arguments(payload)
    skill_name = _skill_name_from_arguments(arguments)
    return [skill_name] if skill_name else []


def _structured_tool_call_names(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        names.extend(_extract_skill_names_from_tool_call(item))
        function = item.get('function')
        if isinstance(function, dict):
            names.extend(_extract_skill_names_from_tool_call(function))
    return names


def _structured_tool_calls_contain_skill(tool_calls: Any) -> bool:
    if not isinstance(tool_calls, list):
        return False
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        function = item.get('function')
        if isinstance(function, dict) and _matches_tool_name(function.get('name')):
            return True
        if _matches_tool_name(item.get('name')) or _matches_tool_name(item.get('tool_name')):
            return True
    return False


def _structured_tool_results_contain_skill(tool_results: Any) -> bool:
    if not isinstance(tool_results, list):
        return False
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if _matches_tool_name(item.get('name')) or _matches_tool_name(item.get('tool_name')):
            return True
    return False


def _text_skill_tool_call_names(text: str) -> list[str]:
    if not text:
        return []
    names: list[str] = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if not raw:
            continue
        tool_call = _json_object(raw)
        if not tool_call:
            continue
        names.extend(_extract_skill_names_from_tool_call(tool_call))
    return names


def _text_contains_skill_tool_call(text: str) -> bool:
    if not text:
        return False
    for match in _TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            tool_call = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if _matches_tool_name(tool_call.get('name')) or _matches_tool_name(tool_call.get('tool_name')):
            return True
    return False


def _matches_tool_name(value: Any) -> bool:
    name = str(value or '').strip()
    if not name:
        return False
    if name in _SKILL_TOOL_NAMES:
        return True
    return any(name.endswith(f'_{base_name}') for base_name in _SKILL_TOOL_NAMES)


def _extract_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ('arguments', 'args', 'parameters'):
        arguments = _json_object(payload.get(key))
        if arguments:
            return arguments
    return {}


def _skill_name_from_arguments(arguments: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        return ''
    for key in _SKILL_NAME_ARGUMENTS:
        value = arguments.get(key)
        if isinstance(value, dict):
            value = value.get('name') or value.get('skill_name')
        name = str(value or '').strip()
        if name:
            return name
    for key in ('rel_path', 'path'):
        path = str(arguments.get(key) or '').strip()
        if path:
            return path
    return ''


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ordered_skill_names(skill_names: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_name in skill_names or []:
        name = str(raw_name or '').strip()
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _build_skill_name_map(skill_names: list[str] | tuple[str, ...] | set[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in _ordered_skill_names(skill_names):
        result[name] = name
        result[name.rsplit('/', 1)[-1]] = name
    return result


def _match_injected_skill_name(raw_name: str, skill_name_map: dict[str, str]) -> str:
    name = str(raw_name or '').strip()
    if not name:
        return ''
    if not skill_name_map:
        return name
    if name in skill_name_map:
        return skill_name_map[name]
    for part in re.split(r'[/\\]+', name):
        if part in skill_name_map:
            return skill_name_map[part]
    return ''


def _safe_non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0
