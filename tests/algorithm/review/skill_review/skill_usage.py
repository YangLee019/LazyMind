from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_SKILL_TOOL_NAMES = {'get_skill', 'read_reference', 'run_script'}


def used_get_skill(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        if _payload_contains_skill_tool(payload):
            return True
    return False


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
