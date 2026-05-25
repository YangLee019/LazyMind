from __future__ import annotations

import re
import unicodedata
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
_PRIMARY_INPUT_KEYS = (
    'query',
    'keyword',
    'keywords',
    'url',
    'link',
    'target',
    'name',
    'action',
    'title',
    'path',
    'file',
    'filename',
    'rel_path',
)
_COLLECTION_KEYS = {'items', 'results', 'records', 'matches', 'documents', 'data'}
_LONG_TEXT_KEYS = {'content', 'text', 'body', 'summary', 'description', 'excerpt', 'snippet', 'reason'}
_ERROR_KEYS = {'error', 'message', 'reason', 'error_type'}
_DECISION_HINTS = (
    '我将', '我会', '我先', '让我', '接下来', '准备', '尝试', '改用',
    'i will', 'i\'ll', 'let me', 'next', 'then i will', 'i should', 'i can try', 'i am going to',
)
_REASONING_HINTS = (
    '因为', '由于', '根据', '需要', '应该', '所以', '为了', '如果', '先', '再',
    'because', 'since', 'based on', 'need to', 'should', 'so that', 'in order to', 'if', 'first', 'then',
)
_RESULT_HINTS = (
    '成功', '失败', '完成', '找到', '获取', '保存', '记录', '无法', '可以', '需要', '主要', '当前',
    'success', 'failed', 'completed', 'found', 'retrieved', 'saved', 'recorded', 'unable', 'can', 'need', 'mainly', 'current',
)
_ERROR_HINTS = (
    'timeout',
    'timed out',
    'unreachable',
    'not available',
    'not found',
    'failed',
    'error',
    'denied',
    'missing',
    '异常',
    '失败',
    '错误',
    '超时',
    '不可达',
    '不可用',
    '缺失',
)
_TAIL_PHRASE_HINTS = (
    '有什么我可以帮助你',
    '有具体任务我可以帮您执行吗',
    '还有什么我可以帮助你',
    '如果你有特定需求',
    'how can i help you',
    'what can i help you with',
    'let me know if you want more details',
    'feel free to ask',
    'anything else i can help with',
    'is there anything else i can help',
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
    cleaned = _clean_text(text)
    summary = _extract_text_summary(cleaned, focus='result', max_sentences=2, limit=180)
    return summary or _shorten(cleaned, 180)


def _compress_assistant_text(text: str) -> str:
    raw_text = str(text or '')
    cleaned = _clean_text(raw_text)
    if not cleaned:
        return ''
    structured = _extract_structured_summary(raw_text)
    if structured:
        return structured
    if _looks_like_content_dump(raw_text):
        return _summarize_content_dump(raw_text)
    summary = _extract_text_summary(raw_text, focus='result', max_sentences=2, limit=260)
    return summary or _shorten(cleaned, 260)


def _compress_reasoning(text: str) -> str:
    cleaned = _clean_text(text)
    summary = _extract_text_summary(cleaned, focus='reasoning', max_sentences=2, limit=220)
    return summary or _shorten(cleaned, 220)


def _compress_tool_input(value: Any) -> Any:
    return _compress_payload(value, depth=0, max_items=4)


def _compress_tool_output(value: Any) -> Any:
    return _compress_payload(value, depth=0, max_items=4)


def _compress_payload(value: Any, *, depth: int, max_items: int) -> Any:
    return _compress_payload_for_field(value, field_name=None, depth=depth, max_items=max_items)


def _compress_payload_for_field(
    value: Any,
    *,
    field_name: str | None,
    depth: int,
    max_items: int,
) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if parsed is not value:
            return _compress_payload_for_field(parsed, field_name=field_name, depth=depth, max_items=max_items)
        return _compress_text_value(value, field_name=field_name)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        items = [(str(key), sub_value) for key, sub_value in value.items() if _has_signal(sub_value)]
        if not items:
            return {}
        prioritized = sorted(
            items,
            key=lambda item: (
                _payload_priority(item[0], item[1], depth),
                item[0],
            ),
        )
        compressed: dict[str, Any] = {}
        for key, sub_value in prioritized[:max_items]:
            compressed[key] = _compress_payload_for_field(
                sub_value,
                field_name=key,
                depth=depth + 1,
                max_items=3 if depth >= 1 else 4,
            )
        omitted = len(prioritized) - len(compressed)
        if omitted > 0:
            compressed['_truncated'] = omitted
        return compressed
    if isinstance(value, list):
        if not value:
            return []
        sample_limit = 3 if field_name in _COLLECTION_KEYS or depth >= 1 else max_items
        sample = [
            _compress_list_item(item, depth=depth + 1)
            for item in value[:sample_limit]
        ]
        if len(value) <= sample_limit and all(not isinstance(item, (dict, list)) for item in value):
            return sample
        if len(value) <= sample_limit and depth >= 2:
            return sample
        return {
            'count': len(value),
            'sample': sample,
        }
    return _normalize_text(str(value), 220)


def _summarize_assistant_action(
    assistant_text: str,
    assistant_reasoning: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    if tool_calls:
        decision = _extract_text_summary(
            assistant_text or assistant_reasoning,
            focus='decision',
            max_sentences=1,
            limit=160,
        )
        if decision:
            return decision
        return _summarize_tool_call(
            str(tool_calls[0].get('name') or '').strip() or None,
            _compress_tool_input(tool_calls[0].get('arguments')),
        )
    if assistant_text:
        return assistant_text
    return assistant_reasoning


def _summarize_assistant_state(
    assistant_text: str,
    assistant_reasoning: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    if assistant_reasoning:
        return assistant_reasoning
    if tool_calls:
        intent = _infer_tool_intent(_compress_tool_input(tool_calls[0].get('arguments')))
        if intent:
            return _shorten(intent, 160)
        return _normalize_text(
            f"Preparing tool call: {_summarize_tool_call(str(tool_calls[0].get('name') or '').strip() or None, _compress_tool_input(tool_calls[0].get('arguments')))}",
            180,
        )
    return _extract_text_summary(assistant_text, focus='state', max_sentences=1, limit=180) or assistant_text


def _summarize_tool_call(tool_name: str | None, tool_input: Any) -> str:
    name = tool_name or 'tool'
    intent = _infer_tool_intent(tool_input)
    if intent:
        return _normalize_text(f"Call {name} to {intent}", 200)
    compact_input = _summarize_value(tool_input, 160)
    if compact_input:
        return _normalize_text(f"Call {name} with {compact_input}", 220)
    return f"Call {name}"


def _summarize_tool_result(tool_name: str | None, tool_output: Any) -> str:
    name = tool_name or 'tool'
    outcome_kind = _classify_tool_outcome(tool_output)
    if outcome_kind == 'failure':
        reason = _extract_error_summary(tool_output) or 'operation failed'
        return _normalize_text(f"{name} failed: {reason}", 220)
    if outcome_kind == 'write_success':
        return _normalize_text(f"{name} completed write successfully", 180)
    if outcome_kind == 'collection':
        count = _extract_collection_count(tool_output)
        noun = 'items' if count != 1 else 'item'
        if count is not None:
            return f"{name} returned {count} {noun}"
    if outcome_kind == 'record':
        summary = _extract_record_summary(tool_output)
        if summary:
            return _normalize_text(f"{name} returned {summary}", 220)
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
    value = _clean_text(text)
    if not value:
        return ''
    return _shorten(value, limit)


def _clean_text(text: Any) -> str:
    value = _prepare_text(text, preserve_newlines=False)
    if _looks_like_corrupted_text(value):
        return '[encoding issue omitted]'
    return value


def _compress_text_value(text: str, *, field_name: str | None) -> str:
    if field_name in _ERROR_KEYS:
        normalized = _normalize_error_text(text)
        return _normalize_text(normalized, 160)
    if field_name in _LONG_TEXT_KEYS:
        summary = _extract_text_summary(text, focus='result', max_sentences=2, limit=180)
        return summary or _normalize_text(text, 180)
    return _normalize_text(text, 220)


def _compress_list_item(value: Any, *, depth: int) -> Any:
    if isinstance(value, dict):
        prioritized = sorted(
            [(str(key), sub_value) for key, sub_value in value.items() if _has_signal(sub_value)],
            key=lambda item: (_record_priority(item[0], item[1]), item[0]),
        )
        result: dict[str, Any] = {}
        for key, sub_value in prioritized[:3]:
            result[key] = _compress_payload_for_field(sub_value, field_name=key, depth=depth + 1, max_items=2)
        omitted = len(prioritized) - len(result)
        if omitted > 0:
            result['_truncated'] = omitted
        return result
    return _compress_payload_for_field(value, field_name=None, depth=depth, max_items=2)


def _payload_priority(key: str, value: Any, depth: int) -> int:
    lowered = key.lower()
    if lowered in _PREFERRED_RESULT_KEYS:
        return 0
    if lowered in _PRIMARY_INPUT_KEYS:
        return 1
    if lowered in _COLLECTION_KEYS:
        return 2
    if lowered == 'result':
        return 2
    if lowered in _ERROR_KEYS:
        return 2
    if lowered in _LONG_TEXT_KEYS:
        return 4
    if lowered in {'debug', 'metadata', 'headers', 'raw', 'unused'}:
        return 6
    if isinstance(value, (dict, list)) and depth >= 1:
        return 5
    return 3


def _record_priority(key: str, value: Any) -> int:
    lowered = key.lower()
    if lowered in {'title', 'name', 'file_name', 'path', 'id', 'status', 'score', 'citation_index', 'type'}:
        return 0
    if lowered in {'url', 'source', 'docid'}:
        return 1
    if lowered in _LONG_TEXT_KEYS:
        return 2
    if isinstance(value, (int, float, bool)):
        return 2
    return 3


def _extract_text_summary(
    text: str,
    *,
    focus: str,
    max_sentences: int,
    limit: int,
) -> str:
    source = _prepare_text_for_summary(text)
    if not source:
        return ''
    sentences = _split_sentences(source)
    if not sentences:
        return _shorten(_clean_text(source), limit)
    scored = [
        (_score_sentence(sentence, focus), index, sentence)
        for index, sentence in enumerate(sentences)
        if sentence
    ]
    if not scored:
        return _shorten(_clean_text(source), limit)
    chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    chosen.sort(key=lambda item: item[1])
    summary = ' '.join(item[2] for item in chosen)
    return _normalize_text(summary, limit)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[。！？.!?])\s+|(?<=[:：；;])\s+|\n+', text)
    sentences: list[str] = []
    for part in parts:
        cleaned = part.strip(' -')
        if not cleaned:
            continue
        if len(cleaned) <= 2 and cleaned not in {'OK', 'ok'}:
            continue
        sentences.append(cleaned)
    return sentences


def _score_sentence(sentence: str, focus: str) -> int:
    score = 0
    if 6 <= len(sentence) <= 120:
        score += 3
    if re.search(r'\d', sentence):
        score += 1
    if focus == 'reasoning':
        if any(hint in sentence for hint in _REASONING_HINTS):
            score += 4
    elif focus == 'decision':
        if any(hint in sentence for hint in _DECISION_HINTS):
            score += 4
    elif focus == 'state':
        if any(hint in sentence for hint in _RESULT_HINTS + _ERROR_HINTS):
            score += 3
    else:
        if any(hint in sentence for hint in _RESULT_HINTS):
            score += 4
    if any(hint in sentence.lower() for hint in _ERROR_HINTS):
        score += 3
    if sentence.startswith(('好的', '当然', '谢谢', '感谢')) and len(sentence) < 20:
        score -= 2
    lowered = sentence.lower()
    if any(hint in lowered or hint in sentence for hint in _TAIL_PHRASE_HINTS):
        score -= 2
    return score


def _looks_like_content_dump(text: str) -> bool:
    if len(text) < 260:
        return False
    if '《' in text and '》' in text:
        return True
    if any(marker in text for marker in ('##', '###', '|', '---', '\n- ', '\n1. ')):
        return True
    if text.count('。') >= 6:
        return True
    return False


def _summarize_content_dump(text: str) -> str:
    title_match = re.search(r'《[^》]{1,30}》', text)
    if title_match:
        return f'展示{title_match.group(0)}全文内容'
    summary = _extract_text_summary(text, focus='result', max_sentences=1, limit=180)
    if summary:
        return summary
    return '提供长篇内容摘要'


def _extract_structured_summary(text: str) -> str:
    raw = str(text or '')
    if not raw:
        return ''
    if not any(marker in raw for marker in ('##', '###', '\n- ', '\n1. ', '结论', '总结', 'Conclusion', 'Summary', '|')):
        return ''
    match = re.search(r'(?:结论|总结|Conclusion|Summary)\s*[:：]?\s*(.+)', raw, flags=re.I | re.S)
    if match:
        conclusion = _extract_text_summary(match.group(1), focus='result', max_sentences=1, limit=180)
        if conclusion:
            return conclusion
    summary = _extract_text_summary(raw, focus='result', max_sentences=2, limit=220)
    return summary


def _looks_like_corrupted_text(text: str) -> bool:
    value = str(text or '')
    if not value:
        return False

    compact = ''.join(ch for ch in value if not ch.isspace())
    if len(compact) < 12:
        return False

    score = 0
    replacement_count = value.count('\ufffd')
    if replacement_count:
        score += min(replacement_count * 3, 9)

    control_count = sum(
        1
        for ch in value
        if unicodedata.category(ch).startswith('C') and ch not in '\n\r\t'
    )
    if control_count:
        score += min(control_count * 2, 6)

    escape_fragment_count = len(re.findall(r'(?:\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})', value))
    if escape_fragment_count >= 2:
        score += min(escape_fragment_count * 2, 8)

    extended_letter_count = sum(1 for ch in compact if _is_extended_non_cjk_letter(ch))
    if extended_letter_count >= 6:
        extended_ratio = extended_letter_count / max(len(compact), 1)
        if extended_ratio >= 0.18:
            score += 3

    suspicious_run_count = len(re.findall(r'[^\x00-\x7F]{3,}', compact))
    if suspicious_run_count and extended_letter_count >= 4 and _readable_token_count(value) <= 1:
        score += 2

    suspicious_symbol_count = sum(
        1
        for ch in compact
        if ord(ch) > 127
        and not _is_cjk_char(ch)
        and not unicodedata.category(ch).startswith(('L', 'N'))
    )
    if suspicious_run_count and suspicious_symbol_count >= 2:
        score += 3

    punctuation_noise = sum(1 for ch in compact if ch in {'�', 'Ã', 'Â', 'ð', 'Ð', '¤', '¦', '¬'})
    if punctuation_noise >= 3:
        score += 2

    return score >= 5


def _is_extended_non_cjk_letter(ch: str) -> bool:
    if len(ch) != 1:
        return False
    if _is_cjk_char(ch):
        return False
    category = unicodedata.category(ch)
    if not category.startswith('L'):
        return False
    return ord(ch) > 127


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _readable_token_count(text: str) -> int:
    return len(re.findall(r'[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}', text or ''))


def _prepare_text_for_summary(text: str) -> str:
    value = _prepare_text(text, preserve_newlines=True)
    if _looks_like_corrupted_text(value):
        return '[encoding issue omitted]'
    return value


def _prepare_text(text: Any, *, preserve_newlines: bool) -> str:
    value = str(text or '').strip()
    if not value:
        return ''
    value = re.sub(r'```(?:json|text|markdown)?\s*', '', value)
    value = value.replace('```', ' ')
    value = re.sub(r'https?://\S+', '[url]', value)
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'^\s{0,3}#{1,6}\s*', '', value, flags=re.MULTILINE)
    value = re.sub(r'^\s*[-*+]\s*', '', value, flags=re.MULTILINE)
    value = re.sub(r'^\s*\d+\.\s*', '', value, flags=re.MULTILINE)
    value = re.sub(r'\|\s*-+\s*\|', '\n' if preserve_newlines else ' ', value)
    value = value.replace('|', ' ')
    if preserve_newlines:
        value = re.sub(r'[ \t]+\n', '\n', value)
        value = re.sub(r'\n[ \t]+', '\n', value)
        value = re.sub(r'\n{2,}', '\n', value).strip()
    else:
        value = re.sub(r'\s+', ' ', value).strip()
    return value


def _infer_tool_intent(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return ''
    if _has_non_empty_key(tool_input, 'query', 'keyword', 'keywords'):
        query = tool_input.get('query') or tool_input.get('keyword') or tool_input.get('keywords')
        query_text = _summarize_value(query, 100)
        return f"search for {query_text}" if query_text else 'search for information'
    if _has_non_empty_key(tool_input, 'url', 'link'):
        url = _summarize_value(tool_input.get('url') or tool_input.get('link'), 100)
        return f"fetch content from {url}" if url else 'fetch page content'
    if _has_non_empty_key(tool_input, 'target') and _has_non_empty_key(tool_input, 'suggestions', 'content'):
        target = _summarize_value(tool_input.get('target'), 60)
        return f"save data to {target}" if target else 'save structured data'
    if _has_non_empty_key(tool_input, 'action', 'name'):
        action = _summarize_value(tool_input.get('action'), 40)
        name = _summarize_value(tool_input.get('name'), 80)
        if action and name:
            return f"{action} {name}"
    if _has_non_empty_key(tool_input, 'path', 'file', 'filename', 'rel_path'):
        path = _summarize_value(
            tool_input.get('path')
            or tool_input.get('file')
            or tool_input.get('filename')
            or tool_input.get('rel_path'),
            80,
        )
        return f"read or run {path}" if path else 'read or run a file'
    compact = _summarize_value(tool_input, 120)
    return compact


def _classify_tool_outcome(tool_output: Any) -> str:
    if isinstance(tool_output, str):
        return 'failure' if any(hint in tool_output.lower() for hint in _ERROR_HINTS) else 'text'
    if not isinstance(tool_output, dict):
        return 'generic'
    if _tool_output_indicates_failure(tool_output):
        return 'failure'
    if _extract_collection_count(tool_output) is not None:
        return 'collection'
    if _looks_like_write_success(tool_output):
        return 'write_success'
    if any(key in tool_output for key in ('name', 'title', 'path', 'id')):
        return 'record'
    return 'generic'


def _tool_output_indicates_failure(tool_output: dict[str, Any]) -> bool:
    success = tool_output.get('success')
    status = str(tool_output.get('status') or '').lower()
    if success is False:
        return True
    if any(key in tool_output for key in _ERROR_KEYS):
        if tool_output.get('error') or tool_output.get('reason'):
            return True
    return status in {'error', 'failed', 'timeout', 'request_timeout', 'network_unreachable', 'unavailable'}


def _looks_like_write_success(tool_output: dict[str, Any]) -> bool:
    if tool_output.get('success') is not True:
        return False
    return any(
        key in tool_output
        for key in ('persisted', 'appended_suggestions', 'updated', 'created', 'deleted', 'saved')
    )


def _extract_collection_count(tool_output: Any) -> int | None:
    if not isinstance(tool_output, dict):
        return None
    for key in ('count', 'total'):
        value = tool_output.get(key)
        if isinstance(value, int):
            return value
    for key in _COLLECTION_KEYS:
        value = tool_output.get(key)
        if isinstance(value, dict) and isinstance(value.get('count'), int):
            return value['count']
        if isinstance(value, list):
            return len(value)
    return None


def _extract_record_summary(tool_output: Any) -> str:
    if not isinstance(tool_output, dict):
        return ''
    parts: list[str] = []
    for key in ('title', 'name', 'path', 'id', 'status'):
        value = tool_output.get(key)
        if value:
            parts.append(f"{key}={_normalize_text(str(value), 60)}")
    return ', '.join(parts)


def _extract_error_summary(tool_output: Any) -> str:
    if isinstance(tool_output, str):
        return _normalize_error_text(tool_output)
    if not isinstance(tool_output, dict):
        return ''
    for key in ('error', 'reason', 'message', 'status', 'error_type'):
        value = tool_output.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_error_text(value)
    return ''


def _normalize_error_text(text: str) -> str:
    lowered = _clean_text(text).lower()
    if 'embedding key' in lowered or ('available keys' in lowered and 'group' in lowered):
        return 'embedding configuration error'
    if 'missing' in lowered and 'key' in lowered:
        return 'missing required key'
    if 'network is unreachable' in lowered or 'unreachable' in lowered:
        return 'network unreachable'
    if 'timed out' in lowered or 'timeout' in lowered:
        return 'request timeout'
    if 'not available' in lowered:
        return 'tool not available'
    if 'not found' in lowered:
        return 'resource not found'
    if 'denied' in lowered:
        return 'permission denied'
    return _shorten(_clean_text(text), 120)


def _has_non_empty_key(data: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = data.get(key)
        if _has_signal(value):
            return True
    return False


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
