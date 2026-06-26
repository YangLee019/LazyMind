from __future__ import annotations

import re
from typing import Any


MAX_STEP_ERROR = 'max_steps_exceeded'
TASK_TYPE_PATTERNS = {
    'look_at_obj_in_light': re.compile(r'/look_at_obj_in_light-'),
    'pick_and_place_simple': re.compile(r'/pick_and_place_simple-'),
    'pick_clean_then_place_in_recep': re.compile(r'/pick_clean_then_place_in_recep-'),
    'pick_cool_then_place_in_recep': re.compile(r'/pick_cool_then_place_in_recep-'),
    'pick_heat_then_place_in_recep': re.compile(r'/pick_heat_then_place_in_recep-'),
    'pick_two_obj_and_place': re.compile(r'/pick_two_obj_and_place-'),
}


def extract_gamefile(info: Any) -> str:
    if not isinstance(info, dict):
        return ''
    value = info.get('extra.gamefile') or info.get('gamefile') or ''
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ''
    return str(value or '')


def extract_won(info: Any) -> bool | None:
    if not isinstance(info, dict):
        return None
    value = info.get('won')
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    return bool(value)


def infer_task_type_from_gamefile(gamefile: str) -> str:
    normalized = str(gamefile or '')
    for task_type, pattern in TASK_TYPE_PATTERNS.items():
        if pattern.search(normalized):
            return task_type
    return 'unknown'


def _build_task_success_rate(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    task_counts = {
        task_type: {'total': 0, 'success': 0}
        for task_type in TASK_TYPE_PATTERNS
    }
    for result in results:
        task_type = str(result.get('task_type') or '')
        if task_type not in task_counts:
            task_type = infer_task_type_from_gamefile(str(result.get('gamefile') or ''))
        if task_type not in task_counts:
            continue
        task_counts[task_type]['total'] += 1
        if bool(result.get('success')):
            task_counts[task_type]['success'] += 1

    return {
        task_type: {
            'total': counts['total'],
            'success': counts['success'],
            'rate': counts['success'] / counts['total'] if counts['total'] else 0.0,
        }
        for task_type, counts in task_counts.items()
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = len(results)
    success_results = [result for result in results if bool(result.get('success'))]
    success_count = len(success_results)
    error_count = sum(1 for result in results if result.get('error'))
    max_step_failures = sum(1 for result in results if result.get('error') == MAX_STEP_ERROR)
    skill_usage_count = sum(1 for result in results if bool(result.get('used_skill')))

    total_steps = sum(int(result.get('steps') or 0) for result in results)
    success_steps = sum(int(result.get('steps') or 0) for result in success_results)

    return {
        'total_tasks': total_tasks,
        'success_count': success_count,
        'success_rate': success_count / total_tasks if total_tasks else 0.0,
        'avg_steps': total_steps / total_tasks if total_tasks else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
        'skill_usage_count': skill_usage_count,
        'skill_usage_rate': skill_usage_count / total_tasks if total_tasks else 0.0,
        'task_success_rate': _build_task_success_rate(results),
        'max_step_failure_rate': max_step_failures / total_tasks if total_tasks else 0.0,
        'error_count': error_count,
    }
