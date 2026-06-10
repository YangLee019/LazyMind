from __future__ import annotations

from typing import Any


MAX_STEP_ERROR = 'max_steps_exceeded'


def _num_tests(evaluation: dict[str, Any]) -> int:
    try:
        return max(int(evaluation.get('num_tests') or 0), 0)
    except (TypeError, ValueError):
        return 0


def _pass_count(evaluation: dict[str, Any], num_tests: int) -> int:
    passes = evaluation.get('passes')
    count = len(passes) if isinstance(passes, list) else 0
    return min(max(count, 0), num_tests)


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = len(results)
    success_results = [result for result in results if bool(result.get('success'))]
    success_count = len(success_results)
    error_count = sum(1 for result in results if result.get('error'))
    max_step_failures = sum(1 for result in results if result.get('error') == MAX_STEP_ERROR)

    total_steps = sum(int(result.get('steps') or 0) for result in results)
    success_steps = sum(int(result.get('steps') or 0) for result in success_results)
    total_tool_call_rounds = sum(_tool_call_rounds(result) for result in results)
    completed_tool_call_rounds = sum(_tool_call_rounds(result) for result in results if bool(result.get('completed')))
    handle_chat_turn_values = [
        int(result.get('handle_chat_tool_call_turns') or 0)
        for result in results
        if result.get('handle_chat_tool_call_turns') is not None
    ]

    total_tests = 0
    total_passes = 0
    completed_count = 0
    for result in results:
        if bool(result.get('completed')):
            completed_count += 1
        evaluation = result.get('evaluation')
        if not isinstance(evaluation, dict):
            continue
        num_tests = _num_tests(evaluation)
        total_tests += num_tests
        total_passes += _pass_count(evaluation, num_tests)

    return {
        'total_tasks': total_tasks,
        'success_count': success_count,
        'success_rate': success_count / total_tasks if total_tasks else 0.0,
        'completed_count': completed_count,
        'completed_rate': completed_count / total_tasks if total_tasks else 0.0,
        'completion_rate': completed_count / total_tasks if total_tasks else 0.0,
        'avg_steps': total_steps / total_tasks if total_tasks else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
        'total_tool_call_rounds': total_tool_call_rounds,
        'avg_tool_call_rounds': total_tool_call_rounds / total_tasks if total_tasks else 0.0,
        'avg_completed_tool_call_rounds': (
            completed_tool_call_rounds / completed_count if completed_count else 0.0
        ),
        'max_tool_call_rounds': max((_tool_call_rounds(result) for result in results), default=0),
        'total_handle_chat_tool_call_turns': sum(handle_chat_turn_values),
        'avg_handle_chat_tool_call_turns': (
            sum(handle_chat_turn_values) / len(handle_chat_turn_values)
            if handle_chat_turn_values else 0.0
        ),
        'max_handle_chat_tool_call_turns': max(handle_chat_turn_values, default=0),
        'max_step_failure_rate': max_step_failures / total_tasks if total_tasks else 0.0,
        'error_count': error_count,
        'total_tests': total_tests,
        'total_passes': total_passes,
        'test_pass_rate': total_passes / total_tests if total_tests else None,
    }


def _tool_call_rounds(result: dict[str, Any]) -> int:
    return int(result.get('tool_call_rounds', result.get('steps') or 0) or 0)
