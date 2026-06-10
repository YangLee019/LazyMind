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
        'avg_steps': total_steps / total_tasks if total_tasks else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
        'max_step_failure_rate': max_step_failures / total_tasks if total_tasks else 0.0,
        'error_count': error_count,
        'total_tests': total_tests,
        'total_passes': total_passes,
        'test_pass_rate': total_passes / total_tests if total_tests else None,
    }
