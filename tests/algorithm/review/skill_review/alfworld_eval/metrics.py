from __future__ import annotations

from typing import Any


MAX_STEP_ERROR = 'max_steps_exceeded'


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = len(results)
    success_results = [result for result in results if bool(result.get('success'))]
    success_count = len(success_results)
    error_count = sum(1 for result in results if result.get('error'))
    max_step_failures = sum(1 for result in results if result.get('error') == MAX_STEP_ERROR)

    total_steps = sum(int(result.get('steps') or 0) for result in results)
    success_steps = sum(int(result.get('steps') or 0) for result in success_results)

    return {
        'total_tasks': total_tasks,
        'success_count': success_count,
        'success_rate': success_count / total_tasks if total_tasks else 0.0,
        'avg_steps': total_steps / total_tasks if total_tasks else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
        'max_step_failure_rate': max_step_failures / total_tasks if total_tasks else 0.0,
        'error_count': error_count,
    }
