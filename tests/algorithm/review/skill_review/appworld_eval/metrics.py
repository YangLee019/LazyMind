from __future__ import annotations

from typing import Any

try:
    from ..skill_usage import aggregate_skill_usage_counts, has_non_empty_trajectory
except ImportError:  # pragma: no cover - direct script execution path
    from skill_usage import aggregate_skill_usage_counts, has_non_empty_trajectory


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


def compute_metrics(
    results: list[dict[str, Any]],
    skill_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    total_tasks = len(results)
    success_results = [result for result in results if bool(result.get('success'))]
    success_count = len(success_results)
    adjusted_success_results = [result for result in results if bool(result.get('adjusted_success'))]
    adjusted_success_count = len(adjusted_success_results)
    total_steps = sum(int(result.get('steps') or 0) for result in results)
    success_steps = sum(int(result.get('steps') or 0) for result in success_results)
    adjusted_success_steps = sum(int(result.get('steps') or 0) for result in adjusted_success_results)
    non_empty_results = [result for result in results if has_non_empty_trajectory(result)]
    skill_usage_denominator = len(non_empty_results)
    empty_trajectory_count = total_tasks - skill_usage_denominator
    skill_usage_count = sum(1 for result in non_empty_results if bool(result.get('used_skill')))
    skill_usage_by_name = aggregate_skill_usage_counts(results, skill_names)
    adjusted_by_reason: dict[str, int] = {}

    total_tests = 0
    total_passes = 0
    adjusted_total_passes = 0
    completed_count = 0
    for result in results:
        if bool(result.get('completed')):
            completed_count += 1
        evaluation = result.get('evaluation')
        if not isinstance(evaluation, dict):
            continue
        num_tests = _num_tests(evaluation)
        pass_count = _pass_count(evaluation, num_tests)
        total_tests += num_tests
        total_passes += pass_count
        if bool(result.get('adjusted_success')):
            adjusted_total_passes += num_tests
        else:
            adjusted_total_passes += pass_count
        reason = str(result.get('adjusted_success_reason') or '').strip()
        if reason:
            adjusted_by_reason[reason] = adjusted_by_reason.get(reason, 0) + 1

    return {
        'total_tasks': total_tasks,
        'success_count': success_count,
        'success_rate': success_count / total_tasks if total_tasks else 0.0,
        'adjusted_success_count': adjusted_success_count,
        'adjusted_success_rate': adjusted_success_count / total_tasks if total_tasks else 0.0,
        'completed_count': completed_count,
        'completed_rate': completed_count / total_tasks if total_tasks else 0.0,
        'avg_steps': total_steps / total_tasks if total_tasks else 0.0,
        'avg_success_steps': success_steps / success_count if success_count else 0.0,
        'avg_adjusted_success_steps': (
            adjusted_success_steps / adjusted_success_count if adjusted_success_count else 0.0
        ),
        'non_empty_trajectory_count': skill_usage_denominator,
        'empty_trajectory_count': empty_trajectory_count,
        'skill_usage_count': skill_usage_count,
        'skill_usage_denominator': skill_usage_denominator,
        'skill_usage_rate': skill_usage_count / skill_usage_denominator if skill_usage_denominator else 0.0,
        'skill_usage_by_name': skill_usage_by_name,
        'total_tests': total_tests,
        'total_passes': total_passes,
        'test_pass_rate': total_passes / total_tests if total_tests else None,
        'adjusted_test_pass_rate': adjusted_total_passes / total_tests if total_tests else None,
        'adjusted_success_breakdown': adjusted_by_reason,
    }
