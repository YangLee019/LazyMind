from __future__ import annotations

from dataclasses import dataclass
from typing import Any


APPWORLD_TOOL_NAMES = [
    'appworld_execute',
    'appworld_task_info',
    'appworld_api_docs',
    'appworld_status',
]


@dataclass
class AppWorldTool:
    """Runtime configuration for one AppWorld benchmark session at a time."""

    repo_root: str = ''
    data_root: str = ''
    environment_url: str = ''
    apis_url: str = ''
    experiment_name: str = 'lazyrag'
    timeout_seconds: float | None = None
    max_interactions: int | None = None
    max_api_calls_per_interaction: int | None = None
    show_api_response_schemas: bool | None = None

    __public_apis__ = APPWORLD_TOOL_NAMES

    @staticmethod
    def appworld_task_info() -> dict[str, Any]:
        """Return the prepared AppWorld task instruction, supervisor, and task time."""
        from .appworld_runtime import appworld_task_info

        return appworld_task_info()

    @staticmethod
    def appworld_api_docs(
        app_name: str = '',
        api_name: str = '',
        query: str = '',
        page_limit: int = 5,
    ) -> dict[str, Any]:
        """Look up AppWorld API documentation by app name, API name, or query text."""
        from .appworld_runtime import appworld_api_docs

        return appworld_api_docs(
            app_name=app_name,
            api_name=api_name,
            query=query,
            page_limit=page_limit,
        )

    @staticmethod
    def appworld_execute(code: str) -> dict[str, Any]:
        """Execute Python code in the prepared AppWorld task environment."""
        from .appworld_runtime import appworld_execute

        return appworld_execute(code)

    @staticmethod
    def appworld_status() -> dict[str, Any]:
        """Return the current AppWorld task status and execution trace."""
        from .appworld_runtime import appworld_status

        return appworld_status()

    def prepare(self, *, session_id: str, task_id: str) -> dict[str, Any]:
        from .appworld_runtime import prepare_appworld_task

        return prepare_appworld_task(
            session_id=session_id,
            task_id=task_id,
            data_root=self.data_root,
            environment_url=self.environment_url,
            apis_url=self.apis_url,
            experiment_name=self.experiment_name,
            timeout_seconds=self.timeout_seconds or '',
            max_interactions=self.max_interactions or '',
            max_api_calls_per_interaction=self.max_api_calls_per_interaction or '',
            repo_root=self.repo_root,
            show_api_response_schemas=self.show_api_response_schemas,
        )

    def environment_context(
        self,
        *,
        task_id: str,
        task_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        appworld_ctx: dict[str, Any] = {
            'enabled': True,
            'prepared': True,
            'task_id': task_id,
        }
        for key in (
            'repo_root',
            'data_root',
            'environment_url',
            'apis_url',
            'experiment_name',
            'timeout_seconds',
            'max_interactions',
            'max_api_calls_per_interaction',
            'show_api_response_schemas',
        ):
            value = getattr(self, key)
            if value is None or value == '':
                continue
            appworld_ctx[key] = value

        context: dict[str, Any] = {'appworld': appworld_ctx}
        task_time = str((task_info or {}).get('datetime') or '').strip()
        if task_time:
            context['time'] = {'now': task_time}
        return context

    def status(self, session_id: str) -> dict[str, Any]:
        from .appworld_runtime import get_appworld_session_status

        return get_appworld_session_status(session_id)

    def evaluate(self, session_id: str) -> dict[str, Any]:
        from .appworld_runtime import evaluate_appworld_session

        return evaluate_appworld_session(session_id)

    def runtime_trace(self, session_id: str) -> dict[str, Any]:
        from .appworld_runtime import get_appworld_runtime_trace

        return get_appworld_runtime_trace(session_id) or {}

    def cleanup(self, session_id: str) -> None:
        from .appworld_runtime import clear_appworld_runtime

        clear_appworld_runtime(session_id)
