from __future__ import annotations

import copy
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


_RUNTIMES: dict[str, AppWorldRuntime] = {}
_APP_DOCS_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def _current_session_id() -> str:
    try:
        import lazyllm

        return str(getattr(lazyllm.globals, '_sid', '') or '').strip()
    except Exception:  # noqa: BLE001
        return ''


def _init_lazyllm_session(session_id: str) -> None:
    try:
        import lazyllm

        lazyllm.globals._init_sid(sid=session_id)
        lazyllm.locals._init_sid(sid=session_id)
    except Exception:  # noqa: BLE001
        return


def _ensure_repo_root(repo_root: str) -> None:
    repo_path = Path(str(repo_root or '').strip()).expanduser()
    if not repo_path.exists():
        return
    source_path = repo_path / 'src'
    for path in (source_path, repo_path):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _normalize_url(url: str) -> str:
    cleaned = str(url or '').strip().rstrip('/')
    if not cleaned:
        return ''
    parts = urlsplit(cleaned)
    hostname = (parts.hostname or '').lower()
    if hostname not in {'localhost', '127.0.0.1'}:
        return cleaned
    netloc = parts.netloc
    if hostname == 'localhost':
        netloc = netloc.replace('localhost', '127.0.0.1', 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip('/')


def _json_post(url: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    return value if isinstance(value, dict) else {'output': value}


def _json_get(url: str, timeout: float | None = None) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    return value if isinstance(value, dict) else {'output': value}


def _shorten(value: Any, limit: int = 1000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + '...'


def _execution_succeeded(output: str) -> bool:
    lowered = str(output or '').lower()
    return 'traceback (most recent call last)' not in lowered and 'error' not in lowered[:80]


def _parse_import_targets(raw: str) -> list[str]:
    names: list[str] = []
    for piece in raw.split(','):
        item = piece.strip()
        if not item:
            continue
        item = re.split(r'\s+as\s+', item, maxsplit=1)[0].strip()
        if item:
            names.append(item)
    return names


def _rewrite_appworld_code(code: str) -> str:
    """Map simple `from x import y` app imports onto the prepared `apis` object."""
    rewritten: list[str] = []
    pattern = re.compile(r'^\s*from\s+([a-zA-Z_]\w*)\s+import\s+(.+?)\s*$')
    for line in str(code or '').splitlines():
        match = pattern.match(line)
        if not match:
            rewritten.append(line)
            continue
        app_name, raw_targets = match.groups()
        if app_name in {'appworld', 'typing', 'datetime', 'collections', 'functools', 'itertools'}:
            rewritten.append(line)
            continue
        targets = _parse_import_targets(raw_targets)
        if not targets:
            rewritten.append(line)
            continue
        rewritten.extend(f'{target} = apis.{app_name}.{target}' for target in targets)
    return '\n'.join(rewritten)


def _read_task_specs(data_root: str, task_id: str) -> dict[str, Any]:
    path = Path(data_root).expanduser() / 'data' / 'tasks' / task_id / 'specs.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    return value if isinstance(value, dict) else {}


def _load_app_docs(data_root: str, app_name: str) -> dict[str, dict[str, Any]]:
    key = f'{Path(data_root).expanduser()}::{app_name}'
    cached = _APP_DOCS_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    docs_path = Path(data_root).expanduser() / 'data' / 'api_docs' / 'standard' / f'{app_name}.json'
    if not docs_path.exists():
        return {}
    value = json.loads(docs_path.read_text(encoding='utf-8'))
    docs = value if isinstance(value, dict) else {}
    _APP_DOCS_CACHE[key] = docs
    return copy.deepcopy(docs)


def _normalize_search_text(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(value or '').lower()).strip()


def _available_app_names(data_root: str) -> list[str]:
    docs_root = Path(data_root).expanduser() / 'data' / 'api_docs' / 'standard'
    return sorted(path.stem for path in docs_root.glob('*.json')) if docs_root.exists() else []


def _resolve_app_name(data_root: str, app_name: str) -> str:
    requested = _normalize_search_text(app_name)
    if not requested:
        return ''
    for candidate in _available_app_names(data_root):
        if _normalize_search_text(candidate) == requested:
            return candidate
    return ''


def _resolve_api_name(docs: dict[str, dict[str, Any]], api_name: str) -> str:
    requested = _normalize_search_text(api_name)
    if not requested or requested in {'all', 'any'}:
        return ''
    for candidate in docs:
        if _normalize_search_text(candidate) == requested:
            return candidate
    return ''


def _query_terms(query: str) -> set[str]:
    return {
        term
        for term in _normalize_search_text(query).split()
        if len(term) > 1 and term not in {'api', 'apis', 'app', 'apps', 'the', 'and', 'for'}
    }


def _compact_api_doc(doc: dict[str, Any]) -> dict[str, Any]:
    parameters = []
    for param in doc.get('parameters') or []:
        if not isinstance(param, dict):
            continue
        parameters.append({
            'name': param.get('name'),
            'type': param.get('type'),
            'required': param.get('required'),
        })
    return {
        'app_name': doc.get('app_name'),
        'api_name': doc.get('api_name'),
        'method': doc.get('method'),
        'path': doc.get('path'),
        'description': doc.get('description'),
        'parameters': parameters,
    }


def _summarize_schema(value: Any, *, depth: int = 0, max_depth: int = 2, max_items: int = 8) -> Any:
    if depth >= max_depth:
        if isinstance(value, dict):
            return '{...}'
        if isinstance(value, list):
            return ['...']
        return value
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        summary = {key: _summarize_schema(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
                   for key, item in items}
        if len(value) > max_items:
            summary['...'] = f'+{len(value) - max_items} more keys'
        return summary
    if isinstance(value, list):
        if not value:
            return []
        summary = [
            _summarize_schema(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for item in value[:1]
        ]
        if len(value) > 1:
            summary.append(f'+{len(value) - 1} more items')
        return summary
    return value


def _summarize_response_schemas(doc: dict[str, Any]) -> dict[str, Any]:
    schemas = doc.get('response_schemas') or {}
    if not isinstance(schemas, dict):
        return {}
    return {
        key: _summarize_schema(value)
        for key, value in list(schemas.items())[:4]
    }


def _detailed_api_doc(doc: dict[str, Any], *, include_response_schemas: bool = True) -> dict[str, Any]:
    parameters = []
    for param in doc.get('parameters') or []:
        if not isinstance(param, dict):
            continue
        parameters.append({
            'name': param.get('name'),
            'type': param.get('type'),
            'required': param.get('required'),
            'description': param.get('description'),
        })
    return {
        'app_name': doc.get('app_name'),
        'api_name': doc.get('api_name'),
        'method': doc.get('method'),
        'path': doc.get('path'),
        'description': doc.get('description'),
        'parameters': parameters,
        **(
            {'response_schemas': _summarize_response_schemas(doc)}
            if include_response_schemas
            else {}
        ),
    }


def _rank_api_docs(docs: dict[str, dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return [_compact_api_doc(doc) for doc in list(docs.values())[:limit]]

    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in docs.values():
        haystack = _normalize_search_text({
            'api_name': doc.get('api_name'),
            'path': doc.get('path'),
            'description': doc.get('description'),
            'parameters': [
                {
                    'name': param.get('name'),
                    'description': param.get('description'),
                }
                for param in doc.get('parameters') or []
                if isinstance(param, dict)
            ],
        })
        score = sum(1 for term in terms if term in haystack)
        api_name = _normalize_search_text(doc.get('api_name'))
        score += sum(2 for term in terms if term in api_name)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_compact_api_doc(doc) for _, doc in scored[:limit]]


@dataclass
class AppWorldRuntime:
    session_id: str
    task_id: str
    data_root: str
    repo_root: str
    environment_url: str
    apis_url: str
    experiment_name: str
    timeout_seconds: float | None = None
    max_interactions: int = 200
    max_api_calls_per_interaction: int = 1000
    show_api_response_schemas: bool = False
    initialized: bool = False
    task_info: dict[str, Any] = field(default_factory=dict)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    environment_trace: list[dict[str, Any]] = field(default_factory=list)
    last_output: str = ''
    interaction_count: int = 0
    completed: bool = False
    evaluation: dict[str, Any] = field(default_factory=dict)

    def initialize(self) -> dict[str, Any]:
        _ensure_repo_root(self.repo_root)
        payload = {
            'task_id': self.task_id,
            'experiment_name': self.experiment_name,
            'remote_apis_url': self.apis_url,
            'remote_environment_url': None,
            'max_interactions': self.max_interactions,
            'max_api_calls_per_interaction': self.max_api_calls_per_interaction,
            'timeout_seconds': self.timeout_seconds,
            'show_api_response_schemas': self.show_api_response_schemas,
        }
        response = _json_post(f'{self.environment_url}/initialize', payload, self.timeout_seconds)
        output = response.get('output')
        self.task_info = output if isinstance(output, dict) else {}
        if not self.task_info:
            self.task_info = _read_task_specs(self.data_root, self.task_id)
        self.initialized = True
        self.environment_trace.append({'event': 'initialize', 'task_id': self.task_id})
        return self.task_info

    def execute(self, code: str) -> str:
        if self.interaction_count >= self.max_interactions:
            raise RuntimeError('max_interactions exceeded')
        rewritten = _rewrite_appworld_code(code)
        response = _json_post(
            f'{self.environment_url}/execute',
            {'task_id': self.task_id, 'code': rewritten},
            self.timeout_seconds,
        )
        output = str(response.get('output') or '')
        self.interaction_count += 1
        self.last_output = output
        self.environment_trace.append({
            'number': self.interaction_count,
            'input': rewritten,
            'output': output,
        })
        return output

    def task_completed(self) -> bool:
        try:
            response = _json_post(
                f'{self.environment_url}/task_completed',
                {'task_id': self.task_id},
                self.timeout_seconds,
            )
            self.completed = bool(response.get('output'))
        except Exception:  # noqa: BLE001
            self.completed = False
        return self.completed

    def evaluate(self) -> dict[str, Any]:
        response = _json_post(
            f'{self.environment_url}/evaluate',
            {'task_id': self.task_id, 'suppress_errors': True, 'report': False},
            self.timeout_seconds,
        )
        output = response.get('output')
        self.evaluation = output if isinstance(output, dict) else {}
        return self.evaluation

    def close(self) -> None:
        if not self.initialized:
            return
        try:
            _json_post(f'{self.environment_url}/close', {'task_id': self.task_id}, self.timeout_seconds)
        finally:
            self.initialized = False

    def status(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'task_id': self.task_id,
            'task_info': copy.deepcopy(self.task_info),
            'interaction_count': self.interaction_count,
            'completed': self.completed,
            'last_output': self.last_output,
            'evaluation': copy.deepcopy(self.evaluation),
            'trace': self.runtime_trace(),
        }

    def runtime_trace(self) -> dict[str, Any]:
        return {
            'tool_trace': copy.deepcopy(self.tool_trace),
            'environment_trace': copy.deepcopy(self.environment_trace),
            'environment_final': {
                'status': {
                    'interaction_count': self.interaction_count,
                    'completed': self.completed,
                    'last_output': self.last_output,
                },
                'evaluation': copy.deepcopy(self.evaluation),
            },
        }

    def record_tool_call(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        self.tool_trace.append({
            'tool_name': tool_name,
            'arguments': copy.deepcopy(args),
            'result': copy.deepcopy(result),
        })


def _runtime_for_session(session_id: str | None = None) -> AppWorldRuntime:
    resolved = str(session_id or _current_session_id()).strip()
    if not resolved:
        raise RuntimeError('No active AppWorld session id')
    runtime = _RUNTIMES.get(resolved)
    if runtime is None:
        raise RuntimeError(f'AppWorld runtime is not prepared for session {resolved}')
    return runtime


def prepare_appworld_task(
    *,
    session_id: str,
    task_id: str,
    data_root: str,
    environment_url: str,
    apis_url: str,
    experiment_name: str = 'lazyrag',
    timeout_seconds: float | str | None = None,
    max_interactions: int | str | None = None,
    max_api_calls_per_interaction: int | str | None = None,
    repo_root: str = '',
    show_api_response_schemas: bool | None = None,
) -> dict[str, Any]:
    resolved_session_id = str(session_id or '').strip()
    resolved_task_id = str(task_id or '').strip()
    if not resolved_session_id:
        raise ValueError('session_id is required')
    if not resolved_task_id:
        raise ValueError('task_id is required')
    clear_appworld_runtime(resolved_session_id)
    _init_lazyllm_session(resolved_session_id)
    runtime = AppWorldRuntime(
        session_id=resolved_session_id,
        task_id=resolved_task_id,
        data_root=str(data_root or '').strip(),
        repo_root=str(repo_root or '').strip(),
        environment_url=_normalize_url(environment_url),
        apis_url=_normalize_url(apis_url),
        experiment_name=str(experiment_name or 'lazyrag').strip(),
        timeout_seconds=float(timeout_seconds) if str(timeout_seconds or '').strip() else None,
        max_interactions=int(max_interactions) if str(max_interactions or '').strip() else 200,
        max_api_calls_per_interaction=(
            int(max_api_calls_per_interaction)
            if str(max_api_calls_per_interaction or '').strip()
            else 1000
        ),
        show_api_response_schemas=False if show_api_response_schemas is None else bool(show_api_response_schemas),
    )
    _RUNTIMES[resolved_session_id] = runtime
    task_info = runtime.initialize()
    prepared_query = build_appworld_prepared_query(task_info)
    return {
        'session_id': resolved_session_id,
        'task_id': resolved_task_id,
        'task_info': copy.deepcopy(task_info),
        'prepared_query': prepared_query,
        'prepared_prompt': build_appworld_prepared_prompt(task_info),
        'environment_url': runtime.environment_url,
        'apis_url': runtime.apis_url,
        'experiment_name': runtime.experiment_name,
    }


def build_appworld_prepared_query(task_info: dict[str, Any]) -> str:
    return str(task_info.get('instruction') or '').strip()


def build_appworld_prepared_prompt(task_info: dict[str, Any]) -> str:
    instruction = build_appworld_prepared_query(task_info)
    return textwrap.dedent(
        f"""
        AppWorld task is prepared.

        Instruction:
        {instruction}
        """
    ).strip()


def get_appworld_session_status(session_id: str) -> dict[str, Any]:
    runtime = _runtime_for_session(session_id)
    runtime.task_completed()
    return runtime.status()


def evaluate_appworld_session(session_id: str) -> dict[str, Any]:
    runtime = _runtime_for_session(session_id)
    evaluation = runtime.evaluate()
    return {
        'session_id': runtime.session_id,
        'task_id': runtime.task_id,
        'evaluation': evaluation,
    }


def get_appworld_runtime_trace(session_id: str) -> dict[str, Any]:
    runtime = _runtime_for_session(session_id)
    return runtime.runtime_trace()


def clear_appworld_runtime(session_id: str | None = None) -> None:
    resolved = str(session_id or _current_session_id()).strip()
    if not resolved:
        return
    runtime = _RUNTIMES.pop(resolved, None)
    if runtime is not None:
        runtime.close()


def appworld_task_info() -> dict[str, Any]:
    runtime = _runtime_for_session()
    if not runtime.initialized:
        runtime.initialize()
    result = copy.deepcopy(runtime.task_info)
    runtime.record_tool_call('appworld_task_info', {}, result)
    return result


def appworld_api_docs(
    app_name: str = '',
    api_name: str = '',
    query: str = '',
    page_limit: int = 5,
) -> dict[str, Any]:
    runtime = _runtime_for_session()
    raw_app = str(app_name or '').strip()
    raw_api = str(api_name or '').strip()
    raw_query = str(query or '').strip()
    limit = max(1, min(int(page_limit or 5), 20))
    app = _resolve_app_name(runtime.data_root, raw_app)

    if app:
        docs = _load_app_docs(runtime.data_root, app)
        api = _resolve_api_name(docs, raw_api)
        if api:
            result: Any = {
                'app_name': app,
                'api': _detailed_api_doc(
                    docs[api],
                    include_response_schemas=runtime.show_api_response_schemas,
                ),
            }
        elif raw_api and _normalize_search_text(raw_api) not in {'', 'all', 'any'}:
            matches = _rank_api_docs(docs, raw_api, limit)
            result = {
                'app_name': app,
                'message': f'API {raw_api!r} was not found. Use one of the listed api_name values.',
                'matches': matches,
                'available_api_names': sorted(docs.keys())[:50],
            }
        elif raw_query:
            result = {
                'app_name': app,
                'matches': _rank_api_docs(docs, raw_query, limit),
                'message': 'Use api_name to fetch one exact API after narrowing candidates.',
            }
        else:
            result = {
                'app_name': app,
                'total_api_count': len(docs),
                'available_api_names': sorted(docs.keys())[:50],
                'apis': [_compact_api_doc(doc) for doc in list(docs.values())[:limit]],
                'message': 'Pass api_name for one full API doc, or query for ranked matches.',
            }
    else:
        apps = _available_app_names(runtime.data_root)
        if raw_app:
            result = {
                'message': f'App {raw_app!r} was not found. Use one of the listed app names.',
                'available_apps': apps,
            }
        elif raw_query:
            all_docs: dict[str, dict[str, Any]] = {}
            for candidate_app in apps:
                docs = _load_app_docs(runtime.data_root, candidate_app)
                for candidate_api, doc in docs.items():
                    all_docs[f'{candidate_app}.{candidate_api}'] = doc
            result = {
                'query': raw_query,
                'matches': _rank_api_docs(all_docs, raw_query, limit),
                'available_apps': apps[:50],
            }
        else:
            result = {'available_apps': apps[:50]}

    output = {'output': result}
    runtime.record_tool_call(
        'appworld_api_docs',
        {'app_name': app_name, 'api_name': api_name, 'query': query, 'page_limit': page_limit},
        output,
    )
    return output


def appworld_execute(code: str) -> dict[str, Any]:
    cleaned = str(code or '').strip()
    if not cleaned:
        raise ValueError('code is required')
    runtime = _runtime_for_session()
    if not runtime.initialized:
        runtime.initialize()
    output = runtime.execute(cleaned)
    completed = runtime.task_completed() if _execution_succeeded(output) else runtime.completed
    result = {
        'output': output,
        'completed': completed,
        'interaction_count': runtime.interaction_count,
    }
    runtime.record_tool_call('appworld_execute', {'code': _shorten(cleaned)}, result)
    return result


def appworld_status() -> dict[str, Any]:
    runtime = _runtime_for_session()
    result = runtime.status()
    runtime.record_tool_call('appworld_status', {}, result)
    return result
