from __future__ import annotations

import json
import uuid
from typing import Any


def insert_chat_history_row(
    row: dict[str, Any],
    *,
    create_user_id: str,
    create_user_name: str = '',
) -> None:
    """Insert one AppWorld chat_histories row and its conversation."""
    from sqlalchemy import text

    from lazymind.review.skill_review.db import _get_app_conn

    raw_conversation_id = str(row.get('conversation_id') or '').strip()
    if not raw_conversation_id:
        raise ValueError('chat history row requires conversation_id')
    conversation_id = _varchar36_id(raw_conversation_id)
    user_id = str(create_user_id or '').strip()
    if not user_id:
        raise ValueError('create_user_id is required')

    row_id = _varchar36_id(str(row.get('id') or uuid.uuid4()))
    ext = dict(row.get('ext') or {})
    if conversation_id != raw_conversation_id:
        ext['original_conversation_id'] = raw_conversation_id
    if row_id != row.get('id'):
        ext['original_chat_history_id'] = row.get('id')

    created_at = row.get('create_time')
    updated_at = row.get('update_time') or created_at
    benchmark = str(ext.get('benchmark') or 'appworld')
    task_id = str(ext.get('task_id') or row.get('seq', 1))
    with _get_app_conn().begin() as conn:
        conn.execute(
            text(
                """INSERT INTO conversations
                       (id, display_name, channel_id, search_config, application_id,
                        ext, model, models, chat_times, create_user_id,
                        create_user_name, created_at, updated_at, deleted_at)
                   VALUES
                       (:id, :display_name, 'appworld_eval', CAST(:search_config AS JSON),
                        '', CAST(:ext AS JSON), '', CAST(:models AS JSON), 1,
                        :create_user_id, :create_user_name, :created_at, :updated_at, NULL)
                   ON CONFLICT (id) DO UPDATE SET
                       chat_times = GREATEST(conversations.chat_times, 1),
                       create_user_id = EXCLUDED.create_user_id,
                       create_user_name = EXCLUDED.create_user_name,
                       updated_at = EXCLUDED.updated_at"""
            ),
            {
                'id': conversation_id,
                'display_name': f'AppWorld task {task_id}',
                'search_config': json.dumps({}, ensure_ascii=False),
                'ext': json.dumps({'benchmark': benchmark, 'task_id': task_id}, ensure_ascii=False),
                'models': json.dumps([], ensure_ascii=False),
                'create_user_id': user_id,
                'create_user_name': str(create_user_name or user_id),
                'created_at': created_at,
                'updated_at': updated_at,
            },
        )
        conn.execute(
            text(
                """INSERT INTO chat_histories
                       (id, seq, conversation_id, raw_content, retrieval_result,
                        content, result, feed_back, reason, expected_answer,
                        ext, version, create_time, update_time)
                   VALUES
                       (:id, :seq, :conversation_id, :raw_content,
                        CAST(:retrieval_result AS JSON), :content, :result,
                        :feed_back, :reason, :expected_answer, CAST(:ext AS JSON),
                        :version, :create_time, :update_time)
                   ON CONFLICT (id) DO UPDATE SET
                       raw_content = EXCLUDED.raw_content,
                       retrieval_result = EXCLUDED.retrieval_result,
                       content = EXCLUDED.content,
                       result = EXCLUDED.result,
                       feed_back = EXCLUDED.feed_back,
                       reason = EXCLUDED.reason,
                       expected_answer = EXCLUDED.expected_answer,
                       ext = EXCLUDED.ext,
                       version = EXCLUDED.version,
                       update_time = EXCLUDED.update_time"""
            ),
            {
                'id': row_id,
                'seq': row['seq'],
                'conversation_id': conversation_id,
                'raw_content': row.get('raw_content') or '',
                'retrieval_result': json.dumps(row.get('retrieval_result') or {'sources': None}, ensure_ascii=False),
                'content': row.get('content') or '',
                'result': _text_value(row.get('result')),
                'feed_back': int(row.get('feed_back') or 0),
                'reason': row.get('reason') or '',
                'expected_answer': row.get('expected_answer') or '',
                'ext': json.dumps(ext, ensure_ascii=False, default=str),
                'version': row.get('version') or '2.3',
                'create_time': created_at,
                'update_time': updated_at,
            },
        )


def _varchar36_id(value: str) -> str:
    normalized = str(value or '').strip()
    if normalized and len(normalized) <= 36:
        return normalized
    return str(uuid.uuid5(uuid.NAMESPACE_URL, normalized or uuid.uuid4().hex))


def _text_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
