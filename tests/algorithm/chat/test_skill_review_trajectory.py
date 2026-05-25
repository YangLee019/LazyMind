from __future__ import annotations

from chat.components.skill_review.craft import build_skill_craft
from chat.components.skill_review.schemas import SessionData, SessionMessage
from chat.components.skill_review.trajectory import build_trajectory


def _build_session(messages: list[SessionMessage], session_id: str = 'session-1') -> SessionData:
    return SessionData(
        session_id=session_id,
        source_db='test',
        tables=[],
        messages=messages,
        metadata={'user_id': 'user-1'},
    )


class _CaptureLLM:
    def __init__(self) -> None:
        self.prompt = ''

    def complete_json(self, prompt: str) -> dict:
        self.prompt = prompt
        return {
            'contextual_description': {
                'task_goal': '测试任务',
                'applicable_scenario': '测试场景',
                'execution_summary': '测试摘要',
                'key_result': '测试结果',
                'environment': {},
            },
            'refined_trajectory': {
                'steps': [],
            },
            'guidelines': {
                'success_patterns': [],
                'failure_patterns': [],
            },
        }


def test_build_trajectory_extracts_user_assistant_tool_and_task_end():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='帮我查一下今天上海天气，并告诉我是否适合跑步。',
                raw={'role': 'user', 'content': '帮我查一下今天上海天气，并告诉我是否适合跑步。'},
            ),
            SessionMessage(
                role='assistant',
                content='我先查一下上海天气。',
                raw={
                    'role': 'assistant',
                    'content': '我先查一下上海天气。',
                    'reasoning_content': '需要先获取天气数据，再判断是否适合户外跑步。',
                    'tool_calls': [
                        {
                            'id': 'call_weather_1',
                            'name': 'weather',
                            'arguments': {'location': 'Shanghai'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='晴，23到28度，空气质量良。',
                tool_name='weather',
                raw={
                    'role': 'tool',
                    'name': 'weather',
                    'tool_call_id': 'call_weather_1',
                    'result': {'condition': '晴', 'temp_low': 23, 'temp_high': 28, 'aqi': '良'},
                },
            ),
            SessionMessage(
                role='assistant',
                content='今天上海天气晴朗，气温适中，空气质量良，适合跑步。',
                raw={
                    'role': 'assistant',
                    'content': '今天上海天气晴朗，气温适中，空气质量良，适合跑步。',
                },
            ),
        ]
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assert trajectory.session_id == 'session-1'
    assert trajectory.qualified is True
    assert trajectory.user_turns == 1
    assert trajectory.tool_turns == 2
    assert trajectory.called_tools == ['weather']

    assert [step.kind for step in trajectory.steps] == [
        'user_message',
        'assistant_message',
        'tool_call',
        'tool_result',
        'assistant_message',
    ]

    user_step = trajectory.steps[0]
    assert user_step.user_message == '帮我查一下今天上海天气，并告诉我是否适合跑步。'

    assistant_step = trajectory.steps[1]
    assert assistant_step.reasoning == '需要先获取天气数据，再判断是否适合户外跑步。'
    assert assistant_step.result == '我先查一下上海天气。'
    assert assistant_step.tool_input == {'location': 'Shanghai'}

    tool_call_step = trajectory.steps[2]
    assert tool_call_step.tool_name == 'weather'
    assert tool_call_step.tool_input == {'location': 'Shanghai'}
    assert tool_call_step.sub_index == 1

    tool_result_step = trajectory.steps[3]
    assert tool_result_step.tool_name == 'weather'
    assert tool_result_step.tool_output == {
        'condition': '晴',
        'temp_low': 23,
        'temp_high': 28,
        'aqi': '良',
    }

    final_step = trajectory.steps[4]
    assert [step.task_segment_id for step in trajectory.steps] == [1, 1, 1, 1, 1]
    assert final_step.is_task_end is True
    assert final_step.result == '今天上海天气晴朗，气温适中，空气质量良，适合跑步。'


def test_build_trajectory_marks_unqualified_when_tool_turns_not_enough():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='把这句话润色一下。',
                raw={'role': 'user', 'content': '把这句话润色一下。'},
            ),
            SessionMessage(
                role='assistant',
                content='当然，请把原句发给我。',
                raw={'role': 'assistant', 'content': '当然，请把原句发给我。'},
            ),
        ],
        session_id='session-2',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assert trajectory.session_id == 'session-2'
    assert trajectory.user_turns == 1
    assert trajectory.tool_turns == 0
    assert trajectory.qualified is False
    assert trajectory.skip_reason is not None
    assert trajectory.steps[1].is_task_end is True


def test_build_trajectory_uses_only_content_for_assistant_messages():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='帮我记录一个偏好。',
                raw={'role': 'user', 'content': '帮我记录一个偏好。'},
            ),
            SessionMessage(
                role='assistant',
                content='我先帮你保存。',
                raw={
                    'role': 'assistant',
                    'content': '我先帮你保存。',
                    'result': '{"success": true, "should_not": "appear_as_assistant_text"}',
                    'tool_calls': [
                        {
                            'id': 'call_memory_1',
                            'name': 'memory',
                            'arguments': {'target': 'user'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"success": true}',
                tool_name='memory',
                raw={
                    'role': 'tool',
                    'name': 'memory',
                    'result': {'success': True},
                },
            ),
            SessionMessage(
                role='assistant',
                content='已经保存好了。',
                raw={'role': 'assistant', 'content': '已经保存好了。'},
            ),
        ],
        session_id='session-assistant-content-only',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assistant_step = trajectory.steps[1]
    assert assistant_step.result == '我先帮你保存。'
    assert 'should_not' not in assistant_step.result
    assert trajectory.steps[4].is_task_end is True


def test_build_trajectory_marks_task_end_per_user_segment():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='先查天气，再总结。',
                raw={'role': 'user', 'content': '先查天气，再总结。'},
            ),
            SessionMessage(
                role='assistant',
                content='我先查一下天气。',
                raw={
                    'role': 'assistant',
                    'content': '我先查一下天气。',
                    'tool_calls': [
                        {
                            'id': 'call_weather_1',
                            'name': 'weather',
                            'arguments': {'location': 'Shanghai'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='晴，23到28度。',
                tool_name='weather',
                raw={
                    'role': 'tool',
                    'name': 'weather',
                    'result': {'condition': '晴'},
                },
            ),
            SessionMessage(
                role='assistant',
                content='今天适合外出。',
                raw={'role': 'assistant', 'content': '今天适合外出。'},
            ),
            SessionMessage(
                role='user',
                content='顺便提醒我带水。',
                raw={'role': 'user', 'content': '顺便提醒我带水。'},
            ),
            SessionMessage(
                role='assistant',
                content='记得带水，今天适合外出。',
                raw={'role': 'assistant', 'content': '记得带水，今天适合外出。'},
            ),
        ],
        session_id='session-task-end-per-segment',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assistant_steps = [step for step in trajectory.steps if step.role == 'assistant' and step.kind == 'assistant_message']
    assert [step.task_segment_id for step in trajectory.steps] == [1, 1, 1, 1, 1, 2, 2]
    assert [step.is_task_end for step in assistant_steps] == [False, True, True]


def test_build_trajectory_marks_tool_result_as_task_end_without_followup_assistant():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='帮我查一下网络状态。',
                raw={'role': 'user', 'content': '帮我查一下网络状态。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来检查。',
                raw={
                    'role': 'assistant',
                    'content': '我来检查。',
                    'tool_calls': [
                        {
                            'id': 'call_web_1',
                            'name': 'web_search',
                            'arguments': {'query': 'network status'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"success":false,"status":"timeout"}',
                tool_name='web_search',
                raw={
                    'role': 'tool',
                    'name': 'web_search',
                    'content': '{"success":false,"status":"timeout"}',
                    'tool_call_id': 'call_web_1',
                },
            ),
            SessionMessage(
                role='user',
                content='那先算了。',
                raw={'role': 'user', 'content': '那先算了。'},
            ),
        ],
        session_id='session-no-task-end',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assert [step.task_segment_id for step in trajectory.steps] == [1, 1, 1, 1, 2]
    assert trajectory.steps[3].kind == 'tool_result'
    assert trajectory.steps[3].is_task_end is True
    assert not any(step.is_task_end for step in trajectory.steps[:3])


def test_build_trajectory_extracts_tool_output_from_tool_content_json():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='查一下网络状态。',
                raw={'role': 'user', 'content': '查一下网络状态。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来检查。',
                raw={
                    'role': 'assistant',
                    'content': '我来检查。',
                    'tool_calls': [
                        {
                            'id': 'call_web_1',
                            'name': 'web_search',
                            'arguments': {'query': 'network status'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"success":false,"status":"network_unreachable","error":"timeout"}',
                tool_name='web_search',
                raw={
                    'role': 'tool',
                    'name': 'web_search',
                    'content': '{"success":false,"status":"network_unreachable","error":"timeout"}',
                },
            ),
        ],
        session_id='session-tool-output-from-content-json',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[3]
    assert tool_result_step.tool_output == {
        'success': False,
        'status': 'network_unreachable',
        'error': 'request timeout',
    }


def test_build_trajectory_extracts_tool_output_from_tool_content_text():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='用不可用工具试一下。',
                raw={'role': 'user', 'content': '用不可用工具试一下。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来试一下。',
                raw={
                    'role': 'assistant',
                    'content': '我来试一下。',
                    'tool_calls': [
                        {
                            'id': 'call_tool_1',
                            'name': 'kb_search',
                            'arguments': {'query': 'test'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='Tool [kb_search] is not available. Please choose from the available tools.',
                tool_name='kb_search',
                raw={
                    'role': 'tool',
                    'name': 'kb_search',
                    'content': 'Tool [kb_search] is not available. Please choose from the available tools.',
                },
            ),
        ],
        session_id='session-tool-output-from-content-text',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[3]
    assert tool_result_step.tool_output == 'Tool [kb_search] is not available. Please choose from the available tools.'


def test_build_trajectory_infers_skill_name_from_skill_tools():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='帮我创建一个春天写作技能。',
                raw={'role': 'user', 'content': '帮我创建一个春天写作技能。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来创建这个技能。',
                raw={
                    'role': 'assistant',
                    'content': '我来创建这个技能。',
                    'tool_calls': [
                        {
                            'id': 'call_skill_manage_1',
                            'name': 'skill_manage',
                            'arguments': {
                                'name': 'spring-prose-writing',
                                'action': 'create',
                                'category': 'writing',
                            },
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"success": true}',
                tool_name='skill_manage',
                raw={
                    'role': 'tool',
                    'name': 'skill_manage',
                    'content': '{"success": true}',
                    'tool_call_id': 'call_skill_manage_1',
                },
            ),
            SessionMessage(
                role='assistant',
                content='我再读取一下这个技能。',
                raw={
                    'role': 'assistant',
                    'content': '我再读取一下这个技能。',
                    'tool_calls': [
                        {
                            'id': 'call_get_skill_1',
                            'name': 'get_skill',
                            'arguments': {'name': 'spring-prose-writing'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"status": "ok"}',
                tool_name='get_skill',
                raw={
                    'role': 'tool',
                    'name': 'get_skill',
                    'content': '{"status": "ok"}',
                    'tool_call_id': 'call_get_skill_1',
                },
            ),
            SessionMessage(
                role='assistant',
                content='接着运行它的脚本。',
                raw={
                    'role': 'assistant',
                    'content': '接着运行它的脚本。',
                    'tool_calls': [
                        {
                            'id': 'call_run_script_1',
                            'name': 'run_script',
                            'arguments': {'name': 'spring-prose-writing', 'rel_path': 'scripts/create.sh'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='{"status": "error"}',
                tool_name='run_script',
                raw={
                    'role': 'tool',
                    'name': 'run_script',
                    'content': '{"status": "error"}',
                    'tool_call_id': 'call_run_script_1',
                },
            ),
            SessionMessage(
                role='assistant',
                content='处理完了。',
                raw={'role': 'assistant', 'content': '处理完了。'},
            ),
        ],
        session_id='session-infer-skill-name',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assert trajectory.called_skills == ['spring-prose-writing']
    assert 'skill_manage' in trajectory.called_tools
    assert 'get_skill' in trajectory.called_tools
    assert 'run_script' in trajectory.called_tools
    assert 'spring-prose-writing' not in trajectory.called_tools
    skill_steps = [step for step in trajectory.steps if step.skill_name == 'spring-prose-writing']
    assert skill_steps


def test_build_trajectory_compresses_tool_fields_for_llm_readability():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='请帮我分析这个很长的结果。',
                raw={'role': 'user', 'content': '请帮我分析这个很长的结果。'},
            ),
            SessionMessage(
                role='assistant',
                content='我先调用搜索工具，并重点关注 status、message 和结果列表。',
                raw={
                    'role': 'assistant',
                    'content': '我先调用搜索工具，并重点关注 status、message 和结果列表。',
                    'reasoning_content': '先确认状态，再看返回的结果列表里有没有可用条目。',
                    'tool_calls': [
                        {
                            'id': 'call_search_1',
                            'name': 'web_search',
                            'arguments': {
                                'query': 'very long query',
                                'page': 1,
                                'debug': '',
                                'metadata': {
                                    'trace_id': 'abc',
                                    'session': 's1',
                                    'extra': 'x',
                                    'ignored': '',
                                },
                                'tags': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
                            },
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='',
                tool_name='web_search',
                raw={
                    'role': 'tool',
                    'name': 'web_search',
                    'result': {
                        'status': 'ok',
                        'message': 'found results',
                        'results': [
                            {'title': 'Result A', 'snippet': 'A' * 200, 'url': 'https://example.com/a'},
                            {'title': 'Result B', 'snippet': 'B' * 200, 'url': 'https://example.com/b'},
                            {'title': 'Result C', 'snippet': 'C' * 200, 'url': 'https://example.com/c'},
                            {'title': 'Result D', 'snippet': 'D' * 200, 'url': 'https://example.com/d'},
                            {'title': 'Result E', 'snippet': 'E' * 200, 'url': 'https://example.com/e'},
                        ],
                        'debug': {
                            'raw_html': '<html>...</html>',
                            'timing_ms': 132,
                            'extra': 'unused',
                        },
                        'unused': '',
                    },
                },
            ),
        ],
        session_id='session-compress-tool-fields',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    assistant_step = trajectory.steps[1]
    tool_call_step = trajectory.steps[2]
    tool_result_step = trajectory.steps[3]

    assert assistant_step.reasoning == '先确认状态，再看返回的结果列表里有没有可用条目。'
    assert tool_call_step.action == 'Call web_search to search for very long query'
    assert tool_call_step.tool_input == {
        'metadata': {'extra': 'x', 'session': 's1', 'trace_id': 'abc'},
        'page': 1,
        'query': 'very long query',
        'tags': {'count': 5, 'sample': ['alpha', 'beta', 'gamma']},
    }
    assert tool_result_step.action == 'web_search returned 5 items'
    assert tool_result_step.tool_output == {
        'message': 'found results',
        'status': 'ok',
        'results': {
            'count': 5,
            'sample': [
                {'title': 'Result A', 'url': '[url]', 'snippet': 'A' * 177 + '...'},
                {'title': 'Result B', 'url': '[url]', 'snippet': 'B' * 177 + '...'},
                {'title': 'Result C', 'url': '[url]', 'snippet': 'C' * 177 + '...'},
            ],
        },
        'debug': {'extra': 'unused', 'raw_html': '...', 'timing_ms': 132},
    }


def test_build_trajectory_semantically_compresses_structured_assistant_reply():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='总结一下EVO研究。',
                raw={'role': 'user', 'content': '总结一下EVO研究。'},
            ),
            SessionMessage(
                role='assistant',
                content=(
                    '根据知识库搜索结果，我为你整理如下：\n\n'
                    '## 主要方向\n'
                    '- CoEvoSkills：协同进化框架\n'
                    '- Evo Skills：自动化技能发现\n'
                    '- 实验结果：完整框架效果更好\n\n'
                    '### 结论\n'
                    'EVO在这里主要指AI技能进化相关研究。'
                ),
                raw={
                    'role': 'assistant',
                    'content': (
                        '根据知识库搜索结果，我为你整理如下：\n\n'
                        '## 主要方向\n'
                        '- CoEvoSkills：协同进化框架\n'
                        '- Evo Skills：自动化技能发现\n'
                        '- 实验结果：完整框架效果更好\n\n'
                        '### 结论\n'
                        'EVO在这里主要指AI技能进化相关研究。'
                    ),
                    'reasoning_content': '我已经拿到知识库结果，需要先给结论，再补充重点。',
                },
            ),
        ],
        session_id='session-structured-assistant-summary',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=0,
    )

    assistant_step = trajectory.steps[1]
    assert assistant_step.action == 'EVO在这里主要指AI技能进化相关研究。'
    assert assistant_step.result == 'EVO在这里主要指AI技能进化相关研究。'
    assert assistant_step.reasoning == '我已经拿到知识库结果，需要先给结论，再补充重点。'
    assert '##' not in assistant_step.result
    assert '- CoEvoSkills' not in assistant_step.result


def test_build_trajectory_semantically_compresses_english_structured_reply():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='Summarize the EVO research.',
                raw={'role': 'user', 'content': 'Summarize the EVO research.'},
            ),
            SessionMessage(
                role='assistant',
                content=(
                    'Based on the search results, here is a structured summary:\n\n'
                    '## Main directions\n'
                    '- CoEvoSkills: a collaborative evolution framework\n'
                    '- Evo Skills: automated skill discovery\n'
                    '- Experimental results: the full framework performs better\n\n'
                    '### Conclusion\n'
                    'EVO here mainly refers to AI skill evolution research.'
                ),
                raw={
                    'role': 'assistant',
                    'content': (
                        'Based on the search results, here is a structured summary:\n\n'
                        '## Main directions\n'
                        '- CoEvoSkills: a collaborative evolution framework\n'
                        '- Evo Skills: automated skill discovery\n'
                        '- Experimental results: the full framework performs better\n\n'
                        '### Conclusion\n'
                        'EVO here mainly refers to AI skill evolution research.'
                    ),
                    'reasoning_content': 'I have enough evidence and should provide the conclusion first.',
                },
            ),
        ],
        session_id='session-structured-assistant-summary-en',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=0,
    )

    assistant_step = trajectory.steps[1]
    assert assistant_step.action == 'EVO here mainly refers to AI skill evolution research.'
    assert assistant_step.result == 'EVO here mainly refers to AI skill evolution research.'
    assert assistant_step.reasoning == 'I have enough evidence and should provide the conclusion first.'
    assert '##' not in assistant_step.result
    assert '- CoEvoSkills' not in assistant_step.result


def test_build_trajectory_drops_english_tail_phrase_when_extracting_summary():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='Tell me what tools you have.',
                raw={'role': 'user', 'content': 'Tell me what tools you have.'},
            ),
            SessionMessage(
                role='assistant',
                content=(
                    'I currently have search, memory, and skill management tools available. '
                    'These let me retrieve information, save preferences, and manage reusable skills. '
                    'Let me know if you want more details.'
                ),
                raw={
                    'role': 'assistant',
                    'content': (
                        'I currently have search, memory, and skill management tools available. '
                        'These let me retrieve information, save preferences, and manage reusable skills. '
                        'Let me know if you want more details.'
                    ),
                },
            ),
        ],
        session_id='session-english-tail-phrase',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=0,
    )

    assistant_step = trajectory.steps[1]
    assert 'Let me know if you want more details.' not in assistant_step.result
    assert 'search, memory, and skill management tools' in assistant_step.result


def test_build_trajectory_filters_generic_corrupted_text():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='看一下这个抓取结果。',
                raw={'role': 'user', 'content': '看一下这个抓取结果。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来检查这个结果。',
                raw={'role': 'assistant', 'content': '我来检查这个结果。'},
            ),
            SessionMessage(
                role='tool',
                content='',
                tool_name='url_fetch',
                raw={
                    'role': 'tool',
                    'name': 'url_fetch',
                    'result': {
                        'status': 'ok',
                        'content': '� � \\xE4\\xB8\\xAD Ã¤Â¸Â­ ð\x9f\x8e\x89 Ã¥Â¥Â½ Ã¤Â½Â\xa0',
                    },
                },
            ),
        ],
        session_id='session-corrupted-text',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[2]
    assert tool_result_step.tool_output == {
        'status': 'ok',
        'content': '[encoding issue omitted]',
    }


def test_build_trajectory_keeps_normal_accented_text():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='Check the page summary.',
                raw={'role': 'user', 'content': 'Check the page summary.'},
            ),
            SessionMessage(
                role='assistant',
                content='I will review the fetched page.',
                raw={'role': 'assistant', 'content': 'I will review the fetched page.'},
            ),
            SessionMessage(
                role='tool',
                content='',
                tool_name='url_fetch',
                raw={
                    'role': 'tool',
                    'name': 'url_fetch',
                    'result': {
                        'status': 'ok',
                        'content': "Résumé de l'étude sur l'évolution de l'IA à Montréal.",
                    },
                },
            ),
        ],
        session_id='session-accented-text',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[2]
    assert tool_result_step.tool_output == {
        'status': 'ok',
        'content': "Résumé de l'étude sur l'évolution de l'IA à Montréal.",
    }


def test_build_trajectory_filters_short_mojibake_title():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='Check the fetched title.',
                raw={'role': 'user', 'content': 'Check the fetched title.'},
            ),
            SessionMessage(
                role='assistant',
                content='I will inspect the fetched page.',
                raw={'role': 'assistant', 'content': 'I will inspect the fetched page.'},
            ),
            SessionMessage(
                role='tool',
                content='',
                tool_name='url_fetch',
                raw={
                    'role': 'tool',
                    'name': 'url_fetch',
                    'result': {
                        'status': 'ok',
                        'title': 'DeepSeek æ·±åº¦æ±ç´¢',
                    },
                },
            ),
        ],
        session_id='session-short-mojibake-title',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[2]
    assert tool_result_step.tool_output == {
        'status': 'ok',
        'title': '[encoding issue omitted]',
    }


def test_build_trajectory_normalizes_embedding_key_error_as_configuration_issue():
    session = _build_session(
        [
            SessionMessage(
                role='user',
                content='查一下知识库。',
                raw={'role': 'user', 'content': '查一下知识库。'},
            ),
            SessionMessage(
                role='assistant',
                content='我来检索。',
                raw={
                    'role': 'assistant',
                    'content': '我来检索。',
                    'tool_calls': [
                        {
                            'id': 'call_kb_1',
                            'name': 'kb_search',
                            'arguments': {'query': 'evo'},
                        }
                    ],
                },
            ),
            SessionMessage(
                role='tool',
                content='',
                tool_name='kb_search',
                raw={
                    'role': 'tool',
                    'name': 'kb_search',
                    'result': {
                        'success': False,
                        'reason': 'kb_search failed: Embedding key embed_image not found in group image from document http://x, available keys: []',
                        'error': 'Embedding key embed_image not found in group image from document http://x, available keys: []',
                        'error_type': 'RuntimeError',
                    },
                },
            ),
        ],
        session_id='session-embedding-key-error',
    )

    trajectory = build_trajectory(
        session,
        min_user_turns=1,
        min_tool_turns=1,
    )

    tool_result_step = trajectory.steps[3]
    assert tool_result_step.action == 'kb_search failed: embedding configuration error'
    assert tool_result_step.tool_output == {
        'success': False,
        'error': 'embedding configuration error',
        'error_type': 'RuntimeError',
        'reason': 'embedding configuration error',
    }


def test_build_skill_craft_filters_step_raw_from_prompt():
    session = SessionData(
        session_id='craft-session',
        source_db='test',
        tables=[],
        metadata={'user_id': 'user-1'},
        messages=[
            SessionMessage(
                role='user',
                content='帮我查天气。',
                raw={'role': 'user', 'content': '帮我查天气。', 'secret_raw': 'should-not-be-in-prompt'},
            ),
            SessionMessage(
                role='assistant',
                content='我先查一下。',
                raw={
                    'role': 'assistant',
                    'content': '我先查一下。',
                    'tool_calls': [
                        {'id': 'call_weather_1', 'name': 'weather', 'arguments': {'location': 'Shanghai'}}
                    ],
                    'secret_raw': 'assistant-hidden',
                },
            ),
            SessionMessage(
                role='tool',
                content='{"condition":"晴"}',
                raw={
                    'role': 'tool',
                    'name': 'weather',
                    'content': '{"condition":"晴"}',
                    'secret_raw': 'tool-hidden',
                },
            ),
        ],
    )
    trajectory = build_trajectory(session, min_user_turns=1, min_tool_turns=1)
    llm = _CaptureLLM()

    build_skill_craft(trajectory, llm)

    assert 'secret_raw' not in llm.prompt
    assert '"raw"' not in llm.prompt
    assert '"tool_output"' in llm.prompt
    assert '"action"' in llm.prompt
