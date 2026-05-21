from __future__ import annotations

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


def test_build_trajectory_extracts_user_assistant_tool_and_final_answer():
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
    assert trajectory.final_answer == '今天上海天气晴朗，气温适中，空气质量良，适合跑步。'

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
    assert final_step.is_final is True
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
    assert trajectory.final_answer == '当然，请把原句发给我。'
