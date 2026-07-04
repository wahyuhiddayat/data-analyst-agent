from eval.collect import serialize_messages


class _FakeMessage:
    """Stand-in for an SDK assistant message exposing model_dump()."""

    def __init__(self, data):
        self._data = data

    def model_dump(self, exclude_none=True):
        if exclude_none:
            return {k: v for k, v in self._data.items() if v is not None}
        return dict(self._data)


def test_dict_turns_pass_through():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "abc", "content": "42"},
    ]
    assert serialize_messages(messages) == messages


def test_assistant_tool_call_is_serialized():
    msg = _FakeMessage({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "run_python", "arguments": '{"code": "print(1)"}'}}
        ],
        "reasoning_content": "secret chain of thought",
    })
    out = serialize_messages([msg])
    assert len(out) == 1
    turn = out[0]
    assert turn["role"] == "assistant"
    assert turn["content"] == ""
    assert "reasoning_content" not in turn
    assert turn["tool_calls"][0]["function"]["arguments"] == '{"code": "print(1)"}'


def test_final_assistant_answer_keeps_content():
    msg = _FakeMessage({"role": "assistant", "content": "@mean[3.0]", "tool_calls": None})
    out = serialize_messages([msg])
    assert out[0]["content"] == "@mean[3.0]"
    assert "tool_calls" not in out[0]
