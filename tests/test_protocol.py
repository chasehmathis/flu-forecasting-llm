"""The tool-call wire format is the one thing that must not break silently."""

from flufc import protocol


def test_parses_multiple_calls():
    reply = """Let me check two things.

<tool_call>
<function=get_ili>
<parameter=region>
nat
</parameter>
<parameter=weeks>
8
</parameter>
</function>
</tool_call>
<tool_call>
<function=calculate>
<parameter=expression>
1 + 1
</parameter>
</function>
</tool_call>"""
    text, calls = protocol.parse_calls(reply)
    assert text == "Let me check two things."
    assert [c.name for c in calls] == ["get_ili", "calculate"]
    assert calls[0].arguments == {"region": "nat", "weeks": "8"}


def test_multiline_parameter_survives():
    call = protocol.render_tool_call("submit_forecast",
                                     {"rationale": "line one\nline two"})
    _, parsed = protocol.parse_calls(call)
    assert parsed[0].arguments["rationale"] == "line one\nline two"


def test_reply_with_no_calls_is_just_text():
    text, calls = protocol.parse_calls("The answer is 42.")
    assert calls == [] and text == "The answer is 42."


def test_truncated_call_is_detected():
    assert protocol.looks_truncated("<tool_call>\n<function=get_ili>\n<param")
    assert not protocol.looks_truncated("done")


def test_native_calls_convert():
    calls = protocol.parse_native_calls(
        [{"function": {"name": "calculate", "arguments": '{"expression": "2*2"}'}}])
    assert calls[0].name == "calculate"
    assert calls[0].arguments == {"expression": "2*2"}
