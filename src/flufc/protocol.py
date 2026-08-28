"""Qwen's tool-call format.

The model's own chat template teaches it to call functions with an XML block:

    <tool_call>
    <function=search_web>
    <parameter=query>
    CDC flu hospitalizations
    </parameter>
    </function>
    </tool_call>

vLLM will only assemble that block for us if the server was started with
--enable-auto-tool-choice --tool-call-parser qwen3_xml. When it wasn't, we put
the same instructions in the system prompt ourselves and parse the block out of
the reply. Same wire format either way, so the model sees what it was trained
on and the agent loop does not care which mode is in play.
"""

import json
import re
from dataclasses import dataclass, field

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)

# Lifted from the model's chat template so prompt mode looks like native mode.
TOOL_PROMPT = """# Tools

You have access to the following functions:

<tools>
{specs}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- Call at most a few functions per turn, then wait for their results
</IMPORTANT>"""


@dataclass
class ToolCall:
    """One requested call, with arguments still as raw strings."""
    name: str
    arguments: dict = field(default_factory=dict)
    call_id: str = ""          # native mode only; results must reference it

    def describe(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args[:160]})"


def render_tool_prompt(specs):
    """The system-prompt block that teaches prompt mode. `specs` is OpenAI-style."""
    body = "\n".join(json.dumps(s) for s in specs)
    return TOOL_PROMPT.format(specs=body)


def render_tool_call(name, arguments):
    """Build a call block. Used by tests and to echo calls back into history."""
    params = "".join(
        f"<parameter={key}>\n{_render_value(value)}\n</parameter>\n"
        for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _render_value(value):
    return value if isinstance(value, str) else json.dumps(value)


def parse_calls(content):
    """Split an assistant reply into (visible text, [ToolCall]).

    Arguments come back as strings; coercion to the declared JSON types happens
    in tools.call(), which is where the schema lives.
    """
    calls = []
    for name, body in TOOL_CALL_RE.findall(content or ""):
        arguments = {key: value.strip("\n")
                     for key, value in PARAM_RE.findall(body)}
        calls.append(ToolCall(name=name.strip(), arguments=arguments))
    text = TOOL_CALL_RE.sub("", content or "").strip()
    return text, calls


def parse_native_calls(raw_calls):
    """Convert vLLM's OpenAI-shaped tool_calls into our ToolCall objects."""
    calls = []
    for raw in raw_calls:
        fn = raw.get("function", {})
        arguments = fn.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__raw": arguments}
        calls.append(ToolCall(name=fn.get("name", ""), arguments=arguments,
                              call_id=raw.get("id", "")))
    return calls


def looks_truncated(content):
    """True if the reply opened a tool call it never closed.

    Worth catching: a call cut off by max_tokens otherwise reads to the parser
    as 'the model chose to say nothing', and the loop ends for no reason.
    """
    content = content or ""
    return content.count("<tool_call>") > content.count("</tool_call>")
