"""The loop itself, driven by a scripted model so the tests are deterministic."""

import json
import math
from dataclasses import dataclass
from statistics import NormalDist

import pytest

from flufc.agent import Agent
from flufc.config import Settings
from flufc.forecast import QUANTILES
from flufc.llm import Reply
from flufc.protocol import render_tool_call


class ScriptedClient:
    """Stands in for QwenClient: replays canned replies, records the history."""

    def __init__(self, replies, settings=None):
        self.replies = list(replies)
        self.settings = settings or Settings.load(tool_mode="prompt")
        self.seen = []

    def supports_native_tools(self):
        return False

    def chat(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        content = self.replies.pop(0) if self.replies else "done"
        return Reply(content=content, usage={"prompt_tokens": 10,
                                             "completion_tokens": 5})


def settings(**kw):
    return Settings.load(tool_mode="prompt", **kw)


def test_runs_a_tool_then_answers():
    client = ScriptedClient([
        render_tool_call("calculate", {"expression": "2 + 2"}),
        "The answer is 4.",
    ], settings())
    result = Agent(client, "sys", ["calculate"], client.settings).run("add")
    assert result.stopped == "answered"
    assert result.answer == "The answer is 4."
    assert result.tool_calls == 1
    assert result.steps[0].results[0].output.endswith("= 4")


def test_tool_results_go_back_as_tool_messages():
    client = ScriptedClient(
        [render_tool_call("calculate", {"expression": "1+1"}), "two"], settings())
    Agent(client, "sys", ["calculate"], client.settings).run("go")
    roles = [m["role"] for m in client.seen[-1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "= 2" in client.seen[-1][-1]["content"]


def test_terminal_tool_stops_the_loop_immediately():
    grid = json.dumps([round(800 * math.exp(0.35 * NormalDist().inv_cdf(q)), 1)
                       for q in QUANTILES])
    client = ScriptedClient([
        render_tool_call("submit_forecast", {
            "location": "US", "reference_date": "2026-08-22",
            "quantiles": '{"0": %s}' % grid, "rationale": "flat"}),
        "this reply should never be requested",
    ], settings())
    result = Agent(client, "sys", ["submit_forecast"], client.settings).run("go")
    assert result.stopped == "terminal"
    assert result.value.median(0) == 800
    assert len(client.seen) == 1


def test_a_failed_tool_call_does_not_end_the_run():
    client = ScriptedClient([
        render_tool_call("calculate", {"expression": "1/0"}),
        render_tool_call("calculate", {"expression": "1/2"}),
        "recovered",
    ], settings())
    result = Agent(client, "sys", ["calculate"], client.settings).run("go")
    assert result.stopped == "answered" and result.answer == "recovered"
    assert not result.steps[0].results[0].ok


def test_truncated_call_is_retried_rather_than_treated_as_an_answer():
    client = ScriptedClient([
        "<tool_call>\n<function=calculate>\n<parameter=expression>\n1+1",
        render_tool_call("calculate", {"expression": "1+1"}),
        "two",
    ], settings())
    result = Agent(client, "sys", ["calculate"], client.settings).run("go")
    assert result.answer == "two"
    assert "cut off" in client.seen[1][-1]["content"]


def test_the_loop_stops_at_max_steps():
    client = ScriptedClient(
        [render_tool_call("calculate", {"expression": "1+1"})] * 10,
        settings(max_steps=3))
    result = Agent(client, "sys", ["calculate"], client.settings).run("go")
    assert result.stopped == "max_steps" and len(result.steps) == 3


def test_the_tool_spec_is_injected_in_prompt_mode():
    client = ScriptedClient(["done"], settings())
    agent = Agent(client, "You forecast flu.", ["calculate"], client.settings)
    assert "<tool_call>" in agent.system_prompt
    assert '"name": "calculate"' in agent.system_prompt
    assert "submit_forecast" not in agent.system_prompt   # not in this toolset


def test_oversized_context_elides_the_oldest_tool_output():
    client = ScriptedClient(
        [render_tool_call("calculate", {"expression": "1+1"})] * 4 + ["done"],
        settings(max_steps=5, context_budget=1))     # force trimming every step
    result = Agent(client, "sys", ["calculate"], client.settings).run("go")
    elided = [m for m in result.messages
              if m["role"] == "tool" and m["content"].startswith("[elided")]
    assert elided


def test_reply_cut_off_by_the_token_limit_is_not_mistaken_for_an_answer():
    """The failure that loses a forecast: the model thinks past max_tokens."""

    class CutOffClient(ScriptedClient):
        def chat(self, messages, **kwargs):
            self.seen.append([dict(m) for m in messages])
            content = self.replies.pop(0) if self.replies else "done"
            # the first reply is truncated mid-thought, with no tool call in it
            reason = "length" if len(self.seen) == 1 else "stop"
            return Reply(content=content, finish_reason=reason,
                         usage={"prompt_tokens": 10, "completion_tokens": 5})

    client = CutOffClient([
        "Let me think about the trend. The recent weeks are 664, 695, 651, 62",
        render_tool_call("calculate", {"expression": "1+1"}),
        "two",
    ], settings())
    result = Agent(client, "sys", ["calculate"], client.settings).run("go")
    assert result.answer == "two"
    assert result.tool_calls == 1
    assert "token limit" in client.seen[1][-1]["content"]


def test_an_unknown_toolset_name_fails_loudly():
    client = ScriptedClient(["done"], settings())
    with pytest.raises(ValueError, match="unknown toolset"):
        Agent(client, "sys", "reserch", client.settings)     # typo
