"""The agent loop: think, call tools, observe, answer.

Kept deliberately small and inspectable. Every run produces a transcript --
messages, tool calls, timings, token counts -- because when a forecast comes
out strange the question is always "what did it actually look at?"
"""

import datetime as dt
import json
import time
from dataclasses import dataclass, field, asdict

from . import protocol, tools
from .prompts import final_nudge


@dataclass
class Step:
    """One turn of the loop."""
    index: int
    text: str = ""
    reasoning: str = ""
    calls: list = field(default_factory=list)      # [(name, args)]
    results: list = field(default_factory=list)    # [ToolResult]
    seconds: float = 0.0
    prompt_tokens: int = 0


@dataclass
class AgentResult:
    answer: str = ""
    value: object = None            # what a terminal tool returned
    stopped: str = ""               # terminal | answered | max_steps | error
    steps: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tool_calls(self):
        return sum(len(s.calls) for s in self.steps)


class Agent:
    """Runs one task to completion against one toolset."""

    def __init__(self, client, system_prompt, toolset="research",
                 settings=None, on_event=None):
        self.client = client
        self.settings = settings or client.settings
        self.on_event = on_event or (lambda *a, **k: None)
        if isinstance(toolset, str):
            if toolset not in tools.TOOLSETS:
                raise ValueError(
                    f"unknown toolset {toolset!r}; known: "
                    f"{', '.join(sorted(tools.TOOLSETS))}")
            self.tool_names = list(tools.TOOLSETS[toolset])
        else:
            self.tool_names = list(toolset)
        self.specs = tools.specs(self.tool_names)

        mode = self.settings.tool_mode
        if mode == "auto":
            mode = "native" if client.supports_native_tools() else "prompt"
        self.mode = mode

        self.system_prompt = system_prompt
        if self.mode == "prompt":
            self.system_prompt += "\n\n" + protocol.render_tool_prompt(self.specs)

    # -- main loop ------------------------------------------------------------

    def run(self, task):
        started = time.time()
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}]
        result = AgentResult(messages=messages)
        terminal_available = any(tools.REGISTRY[n].terminal
                                 for n in self.tool_names
                                 if n in tools.REGISTRY)

        for index in range(self.settings.max_steps):
            step = Step(index=index)
            self.on_event("step", step=step)

            reply = self.client.chat(
                messages,
                tools=self.specs if self.mode == "native" else None)
            step.seconds = reply.seconds
            step.reasoning = reply.reasoning
            step.prompt_tokens = reply.usage.get("prompt_tokens", 0)
            result.prompt_tokens = max(result.prompt_tokens, step.prompt_tokens)
            result.completion_tokens += reply.usage.get("completion_tokens", 0)

            if self.mode == "native" and reply.tool_calls:
                calls = protocol.parse_native_calls(reply.tool_calls)
                step.text = reply.content
            else:
                step.text, calls = protocol.parse_calls(reply.content)

            # A reply cut off by max_tokens parses as "no calls at all". Left
            # alone, a model that thinks past the token limit looks like a model
            # that chose to stop, and the run ends with nothing to show for it.
            if not calls and reply.finish_reason == "length":
                messages.append({"role": "user", "content": (
                    "Your reply hit the token limit before you called a tool. "
                    "Stop analysing and act: make the call now, with no "
                    "preamble.")})
                result.steps.append(step)
                continue
            if not calls and protocol.looks_truncated(reply.content):
                messages.append({"role": "user", "content":
                                 "Your last tool call was cut off. Repeat it, "
                                 "complete and on its own."})
                result.steps.append(step)
                continue

            if not calls:
                result.answer = step.text or reply.content
                result.stopped = "answered"
                result.steps.append(step)
                self.on_event("answer", text=result.answer)
                break

            self._append_assistant(messages, reply, calls)

            for call in calls:
                self.on_event("call", call=call)
                outcome = tools.call(
                    call.name, call.arguments,
                    max_chars=self.settings.max_tool_output_chars)
                step.calls.append((call.name, call.arguments))
                step.results.append(outcome)
                self.on_event("result", call=call, result=outcome)
                self._append_result(messages, call, outcome)

                entry = tools.REGISTRY.get(call.name)
                if entry is not None and entry.terminal and outcome.ok:
                    result.value = outcome.value
                    result.answer = outcome.output
                    result.stopped = "terminal"

            result.steps.append(step)
            if result.stopped:
                break

            self._trim(messages, result.prompt_tokens)

            remaining = self.settings.max_steps - index - 1
            if terminal_available and remaining <= 1:
                messages.append({"role": "user",
                                 "content": final_nudge(remaining)})
        else:
            result.stopped = "max_steps"

        result.seconds = time.time() - started
        result.messages = messages
        return result

    # -- history ---------------------------------------------------------------

    def _append_assistant(self, messages, reply, calls):
        """Echo the assistant turn back into history, calls included."""
        if self.mode == "native" and reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": reply.tool_calls})
        else:
            # In prompt mode the raw text already carries the XML call blocks,
            # so it round-trips exactly as the model wrote it.
            messages.append({"role": "assistant", "content": reply.content})

    def _append_result(self, messages, call, outcome):
        message = {"role": "tool", "name": call.name, "content": outcome.output}
        if call.call_id:
            message["tool_call_id"] = call.call_id
        messages.append(message)

    def _trim(self, messages, prompt_tokens):
        """Elide the oldest tool outputs once the context gets tight.

        The server is started with max_model_len 32768; running into that mid
        run turns a good forecast into an HTTP 400.
        """
        if prompt_tokens < self.settings.context_budget:
            return
        for message in messages:
            if (message.get("role") == "tool"
                    and not message["content"].startswith("[elided")):
                message["content"] = (
                    f"[elided {len(message['content'])} characters of "
                    f"{message.get('name', 'tool')} output to save context]")
                return


def save_transcript(result, path, extra=None):
    """Write a run's full transcript as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stopped": result.stopped,
        "seconds": round(result.seconds, 1),
        "tool_calls": result.tool_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "answer": result.answer,
        "steps": [
            {**asdict(step),
             "results": [asdict(r) | {"value": None} for r in step.results]}
            for step in result.steps
        ],
        "messages": result.messages,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
