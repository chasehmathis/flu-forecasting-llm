"""Transport to the Qwen vLLM server (OpenAI-compatible /chat/completions).

Stdlib only, on purpose: this has to run on a login node with nothing
installed. The client knows nothing about tools beyond passing them through --
encoding and parsing live in protocol.py.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    """One assistant turn, as the server returned it."""
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)   # native calls, if any
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


class QwenClient:
    """A thin, retrying client for one vLLM endpoint."""

    def __init__(self, settings):
        self.settings = settings
        self.endpoint = settings.endpoint
        self.model = settings.model
        self._native_tools = None       # unknown until probed

    # -- capability -----------------------------------------------------------

    def supports_native_tools(self):
        """True if the server was started with --enable-auto-tool-choice.

        Without it, vLLM rejects a request carrying `tools` outright, so we
        find out by asking once and remember the answer.
        """
        if self._native_tools is None:
            probe = {
                "model": self.model, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {
                    "name": "noop", "description": "does nothing",
                    "parameters": {"type": "object", "properties": {}}}}],
                "tool_choice": "auto",
            }
            try:
                self._post(probe, timeout=60)
                self._native_tools = True
            except LLMError as e:
                if "tool" in str(e).lower():
                    self._native_tools = False
                else:
                    raise
        return self._native_tools

    def health(self):
        """(ok, message) -- is the server up and serving the expected model?"""
        try:
            req = urllib.request.Request(f"{self.endpoint}/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                served = [m["id"] for m in json.load(resp)["data"]]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            return False, (f"no server at {self.endpoint} ({e}). "
                           "Start one with:  qwen up")
        if self.model not in served:
            return False, f"server serves {served}, not {self.model!r}"
        return True, f"{self.endpoint} serving {self.model}"

    # -- generation -----------------------------------------------------------

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None,
             stop=None):
        """One completion. `tools` is only sent when the server supports it."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": (self.settings.temperature if temperature is None
                            else temperature),
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if stop:
            body["stop"] = stop
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        started = time.time()
        data = self._post(body, timeout=self.settings.request_timeout)
        choice = data["choices"][0]
        message = choice.get("message", {})

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        # In prompt mode the server hands back one blob: the model's thinking
        # runs until </think>, and the answer follows it.
        if not reasoning and "</think>" in content:
            reasoning, _, content = content.partition("</think>")
            reasoning = reasoning.replace("<think>", "").strip()

        return Reply(
            content=content.strip(),
            reasoning=reasoning.strip(),
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage") or {},
            seconds=time.time() - started,
        )

    # -- plumbing -------------------------------------------------------------

    def _post(self, body, *, timeout, attempts=3):
        payload = json.dumps(body).encode()
        req_url = f"{self.endpoint}/chat/completions"
        last = None
        for attempt in range(attempts):
            req = urllib.request.Request(
                req_url, data=payload,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                # 4xx means the request itself is wrong; retrying cannot help.
                if 400 <= e.code < 500:
                    raise LLMError(f"HTTP {e.code}: {detail}") from e
                last = LLMError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = LLMError(f"{self.endpoint} unreachable: {e}")
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
        raise last
