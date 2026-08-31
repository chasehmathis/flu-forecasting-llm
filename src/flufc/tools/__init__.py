"""The tool registry.

A tool is a plain Python function with type hints and a Google-style docstring;
the decorator turns that into the JSON schema the model sees, so the signature
and the advertised contract cannot drift apart.

    @tool
    def search_web(query: str, max_results: int = 6) -> str:
        '''Search the web.

        Args:
            query: What to search for.
            max_results: How many results to return.
        '''
"""

import inspect
import json
import os
import re
import time
import traceback
import typing
from dataclasses import dataclass, field

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}


class ToolError(Exception):
    """Raised inside a tool when the model can fix the problem by retrying."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: callable
    terminal: bool = False      # calling it ends the agent loop

    def spec(self):
        """OpenAI-style function spec -- what goes in the prompt or `tools`."""
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters}}


@dataclass
class ToolResult:
    name: str
    arguments: dict
    output: str = ""
    value: object = None        # the raw return, for terminal tools
    ok: bool = True
    seconds: float = 0.0


REGISTRY = {}

# Named bundles, so a task only sees the tools it should use.
_SURVEILLANCE = ["list_locations", "get_flu_hospitalizations", "get_ili"]
_RESEARCH = ["search_web", "fetch_url", "search_literature", "read_paper"]
_QUANTITATIVE = ["fit_forecast_model", "compare_forecast_models",
                 "get_hub_forecasts", "list_hub_models", "calculate"]

TOOLSETS = {
    "research": _SURVEILLANCE + _RESEARCH + _QUANTITATIVE,
    "forecast": _SURVEILLANCE + _RESEARCH + _QUANTITATIVE + ["submit_forecast"],
    # Numbers only: the statistical models, the hub, and no reading.
    "quant": _SURVEILLANCE + _QUANTITATIVE + ["submit_forecast"],
}


def tool(fn=None, *, name=None, terminal=False):
    """Register a function as a tool. Usable bare or with keywords."""
    def register(target):
        names = tuple(inspect.signature(target).parameters)
        description, arg_docs = _parse_docstring(target.__doc__ or "", names)
        entry = Tool(
            name=name or target.__name__,
            description=description,
            parameters=_build_schema(target, arg_docs),
            fn=target,
            terminal=terminal,
        )
        REGISTRY[entry.name] = entry
        return target
    return register(fn) if fn is not None else register


def _parse_docstring(doc, names=()):
    """Split a docstring into (description, {arg: description}).

    `names` is the function's actual parameters. Without it a continuation line
    that happens to start "something:" reads as a new argument and silently
    truncates the description above it -- which loses the only contract the
    model ever sees for that parameter.
    """
    doc = inspect.cleandoc(doc)
    head, _, tail = doc.partition("\nArgs:")
    arg_docs, current = {}, None
    for line in tail.splitlines():
        m = re.match(r"\s+(\w+):\s*(.*)$", line)
        if m and (not names or m.group(1) in names):
            current = m.group(1)
            arg_docs[current] = m.group(2).strip()
        elif current and line.strip():
            arg_docs[current] += " " + line.strip()
    return head.strip(), arg_docs


def _build_schema(fn, arg_docs):
    """JSON schema for a function's parameters, from its annotations."""
    hints = typing.get_type_hints(fn)
    properties, required = {}, []
    for pname, param in inspect.signature(fn).parameters.items():
        annotation = hints.get(pname, str)
        origin = typing.get_origin(annotation)
        if origin is typing.Union:          # Optional[X] -> X, not required
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = args[0] if args else str
        schema = {"type": JSON_TYPES.get(annotation, "string")}
        if pname in arg_docs:
            schema["description"] = arg_docs[pname]
        properties[pname] = schema
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


def specs(names=None):
    """Tool specs for `names` (default: everything registered)."""
    chosen = names if names is not None else list(REGISTRY)
    return [REGISTRY[n].spec() for n in chosen if n in REGISTRY]


def coerce(entry, arguments):
    """Cast raw string arguments to the types the schema declares."""
    out = {}
    props = entry.parameters["properties"]
    for key, value in arguments.items():
        if key not in props:
            raise ToolError(
                f"{entry.name} has no parameter {key!r}; "
                f"valid: {', '.join(props) or '(none)'}")
        want = props[key]["type"]
        if not isinstance(value, str):
            out[key] = value
            continue
        text = value.strip()
        try:
            if want == "integer":
                out[key] = int(float(text))
            elif want == "number":
                out[key] = float(text)
            elif want == "boolean":
                out[key] = text.lower() in ("true", "1", "yes")
            elif want in ("array", "object"):
                out[key] = json.loads(text)
            else:
                out[key] = text
        except (ValueError, json.JSONDecodeError) as e:
            raise ToolError(
                f"{entry.name}: {key}={value!r} is not a valid {want} ({e})")
    missing = [r for r in entry.parameters["required"] if r not in out]
    if missing:
        raise ToolError(f"{entry.name} is missing required "
                        f"parameter(s): {', '.join(missing)}")
    return out


def call(name, arguments, *, context=None, max_chars=6000):
    """Run a tool and return a ToolResult whose `output` is safe to show a model.

    Tools never raise into the agent loop: a failure becomes an error string the
    model can read and react to, which is usually more useful than a crash.
    """
    started = time.time()
    entry = REGISTRY.get(name)
    if entry is None:
        return ToolResult(name, arguments, ok=False, seconds=0.0,
                          output=f"ERROR: no tool named {name!r}. "
                                 f"Available: {', '.join(sorted(REGISTRY))}")
    try:
        kwargs = coerce(entry, arguments)
        if "context" in inspect.signature(entry.fn).parameters:
            kwargs["context"] = context
        value = entry.fn(**kwargs)
        output = value if isinstance(value, str) else json.dumps(value, indent=2,
                                                                default=str)
        ok = True
    except ToolError as e:
        value, output, ok = None, f"ERROR: {e}", False
    except Exception as e:                    # a bug in a tool, not the model
        value, ok = None, False
        output = f"ERROR: {type(e).__name__}: {e}"
        if os.environ.get("FLUFC_DEBUG"):
            output += "\n" + traceback.format_exc()
    if len(output) > max_chars:
        output = (output[:max_chars]
                  + f"\n... [truncated at {max_chars} characters]")
    return ToolResult(name, arguments, output=output, value=value, ok=ok,
                      seconds=time.time() - started)


# Importing the modules is what populates REGISTRY.
from . import (web, surveillance, compute, models, hub, literature,
               submit)                              # noqa: E402,F401
