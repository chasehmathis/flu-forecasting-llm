"""Arithmetic for the model.

Language models are unreliable at multi-step arithmetic and very reliable at
deciding what arithmetic to do, so give them a calculator rather than trusting
mental math about growth rates and ratios. This evaluates an expression from a
whitelisted AST -- no name lookup, no attribute access, no calls to anything
but the functions below.
"""

import ast
import math
import operator
import statistics

from . import ToolError, tool

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "sorted": sorted,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "mean": statistics.fmean, "median": statistics.median,
    "stdev": lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0,
    "quantile": lambda xs, q: _quantile(xs, q),
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _quantile(values, q):
    """Type-7 quantile, the one R and numpy use by default."""
    ordered = sorted(values)
    if not ordered:
        raise ToolError("quantile() needs at least one value")
    position = (len(ordered) - 1) * float(q)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"only numbers are allowed, not {node.value!r}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS:
            raise ToolError(f"unknown function {name!r}. Available: "
                            f"{', '.join(sorted(_FUNCTIONS))}")
        return _FUNCTIONS[name](*[_eval(a) for a in node.args])
    raise ToolError(f"expression element not allowed: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / // % **, lists, and the functions abs, round, min, max,
    sum, len, sorted, sqrt, exp, log, log10, floor, ceil, mean, median, stdev
    and quantile(values, q). Use it for growth rates, ratios and percentiles
    instead of doing them in your head.

    Args:
        expression: e.g. "mean([412, 455, 501]) * 1.15" or "501 / 455 - 1".
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"could not parse {expression!r}: {e}")
    try:
        value = _eval(tree)
    except ZeroDivisionError:
        raise ToolError("division by zero")
    except (TypeError, ValueError, statistics.StatisticsError) as e:
        raise ToolError(f"{type(e).__name__}: {e}")
    if isinstance(value, float):
        return f"{expression.strip()} = {value:.6g}"
    return f"{expression.strip()} = {value}"
