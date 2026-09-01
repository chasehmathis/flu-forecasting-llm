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
_COMPARE = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}

# A comprehension iterates a literal list, so its length is already bounded by
# what the model was willing to type. This is the backstop against the short
# expressions that are expensive to run -- [0] * 10**9 and its relatives.
MAX_ITEMS = 1024

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


def _eval(node, env=None):
    env = env or {}
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"only numbers are allowed, not {node.value!r}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left, env), _eval(node.right, env)
        # [0] * n allocates before anything downstream could reject it, so the
        # length has to be checked before the multiply, not after.
        if type(node.op) is ast.Mult:
            for a, b in ((left, right), (right, left)):
                if isinstance(a, list) and isinstance(b, (int, float)) \
                        and len(a) * b > MAX_ITEMS:
                    raise ToolError(f"that would build a list of "
                                    f"{int(len(a) * b)} items, over the "
                                    f"{MAX_ITEMS} limit")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 \
            and type(node.ops[0]) in _COMPARE:
        return _COMPARE[type(node.ops[0])](_eval(node.left, env),
                                           _eval(node.comparators[0], env))
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return _comprehension(node, env)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS:
            raise ToolError(f"unknown function {name!r}. Available: "
                            f"{', '.join(sorted(_FUNCTIONS))}")
        return _FUNCTIONS[name](*[_eval(a, env) for a in node.args])
    raise ToolError(f"expression element not allowed: {type(node).__name__}")


def _comprehension(node, env):
    """[f(x) for x in xs if p(x)] -- one generator, over an already-safe list.

    A forecast is 23 quantiles per horizon, and writing them as 23 separate
    calls costs the agent more steps than it has. The comprehension is the
    natural way to say it once, so it is worth supporting; the restriction to
    a single plain loop variable keeps the evaluator as closed as it was.
    """
    if len(node.generators) != 1:
        raise ToolError("only one 'for' clause is supported")
    gen = node.generators[0]
    if gen.is_async:
        raise ToolError("'async for' is not supported")
    if not isinstance(gen.target, ast.Name):
        raise ToolError("the loop variable must be a plain name, as in "
                        "[x * 2 for x in [1, 2, 3]]")

    items = _eval(gen.iter, env)
    if not isinstance(items, list):
        raise ToolError("the 'in' part must be a list, as in "
                        "[x * 2 for x in [1, 2, 3]]")
    if len(items) > MAX_ITEMS:
        raise ToolError(f"too many items to iterate: {len(items)} > {MAX_ITEMS}")

    out = []
    for item in items:
        scope = dict(env, **{gen.target.id: item})
        if all(_eval(cond, scope) for cond in gen.ifs):
            out.append(_eval(node.elt, scope))
    return out


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / // % **, comparisons, lists, list comprehensions, and the
    functions abs, round, min, max, sum, len, sorted, sqrt, exp, log, log10,
    floor, ceil, mean, median, stdev and quantile(values, q). Use it for growth
    rates, ratios and percentiles instead of doing them in your head.

    A comprehension is the cheap way to build a whole quantile grid in one
    call, rather than spending a step on each of the 23 levels:

        [round(830 * exp(z * 0.166)) for z in [-1.96, -1.28, 0, 1.28, 1.96]]

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
