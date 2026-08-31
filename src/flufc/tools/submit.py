"""The terminal tool: how a forecasting run ends.

Making the answer a tool call rather than free text buys two things. The loop
knows unambiguously when it is finished, and a malformed forecast comes back to
the model as a tool error it can fix on the next step -- which is exactly the
behaviour you want, since the failure mode of an LLM forecaster is not bad
judgement so much as quantiles that do not parse.

The model states all 23 FluSight levels itself. That is 23 numbers per horizon
rather than five, so the tool takes them either as an object keyed by level or,
much cheaper in tokens, as a bare list of 23 values in grid order.
"""

from . import ToolError, tool
from ..forecast import ForecastError, QUANTILES, from_quantiles

GRID = ", ".join(str(q) for q in QUANTILES)
EXAMPLE = ('{"0": [610, 640, 668, 702, 725, 744, 761, 777, 792, 806, 820, 834, '
           '848, 863, 879, 897, 917, 941, 970, 1010, 1070, 1120, 1190], '
           '"1": [...], "2": [...], "3": [...]}')


def _parse_levels(key, levels):
    """One horizon's quantiles, as a list in grid order or a level->value map."""
    if isinstance(levels, (list, tuple)):
        if len(levels) != len(QUANTILES):
            raise ToolError(
                f"horizon {key}: a list must have exactly {len(QUANTILES)} "
                f"values, one per level in the order {GRID} -- got "
                f"{len(levels)}")
        pairs = zip(QUANTILES, levels)
    elif isinstance(levels, dict):
        pairs = []
        for level, value in levels.items():
            try:
                pairs.append((float(level), value))
            except (TypeError, ValueError):
                raise ToolError(f"horizon {key}: {level!r} is not a quantile "
                                f"level. The levels are {GRID}.")
    else:
        raise ToolError(
            f"horizon {key}: expected {len(QUANTILES)} values as a list or an "
            f"object of quantile levels, got {type(levels).__name__}")

    parsed = {}
    for level, value in pairs:
        try:
            parsed[level] = float(value)
        except (TypeError, ValueError):
            raise ToolError(
                f"horizon {key}: {level}: {value!r} is not a number")
    return parsed


@tool(terminal=True)
def submit_forecast(location: str, reference_date: str, quantiles: dict,
                    rationale: str) -> object:
    """Submit the final forecast. Call this exactly once, when you are done.

    Args:
        location: FluSight location code, e.g. 'US' or '06'.
        reference_date: Saturday ending the reference week, as YYYY-MM-DD.
        quantiles: JSON object mapping each horizon to its full predictive
            distribution: all 23 FluSight quantile levels, in this order --
            0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99.
            Give each horizon a list of 23 numbers in exactly that order (much
            shorter to write), or an object keyed by level. The values must not
            decrease across the list. Example: {"0": [610, 640, 668, 702, 725,
            744, 761, 777, 792, 806, 820, 834, 848, 863, 879, 897, 917, 941,
            970, 1010, 1070, 1120, 1190], "1": [...], "2": [...], "3": [...]}.
        rationale: A few sentences on what drove the forecast: the recent
            trajectory, what the evidence says, and why the interval is as wide
            as it is.
    """
    if not isinstance(quantiles, dict) or not quantiles:
        raise ToolError(
            f"quantiles must be a JSON object of horizons, e.g. {EXAMPLE}")

    grids = {}
    for horizon, levels in quantiles.items():
        try:
            key = int(str(horizon).strip())
        except ValueError:
            raise ToolError(f"horizon {horizon!r} is not an integer")
        grids[key] = _parse_levels(key, levels)

    try:
        forecast = from_quantiles(reference_date, location, grids,
                                  rationale=rationale)
    except ForecastError as e:
        raise ToolError(f"forecast rejected: {e}")
    except ValueError as e:
        raise ToolError(f"reference_date must be YYYY-MM-DD ({e})")
    return forecast
