"""The terminal tool: how a forecasting run ends.

Making the answer a tool call rather than free text buys two things. The loop
knows unambiguously when it is finished, and a malformed forecast comes back to
the model as a tool error it can fix on the next step -- which is exactly the
behaviour you want, since the failure mode of an LLM forecaster is not bad
judgement so much as quantiles that do not parse.
"""

from . import ToolError, tool
from ..forecast import ANCHOR_QUANTILES, ForecastError, from_anchors

EXAMPLE = ('{"0": {"0.025": 480, "0.25": 700, "0.5": 820, "0.75": 960, '
           '"0.975": 1300}, "1": {...}}')


@tool(terminal=True)
def submit_forecast(location: str, reference_date: str, quantiles: dict,
                    rationale: str) -> object:
    """Submit the final forecast. Call this exactly once, when you are done.

    Args:
        location: FluSight location code, e.g. 'US' or '06'.
        reference_date: Saturday ending the reference week, as YYYY-MM-DD.
        quantiles: JSON object mapping horizon to quantile levels, e.g.
            {"0": {"0.025": 480, "0.25": 700, "0.5": 820, "0.75": 960,
            "0.975": 1300}, "1": {...}, "2": {...}, "3": {...}}. Horizons are
            weeks ahead of the reference date. Give every horizon the levels
            0.025, 0.25, 0.5, 0.75 and 0.975.
        rationale: A few sentences on what drove the forecast: the recent
            trajectory, what the evidence says, and why the interval is as wide
            as it is.
    """
    if not isinstance(quantiles, dict) or not quantiles:
        raise ToolError(
            f"quantiles must be a JSON object of horizons, e.g. {EXAMPLE}")

    anchors = {}
    for horizon, levels in quantiles.items():
        try:
            key = int(str(horizon).strip())
        except ValueError:
            raise ToolError(f"horizon {horizon!r} is not an integer")
        if not isinstance(levels, dict):
            raise ToolError(
                f"horizon {key}: expected an object of quantile levels, "
                f"got {type(levels).__name__}")
        parsed = {}
        for level, value in levels.items():
            try:
                parsed[float(level)] = float(value)
            except (TypeError, ValueError):
                raise ToolError(
                    f"horizon {key}: {level!r}: {value!r} is not a number")
        missing = [q for q in ANCHOR_QUANTILES if q not in parsed]
        if missing:
            raise ToolError(
                f"horizon {key} is missing quantile level(s) "
                f"{', '.join(str(q) for q in missing)}. Every horizon needs "
                f"{', '.join(str(q) for q in ANCHOR_QUANTILES)}.")
        anchors[key] = parsed

    try:
        forecast = from_anchors(reference_date, location, anchors,
                                rationale=rationale)
    except ForecastError as e:
        raise ToolError(f"forecast rejected: {e}")
    except ValueError as e:
        raise ToolError(f"reference_date must be YYYY-MM-DD ({e})")
    return forecast
