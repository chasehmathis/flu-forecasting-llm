"""Registry mechanics and the tools that do not need the network."""

import json
import math
from statistics import NormalDist

import pytest

from flufc import tools
from flufc.forecast import QUANTILES


def test_schema_is_built_from_the_signature():
    schema = tools.REGISTRY["get_flu_hospitalizations"].parameters
    assert schema["properties"]["weeks"]["type"] == "integer"
    assert schema["required"] == ["location"]          # weeks has a default
    assert "FIPS" in schema["properties"]["location"]["description"]


def test_string_arguments_are_coerced_to_declared_types():
    entry = tools.REGISTRY["get_flu_hospitalizations"]
    assert tools.coerce(entry, {"location": "US", "weeks": "12"})["weeks"] == 12


def test_missing_required_argument_is_reported_not_raised():
    result = tools.call("get_flu_hospitalizations", {})
    assert not result.ok and "missing required" in result.output


def test_unknown_tool_lists_the_real_ones():
    result = tools.call("summon_oracle", {})
    assert not result.ok and "calculate" in result.output


def test_calculator_evaluates_and_refuses_code():
    assert "524.4" in tools.call(
        "calculate", {"expression": "mean([412,455,501]) * 1.15"}).output
    assert not tools.call(
        "calculate", {"expression": "__import__('os').listdir('.')"}).ok
    assert not tools.call("calculate", {"expression": "1/0"}).ok


def test_output_is_truncated_to_protect_the_context():
    result = tools.call("calculate", {"expression": "1"}, max_chars=4)
    assert "truncated" in result.output


def test_a_continuation_line_is_not_mistaken_for_an_argument():
    """"word:" inside a description used to silently truncate it."""
    def fake(alpha: str, beta: str) -> str:
        """Does a thing.

        Args:
            alpha: first, which is a distribution: of some kind,
                and note: it keeps going.
            beta: second.
        """
    _, args = tools._parse_docstring(fake.__doc__, ("alpha", "beta"))
    assert set(args) == {"alpha", "beta"}
    assert args["alpha"].endswith("it keeps going.")


def test_the_submit_schema_carries_the_whole_grid():
    schema = tools.REGISTRY["submit_forecast"].parameters
    description = schema["properties"]["quantiles"]["description"]
    assert "0.99" in description and "23" in description


def _grid(centre=800.0, width=0.35):
    """A plausible 23-value grid, the way the model is asked to write one."""
    return [round(centre * math.exp(width * NormalDist().inv_cdf(q)), 1)
            for q in QUANTILES]


def test_submit_forecast_rejects_a_partial_grid():
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": '{"0": {"0.5": 800}}', "rationale": "flat"})
    assert not result.ok
    assert "missing 22 of 23 quantile levels" in result.output


def test_submit_forecast_rejects_a_list_of_the_wrong_length():
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": '{"0": [1, 2, 3, 4, 5]}', "rationale": "flat"})
    assert not result.ok
    assert "exactly 23 values" in result.output


def test_submit_forecast_takes_a_list_in_grid_order():
    grid = _grid()
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": json.dumps({"0": grid, "1": grid}),
        "rationale": "flat season"})
    assert result.ok, result.output
    assert result.value.median(0) == 800
    assert len(list(result.value.rows())) == 46


def test_submit_forecast_takes_an_object_keyed_by_level():
    levels = {str(q): v for q, v in zip(QUANTILES, _grid())}
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": json.dumps({"0": levels}), "rationale": "flat season"})
    assert result.ok, result.output
    assert result.value.median(0) == 800


def test_submit_forecast_keeps_a_shape_the_anchor_path_would_have_lost():
    """The point of 23 levels: an asymmetric tail reaches the CSV intact."""
    grid = _grid()
    grid[-1] = 4000.0                       # a long, deliberate upper tail
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": json.dumps({"0": grid}), "rationale": "inflection risk"})
    assert result.ok, result.output
    assert result.value.horizons[0][0.99] == 4000.0


def test_submit_forecast_repairs_crossed_neighbours():
    grid = _grid()
    grid[11], grid[12] = grid[12], grid[11]     # 0.5 and 0.55 swapped
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": json.dumps({"0": grid}), "rationale": "slip"})
    assert result.ok, result.output
    values = [result.value.horizons[0][q] for q in QUANTILES]
    assert values == sorted(values)


def test_submit_forecast_rejects_levels_off_the_grid():
    levels = {str(q): v for q, v in zip(QUANTILES, _grid())}
    levels["0.333"] = 750
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": json.dumps({"0": levels}), "rationale": "flat"})
    assert not result.ok and "not on the FluSight grid" in result.output
