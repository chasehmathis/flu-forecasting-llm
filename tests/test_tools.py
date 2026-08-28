"""Registry mechanics and the tools that do not need the network."""

import pytest

from flufc import tools


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


def test_submit_forecast_rejects_incomplete_quantiles():
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": '{"0": {"0.5": 800}}', "rationale": "flat"})
    assert not result.ok and "missing quantile" in result.output


def test_submit_forecast_returns_a_forecast():
    levels = {"0.025": 500, "0.25": 700, "0.5": 800, "0.75": 900, "0.975": 1200}
    result = tools.call("submit_forecast", {
        "location": "US", "reference_date": "2026-08-22",
        "quantiles": '{"0": %s, "1": %s}' % (str(levels).replace("'", '"'),
                                             str(levels).replace("'", '"')),
        "rationale": "flat season"})
    assert result.ok, result.output
    assert result.value.median(0) == 800
    assert len(list(result.value.rows())) == 46
