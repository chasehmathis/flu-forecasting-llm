"""The tools added for anchoring: statistical models, the hub, the literature.

Nothing here touches the network -- fetches are replaced with fixed payloads,
so what is under test is the parsing and the presentation, which is where the
mistakes are.
"""

import datetime as dt

import pytest

from flufc import tools
from flufc.tools import hub, literature, models, surveillance

US = {"abbreviation": "US", "location": "US", "location_name": "US",
      "population": 340_110_988}

HUB_CSV = """reference_date,location,horizon,target,target_end_date,output_type,output_type_id,value
2026-01-10,US,0,wk inc flu hosp,2026-01-10,quantile,0.025,8000
2026-01-10,US,0,wk inc flu hosp,2026-01-10,quantile,0.5,10000
2026-01-10,US,0,wk inc flu hosp,2026-01-10,quantile,0.975,13000
2026-01-10,US,1,wk inc flu hosp,2026-01-17,quantile,0.5,11000
2026-01-10,06,0,wk inc flu hosp,2026-01-10,quantile,0.5,900
2026-01-10,US,0,wk flu hosp rate category,2026-01-10,pmf,large,0.2
"""


def test_toolsets_only_name_registered_tools():
    for name, bundle in tools.TOOLSETS.items():
        unknown = [t for t in bundle if t not in tools.REGISTRY]
        assert not unknown, f"{name} names missing tools: {unknown}"
    assert "submit_forecast" not in tools.TOOLSETS["research"]
    assert "fit_forecast_model" in tools.TOOLSETS["forecast"]
    assert "search_web" not in tools.TOOLSETS["quant"]


def test_horizons_are_parsed_leniently_but_validated():
    assert models.parse_horizons("0,1,2,3") == [0, 1, 2, 3]
    assert models.parse_horizons("3 1 1") == [1, 3]
    assert models.parse_horizons("") == [0, 1, 2, 3]
    with pytest.raises(tools.ToolError):
        models.parse_horizons("0,9")
    with pytest.raises(tools.ToolError):
        models.parse_horizons("soon")


def test_horizons_become_steps_beyond_the_last_reported_week():
    series = [("2026-01-03", 500.0), ("2026-01-10", 600.0)]
    mapping, observed = models._steps(series, "2026-01-17", [-1, 0, 1, 2])
    assert observed[-1] == ("2026-01-10", 600.0)      # already reported
    assert mapping[0] == (1, "2026-01-17")            # data lags by a week
    assert mapping[2] == (3, "2026-01-31")


def test_hub_forecast_keeps_one_location_and_the_quantile_target(monkeypatch):
    monkeypatch.setattr(hub, "fetch", lambda *a, **k: HUB_CSV)
    parsed = hub.hub_forecast("FluSight-ensemble", "2026-01-10", "US")
    assert parsed[0][0.5] == 10000 and parsed[1][0.5] == 11000
    assert 0.2 not in parsed[0].values()              # the pmf rows are dropped
    assert hub.hub_forecast("FluSight-ensemble", "2026-01-10", "06")[0][0.5] == 900


def test_get_hub_forecasts_reports_the_median_across_models(monkeypatch):
    monkeypatch.setattr(surveillance, "resolve_location", lambda text: US)
    payloads = {"FluSight-ensemble": HUB_CSV,
                "FluSight-baseline": HUB_CSV.replace("10000", "14000")}
    monkeypatch.setattr(hub, "fetch",
                        lambda url, **k: payloads[url.split("/")[-2]])
    result = tools.call("get_hub_forecasts",
                        {"location": "US", "reference_date": "2026-01-10"},
                        max_chars=20000)
    assert result.ok, result.output
    assert "10000" in result.output and "14000" in result.output
    assert "12000" in result.output                   # median of the two


def test_a_week_with_no_submissions_says_so(monkeypatch):
    monkeypatch.setattr(surveillance, "resolve_location", lambda text: US)

    def missing(*a, **k):
        raise hub.FetchError("HTTP 404 from raw.githubusercontent.com")
    monkeypatch.setattr(hub, "fetch", missing)
    monkeypatch.setattr(hub, "_nearest_dates", lambda *a, **k: ["2026-05-30"])
    result = tools.call("get_hub_forecasts",
                        {"location": "US", "reference_date": "2026-08-22"},
                        max_chars=20000)
    assert result.ok
    assert "no submission" in result.output and "2026-05-30" in result.output


def test_literature_search_formats_records(monkeypatch):
    search = {"esearchresult": {"count": "2", "idlist": ["111", "222"]}}
    summary = {"result": {
        "111": {"title": "Subclade K and severity", "source": "Nat Commun",
                "pubdate": "2026 Jan 2", "authors": [{"name": "Doe J"},
                                                     {"name": "Roe A"}],
                "articleids": [{"idtype": "doi", "value": "10.1000/x"}]},
        "222": {"title": "Forecast evaluation", "source": "Epidemics",
                "pubdate": "2025", "authors": [{"name": "Poe R"}],
                "articleids": []}}}
    calls = iter([search, summary])
    monkeypatch.setattr(literature, "fetch_json", lambda *a, **k: next(calls))
    result = tools.call("search_literature", {"query": "subclade K"},
                        max_chars=20000)
    assert result.ok, result.output
    assert "PMID 111" in result.output and "Doe J et al." in result.output
    assert "doi:10.1000/x" in result.output


def test_read_paper_rejects_anything_that_is_not_a_pmid():
    result = tools.call("read_paper", {"pmid": "the H3N2 one"})
    assert not result.ok and "not a PMID" in result.output
