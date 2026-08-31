"""Scoring the archive and turning it into pages, against a fake hub."""

import json

import pytest

from flufc import archive, charts, site, standings
from flufc.forecast import Forecast, expand_quantiles
from flufc.tools import ToolError

REFERENCE = "2026-02-07"
ENDS = ["2026-02-07", "2026-02-14"]
TRUTH = {(ENDS[0], "US"): 1000.0, (ENDS[1], "US"): 1200.0,
         ("2026-02-21", "US"): 1300.0}
HISTORY = [("2026-01-17", 700.0), ("2026-01-24", 820.0),
           ("2026-01-31", 910.0)]


def band(centre, width=0.2):
    return expand_quantiles({0.025: centre * (1 - 2 * width),
                             0.25: centre * (1 - width),
                             0.5: centre,
                             0.75: centre * (1 + width),
                             0.975: centre * (1 + 2 * width)})


def archived(root, reference_date=REFERENCE, centres=None, rationale="anchored"):
    centres = centres or {0: 1050, 1: 1150}
    forecast = Forecast(reference_date=reference_date, location="US",
                        horizons={h: band(c) for h, c in centres.items()},
                        rationale=rationale, model_id="qwen-agent/test")
    return archive.save(root, forecast, location_name="US")


@pytest.fixture
def hub(monkeypatch):
    """A small fake hub, and truth that stops after the second week."""
    forecasts = {
        "Sharp-Team": {0: band(1000, 0.05), 1: band(1200, 0.05)},
        "FluSight-baseline": {0: band(900, 0.3), 1: band(900, 0.3)},
        "FluSight-ensemble": {0: band(1020, 0.15), 1: band(1180, 0.15)},
        "Broken-Team": {},
    }

    def hub_forecast(model, reference_date, location):
        if model == "Broken-Team":
            raise ToolError(f"no submission from {model}")
        return forecasts[model]

    monkeypatch.setattr("flufc.tools.hub.hub_models", lambda: sorted(forecasts))
    monkeypatch.setattr("flufc.tools.hub.hub_forecast", hub_forecast)
    return forecasts


def refresh(root, **kw):
    return standings.refresh(root, truth=TRUTH, **kw)


# -- scoring ------------------------------------------------------------------

def test_refresh_scores_and_ranks(tmp_path, hub):
    archived(tmp_path)
    (record,) = refresh(tmp_path)

    score = record["score"]
    assert score["scored_horizons"] == [0, 1]
    # Sharp-Team and the ensemble both land closer; the flat baseline does not.
    assert score["you"]["rank"] == 3
    assert score["you"]["field"] == 3
    assert score["you"]["beat"] == 1
    assert score["you"]["rel_wis"] < 1               # still beat doing nothing
    assert score["vs_ensemble"] > 1                  # but not the ensemble
    assert score["hub_medians"]["FluSight-ensemble"]["0"] == pytest.approx(1020)


def test_the_score_is_written_beside_the_forecast(tmp_path, hub):
    archived(tmp_path)
    refresh(tmp_path)

    written = json.loads((tmp_path / REFERENCE / "US.score.json").read_text())
    assert written["version"] == archive.SCORE_VERSION
    assert written["you"]["rank"] == 3


def test_a_settled_score_is_not_recomputed(tmp_path, hub, monkeypatch):
    archived(tmp_path)
    refresh(tmp_path)                                 # settles: truth runs past

    calls = []
    monkeypatch.setattr("flufc.standings.leaderboard",
                        lambda *a, **kw: calls.append(a) or {})
    (record,) = refresh(tmp_path)

    assert calls == [], "a week that is over should never be scored twice"
    assert record["score"]["you"]["rank"] == 3


def test_an_unsettled_score_is_recomputed(tmp_path, hub):
    # Truth reaches 2026-02-21, so a forecast whose last week is 2026-02-21 is
    # still open to revision and has to be scored again every time.
    archived(tmp_path, reference_date="2026-02-14", centres={0: 1200, 1: 1300})
    events = []
    refresh(tmp_path, on_event=lambda e, **kw: events.append(e))
    refresh(tmp_path, on_event=lambda e, **kw: events.append(e))

    assert events.count("scored") == 2


def test_a_forecast_with_no_truth_yet_is_pending_not_broken(tmp_path, hub):
    archived(tmp_path, reference_date="2026-03-07", centres={0: 1400, 1: 1500})
    (record,) = refresh(tmp_path)

    assert record["score"]["you"] is None
    assert record["score"]["scored_horizons"] == []
    assert standings.summary([record])["n_pending"] == 1


def test_a_network_failure_keeps_the_previous_score(tmp_path, hub, monkeypatch):
    archived(tmp_path, reference_date="2026-02-14", centres={0: 1200, 1: 1300})
    refresh(tmp_path)

    def boom(*a, **kw):
        raise OSError("the hub is down")

    monkeypatch.setattr("flufc.standings.leaderboard", boom)
    (record,) = refresh(tmp_path)

    assert "the hub is down" in record["score_error"]
    assert record["score"]["you"]["rank"] is not None, "kept the last good score"


def test_summary_counts_the_whole_record(tmp_path, hub):
    archived(tmp_path)
    archived(tmp_path, reference_date="2026-03-07", centres={0: 1400, 1: 1500})
    totals = standings.summary(refresh(tmp_path))

    assert totals == {"n_forecasts": 2, "n_scored": 1, "n_pending": 1,
                      "mean_rel_wis": 0.553, "beat_baseline": 1,
                      "mean_rank": 3.0, "mean_field": 3.0,
                      "beat_share": pytest.approx(1 / 3, rel=1e-2),
                      "beat_ensemble": 0, "n_vs_ensemble": 1}


# -- the pages ----------------------------------------------------------------

def build(root, out, **kw):
    return site.build(root, out, records=refresh(root),
                      history_fn=lambda location: HISTORY, **kw)


def test_build_writes_a_page_for_every_forecast(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    archived(root, reference_date="2026-02-14", centres={0: 1200, 1: 1300})
    build(root, out)

    assert (out / "index.html").exists()
    assert (out / "style.css").exists()
    assert (out / ".nojekyll").exists()               # or Pages hides f/
    assert (out / "f" / "2026-02-07-US.html").exists()
    assert (out / "f" / "2026-02-14-US.html").exists()


def test_the_index_says_where_the_forecast_stands(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    build(root, out)
    page = (out / "index.html").read_text()

    assert "Ranked <b>3 of 4</b> teams" in page
    assert "the flat baseline" in page
    assert "anchored" in page, "the model's reasoning belongs on the page"
    assert "<svg" in page


def test_a_pending_forecast_also_shows_the_last_result(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    archived(root, reference_date="2026-03-07", centres={0: 1400, 1: 1500})
    build(root, out)
    page = (out / "index.html").read_text()

    assert "Waiting on truth" in page
    assert "most recent result" in page, \
        "a page whose only card is pending tells a visitor nothing"


def test_the_detail_page_carries_the_whole_field(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    build(root, out)
    page = (out / "f" / "2026-02-07-US.html").read_text()

    for model in ("Sharp-Team", "FluSight-baseline", "FluSight-ensemble"):
        assert model in page
    assert "Broken-Team" not in page, "a team that did not submit is not ranked"
    assert "The submission" in page and "97.5%" in page


def test_the_pages_escape_what_the_model_wrote(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root, rationale="growth <script>alert(1)</script> & 5 > 3")
    build(root, out)
    page = (out / "f" / "2026-02-07-US.html").read_text()

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page and "5 &gt; 3" in page


def test_the_feed_is_machine_readable(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    build(root, out)
    feed = json.loads((out / "standings.json").read_text())

    assert feed["summary"]["n_scored"] == 1
    assert feed["forecasts"][0]["score"]["you"]["rank"] == 3


def test_an_empty_archive_still_builds(tmp_path):
    out = tmp_path / "docs"
    site.build(tmp_path / "forecasts", out, records=[])

    assert "Nothing forecast yet" in (out / "index.html").read_text()


def test_a_missing_history_does_not_stop_the_build(tmp_path, hub):
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)

    def no_history(location):
        raise OSError("NHSN is down")

    site.build(root, out, records=refresh(root), history_fn=no_history)
    assert (out / "index.html").exists()


# -- the charts ---------------------------------------------------------------

def test_the_fan_places_weeks_on_a_date_scale():
    """A gap in the series has to show as a gap, not be closed up."""
    even = charts.fan([("2026-01-03", 10.0), ("2026-01-10", 20.0)],
                      {"2026-01-17": band(30)})
    gapped = charts.fan([("2026-01-03", 10.0), ("2026-01-10", 20.0)],
                        {"2026-03-07": band(30)})

    assert "2026" not in even                       # labels are short dates
    assert even != gapped
    assert "17 Jan" in even and "7 Mar" in gapped


def test_the_fan_needs_two_points():
    assert charts.fan([], {}) == ""


def test_the_track_chart_marks_the_baseline():
    svg = charts.track([{"rel_wis": 0.8, "label": "a week", "short": "02-07"},
                        {"rel_wis": 1.4, "label": "another", "short": "02-14"}])

    assert "bar-good" in svg and "bar-bad" in svg
    assert "1x" in svg
    assert charts.track([]) == ""


def test_a_rebuild_that_changes_nothing_leaves_the_files_alone(tmp_path, hub):
    """Otherwise the nightly job commits a clock every day."""
    root, out = tmp_path / "forecasts", tmp_path / "docs"
    archived(root)
    build(root, out)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}

    build(root, out)
    after = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert before == after

    # ...but a real change still lands.
    archived(root, reference_date="2026-03-07", centres={0: 1400, 1: 1500})
    build(root, out)
    assert (out / "index.html").stat().st_mtime_ns != before[out / "index.html"]
