"""The committed record: saving forecasts and knowing when a score is final."""

import json

import pytest

from flufc import archive
from flufc.forecast import Forecast, expand_quantiles

REFERENCE = "2026-02-07"


def band(centre, width=0.2):
    return expand_quantiles({0.025: centre * (1 - 2 * width),
                             0.25: centre * (1 - width),
                             0.5: centre,
                             0.75: centre * (1 + width),
                             0.975: centre * (1 + 2 * width)})


class FakeRun:
    stopped = "terminal"
    tool_calls = 4
    seconds = 91.5
    prompt_tokens = 1000
    completion_tokens = 200
    steps = [type("Step", (), {"calls": [("get_ili", {}), ("calculate", {})]})()]


def make(root, reference_date=REFERENCE, location="US", centres=None,
         rationale="because the trees said so", run=None):
    centres = centres or {0: 1000, 1: 1100}
    forecast = Forecast(reference_date=reference_date, location=location,
                        horizons={h: band(c) for h, c in centres.items()},
                        rationale=rationale, model_id="qwen-agent/test")
    return archive.save(root, forecast, location_name="US", run=run)


def test_save_writes_the_csv_and_its_provenance(tmp_path):
    record = make(tmp_path, run=FakeRun())

    csv_path, meta_path, score_path = archive.paths(tmp_path, REFERENCE, "US")
    assert csv_path.exists() and meta_path.exists()
    assert not score_path.exists()          # scored later, by somebody else

    meta = json.loads(meta_path.read_text())
    assert meta["medians"] == {"0": 1000.0, "1": 1100.0}
    assert meta["target_end_dates"]["1"] == "2026-02-14"
    assert meta["rationale"] == "because the trees said so"
    assert meta["run"]["tool_calls"] == 4
    assert meta["run"]["tools"] == ["get_ili", "calculate"]
    assert record["slug"] == "2026-02-07-US"


def test_the_csv_is_a_readable_submission(tmp_path):
    record = make(tmp_path)
    quantiles = archive.quantiles(record)

    assert sorted(quantiles) == [0, 1]
    assert quantiles[0][0.5] == pytest.approx(1000, rel=1e-6)
    assert len(quantiles[0]) == 23                     # the full FluSight grid


def test_records_come_back_newest_first(tmp_path):
    make(tmp_path, reference_date="2026-01-31")
    make(tmp_path, reference_date="2026-02-14")
    make(tmp_path, reference_date=REFERENCE)

    dates = [r["reference_date"] for r in archive.records(tmp_path)]
    assert dates == ["2026-02-14", "2026-02-07", "2026-01-31"]


def test_records_survive_a_stray_file(tmp_path):
    make(tmp_path)
    (tmp_path / REFERENCE / "notes.json").write_text("{ not json")
    (tmp_path / REFERENCE / "CA.json").write_text('{"reference_date": "x"}')

    # The unparseable one and the one with no CSV beside it are both skipped
    # rather than taking the whole listing down.
    assert [r["location"] for r in archive.records(tmp_path)] == ["US"]


def test_has_sees_an_existing_forecast(tmp_path):
    assert not archive.has(tmp_path, REFERENCE, "US")
    make(tmp_path)
    assert archive.has(tmp_path, REFERENCE, "US")


def test_a_score_round_trips(tmp_path):
    record = make(tmp_path)
    assert archive.read_score(record) is None

    archive.write_score(record, {"scored_horizons": [0, 1]})
    assert archive.read_score(record)["scored_horizons"] == [0, 1]


def test_a_score_from_an_older_layout_is_ignored(tmp_path):
    record = make(tmp_path)
    archive.write_score(record, {"scored_horizons": [0]})

    path = tmp_path / REFERENCE / "US.score.json"
    stale = json.loads(path.read_text()) | {"version": archive.SCORE_VERSION - 1}
    path.write_text(json.dumps(stale))

    assert archive.read_score(record) is None


ENDS = {"0": "2026-02-07", "1": "2026-02-14"}


@pytest.mark.parametrize("score, settled, why", [
    (None, False, "no score at all"),
    ({"scored_horizons": [], "target_end_dates": ENDS,
      "truth_last_date": "2026-03-01"}, False, "nothing scored yet"),
    ({"scored_horizons": [0], "unscored_horizons": [1],
      "target_end_dates": ENDS, "truth_last_date": "2026-02-07"},
     False, "one horizon still waiting"),
    ({"scored_horizons": [0, 1], "unscored_horizons": [],
      "target_end_dates": ENDS, "truth_last_date": "2026-02-14"},
     False, "truth has not moved past the last week, so it can still revise"),
    ({"scored_horizons": [0, 1], "unscored_horizons": [],
      "target_end_dates": ENDS, "truth_last_date": "2026-02-21"},
     True, "every horizon scored and the series has moved on"),
])
def test_is_settled(score, settled, why):
    assert archive.is_settled(score) is settled, why
