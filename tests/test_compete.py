"""The head-to-head leaderboard, driven against a fake hub."""

import pytest

from flufc import compete as compete_mod
from flufc.compete import CompetitionError, leaderboard
from flufc.forecast import Forecast, expand_quantiles
from flufc.tools import ToolError

REFERENCE = "2026-02-07"
ENDS = ["2026-02-07", "2026-02-14"]
TRUTH = {(ENDS[0], "US"): 1000.0, (ENDS[1], "US"): 1200.0}


def band(centre, width=0.2):
    """Quantiles centred on `centre`, `width` wide in relative terms."""
    return expand_quantiles({0.025: centre * (1 - 2 * width),
                             0.25: centre * (1 - width),
                             0.5: centre,
                             0.75: centre * (1 + width),
                             0.975: centre * (1 + 2 * width)})


def submission(tmp_path, name, centres, width=0.2):
    forecast = Forecast(reference_date=REFERENCE, location="US",
                        horizons={h: band(c, width) for h, c in centres.items()})
    return forecast.write_csv(tmp_path / f"{name}.csv")


@pytest.fixture
def hub(monkeypatch):
    """A fake hub: {model: {horizon: quantiles}}, plus models that fail."""
    forecasts = {
        "Sharp-Team": {0: band(1000, 0.05), 1: band(1200, 0.05)},
        "Vague-Team": {0: band(1000, 0.8), 1: band(1200, 0.8)},
        "FluSight-baseline": {0: band(900, 0.3), 1: band(900, 0.3)},
        "Half-Team": {0: band(1000, 0.05)},
        "Absent-Team": {},
    }

    def hub_models():
        return sorted(forecasts) + ["Broken-Team"]

    def hub_forecast(model, reference_date, location):
        assert (reference_date, location) == (REFERENCE, "US")
        if model == "Broken-Team":
            raise ToolError(f"no submission from {model} for {reference_date}")
        return forecasts[model]

    monkeypatch.setattr("flufc.tools.hub.hub_models", hub_models)
    monkeypatch.setattr("flufc.tools.hub.hub_forecast", hub_forecast)
    return forecasts


def test_ranks_you_among_the_teams(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050, 1: 1150})
    board = leaderboard(path, truth=TRUTH)

    order = [e["model"] for e in board["entries"]]
    assert order[0] == "Sharp-Team"                  # right and sharp wins
    assert order.index("you (mine)") < order.index("Vague-Team")
    assert board["you"]["rank"] == order.index("you (mine)") + 1
    assert board["you"]["field"] == len(order) - 1
    assert board["you"]["beat"] == sum(
        1 for e in board["entries"] if not e["is_you"]
        and e["mean_wis"] > board["you"]["mean_wis"])


def test_relative_wis_is_against_the_flat_baseline(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050, 1: 1150})
    board = leaderboard(path, truth=TRUTH)

    baseline = next(e for e in board["entries"]
                    if e["model"] == "FluSight-baseline")
    assert baseline["rel_wis"] == 1.0
    assert board["you"]["rel_wis"] == pytest.approx(
        board["you"]["mean_wis"] / baseline["mean_wis"], rel=1e-3)
    assert board["baseline_wis"] == baseline["mean_wis"]


def test_teams_missing_a_horizon_are_set_aside_not_ranked(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050, 1: 1150})
    board = leaderboard(path, truth=TRUTH)

    ranked = {e["model"] for e in board["entries"]}
    assert "Half-Team" not in ranked                 # only submitted horizon 0
    assert {p["model"] for p in board["partial"]} == {"Half-Team"}
    assert {m["model"] for m in board["missing"]} == {"Absent-Team",
                                                      "Broken-Team"}


def test_horizons_without_truth_are_dropped_from_the_comparison(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050, 1: 1150, 2: 1300})
    board = leaderboard(path, truth=TRUTH)

    assert board["scored_horizons"] == [0, 1]
    assert board["unscored_horizons"] == [2]
    assert all(set(e["wis_by_horizon"]) == {0, 1} for e in board["entries"])


def test_no_truth_at_all_returns_an_empty_board(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050})
    board = leaderboard(path, truth={})

    assert board["scored_horizons"] == []
    assert board["entries"] == []
    assert board["you"] is None
    assert board["truth_last_date"] is None


def test_the_field_can_be_narrowed(tmp_path, hub):
    path = submission(tmp_path, "mine", {0: 1050, 1: 1150})
    board = leaderboard(path, models=["Sharp-Team"], truth=TRUTH)

    assert [e["model"] for e in board["entries"]] == ["Sharp-Team",
                                                      "you (mine)"]
    assert board["you"]["rel_wis"] is None           # no baseline in the field


def test_two_locations_is_an_error(tmp_path, hub):
    path = tmp_path / "two.csv"
    rows = list(Forecast(reference_date=REFERENCE, location="US",
                         horizons={0: band(1000)}).rows())
    rows += list(Forecast(reference_date=REFERENCE, location="06",
                          horizons={0: band(100)}).rows())
    import csv
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(CompetitionError):
        leaderboard(path, truth=TRUTH)


def test_a_sharp_perfect_forecast_beats_everyone(tmp_path, hub):
    path = submission(tmp_path, "oracle", {0: 1000, 1: 1200}, width=0.01)
    board = leaderboard(path, truth=TRUTH)
    assert board["you"]["rank"] == 1
    assert board["you"]["beat"] == board["you"]["field"]
