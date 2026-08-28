"""The statistical models: do they fit what they claim to, and stay honest?"""

import datetime as dt

import pytest

from flufc import stats


def make_series(values, end="2026-08-22"):
    """(dates, values) ending on `end`, weekly, oldest first."""
    last = dt.date.fromisoformat(end)
    dates = [(last - dt.timedelta(days=7 * (len(values) - 1 - i))).isoformat()
             for i in range(len(values))]
    return dates, [float(v) for v in values]


def test_persistence_carries_the_last_week_forward():
    dates, values = make_series([100] * 30 + [140])
    assert stats._persistence(dates, values, [1, 3]) == {1: 140.0, 3: 140.0}


def test_loglinear_follows_exponential_growth_and_damps_it():
    dates, values = make_series([100 * 1.3 ** i for i in range(12)])
    points = stats._loglinear(dates, values, [1, 4])
    assert points[1] == pytest.approx(values[-1] * 1.3, rel=0.05)
    # Damped: four weeks of 30% growth compounded would be 2.9x, not 2.6x.
    assert points[4] < values[-1] * 1.3 ** 4


def test_loglinear_refuses_to_forecast_a_collapse_or_an_explosion():
    dates, values = make_series([1, 10, 100, 1000, 10000, 100000] * 2)
    rate = stats._loglinear(dates, values, [1])[1] / values[-1]
    assert rate <= 1.5                       # the weekly log-rate cap bites


def test_seasonal_reads_the_same_weeks_in_previous_years():
    # Flat at 100 except a spike 52 weeks (364 days) back, two weeks ahead.
    values = [100.0] * 120
    values[-52 + 2 - 1] = 300.0
    dates, values = make_series(values)
    points = stats._seasonal(dates, values, [1, 2])
    assert points[2] > points[1]             # last year, week +2 tripled
    assert points[1] == pytest.approx(100, abs=5)


def test_quantiles_are_ordered_and_widen_with_horizon():
    dates, values = make_series([100 + 5 * (i % 7) for i in range(80)])
    fit = stats.predict(dates, values, [1, 2, 3], method="loglinear", folds=6)
    for step in (1, 2, 3):
        q = fit.quantiles(step)
        assert sorted(q.values()) == [q[level] for level in sorted(q)]
        assert q[0.025] >= 0
    # Relative to the median, since the absolute width scales with the level.
    width = {s: ((fit.quantiles(s)[0.975] - fit.quantiles(s)[0.025])
                 / fit.quantiles(s)[0.5]) for s in (1, 2, 3)}
    assert width[1] < width[2] < width[3]     # spread is kept monotone


def test_backtest_never_sees_the_week_it_is_predicting():
    dates, values = make_series(list(range(1, 61)))
    pairs = stats.walk_forward(dates, values, [2], "persistence", folds=5)[2]
    # Persistence at origin t predicts values[t] for values[t + 2].
    assert all(actual - predicted == 2 for actual, predicted in pairs)
    assert len(pairs) == 5


def test_skill_is_measured_against_carrying_the_last_week_forward():
    dates, values = make_series([100 + 3 * i for i in range(60)])
    fit = stats.predict(dates, values, [1], method="linear", folds=6)
    skill = fit.skill[1]
    assert skill["mae"] < skill["mae_persistence"]      # a trend line should win
    assert skill["skill_vs_persistence"] > 0


def test_boosted_trees_learn_a_split():
    rows = [[x, 0.0] for x in range(-50, 50)]
    targets = [1.0 if row[0] > 0 else -1.0 for row in rows]
    model = stats._fit_gbt(rows, targets, rounds=60, min_leaf=4)
    assert stats._predict_gbt(model, [30.0, 0.0]) > 0.4
    assert stats._predict_gbt(model, [-30.0, 0.0]) < -0.4


def test_gbt_can_forecast_above_every_level_it_was_trained_on():
    """It predicts growth rates, so a rising series is not capped at the max."""
    values = []
    for _ in range(6):                        # six seasons of the same shape
        values += [50] * 30 + [80, 140, 260, 420, 300, 160, 90] + [50] * 15
    dates, values = make_series(values)
    fit = stats.predict(dates, values, [1, 2], method="gbt", folds=4)
    assert fit.point(1) > 0


def test_unusable_requests_are_rejected():
    dates, values = make_series([100] * 30)
    with pytest.raises(stats.StatsError):
        stats.predict(dates, values, [1], method="prophecy")
    with pytest.raises(stats.StatsError):
        stats.predict(dates[:5], values[:5], [1])
    with pytest.raises(stats.StatsError):
        stats.predict(dates, values, [0])
