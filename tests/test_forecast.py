"""Quantile expansion and the FluSight submission shape."""

import pytest

from flufc.forecast import (ForecastError, QUANTILES, expand_quantiles,
                            from_anchors, most_recent_saturday)


def test_expansion_is_monotone_and_hits_its_anchors():
    anchors = {0.025: 500, 0.25: 800, 0.5: 950, 0.75: 1150, 0.975: 1600}
    q = expand_quantiles(anchors)
    assert set(q) == set(QUANTILES)
    values = [q[level] for level in sorted(q)]
    assert values == sorted(values)
    for level, value in anchors.items():
        assert q[level] == pytest.approx(value, abs=1e-6)


def test_tails_extend_beyond_the_anchors():
    q = expand_quantiles({0.025: 500, 0.5: 950, 0.975: 1600})
    assert q[0.01] < q[0.025] and q[0.99] > q[0.975]


def test_quantile_crossings_are_repaired_not_rejected():
    q = expand_quantiles({0.025: 500, 0.5: 400, 0.975: 1600})   # median too low
    values = [q[level] for level in sorted(q)]
    assert values == sorted(values)


def test_forecast_is_clamped_at_zero():
    q = expand_quantiles({0.025: 1, 0.5: 3, 0.975: 9})
    assert q[0.01] >= 0


def test_needs_more_than_one_quantile():
    with pytest.raises(ForecastError):
        expand_quantiles({0.5: 900})


def test_submission_rows_and_dates():
    f = from_anchors("2026-08-22", "US",
                     {0: {0.025: 500, 0.5: 800, 0.975: 1200},
                      3: {0.025: 400, 0.5: 900, 0.975: 2000}})
    rows = list(f.rows())
    assert len(rows) == 2 * len(QUANTILES)
    assert {r["output_type"] for r in rows} == {"quantile"}
    horizon_3 = [r for r in rows if r["horizon"] == 3][0]
    assert horizon_3["target_end_date"] == "2026-09-12"


def test_horizon_outside_flusight_range_is_rejected():
    with pytest.raises(ForecastError):
        from_anchors("2026-08-22", "US", {9: {0.025: 1, 0.5: 2, 0.975: 3}})


def test_reference_date_is_a_saturday():
    import datetime as dt
    assert dt.date.fromisoformat(most_recent_saturday()).weekday() == 5
