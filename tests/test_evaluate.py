"""Scoring. The WIS invariants are worth pinning down precisely."""

import pytest

from flufc.evaluate import covers, interval_score, wis
from flufc.forecast import QUANTILES, expand_quantiles


def test_point_forecast_scores_as_absolute_error():
    point = {q: 100.0 for q in QUANTILES}
    assert wis(130.0, point) == pytest.approx(30.0)
    assert wis(100.0, point) == pytest.approx(0.0)


def test_sharp_beats_wide_when_right_and_loses_when_wrong():
    sharp = expand_quantiles({0.025: 80, 0.25: 95, 0.5: 100, 0.75: 110, 0.975: 140})
    wide = expand_quantiles({0.025: 20, 0.25: 70, 0.5: 100, 0.75: 150, 0.975: 300})
    assert wis(100.0, sharp) < wis(100.0, wide)
    assert wis(300.0, sharp) > wis(300.0, wide)


def test_interval_score_penalises_only_misses():
    assert interval_score(50, 40, 60, 0.05) == 20            # inside: width only
    assert interval_score(70, 40, 60, 0.05) > 20             # outside: penalised


def test_coverage_flags():
    q = expand_quantiles({0.025: 80, 0.5: 100, 0.975: 140})
    assert covers(100, q, 0.025, 0.975)
    assert not covers(500, q, 0.025, 0.975)


def test_median_is_required():
    with pytest.raises(ValueError):
        wis(10.0, {0.1: 1.0, 0.9: 2.0})
