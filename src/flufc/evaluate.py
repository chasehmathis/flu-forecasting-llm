"""Scoring forecasts against what actually happened.

The weighted interval score is what FluSight ranks on. It rewards sharp
intervals and punishes ones that miss, and it reduces to absolute error for a
point forecast -- which makes it directly comparable across models that express
different amounts of uncertainty.
"""

import csv
import io
from collections import defaultdict

from .cache import fetch
from .forecast import QUANTILES

TRUTH_URL = ("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub"
             "/main/target-data/target-hospital-admissions.csv")


def load_truth(cache_dir=None, ttl=86400):
    """{(date, location): admissions} from the FluSight hub's target data."""
    body = fetch(TRUTH_URL, timeout=90, ttl=ttl, cache_dir=cache_dir,
                 max_bytes=8_000_000)
    truth = {}
    for row in csv.DictReader(io.StringIO(body)):
        try:
            truth[(row["date"], row["location"])] = float(row["value"])
        except (KeyError, ValueError):
            continue
    return truth


def interval_score(observed, lower, upper, alpha):
    """Interval score for a central (1-alpha) prediction interval."""
    score = upper - lower
    if observed < lower:
        score += (2.0 / alpha) * (lower - observed)
    elif observed > upper:
        score += (2.0 / alpha) * (observed - upper)
    return score


def wis(observed, quantiles):
    """Weighted interval score. Equals |observed - median| for a point forecast."""
    levels = sorted(quantiles)
    pairs = [(lo, hi) for lo, hi in zip(levels, reversed(levels)) if lo < hi]
    if 0.5 not in quantiles:
        raise ValueError("WIS needs the median (quantile level 0.5)")

    total = 0.5 * abs(observed - quantiles[0.5])
    for lower_level, upper_level in pairs:
        alpha = 2 * lower_level
        total += (alpha / 2) * interval_score(
            observed, quantiles[lower_level], quantiles[upper_level], alpha)
    return total / (len(pairs) + 0.5)


def pinball(observed, quantiles):
    """Mean pinball (quantile) loss across the supplied levels."""
    losses = []
    for level, value in quantiles.items():
        delta = observed - value
        losses.append(level * delta if delta >= 0 else (level - 1) * delta)
    return sum(losses) / len(losses)


def covers(observed, quantiles, lower_level, upper_level):
    return quantiles[lower_level] <= observed <= quantiles[upper_level]


def read_submission(path):
    """Read a FluSight CSV into {(location, reference_date, horizon): {q: v}}."""
    grouped = defaultdict(dict)
    meta = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("output_type") != "quantile":
                continue
            key = (row["location"], row["reference_date"], int(row["horizon"]))
            grouped[key][float(row["output_type_id"])] = float(row["value"])
            meta[key] = row["target_end_date"]
    return grouped, meta


def score_submission(path, truth=None, cache_dir=None):
    """Score every (location, horizon) in a submission. Returns rows + totals."""
    truth = truth if truth is not None else load_truth(cache_dir=cache_dir)
    grouped, ends = read_submission(path)

    rows = []
    for (location, reference_date, horizon), quantiles in sorted(grouped.items()):
        target_end = ends[(location, reference_date, horizon)]
        observed = truth.get((target_end, location))
        row = {
            "location": location,
            "reference_date": reference_date,
            "horizon": horizon,
            "target_end_date": target_end,
            "median": quantiles.get(0.5),
            "observed": observed,
        }
        if observed is None:
            row.update({"wis": None, "abs_error": None,
                        "cover_50": None, "cover_95": None,
                        "note": "no truth published yet"})
        else:
            row.update({
                "wis": round(wis(observed, quantiles), 3),
                "abs_error": round(abs(observed - quantiles[0.5]), 3),
                "cover_50": covers(observed, quantiles, 0.25, 0.75),
                "cover_95": covers(observed, quantiles, 0.025, 0.975),
                "note": "",
            })
        rows.append(row)

    scored = [r for r in rows if r["wis"] is not None]
    totals = {"n_scored": len(scored), "n_total": len(rows)}
    if scored:
        totals.update({
            "mean_wis": round(sum(r["wis"] for r in scored) / len(scored), 3),
            "mean_abs_error": round(
                sum(r["abs_error"] for r in scored) / len(scored), 3),
            "coverage_50": round(
                sum(r["cover_50"] for r in scored) / len(scored), 3),
            "coverage_95": round(
                sum(r["cover_95"] for r in scored) / len(scored), 3),
        })
    return rows, totals


def baseline_flat(recent_values, horizons, spread=0.35):
    """A flat random-walk baseline: last value, with multiplicative spread.

    FluSight's own baseline is roughly this. A harness that cannot beat it is
    not yet doing anything, so it is worth having on hand.
    """
    last = recent_values[-1]
    out = {}
    for horizon in horizons:
        width = spread * (1 + int(horizon))
        out[int(horizon)] = {
            q: max(0.0, last * (1 + width * _z(q)))
            for q in QUANTILES
        }
    return out


def _z(level):
    from statistics import NormalDist
    return NormalDist().inv_cdf(level)
