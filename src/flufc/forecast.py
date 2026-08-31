"""Quantile forecasts in the FluSight format.

The model states the whole distribution: all 23 FluSight quantile levels per
horizon, which is what actually gets scored. `repair_quantiles` takes that grid
as given -- it only enforces the two things a submission cannot violate,
monotonicity and non-negativity.

`expand_quantiles` is the older, smaller-mouthful path: a few anchor levels
interpolated onto the grid on the normal-score scale, linear in z = Phi^-1(q),
which is monotone by construction and gives sane tails from few anchors. It is
still here because it is the right way to build a forecast from a summary --
the statistical models use it, and so do the tests -- but the agent no longer
goes through it, so a shape it wants that a lognormal cannot make (a fat upper
tail against a tight floor, say) now survives to the CSV.
"""

import csv
import datetime as dt
from dataclasses import dataclass, field
from statistics import NormalDist

# The quantile levels FluSight scores on.
QUANTILES = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
             0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
             0.95, 0.975, 0.99]

# What we ask the model for. Enough to pin the centre and both tails.
ANCHOR_QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

TARGET = "wk inc flu hosp"
HORIZONS = [-1, 0, 1, 2, 3]

_NORM = NormalDist()


class ForecastError(ValueError):
    """The model's forecast is not usable as submitted."""


def expand_quantiles(anchors, *, levels=QUANTILES, non_negative=True):
    """Expand anchor quantiles {level: value} onto the full `levels` grid."""
    points = sorted((float(q), float(v)) for q, v in anchors.items())
    if len(points) < 2:
        raise ForecastError(
            "need at least two quantiles to describe uncertainty, got "
            f"{len(points)}")
    for q, _ in points:
        if not 0.0 < q < 1.0:
            raise ForecastError(f"quantile level {q} is not strictly in (0, 1)")

    # Enforce monotonicity rather than rejecting: a model that says the 25th
    # percentile is above the median made a slip, not an unrecoverable error.
    values, running = [], float("-inf")
    for _, v in points:
        running = max(running, v)
        values.append(running)
    zs = [_NORM.inv_cdf(q) for q, _ in points]

    out = {}
    for level in levels:
        z = _NORM.inv_cdf(level)
        if z <= zs[0]:
            value = _extrapolate(zs[0], values[0], zs[1], values[1], z)
        elif z >= zs[-1]:
            value = _extrapolate(zs[-2], values[-2], zs[-1], values[-1], z)
        else:
            i = next(k for k in range(len(zs) - 1) if zs[k] <= z <= zs[k + 1])
            value = _extrapolate(zs[i], values[i], zs[i + 1], values[i + 1], z)
        if non_negative:
            value = max(0.0, value)
        out[level] = round(value, 4)
    return out


def repair_quantiles(values_by_level, *, levels=QUANTILES, non_negative=True):
    """Make a full quantile grid submittable, without reshaping it.

    Unlike `expand_quantiles` this invents nothing: every level must already be
    present. It enforces monotonicity by running maximum and clips at zero, so
    a model that crosses two neighbouring levels has made a slip we can fix
    rather than an error we have to reject.
    """
    missing = [q for q in levels if q not in values_by_level]
    if missing:
        raise ForecastError(
            f"missing {len(missing)} of {len(levels)} quantile levels: "
            f"{', '.join(str(q) for q in missing)}")
    extra = [q for q in sorted(values_by_level) if q not in levels]
    if extra:
        raise ForecastError(
            f"quantile level(s) {', '.join(str(q) for q in extra)} are not on "
            f"the FluSight grid")

    out, running = {}, float("-inf")
    for level in levels:
        running = max(running, float(values_by_level[level]))
        value = max(0.0, running) if non_negative else running
        out[level] = round(value, 4)
    return out


def _extrapolate(z0, v0, z1, v1, z):
    if z1 == z0:
        return v0
    return v0 + (v1 - v0) * (z - z0) / (z1 - z0)


@dataclass
class Forecast:
    """One location's forecast across horizons, plus the model's reasoning."""
    reference_date: str                       # Saturday ending the current week
    location: str                             # FluSight code: "US", "06", ...
    horizons: dict = field(default_factory=dict)   # {horizon: {level: value}}
    rationale: str = ""
    target: str = TARGET
    model_id: str = "qwen-agent"

    def validate(self):
        if not self.horizons:
            raise ForecastError("forecast has no horizons")
        for horizon, quantiles in self.horizons.items():
            if int(horizon) not in HORIZONS:
                raise ForecastError(
                    f"horizon {horizon} outside FluSight range {HORIZONS}")
            levels = sorted(quantiles)
            values = [quantiles[q] for q in levels]
            if any(b < a - 1e-9 for a, b in zip(values, values[1:])):
                raise ForecastError(f"horizon {horizon}: quantiles decrease")
            if any(v < 0 for v in values):
                raise ForecastError(f"horizon {horizon}: negative admissions")
        return self

    def target_end_date(self, horizon):
        ref = dt.date.fromisoformat(self.reference_date)
        return (ref + dt.timedelta(days=7 * int(horizon))).isoformat()

    def median(self, horizon):
        return self.horizons[horizon].get(0.5)

    def rows(self):
        """FluSight submission rows, one per (horizon, quantile)."""
        for horizon in sorted(self.horizons, key=int):
            for level in sorted(self.horizons[horizon]):
                yield {
                    "reference_date": self.reference_date,
                    "target": self.target,
                    "horizon": int(horizon),
                    "target_end_date": self.target_end_date(horizon),
                    "location": self.location,
                    "output_type": "quantile",
                    "output_type_id": level,
                    "value": self.horizons[horizon][level],
                }

    def write_csv(self, path):
        rows = list(self.rows())
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def summary(self):
        """A compact table for the terminal."""
        lines = [f"{self.location}  reference_date={self.reference_date}  "
                 f"target={self.target}",
                 f"{'horizon':>8}  {'week ending':>12}  {'2.5%':>9}  "
                 f"{'median':>9}  {'97.5%':>9}"]
        for horizon in sorted(self.horizons, key=int):
            q = self.horizons[horizon]
            lines.append(
                f"{int(horizon):>8}  {self.target_end_date(horizon):>12}  "
                f"{q.get(0.025, float('nan')):>9.1f}  "
                f"{q.get(0.5, float('nan')):>9.1f}  "
                f"{q.get(0.975, float('nan')):>9.1f}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "reference_date": self.reference_date,
            "location": self.location,
            "target": self.target,
            "model_id": self.model_id,
            "rationale": self.rationale,
            "horizons": {str(h): {str(q): v for q, v in qs.items()}
                         for h, qs in self.horizons.items()},
        }


def from_anchors(reference_date, location, anchors_by_horizon, rationale=""):
    """Build a Forecast from {horizon: {anchor level: value}}."""
    horizons = {int(h): expand_quantiles(a)
                for h, a in anchors_by_horizon.items()}
    return Forecast(reference_date=reference_date, location=location,
                    horizons=horizons, rationale=rationale).validate()


def from_quantiles(reference_date, location, quantiles_by_horizon, rationale=""):
    """Build a Forecast from {horizon: {level: value}} on the full grid."""
    horizons = {int(h): repair_quantiles(q)
                for h, q in quantiles_by_horizon.items()}
    return Forecast(reference_date=reference_date, location=location,
                    horizons=horizons, rationale=rationale).validate()


def most_recent_saturday(today=None):
    """FluSight reference dates are Saturdays; find the current one."""
    today = today or dt.date.today()
    return (today - dt.timedelta(days=(today.weekday() + 2) % 7)).isoformat()
