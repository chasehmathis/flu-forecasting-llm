"""Statistical forecasts, so the model has something to argue with.

A superforecaster does not start from a vibe; they start from a base rate and
then adjust. These are the base rates: five cheap models fit to the same NHSN
series the agent can read, each producing a median path and an uncertainty band
calibrated on its own walk-forward errors.

The point is not that any of them is good. It is that the model can see what a
purely mechanical extrapolation says, see how wrong that extrapolation has been
at this horizon in the past, and then decide -- with a reason it has to write
down -- whether its judgement moves the number.

Everything here is stdlib and pure Python, including the boosted trees, so the
tools work on a login node with nothing installed.

    dates, values = ...                       # weekly series, oldest first
    fc = predict(dates, values, [1, 2, 3, 4], method="loglinear")
    fc.steps[1]["quantiles"][0.5]

Uncertainty comes from the backtest, not from the fit: for each step ahead we
collect log-scale errors from walk-forward origins, take a robust spread, and
read quantiles off a lognormal with that spread. A model whose errors have been
large gets a wide band automatically, which is the property that matters.
"""

import datetime as dt
import math
import statistics
from dataclasses import dataclass, field
from statistics import NormalDist

from .forecast import ANCHOR_QUANTILES

_NORM = NormalDist()

# Below this many weekly observations nothing here is trustworthy.
MIN_HISTORY = 20

# Uncertainty never shrinks below this, in log space, however calm the backtest
# looked. Flu turns faster than a quiet stretch of series suggests.
SIGMA_FLOOR = 0.08

DEFAULT_FOLDS = 8


class StatsError(ValueError):
    """The series cannot support the model that was asked for."""


# -- small numerics -----------------------------------------------------------

def _ols(xs, ys):
    """Least-squares intercept and slope."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def _quantile(sorted_values, q):
    """Type-7 quantile of an already-sorted list."""
    if not sorted_values:
        raise StatsError("no values to take a quantile of")
    position = (len(sorted_values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    return (sorted_values[low]
            + (sorted_values[high] - sorted_values[low]) * (position - low))


def _spread(residuals, step):
    """Robust log-scale spread of backtest errors, with a floor."""
    floor = SIGMA_FLOOR * math.sqrt(step)
    if len(residuals) < 3:
        return max(floor, 0.18 * math.sqrt(step))
    ordered = sorted(residuals)
    iqr = _quantile(ordered, 0.75) - _quantile(ordered, 0.25)
    sigma = max(statistics.stdev(residuals), iqr / 1.349)
    return max(sigma, floor)


def _log1p(value):
    return math.log1p(max(0.0, value))


# -- the models ---------------------------------------------------------------
#
# Each takes the history (oldest first) and a list of steps ahead of the last
# observation, and returns {step: point forecast}. They see only data up to the
# last element, which is what makes the walk-forward backtest honest.

def _persistence(dates, values, steps):
    """Last observed week, carried forward. The reference every model must beat."""
    return {s: max(0.0, values[-1]) for s in steps}


def _linear(dates, values, steps, window=8):
    """Straight line through the last `window` weeks, extrapolated."""
    ys = values[-window:]
    intercept, slope = _ols(list(range(len(ys))), ys)
    last = len(ys) - 1
    return {s: max(0.0, intercept + slope * (last + s)) for s in steps}


def _loglinear(dates, values, steps, window=8, damping=0.9, max_rate=0.35):
    """Constant growth rate on the log scale, damped with horizon.

    Flu grows multiplicatively, so this is the natural shape for a turn. The
    weekly rate is capped and the trend is damped, because an unconstrained
    exponential fitted to four rising weeks will happily forecast the hospital
    system collapsing by Christmas.
    """
    ys = [_log1p(v) for v in values[-window:]]
    intercept, slope = _ols(list(range(len(ys))), ys)
    slope = max(-max_rate, min(max_rate, slope))
    level = intercept + slope * (len(ys) - 1)
    out = {}
    for s in steps:
        drift = sum(slope * damping ** i for i in range(1, s + 1))
        out[s] = max(0.0, math.expm1(level + drift))
    return out


def _seasonal(dates, values, steps, seasons=4, clamp=(0.4, 3.0)):
    """What happened over the next few weeks in previous years, from here.

    Anchors on the same calendar week in each of the last few years -- 364-day
    steps, so the weekday and hence the week ending date line up -- and applies
    the median observed ratio to the current level. This is the only model that
    knows a season turns in the autumn.
    """
    by_date = dict(zip(dates, values))
    last_date = dt.date.fromisoformat(dates[-1])
    last_value = values[-1]
    out = {}
    for s in steps:
        ratios = []
        for back in range(1, seasons + 1):
            anchor = last_date - dt.timedelta(days=364 * back)
            target = anchor + dt.timedelta(days=7 * s)
            before = by_date.get(anchor.isoformat())
            after = by_date.get(target.isoformat())
            if before is None or after is None:
                continue
            ratio = (after + 1.0) / (before + 1.0)
            ratios.append(max(clamp[0], min(clamp[1], ratio)))
        if not ratios:
            out[s] = max(0.0, last_value)          # no matching weeks: flat
        else:
            out[s] = max(0.0, (last_value + 1.0) * statistics.median(ratios) - 1.0)
    return out


def _gbt(dates, values, steps, **kwargs):
    """Gradient-boosted regression trees on the log growth rate.

    The xgboost-shaped model, in about a hundred lines of stdlib. Trees cannot
    extrapolate past the range of the training data, so predicting the *level*
    would cap every forecast at the worst week ever seen; predicting
    log(y[t+s] / y[t]) instead leaves the level to the anchor and asks the trees
    only for the growth rate, which they can do.

    Features are the current level, one-, two- and four-week log changes, the
    gap to the eight-week mean, and where we are in the year.
    """
    out = {}
    for s in steps:
        rows, targets = _gbt_training_rows(dates, values, s)
        if len(rows) < 30:
            raise StatsError(
                f"gbt needs at least 30 training rows for step {s}, "
                f"got {len(rows)}; use a longer history or another method")
        model = _fit_gbt(rows, targets, **kwargs)
        rate = _predict_gbt(model, _features(dates, values, len(values) - 1))
        out[s] = max(0.0, (values[-1] + 1.0) * math.exp(rate) - 1.0)
    return out


MODELS = {
    "persistence": _persistence,
    "linear": _linear,
    "loglinear": _loglinear,
    "seasonal": _seasonal,
    "gbt": _gbt,
}

DESCRIPTIONS = {
    "persistence": "last observed week carried forward (the FluSight-style "
                   "baseline; hard to beat when nothing is happening)",
    "linear": "ordinary least squares on the last 8 weeks, extrapolated",
    "loglinear": "constant growth rate fitted on the last 8 weeks in log space, "
                 "capped and damped with horizon",
    "seasonal": "the median trajectory over the same calendar weeks in the last "
                "4 years, applied to the current level",
    "gbt": "gradient-boosted regression trees predicting the log growth rate "
           "from lags, momentum and time of year",
}


# -- boosted trees ------------------------------------------------------------

def _features(dates, values, t):
    """Feature row describing the series as of index `t`."""
    level = _log1p(values[t])
    window = [_log1p(values[i]) for i in range(max(0, t - 7), t + 1)]
    mean8 = sum(window) / len(window)
    day = dt.date.fromisoformat(dates[t]).timetuple().tm_yday
    angle = 2 * math.pi * day / 365.25
    return [level,
            level - _log1p(values[t - 1]),
            level - _log1p(values[t - 2]),
            level - _log1p(values[t - 4]),
            level - mean8,
            math.sin(angle),
            math.cos(angle)]


def _gbt_training_rows(dates, values, step, min_lag=8, max_rows=300):
    rows, targets = [], []
    for t in range(min_lag, len(values) - step):
        rows.append(_features(dates, values, t))
        targets.append(math.log((values[t + step] + 1.0) / (values[t] + 1.0)))
    return rows[-max_rows:], targets[-max_rows:]


def _bin_edges(rows, bins):
    """Quantile bin edges per feature, so splitting is a histogram scan."""
    edges = []
    for f in range(len(rows[0])):
        column = sorted(row[f] for row in rows)
        cuts = [_quantile(column, (i + 1) / bins) for i in range(bins - 1)]
        deduped = []
        for c in cuts:                       # ties collapse; keep them ordered
            if not deduped or c > deduped[-1]:
                deduped.append(c)
        edges.append(deduped)
    return edges


def _binned(row, edges):
    out = []
    for value, cuts in zip(row, edges):
        index = 0
        while index < len(cuts) and value > cuts[index]:
            index += 1
        out.append(index)
    return out


def _grow(bins, residuals, index, depth, min_leaf, n_bins):
    """One regression tree over binned features, squared loss, greedy splits."""
    total_sum = sum(residuals[i] for i in index)
    leaf = {"value": total_sum / len(index)}
    if depth == 0 or len(index) < 2 * min_leaf:
        return leaf

    best = None
    for feature in range(len(bins[0])):
        sums = [0.0] * n_bins
        counts = [0] * n_bins
        for i in index:
            b = bins[i][feature]
            sums[b] += residuals[i]
            counts[b] += 1
        left_sum = left_count = 0
        for b in range(n_bins - 1):
            left_sum += sums[b]
            left_count += counts[b]
            right_count = len(index) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            right_sum = total_sum - left_sum
            gain = left_sum ** 2 / left_count + right_sum ** 2 / right_count
            if best is None or gain > best[0]:
                best = (gain, feature, b)
    if best is None:
        return leaf

    _, feature, split = best
    left = [i for i in index if bins[i][feature] <= split]
    right = [i for i in index if bins[i][feature] > split]
    return {"feature": feature, "split": split,
            "left": _grow(bins, residuals, left, depth - 1, min_leaf, n_bins),
            "right": _grow(bins, residuals, right, depth - 1, min_leaf, n_bins)}


def _tree_value(node, row_bins):
    while "value" not in node:
        node = (node["left"] if row_bins[node["feature"]] <= node["split"]
                else node["right"])
    return node["value"]


def _fit_gbt(rows, targets, rounds=40, rate=0.12, depth=2, bins=12, min_leaf=8):
    edges = _bin_edges(rows, bins)
    binned = [_binned(row, edges) for row in rows]
    base = sum(targets) / len(targets)
    residuals = [t - base for t in targets]
    index = list(range(len(rows)))
    trees = []
    for _ in range(rounds):
        tree = _grow(binned, residuals, index, depth, min_leaf, bins)
        predictions = [_tree_value(tree, b) for b in binned]
        residuals = [r - rate * p for r, p in zip(residuals, predictions)]
        trees.append(tree)
    return {"base": base, "rate": rate, "edges": edges, "trees": trees}


def _predict_gbt(model, row):
    binned = _binned(row, model["edges"])
    return model["base"] + model["rate"] * sum(
        _tree_value(tree, binned) for tree in model["trees"])


# -- backtesting --------------------------------------------------------------

@dataclass
class ModelForecast:
    """One method's forecast, plus how that method has actually performed."""
    method: str
    description: str
    last_date: str
    last_value: float
    n_history: int
    folds: int = 0
    steps: dict = field(default_factory=dict)    # {step: {point, quantiles}}
    skill: dict = field(default_factory=dict)    # {step: {mae, skill, ...}}

    def point(self, step):
        return self.steps[step]["point"]

    def quantiles(self, step):
        return self.steps[step]["quantiles"]


def walk_forward(dates, values, steps, method, folds=DEFAULT_FOLDS,
                 min_train=MIN_HISTORY):
    """Refit at successively earlier origins and record what the model missed.

    Returns {step: [(actual, predicted), ...]}, newest origin first. Each fit
    sees only the data that existed at that origin, so the errors are the
    errors the method would really have made.
    """
    fn = MODELS[method]
    largest = max(steps)
    origins = []
    last = len(values) - 1 - largest
    for t in range(last, min_train - 1, -1):
        if len(origins) >= folds:
            break
        origins.append(t)

    out = {s: [] for s in steps}
    for t in origins:
        try:
            predicted = fn(dates[:t + 1], values[:t + 1], steps)
        except StatsError:
            continue                        # not enough history that far back
        for s in steps:
            out[s].append((values[t + s], predicted[s]))
    return out


def predict(dates, values, steps, method="loglinear", folds=DEFAULT_FOLDS):
    """Fit `method`, calibrate its spread on its own backtest, return quantiles."""
    if method not in MODELS:
        raise StatsError(f"unknown method {method!r}. Available: "
                         f"{', '.join(MODELS)}")
    if len(values) < MIN_HISTORY:
        raise StatsError(f"need at least {MIN_HISTORY} weeks of history, "
                         f"got {len(values)}")
    steps = sorted({int(s) for s in steps})
    if any(s < 1 for s in steps):
        raise StatsError("steps must be at least 1 week beyond the last "
                         "observation")

    key = (method, tuple(steps), folds, dates[-1], len(values),
           round(sum(values), 3))
    if key in _CACHE:
        return _CACHE[key]

    points = MODELS[method](dates, values, steps)
    errors = walk_forward(dates, values, steps, method, folds)
    reference = walk_forward(dates, values, steps, "persistence", folds)

    result = ModelForecast(
        method=method, description=DESCRIPTIONS[method], last_date=dates[-1],
        last_value=values[-1], n_history=len(values),
        folds=min(len(v) for v in errors.values()) if errors else 0)

    # Spread never shrinks with horizon. A handful of backtest origins can say
    # step 3 was calmer than step 2; that is noise, and a forecast that gets
    # narrower further out is indefensible whatever the sample says.
    spreads, running = {}, 0.0
    for step in steps:
        residuals = [math.log((a + 1.0) / (p + 1.0)) for a, p in errors[step]]
        running = max(running, _spread(residuals, step))
        spreads[step] = running

    for step in steps:
        pairs = errors[step]
        residuals = [math.log((a + 1.0) / (p + 1.0)) for a, p in pairs]
        bias = statistics.median(residuals) if len(residuals) >= 4 else 0.0
        sigma = spreads[step]
        point = points[step]
        quantiles = {}
        for level in ANCHOR_QUANTILES:
            z = _NORM.inv_cdf(level)
            quantiles[level] = max(
                0.0, (point + 1.0) * math.exp(bias + z * sigma) - 1.0)
        result.steps[step] = {"point": point, "quantiles": quantiles}

        mae = (statistics.fmean(abs(a - p) for a, p in pairs) if pairs else None)
        flat = reference.get(step) or []
        mae_flat = (statistics.fmean(abs(a - p) for a, p in flat)
                    if flat else None)
        covered = [1 for (a, _), r in zip(pairs, residuals)
                   if abs(r - bias) <= 1.96 * sigma]
        result.skill[step] = {
            "mae": mae,
            "mae_persistence": mae_flat,
            "skill_vs_persistence": (None if not mae_flat
                                     else 1.0 - mae / mae_flat),
            "sigma": sigma,
            "bias": bias,
            "coverage_95": (len(covered) / len(pairs)) if pairs else None,
            "n": len(pairs),
        }

    _CACHE[key] = result
    return result


def consensus(forecasts, steps):
    """Median across several methods -- the cheapest ensemble there is."""
    out = {}
    for step in steps:
        medians = [f.quantiles(step)[0.5] for f in forecasts if step in f.steps]
        if medians:
            out[step] = statistics.median(medians)
    return out


# Fits are deterministic and a forecasting run asks for the same one repeatedly,
# so memoise within the process. Keyed on the data, not just the location.
_CACHE = {}
