"""Statistical forecasts on demand.

The agent can now do what a forecaster with a laptop would do before writing
anything down: fit a few mechanical models to the series, look at how those
models have scored at this horizon in the past, and take the best of them as a
starting point rather than a conclusion.

`flufc.stats` does the fitting; this module is the part the model talks to, so
it deals in reference dates and horizons (what FluSight asks for) rather than
steps ahead of the last observation (what the models need).
"""

import datetime as dt

from .. import stats
from ..forecast import HORIZONS, most_recent_saturday
from . import ToolError, tool
from .surveillance import admissions_series


def parse_horizons(text, default=(0, 1, 2, 3)):
    """'0,1,2,3' or '0 1 2 3' or '' -> [0, 1, 2, 3]."""
    if text is None or not str(text).strip():
        return list(default)
    out = []
    for chunk in str(text).replace(",", " ").split():
        try:
            horizon = int(chunk)
        except ValueError:
            raise ToolError(f"horizon {chunk!r} is not an integer")
        if horizon not in HORIZONS:
            raise ToolError(f"horizon {horizon} is outside the FluSight range "
                            f"{HORIZONS}")
        out.append(horizon)
    return sorted(set(out))


def _steps(series, reference_date, horizons):
    """Map FluSight horizons to steps beyond the last observed week.

    Reported data runs a week or two behind the reference date, so horizon 0 is
    usually already one or two steps into the future. Anything the data already
    covers comes back as an observation, not a forecast.
    """
    last = dt.date.fromisoformat(series[-1][0])
    try:
        reference = dt.date.fromisoformat(reference_date)
    except ValueError as e:
        raise ToolError(f"reference_date must be YYYY-MM-DD ({e})")
    mapping, observed = {}, {}
    for horizon in horizons:
        target = reference + dt.timedelta(days=7 * horizon)
        step = (target - last).days // 7
        if step <= 0:
            observed[horizon] = (target.isoformat(),
                                 dict(series).get(target.isoformat()))
        else:
            mapping[horizon] = (step, target.isoformat())
    return mapping, observed


def _header(row, series, reference_date, horizons, observed):
    lines = [f"{row['location_name']} ({row['location']}) -- "
             f"{len(series)} weeks of NHSN history, "
             f"last observed {series[-1][0]} = {series[-1][1]:.0f} admissions",
             f"reference_date {reference_date}, horizons "
             f"{', '.join(str(h) for h in horizons)}"]
    for horizon, (date, value) in sorted(observed.items()):
        seen = "not reported" if value is None else f"{value:.0f} admissions"
        lines.append(f"  horizon {horizon} (week ending {date}) is already "
                     f"observed: {seen}")
    return lines


@tool
def fit_forecast_model(location: str, method: str = "loglinear",
                       reference_date: str = "", horizons: str = "0,1,2,3",
                       folds: int = 8) -> str:
    """Fit a statistical model to the admissions series and forecast it.

    Gives you a mechanical forecast to anchor on, with quantiles calibrated on
    the model's own walk-forward backtest -- so you also see how wrong this
    method has been at this horizon lately, and whether it beats carrying the
    last week forward. Use it before you commit to numbers, and treat the
    output as a starting point you adjust with what you know, not an answer.

    Args:
        location: 'US', a state abbreviation, a state name, or a FIPS code.
        method: One of persistence (last week carried forward), linear (OLS on
            the last 8 weeks), loglinear (damped exponential growth), seasonal
            (what these calendar weeks did in previous years), gbt
            (gradient-boosted trees on the log growth rate).
        reference_date: Saturday ending the reference week, YYYY-MM-DD.
            Defaults to the most recent Saturday.
        horizons: Comma-separated FluSight horizons, e.g. '0,1,2,3'.
        folds: Backtest origins used to calibrate uncertainty (4-30).
    """
    method = (method or "loglinear").strip().lower()
    if method not in stats.MODELS:
        raise ToolError(f"unknown method {method!r}. Available: "
                        f"{', '.join(stats.MODELS)}")
    reference_date = (reference_date or "").strip() or most_recent_saturday()
    wanted = parse_horizons(horizons)
    folds = max(4, min(int(folds), 30))

    row, series = admissions_series(location, weeks=400)
    dates = [d for d, _ in series]
    values = [v for _, v in series]
    mapping, observed = _steps(series, reference_date, wanted)

    lines = _header(row, series, reference_date, wanted, observed)
    lines += ["", f"method: {method} -- {stats.DESCRIPTIONS[method]}"]
    if not mapping:
        lines.append("")
        lines.append("Every requested horizon is already observed; there is "
                     "nothing left to forecast.")
        return "\n".join(lines)

    try:
        forecast = stats.predict(dates, values, [s for s, _ in mapping.values()],
                                 method=method, folds=folds)
    except stats.StatsError as e:
        raise ToolError(str(e))

    lines += ["",
              f"{'horizon':>7} {'week ending':<12} {'2.5%':>9} {'25%':>9} "
              f"{'median':>9} {'75%':>9} {'97.5%':>9}"]
    for horizon in sorted(mapping):
        step, target = mapping[horizon]
        q = forecast.quantiles(step)
        lines.append(f"{horizon:>7} {target:<12} {q[0.025]:>9.0f} {q[0.25]:>9.0f} "
                     f"{q[0.5]:>9.0f} {q[0.75]:>9.0f} {q[0.975]:>9.0f}")

    lines += ["", f"backtest over {forecast.folds} walk-forward origins "
                  f"(each fit saw only earlier data)",
              f"{'horizon':>7} {'MAE':>9} {'MAE flat':>9} {'vs flat':>8} "
              f"{'log sd':>7} {'95% hit':>8}"]
    for horizon in sorted(mapping):
        step, _ = mapping[horizon]
        skill = forecast.skill[step]
        versus = ("-" if skill["skill_vs_persistence"] is None
                  else f"{skill['skill_vs_persistence']:+.0%}")
        coverage = ("-" if skill["coverage_95"] is None
                    else f"{skill['coverage_95']:.0%}")
        lines.append(f"{horizon:>7} {skill['mae'] or 0:>9.1f} "
                     f"{skill['mae_persistence'] or 0:>9.1f} {versus:>8} "
                     f"{skill['sigma']:>7.2f} {coverage:>8}")
    lines += ["",
              "Medians are this method's forecast after subtracting the bias it "
              "showed in backtest; the MAE column is the raw method.",
              "'vs flat' is the reduction in mean absolute error against "
              "carrying the last week forward; negative means this method did "
              "worse than doing nothing.",
              "The interval comes from those backtest errors, so it is only as "
              "honest as the recent past is representative. It does not know "
              "about anything you found by reading."]
    return "\n".join(lines)


@tool
def compare_forecast_models(location: str, reference_date: str = "",
                            horizons: str = "0,1,2,3",
                            methods: str = "") -> str:
    """Fit every statistical model to one location and compare them side by side.

    The spread between methods is itself information: when persistence, a
    damped exponential, the seasonal analogue and boosted trees agree, the
    signal is clean and your interval can be tighter; when they disagree by a
    factor of two, that disagreement belongs in your uncertainty. The last
    column is how each method scored in backtest, which is what tells you
    whether to listen to it.

    Args:
        location: 'US', a state abbreviation, a state name, or a FIPS code.
        reference_date: Saturday ending the reference week, YYYY-MM-DD.
        horizons: Comma-separated FluSight horizons, e.g. '0,1,2,3'.
        methods: Comma-separated subset to run; default is all of them.
    """
    reference_date = (reference_date or "").strip() or most_recent_saturday()
    wanted = parse_horizons(horizons)
    chosen = [m.strip().lower() for m in (methods or "").replace(",", " ").split()]
    for name in chosen:
        if name not in stats.MODELS:
            raise ToolError(f"unknown method {name!r}. Available: "
                            f"{', '.join(stats.MODELS)}")
    chosen = chosen or list(stats.MODELS)

    row, series = admissions_series(location, weeks=400)
    dates = [d for d, _ in series]
    values = [v for _, v in series]
    mapping, observed = _steps(series, reference_date, wanted)

    lines = _header(row, series, reference_date, wanted, observed)
    if not mapping:
        lines += ["", "Every requested horizon is already observed."]
        return "\n".join(lines)

    steps = [s for s, _ in mapping.values()]
    fits, failures = [], []
    for name in chosen:
        try:
            fits.append(stats.predict(dates, values, steps, method=name))
        except stats.StatsError as e:
            failures.append(f"{name}: {e}")
    if not fits:
        raise ToolError("no method could be fitted: " + "; ".join(failures))

    order = sorted(mapping)
    lines += ["", "medians by method (95% interval under each)", "",
              f"{'method':<13}" + "".join(
                  f"{'h' + str(h):>12}" for h in order) + f"{'MAE h' + str(order[-1]):>12}"]
    for fit in fits:
        medians = f"{fit.method:<13}"
        bands = " " * 13
        for horizon in order:
            step, _ = mapping[horizon]
            q = fit.quantiles(step)
            medians += f"{q[0.5]:>12.0f}"
            bands += f"{q[0.025]:>5.0f}-{q[0.975]:<6.0f}"
        last_step = mapping[order[-1]][0]
        medians += f"{fit.skill[last_step]['mae'] or 0:>12.1f}"
        lines.append(medians)
        lines.append(bands)

    agreement = stats.consensus(fits, steps)
    lines += ["", f"{'consensus':<13}" + "".join(
        f"{agreement[mapping[h][0]]:>12.0f}" for h in order)
        + "   (median across methods)"]

    lines.append("")
    for horizon in order:
        step, target = mapping[horizon]
        medians = [f.quantiles(step)[0.5] for f in fits]
        low, high = min(medians), max(medians)
        span = "-" if low <= 0 else f"{high / max(low, 1e-9):.2f}x"
        lines.append(f"horizon {horizon} (week ending {target}): methods span "
                     f"{low:.0f} to {high:.0f} ({span})")
    if failures:
        lines += ["", "could not fit: " + "; ".join(failures)]
    lines += ["",
              "Medians are bias-corrected on each method's own backtest, which "
              "is why persistence need not equal the last observed week.",
              "Weight the methods by their backtest error, not by which number "
              "you like. Where they disagree, your interval should cover the "
              "disagreement."]
    return "\n".join(lines)
