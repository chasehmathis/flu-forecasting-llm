"""What everybody else is forecasting.

Every week around thirty teams submit to the CDC's FluSight hub, and their
forecasts sit in a public repository the moment they are due. Ignoring them
would be strange: the hub ensemble is the thing to beat, it is genuinely hard
to beat, and a forecast that lands far from it should have a reason.

So the agent can read them. Not to copy a number -- the ensemble is a prior,
and the whole point of a judgemental forecaster is to know when the prior is
wrong -- but so that any disagreement is deliberate.

Source: github.com/cdcepi/FluSight-forecast-hub, model-output/<model>/
<reference_date>-<model>.csv
"""

import csv
import io
import os
import statistics

from ..cache import FetchError, fetch, fetch_json
from ..config import active
from ..forecast import most_recent_saturday
from . import ToolError, tool

REPO = "cdcepi/FluSight-forecast-hub"
RAW = ("https://raw.githubusercontent.com/" + REPO
       + "/main/model-output/{model}/{date}-{model}.csv")
CONTENTS = "https://api.github.com/repos/" + REPO + "/contents/model-output"

# The ensemble is the hub's official forecast; the baseline is the flat
# reference every team is measured against. Together they bracket "no opinion".
DEFAULT_MODELS = ("FluSight-ensemble", "FluSight-baseline")

TARGET = "wk inc flu hosp"


def _auth():
    """Bearer token for the GitHub API, if the environment has one.

    Anonymous callers get 60 API requests an hour, shared across everyone on
    the same address -- which is fine from a login node and hopeless from a CI
    runner. Anywhere a token exists (GitHub Actions sets one for free), use it.
    Only the directory listing is affected; the submissions themselves come
    from raw.githubusercontent.com, which is a CDN and is not rate limited the
    same way.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _listing(path=""):
    """Directory listing from the GitHub contents API (cached for a day)."""
    url = CONTENTS + (f"/{path}" if path else "")
    try:
        return fetch_json(url, timeout=45, ttl=86400, headers=_auth(),
                          cache_dir=active().cache_dir)
    except FetchError as e:
        raise ToolError(f"could not list the FluSight hub ({e}). GitHub rate "
                        f"limits unauthenticated requests to 60 an hour; set "
                        f"GITHUB_TOKEN, or try again shortly.")


def hub_models():
    """Every model directory in the hub."""
    return sorted(item["name"] for item in _listing()
                  if item.get("type") == "dir")


def hub_dates(model):
    """Reference dates this model has submitted for, oldest first."""
    dates = []
    for item in _listing(model):
        name = item.get("name", "")
        if name.endswith(".csv") and len(name) > 10:
            dates.append(name[:10])
    return sorted(dates)


def hub_forecast(model, reference_date, location):
    """{horizon: {quantile level: value}} for one model, date and location."""
    url = RAW.format(model=model, date=reference_date)
    try:
        body = fetch(url, timeout=60, ttl=active().cache_ttl,
                     cache_dir=active().cache_dir, max_bytes=8_000_000)
    except FetchError as e:
        detail = "" if "HTTP 404" in str(e) else f" ({e})"
        raise ToolError(
            f"no submission from {model} for {reference_date}{detail}")

    horizons = {}
    for row in csv.DictReader(io.StringIO(body)):
        if row.get("location") != location:
            continue
        if row.get("output_type") != "quantile" or row.get("target") != TARGET:
            continue
        try:
            horizon = int(row["horizon"])
            level = float(row["output_type_id"])
            value = float(row["value"])
        except (KeyError, ValueError):
            continue
        horizons.setdefault(horizon, {})[level] = value
    return horizons


def _nearest_dates(model, reference_date, count=4):
    """Dates this model did submit for, closest to the one that was asked for."""
    try:
        dates = hub_dates(model)
    except ToolError:
        return []
    return sorted(dates, key=lambda d: abs(
        (_ordinal(d) - _ordinal(reference_date))))[:count]


def _ordinal(date):
    year, month, day = (int(part) for part in date.split("-"))
    return year * 372 + month * 31 + day


@tool
def list_hub_models(query: str = "") -> str:
    """List the teams submitting to the CDC FluSight hub.

    Args:
        query: Optional substring to filter model names.
    """
    names = hub_models()
    needle = query.strip().lower()
    if needle:
        names = [n for n in names if needle in n.lower()]
    if not names:
        return f"No FluSight hub models match {query!r}."
    lines = [f"{len(names)} model(s) in the FluSight hub:"]
    lines += ["  " + name for name in names]
    lines.append("")
    lines.append("FluSight-ensemble is the hub's combined forecast and "
                 "FluSight-baseline is the flat reference; pass names to "
                 "get_hub_forecasts to read what they predicted.")
    return "\n".join(lines)


@tool
def get_hub_forecasts(location: str, reference_date: str = "",
                      models: str = "") -> str:
    """Read other teams' FluSight forecasts for one location and week.

    Defaults to the hub ensemble and the flat baseline. The ensemble is a
    strong prior and is hard to beat; use it to check whether your own numbers
    are somewhere defensible, and if you depart from it, know why you are
    departing from it. Submissions only exist during the season, and only from
    the Wednesday they are due.

    Args:
        location: FluSight code ('US', '06') or a name this harness can resolve.
        reference_date: Saturday ending the reference week, YYYY-MM-DD.
            Defaults to the most recent Saturday.
        models: Comma-separated model names (see list_hub_models). Default:
            FluSight-ensemble, FluSight-baseline.
    """
    from .surveillance import resolve_location            # avoids a cycle

    row = resolve_location(location)
    code = row["location"]
    reference_date = (reference_date or "").strip() or most_recent_saturday()
    chosen = [m.strip() for m in (models or "").replace(",", " ").split()]
    chosen = chosen or list(DEFAULT_MODELS)

    lines = [f"FluSight hub forecasts -- {row['location_name']} ({code}), "
             f"reference_date {reference_date}"]
    found, missing = {}, []
    for model in chosen:
        try:
            horizons = hub_forecast(model, reference_date, code)
        except ToolError as e:
            missing.append(f"{model}: {e}")
            continue
        if horizons:
            found[model] = horizons
        else:
            missing.append(f"{model}: submitted for {reference_date} but has "
                           f"no {TARGET} quantiles for location {code}")

    if not found:
        lines.append("")
        lines += ["  " + m for m in missing]
        nearby = _nearest_dates(chosen[0], reference_date)
        if nearby:
            lines += ["", f"{chosen[0]} did submit for: "
                          + ", ".join(nearby),
                      "The hub only receives forecasts during the season, so "
                      "off-season weeks are simply empty."]
        return "\n".join(lines)

    horizons = sorted({h for f in found.values() for h in f})
    for model, forecast in found.items():
        lines += ["", f"{model}",
                  f"{'horizon':>7} {'2.5%':>9} {'25%':>9} {'median':>9} "
                  f"{'75%':>9} {'97.5%':>9}"]
        for horizon in horizons:
            q = forecast.get(horizon)
            if not q:
                continue
            lines.append(f"{horizon:>7} {_at(q, 0.025):>9} {_at(q, 0.25):>9} "
                         f"{_at(q, 0.5):>9} {_at(q, 0.75):>9} "
                         f"{_at(q, 0.975):>9}")

    if len(found) > 1:
        lines += ["", "median across the models above",
                  f"{'horizon':>7} {'median':>9}"]
        for horizon in horizons:
            medians = [f[horizon][0.5] for f in found.values()
                       if horizon in f and 0.5 in f[horizon]]
            if medians:
                lines.append(f"{horizon:>7} {statistics.median(medians):>9.0f}")
    if missing:
        lines += ["", "not available: " + "; ".join(missing)]
    lines += ["",
              "These are other people's judgements, made from the same public "
              "data you have. Treat the ensemble as a prior worth deviating "
              "from only with a stated reason."]
    return "\n".join(lines)


def _at(quantiles, level):
    """Format one quantile, falling back to the nearest level available."""
    if level in quantiles:
        return f"{quantiles[level]:.0f}"
    if not quantiles:
        return "-"
    nearest = min(quantiles, key=lambda q: abs(q - level))
    return f"{quantiles[nearest]:.0f}*"
