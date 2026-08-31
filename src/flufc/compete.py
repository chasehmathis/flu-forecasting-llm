"""Head-to-head against the other FluSight teams.

`flufc score` answers "was this forecast any good?". That is only half the
question, because a week that was easy to forecast gives everyone a low WIS and
a week at a turning point gives everyone a high one. The number that means
something is the rank: the same target weeks, the same truth, the same score,
against the thirty-odd teams who submitted for that week.

So this reads every model in the hub's `model-output/` for the submission's
reference date and location, scores them all exactly as the submission is
scored, and sorts. FluSight-baseline is the flat reference every team is
measured against, which makes relative WIS -- yours divided by the baseline's
-- the honest headline: below 1.0 is a forecast that did something.

To keep the comparison fair, a team is ranked only if it submitted every
horizon that has published truth. A model that covered one horizon out of four
is not beating anybody; it is answering a different question, and it is
reported separately.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .evaluate import load_truth, read_submission, score_horizons, summarize

BASELINE = "FluSight-baseline"
ENSEMBLE = "FluSight-ensemble"

# One model-output file is about a megabyte and covers every location, so the
# fetches are worth doing concurrently. The cache makes the second run instant.
WORKERS = 8


class CompetitionError(ValueError):
    """The submission cannot be entered into a competition as it stands."""


def _one_location(grouped, ends):
    """Pull the single (location, reference_date) a submission describes."""
    keys = set(grouped)
    locations = {k[0] for k in keys}
    references = {k[1] for k in keys}
    if not keys:
        raise CompetitionError("no quantile rows in that submission")
    if len(locations) > 1 or len(references) > 1:
        raise CompetitionError(
            "compete handles one location and one reference date at a time; "
            f"this file has {len(locations)} location(s) and "
            f"{len(references)} reference date(s)")
    location, reference_date = locations.pop(), references.pop()
    quantiles = {h: grouped[(location, reference_date, h)]
                 for (_, _, h) in keys}
    week_ends = {h: ends[(location, reference_date, h)] for (_, _, h) in keys}
    return location, reference_date, quantiles, week_ends


def _entry(model, rows, is_you=False):
    entry = {"model": model, "is_you": is_you, "rows": rows,
             "wis_by_horizon": {r["horizon"]: r["wis"] for r in rows}}
    entry.update(summarize(rows))
    return entry


def leaderboard(path, models=None, cache_dir=None, truth=None,
                workers=WORKERS, on_event=None):
    """Score a submission against every hub team for the same week.

    Returns a dict with the ranked entries, the horizons that could be scored,
    and the teams that had to be left out and why.
    """
    from .tools import ToolError                      # network-side imports
    from .tools.hub import hub_forecast, hub_models

    grouped, ends = read_submission(Path(path))
    location, reference_date, mine, week_ends = _one_location(grouped, ends)
    truth = truth if truth is not None else load_truth(cache_dir=cache_dir)

    my_rows = score_horizons(mine, week_ends, location, truth)
    scored_horizons = sorted(r["horizon"] for r in my_rows
                             if r["wis"] is not None)
    unscored = sorted(r["horizon"] for r in my_rows if r["wis"] is None)

    result = {
        "path": str(path),
        "location": location,
        "reference_date": reference_date,
        "scored_horizons": scored_horizons,
        "unscored_horizons": unscored,
        "target_end_dates": {h: week_ends[h] for h in sorted(week_ends)},
        "truth_last_date": _last_truth_date(truth, location),
        "entries": [],
        "partial": [],
        "missing": [],
        "you": None,
    }
    if not scored_horizons:
        return result

    names = list(models) if models else [
        m for m in hub_models() if m != _stem(path)]
    if on_event:
        on_event("start", models=len(names), reference_date=reference_date,
                 location=location)

    def read(model):
        try:
            return model, hub_forecast(model, reference_date, location), None
        except ToolError as e:
            return model, None, str(e)

    fetched = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for done, (model, horizons, error) in enumerate(
                pool.map(read, names), start=1):
            if on_event:
                on_event("fetched", model=model, done=done, total=len(names))
            fetched.append((model, horizons, error))

    entries = [_entry(f"you ({_stem(path)})",
                      [r for r in my_rows if r["horizon"] in scored_horizons],
                      is_you=True)]
    for model, horizons, error in fetched:
        if error is not None or not horizons:
            result["missing"].append({"model": model,
                                      "reason": error or "no quantiles for "
                                                        f"location {location}"})
            continue
        covered = [h for h in scored_horizons if 0.5 in horizons.get(h, {})]
        if len(covered) < len(scored_horizons):
            result["partial"].append({
                "model": model,
                "reason": "submitted horizons "
                          + (", ".join(str(h) for h in sorted(horizons)) or "none")
                          + f"; the comparison needs "
                          + ", ".join(str(h) for h in scored_horizons)})
            continue
        rows = score_horizons({h: horizons[h] for h in scored_horizons},
                              week_ends, location, truth)
        entries.append(_entry(model, rows))

    entries.sort(key=lambda e: e["mean_wis"])
    reference = next((e["mean_wis"] for e in entries
                      if e["model"] == BASELINE), None)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
        entry["rel_wis"] = (round(entry["mean_wis"] / reference, 3)
                            if reference else None)

    you = next(e for e in entries if e["is_you"])
    you["beat"] = sum(1 for e in entries
                      if not e["is_you"] and e["mean_wis"] > you["mean_wis"])
    you["field"] = len(entries) - 1
    result["entries"] = entries
    result["you"] = you
    result["baseline_wis"] = reference
    return result


def _stem(path):
    return Path(path).stem


def _last_truth_date(truth, location):
    dates = [date for date, loc in truth if loc == location]
    return max(dates) if dates else None


# -- the durable record -------------------------------------------------------
#
# A leaderboard is expensive to build (forty-odd megabyte-sized files) and it
# is not stable: the hub repository is rewritten, teams re-submit, and a rank
# computed today may not be reproducible next year. So once a week is settled
# the result is written down next to the forecast and never recomputed.

def summary(board, top=None):
    """Condense a leaderboard into the JSON record that gets committed."""
    you = board.get("you")
    record = {
        "location": board["location"],
        "reference_date": board["reference_date"],
        "computed_at": _now(),
        "scored_horizons": board["scored_horizons"],
        "unscored_horizons": board["unscored_horizons"],
        "target_end_dates": {str(h): d for h, d
                             in board["target_end_dates"].items()},
        "truth_last_date": board["truth_last_date"],
        "n_partial": len(board["partial"]),
        "n_missing": len(board["missing"]),
    }
    if you is None:
        record.update({"status": "pending", "final": False, "entries": [],
                       "you": None})
        return record

    entries = board["entries"]
    kept = entries if top is None else _keep(entries, top)
    record.update({
        "status": "scored",
        "final": not board["unscored_horizons"],
        "baseline_wis": board.get("baseline_wis"),
        "field": you["field"],
        "you": _slim(you),
        "entries": [_slim(e) for e in kept],
        "truncated": len(kept) < len(entries),
        "medians": {str(r["horizon"]): r["median"] for r in you["rows"]},
        "observed": {str(r["horizon"]): r["observed"] for r in you["rows"]},
    })
    ensemble = next((e for e in entries if e["model"] == ENSEMBLE), None)
    baseline = next((e for e in entries if e["model"] == BASELINE), None)
    record["ensemble"] = _slim(ensemble) if ensemble else None
    record["baseline"] = _slim(baseline) if baseline else None
    return record


def _keep(entries, top):
    """The top few, plus you and the two reference models wherever they land."""
    keep = list(entries[:top])
    for entry in entries:
        if entry["is_you"] or entry["model"] in (ENSEMBLE, BASELINE):
            if entry not in keep:
                keep.append(entry)
    keep.sort(key=lambda e: e["rank"])
    return keep


def _slim(entry):
    return {
        "model": entry["model"],
        "rank": entry.get("rank"),
        "is_you": entry.get("is_you", False),
        "mean_wis": entry.get("mean_wis"),
        "rel_wis": entry.get("rel_wis"),
        "mean_abs_error": entry.get("mean_abs_error"),
        "coverage_50": entry.get("coverage_50"),
        "coverage_95": entry.get("coverage_95"),
        "beat": entry.get("beat"),
        "field": entry.get("field"),
        "wis_by_horizon": {str(h): v for h, v
                           in entry.get("wis_by_horizon", {}).items()},
    }


def _now():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def refresh(record, cache_dir=None, truth=None, force=False, top=None,
            on_event=None):
    """Score an archived forecast against the field, unless it is settled.

    Returns (scores, changed). A record whose every horizon has been scored is
    final and is left alone; anything else is recomputed, because the truth for
    its later weeks may have landed since.
    """
    from .archive import save_scores                      # lazy: avoids a cycle
    from .cache import FetchError
    from .tools import ToolError

    if record.is_final() and not force:
        return record.scores, False
    try:
        board = leaderboard(record.csv_path, cache_dir=cache_dir, truth=truth,
                            on_event=on_event)
    except (CompetitionError, ToolError, FetchError, OSError) as e:
        if record.scores:                    # the hub is down; keep what we had
            return record.scores, False
        raise CompetitionError(f"{record.slug}: {e}")

    scores = summary(board, top=top)
    if _same(scores, record.scores):
        return record.scores, False                       # nothing but the clock
    save_scores(record, scores)
    return scores, True


def _same(new, old):
    """Equal but for the timestamp -- so a no-op refresh makes no git diff."""
    if not old:
        return False
    drop = lambda d: {k: v for k, v in d.items() if k != "computed_at"}
    return drop(new) == drop(old)
