"""Keeping the archive's scores current.

A forecast is written once and scored repeatedly. The week it covers has not
happened when it is made; the other teams' submissions appear the same day; the
truth lands a few weeks later and revises for a few weeks after that. So this
walks the archive and works out, for each forecast, where it stands right now
-- against the published truth, and against everybody who submitted for the
same week.

The expensive part is the leaderboard, which reads forty-odd hub files. Once a
week has settled that answer is fixed forever, so it is cached beside the
forecast and never recomputed. Everything still in flight is recomputed every
run, which is what makes a nightly refresh cheap enough to leave running
unattended.
"""

from .archive import is_settled, read_score, records, write_score
from .archive import quantiles as archive_quantiles
from .compete import BASELINE, ENSEMBLE, CompetitionError, leaderboard
from .evaluate import (baseline_flat, load_truth, score_horizons, summarize,
                       wis)

# What the site shows. Carrying every team's per-horizon rows would make the
# archive heavy for nobody's benefit.
ENTRY_KEYS = ("model", "rank", "mean_wis", "rel_wis", "mean_abs_error",
              "coverage_50", "coverage_95", "wis_by_horizon", "is_you")


def refresh(root, *, cache_dir=None, truth=None, force=False, on_event=None):
    """Score every archived forecast, reusing settled results.

    Returns the records, each with a `score` key (None if it has never been
    scoreable). Network failures leave the previous score in place and set
    `score_error` rather than tearing down the whole run: one unreachable week
    should not cost you the site.
    """
    found = records(root)
    if not found:
        return []
    truth = truth if truth is not None else load_truth(cache_dir=cache_dir)

    for record in found:
        cached = None if force else read_score(record)
        if cached is not None and is_settled(cached):
            record["score"] = cached
            _emit(on_event, "settled", record)
            continue
        _emit(on_event, "scoring", record)
        try:
            score = _score(record, truth=truth, cache_dir=cache_dir)
        except (CompetitionError, OSError, ValueError) as e:
            record["score"] = cached
            record["score_error"] = str(e)
            _emit(on_event, "failed", record, error=str(e))
            continue
        record["score"] = write_score(record, score)
        _emit(on_event, "scored", record)
    return found


def _emit(on_event, event, record, **kw):
    if on_event:
        on_event(event, record=record, **kw)


def _score(record, truth, cache_dir):
    """One forecast's standing: WIS, rank, and what the hub said that week."""
    board = leaderboard(record["path"], truth=truth, cache_dir=cache_dir)
    score = {
        "reference_date": board["reference_date"],
        "location": board["location"],
        "scored_horizons": board["scored_horizons"],
        "unscored_horizons": board["unscored_horizons"],
        "target_end_dates": {str(h): d
                             for h, d in board["target_end_dates"].items()},
        "truth_last_date": board["truth_last_date"],
        "baseline_wis": board.get("baseline_wis"),
        "n_partial": len(board["partial"]),
        "n_missing": len(board["missing"]),
        "entries": [{k: e.get(k) for k in ENTRY_KEYS}
                    for e in board["entries"]],
        "hub_medians": _hub_medians(board["location"],
                                    board["reference_date"]),
        "observed": _observed(record, truth),
        "you": None,
    }
    you = board["you"]
    if you is not None:
        score["you"] = {**{k: you.get(k) for k in ENTRY_KEYS},
                        "beat": you["beat"], "field": you["field"],
                        "rows": you["rows"]}
        ensemble = next((e for e in board["entries"]
                         if e["model"] == ENSEMBLE), None)
        if ensemble:
            score["ensemble"] = {k: ensemble.get(k) for k in ENTRY_KEYS}
            score["vs_ensemble"] = (
                round(you["mean_wis"] / ensemble["mean_wis"], 3)
                if ensemble["mean_wis"] else None)
    return score


def _observed(record, truth):
    """Published truth for this forecast's target weeks, as far as it goes."""
    location = record["location"]
    return {h: truth.get((end, location))
            for h, end in (record.get("target_end_dates") or {}).items()}


def _hub_medians(location, reference_date):
    """The ensemble's and the baseline's medians for the same week, if any.

    Kept even when nothing can be scored yet: "here is what everyone else said"
    is the most useful thing a page about an unresolved forecast can show.
    """
    from .tools import ToolError
    from .tools.hub import hub_forecast

    out = {}
    for model in (ENSEMBLE, BASELINE):
        try:
            horizons = hub_forecast(model, reference_date, location)
        except ToolError:
            continue
        medians = {str(h): q[0.5] for h, q in sorted(horizons.items())
                   if 0.5 in q}
        if medians:
            out[model] = medians
    return out


def preliminary(record, series):
    """A provisional score against the raw NHSN series, not the hub's truth.

    The hub's target file stops updating out of season, but NHSN keeps
    publishing, so this is the only number available for a summer forecast.
    It is not what FluSight ranks on: NHSN revises its recent weeks, and there
    is no field to rank against, so the yardstick is a flat baseline built
    here from the last NHSN week before the reference date -- not the hub's
    FluSight-baseline. Never cached; it is recomputed on every build.
    """
    location = record["location"]
    ends = {int(h): d for h, d in (record.get("target_end_dates") or {}).items()}
    truth = {(date, location): value for date, value in series}
    rows = score_horizons(archive_quantiles(record), ends, location, truth)
    out = {"truth_last_date": max((d for d, _ in series), default=None),
           "rows": rows, **summarize(rows)}

    before = [v for d, v in series if d < record["reference_date"]]
    done = [r for r in rows if r["wis"] is not None]
    if before and done:
        flat = baseline_flat(before, [r["horizon"] for r in done])
        base = sum(wis(r["observed"], flat[r["horizon"]])
                   for r in done) / len(done)
        out["baseline_wis"] = round(base, 3)
        out["rel_wis"] = round(out["mean_wis"] / base, 3) if base else None
    return out


def preliminary_summary(found):
    """The NHSN track record, pooled over every horizon that has a value."""
    rows = [r for rec in found for r in (rec.get("nhsn") or {}).get("rows", [])]
    total = {"n_forecasts": len(found),
             "n_with_data": sum(1 for rec in found
                                if (rec.get("nhsn") or {}).get("n_scored")),
             **summarize(rows)}
    rels = [rec["nhsn"]["rel_wis"] for rec in found
            if (rec.get("nhsn") or {}).get("rel_wis") is not None]
    if rels:
        total["mean_rel_wis"] = round(sum(rels) / len(rels), 3)
        total["beat_baseline"] = sum(1 for r in rels if r < 1)
        total["n_rel"] = len(rels)
    return total


def scored(found):
    """The records that have a rank, newest first."""
    return [r for r in found
            if (r.get("score") or {}).get("you")
            and r["score"]["you"].get("rel_wis") is not None]


def summary(found):
    """The track record: how the archive has done, taken together."""
    done = scored(found)
    total = {"n_forecasts": len(found), "n_scored": len(done),
             "n_pending": len(found) - len(done)}
    if not done:
        return total

    yours = [r["score"]["you"] for r in done]
    rels = [y["rel_wis"] for y in yours]
    fields = sum(y["field"] for y in yours)
    versus = [r["score"]["vs_ensemble"] for r in done
              if r["score"].get("vs_ensemble") is not None]
    total.update({
        "mean_rel_wis": round(sum(rels) / len(rels), 3),
        "beat_baseline": sum(1 for r in rels if r < 1),
        "mean_rank": round(sum(y["rank"] for y in yours) / len(yours), 1),
        "mean_field": round(fields / len(yours), 1),
        "beat_share": (round(sum(y["beat"] for y in yours) / fields, 3)
                       if fields else None),
        "beat_ensemble": sum(1 for v in versus if v < 1),
        "n_vs_ensemble": len(versus),
    })
    return total
