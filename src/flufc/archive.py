"""The forecasts we are willing to be judged on.

`outputs/` is a scratch directory. Every run lands there, including the ones
that went wrong, and none of it is in git. But a forecast only means something
if it was written down before the week happened and cannot be quietly revised
afterwards, so the ones that count go somewhere durable and get committed:

    forecasts/2026-02-07/US.csv          the submission, FluSight format
    forecasts/2026-02-07/US.json         who made it, when, and why
    forecasts/2026-02-07/US.score.json   how it did -- filled in later

The CSV is the record. The JSON beside it carries the model's rationale and
what the run cost, which is the part a reader actually wants and the part the
submission format has nowhere to put. The score file is a cache: working out a
leaderboard means pulling forty-odd hub submissions, and once a week is over
and its truth has settled, that answer never changes again.
"""

import csv
import datetime as dt
import json
from pathlib import Path

# Bump when the shape of a score file changes; stale ones are then recomputed
# rather than misread.
SCORE_VERSION = 1


def paths(root, reference_date, location):
    """The three files that make up one archived forecast."""
    base = Path(root) / reference_date / location
    return (base.with_suffix(".csv"), base.with_suffix(".json"),
            Path(str(base) + ".score.json"))


def has(root, reference_date, location):
    return paths(root, reference_date, location)[0].exists()


def save(root, forecast, *, location_name="", run=None, transcript=None,
         created_at=None):
    """Write a Forecast and its provenance into the archive. Returns the meta."""
    csv_path, meta_path, _ = paths(root, forecast.reference_date,
                                   forecast.location)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.write_csv(csv_path)

    horizons = sorted(forecast.horizons, key=int)
    meta = {
        "reference_date": forecast.reference_date,
        "location": forecast.location,
        "location_name": location_name or forecast.location,
        "target": forecast.target,
        "model_id": forecast.model_id,
        "created_at": (created_at
                       or dt.datetime.now().isoformat(timespec="seconds")),
        "rationale": forecast.rationale,
        "medians": {str(h): forecast.median(h) for h in horizons},
        "target_end_dates": {str(h): forecast.target_end_date(h)
                             for h in horizons},
        "csv": csv_path.name,
    }
    if run is not None:
        meta["run"] = {
            "stopped": run.stopped,
            "tool_calls": run.tool_calls,
            "seconds": round(run.seconds, 1),
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "tools": [name for step in run.steps for name, _ in step.calls],
            "transcript": str(transcript) if transcript else "",
        }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return read(meta_path)


def read(meta_path):
    """One archived forecast as a dict, with its file paths resolved."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    csv_path = meta_path.with_suffix(".csv")
    meta.update({
        "path": str(csv_path),
        "meta_path": str(meta_path),
        "score_path": str(meta_path.with_suffix("")) + ".score.json",
        "slug": f"{meta['reference_date']}-{meta['location']}",
    })
    return meta


def records(root):
    """Every archived forecast, newest reference date first."""
    root = Path(root)
    if not root.exists():
        return []
    found = []
    for meta_path in sorted(root.glob("*/*.json")):
        if meta_path.name.endswith(".score.json"):
            continue
        try:
            record = read(meta_path)
        except (OSError, ValueError, KeyError):
            continue
        if Path(record["path"]).exists():
            found.append(record)
    found.sort(key=lambda m: (m["reference_date"], m["location"]), reverse=True)
    return found


def latest_per_location(found):
    """The most recent forecast for each location, newest first."""
    seen, out = set(), []
    for record in found:
        if record["location"] not in seen:
            seen.add(record["location"])
            out.append(record)
    return out


def read_score(record):
    """The cached score for a record, if it is there and still readable."""
    path = Path(record["score_path"])
    if not path.exists():
        return None
    try:
        score = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return score if score.get("version") == SCORE_VERSION else None


def write_score(record, score):
    score = dict(score, version=SCORE_VERSION,
                 computed_at=dt.datetime.now().isoformat(timespec="seconds"))
    Path(record["score_path"]).write_text(json.dumps(score, indent=2) + "\n")
    return score


def is_settled(score):
    """Can this score still change?

    Only once every horizon has been scored and the published truth has moved
    on past the last target week -- NHSN revises the most recent weeks, so the
    edge of the series is not final just because it exists.
    """
    if not score or not score.get("scored_horizons"):
        return False
    if score.get("unscored_horizons"):
        return False
    ends = score.get("target_end_dates") or {}
    truth_last = score.get("truth_last_date") or ""
    return bool(ends) and truth_last > max(ends.values())


def quantiles(record):
    """{horizon: {level: value}} read back out of the archived CSV."""
    horizons = {}
    with open(record["path"]) as fh:
        for row in csv.DictReader(fh):
            if row.get("output_type") != "quantile":
                continue
            horizon = int(row["horizon"])
            horizons.setdefault(horizon, {})[float(row["output_type_id"])] = \
                float(row["value"])
    return horizons
