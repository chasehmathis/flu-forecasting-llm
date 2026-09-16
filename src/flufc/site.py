"""The forecasts, as a page a person can read.

A CSV of twenty-three quantiles per horizon is the submission format, not a way
of telling anyone what the model thinks. This turns the archive into a small
static site -- one page listing what was forecast most recently and how the
record looks, one page per forecast with the chart, the reasoning and the full
leaderboard -- and writes it into `docs/`, which is what GitHub Pages serves.

Everything is generated from files already in the repository plus public data,
so it rebuilds anywhere. That matters: making a forecast needs the GPU, but
finding out how it did needs nothing but the internet, and the two halves of
the pipeline run in different places because of it.
"""

import datetime as dt
import html
import json
import re
from pathlib import Path

from . import archive, charts
from .compete import BASELINE, ENSEMBLE
from .standings import (preliminary, preliminary_summary, refresh, scored,
                        summary)

REPO_URL = "https://github.com/chasehmathis/flu-forecasting-llm"
HISTORY_WEEKS = 26


def _esc(text):
    return html.escape(str(text if text is not None else ""))


def _n(value, digits=0):
    if value is None:
        return "&mdash;"
    return f"{value:,.{digits}f}"


def _pct(value):
    return "&mdash;" if value is None else f"{value:.0%}"


# -- the build ----------------------------------------------------------------

def build(root, out, *, cache_dir=None, truth=None, force=False,
          history_fn=None, records=None, on_event=None):
    """Regenerate the whole site. Returns the paths written."""
    out = Path(out)
    found = records if records is not None else refresh(
        root, cache_dir=cache_dir, truth=truth, force=force, on_event=on_event)
    series = _histories(found, history_fn, on_event)
    for record in found:
        record["nhsn"] = preliminary(record, series.get(record["location"], []))
    histories = {loc: s[-HISTORY_WEEKS:] for loc, s in series.items()}

    out.mkdir(parents=True, exist_ok=True)
    (out / "f").mkdir(exist_ok=True)
    written = [_write(out / "style.css", STYLE),
               _write(out / ".nojekyll", "")]

    for record in found:
        page = _detail(record, histories.get(record["location"], []))
        written.append(_write(out / "f" / f"{record['slug']}.html", page))

    written.append(_write(out / "index.html",
                          _index(found, histories)))
    written.append(_write(out / "standings.json",
                          json.dumps(_feed(found), indent=2) + "\n"))
    return written


# Every page carries the time it was built, which would otherwise make the
# nightly rebuild look like a change on a day when nothing happened -- a commit
# a week, every week, recording only that the clock moved. Files are compared
# with the timestamps masked out and left alone if that is all that differs.
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?"
                   r"|\d{1,2} \w{3} \d{4}, \d{2}:\d{2} UTC")


def _write(path, text):
    if path.exists():
        try:
            if STAMP.sub("", path.read_text()) == STAMP.sub("", text):
                return path
        except OSError:
            pass
    path.write_text(text)
    return path


def _histories(found, history_fn, on_event=None):
    """NHSN history per location, for the charts and the preliminary scores.
    Optional: no chart is better than no site, so a failed fetch is not fatal.

    Reaches back far enough to cover the oldest forecast's target weeks, not
    just the chart's window."""
    if history_fn is None:
        oldest = min((r["reference_date"] for r in found), default=None)
        weeks = HISTORY_WEEKS
        if oldest:
            age = (dt.date.today() - dt.date.fromisoformat(oldest)).days // 7
            weeks += max(0, age)
        from .tools import ToolError
        from .tools.surveillance import admissions_series

        def history_fn(location):
            try:
                return admissions_series(location, weeks=weeks)[1]
            except (ToolError, OSError):
                return []

    out = {}
    for location in {r["location"] for r in found}:
        if on_event:
            on_event("history", location=location)
        try:
            out[location] = history_fn(location) or []
        except Exception:                       # a chart is never worth a crash
            out[location] = []
    return out


def _feed(found):
    """A machine-readable summary, for anyone who would rather not scrape."""
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc)
                          .isoformat(timespec="seconds"),
        "summary": summary(found),
        "nhsn_preliminary_summary": preliminary_summary(found),
        "forecasts": [{
            "reference_date": r["reference_date"],
            "location": r["location"],
            "location_name": r.get("location_name"),
            "model_id": r.get("model_id"),
            "created_at": r.get("created_at"),
            "medians": r.get("medians"),
            "target_end_dates": r.get("target_end_dates"),
            "score": _slim(r.get("score")),
            "nhsn_preliminary": _slim_nhsn(r.get("nhsn")),
        } for r in found],
    }


def _slim(score):
    if not score:
        return None
    keep = ("scored_horizons", "unscored_horizons", "truth_last_date",
            "baseline_wis", "vs_ensemble", "observed")
    out = {k: score.get(k) for k in keep if score.get(k) is not None}
    if score.get("you"):
        out["you"] = {k: score["you"].get(k)
                      for k in ("rank", "field", "beat", "mean_wis", "rel_wis",
                                "mean_abs_error", "coverage_50", "coverage_95")}
    return out


def _slim_nhsn(nhsn):
    if not nhsn:
        return None
    keep = ("truth_last_date", "n_scored", "mean_wis", "rel_wis",
            "baseline_wis", "mean_abs_error", "coverage_50", "coverage_95")
    return {k: nhsn.get(k) for k in keep if nhsn.get(k) is not None}


# -- reading a score ----------------------------------------------------------

def _status(record):
    """One sentence on where this forecast stands. The headline of every card."""
    score = record.get("score")
    if not score:
        return ("pending", "Not scored yet &mdash; the published truth for "
                           "these weeks has not landed.")
    you = score.get("you")
    if not you:
        weeks = ", ".join(sorted(score.get("target_end_dates", {}).values()))
        return ("pending",
                f"Waiting on truth. The hub has published through "
                f"<b>{_esc(score.get('truth_last_date') or 'nothing yet')}</b>; "
                f"this forecast covers {_esc(weeks)}.")

    bits = [f"Ranked <b>{you['rank']} of {you['field'] + 1}</b> teams"]
    if you.get("rel_wis") is not None:
        verdict = "better than" if you["rel_wis"] < 1 else "worse than"
        bits.append(f"WIS <b>{you['rel_wis']:.2f}x</b> the flat baseline "
                    f"({verdict} doing nothing)")
    if score.get("vs_ensemble") is not None:
        gap = score["vs_ensemble"]
        verdict = "beat" if gap < 1 else "did not beat"
        bits.append(f"{verdict} the hub ensemble ({gap:.2f}x)")
    tone = "good" if (you.get("rel_wis") or 9) < 1 else "bad"
    tail = ""
    if score.get("unscored_horizons"):
        tail = (" Horizons " + ", ".join(str(h) for h in
                                         score["unscored_horizons"])
                + " are still waiting on truth.")
    return (tone, " &middot; ".join(bits) + "." + tail)


def _observed(record):
    score = record.get("score") or {}
    raw = score.get("observed") or {}
    ends = record.get("target_end_dates") or {}
    return {ends[h]: v for h, v in raw.items()
            if h in ends and v is not None}


def _others(record):
    """The hub's medians for the same week, keyed by target week."""
    score = record.get("score") or {}
    ends = record.get("target_end_dates") or {}
    out = {}
    for model, medians in (score.get("hub_medians") or {}).items():
        series = {ends[h]: v for h, v in medians.items() if h in ends}
        if series:
            out[model] = series
    return out


def _chart(record, history, observed=None):
    quantiles = archive.quantiles(record)
    ends = record.get("target_end_dates") or {}
    forecast = {ends[str(h)]: q for h, q in quantiles.items()
                if str(h) in ends}
    if not forecast:
        return ""
    if observed is None:
        observed = _observed(record)
    return charts.fan(history, forecast, observed=observed,
                      others=_others(record),
                      title=f"{record.get('location_name')} forecast from "
                            f"{record['reference_date']}")


# -- the index ----------------------------------------------------------------

def _index(found, histories):
    if not found:
        return _page("No forecasts yet", "", EMPTY, depth=0)

    hub = [_summary_strip(summary(found))]
    for record, note in _front(found):
        hub.append(_card(record, histories.get(record["location"], []), note))
    hub.append(_track_section(found))

    nhsn = [_nhsn_caveat(found), _nhsn_strip(preliminary_summary(found))]
    for record in archive.latest_per_location(found):
        nhsn.append(_nhsn_card(record, histories.get(record["location"], [])))
    nhsn.append(_nhsn_track(found))

    body = [_tabs(hub, nhsn), _glossary()]
    return _page("Flu forecasts", _subtitle(found), "\n".join(body), depth=0)


def _front(found):
    """What goes at the top: the newest forecast for each location, and -- if
    that one is still waiting on truth -- the newest one that has a result.

    A page whose only card says "not scored yet" tells a visitor nothing about
    whether any of this works.
    """
    cards = []
    for record in archive.latest_per_location(found):
        cards.append((record, "latest forecast"))
        if (record.get("score") or {}).get("you"):
            continue
        settled = next((r for r in found
                        if r["location"] == record["location"]
                        and (r.get("score") or {}).get("you")), None)
        if settled:
            cards.append((settled, "most recent result"))
    return cards


def _subtitle(found):
    updated = dt.datetime.now(dt.timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    return (f"A language model forecasts US influenza hospital admissions "
            f"every week, and is scored against the {_esc(ENSEMBLE)} and the "
            f"forty-odd other teams who submit to the CDC's FluSight hub. "
            f"{len(found)} forecast(s) on the record. Rebuilt "
            f"{_esc(updated)}.")


def _summary_strip(totals):
    if not totals.get("n_scored"):
        return (f'<div class="strip"><div class="stat"><b>'
                f'{totals["n_forecasts"]}</b><span>forecasts made</span></div>'
                f'<div class="stat"><b>{totals["n_pending"]}</b>'
                f'<span>waiting on truth</span></div></div>')
    cells = [
        (f"{totals['mean_rel_wis']:.2f}x", "mean WIS vs. baseline"),
        (f"{totals['beat_baseline']}/{totals['n_scored']}", "beat the baseline"),
        (f"{totals['mean_rank']:.0f}",
         f"mean rank of ~{totals['mean_field'] + 1:.0f}"),
        (f"{totals['beat_share']:.0%}" if totals.get("beat_share") else "&mdash;",
         "of teams beaten"),
        (f"{totals['beat_ensemble']}/{totals['n_vs_ensemble']}",
         "beat the ensemble"),
        (f"{totals['n_pending']}", "still pending"),
    ]
    stats = "".join(f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
                    for value, label in cells)
    return f'<div class="strip">{stats}</div>'


def _card(record, history, note=""):
    tone, status = _status(record)
    chart = _chart(record, history)
    tag = f'<span class="tag">{_esc(note)}</span>' if note else ""
    return f"""
<section class="card">
  <div class="card-head">
    <h2>{_esc(record.get('location_name') or record['location'])}
        <span class="muted">week of {_esc(record['reference_date'])}</span>
        {tag}</h2>
    <a class="more" href="f/{_esc(record['slug'])}.html">full detail &rarr;</a>
  </div>
  <p class="status {tone}">{status}</p>
  {chart}
  {_legend(record)}
  {_horizon_table(record)}
  {_rationale(record, short=True)}
</section>"""


def _legend(record, observed=None):
    others = _others(record)
    keys = ["<span class=\"key k-history\"></span>observed admissions",
            "<span class=\"key k-median\"></span>model median",
            "<span class=\"key k-band\"></span>50% / 95% interval"]
    for i, model in enumerate(sorted(others)):
        keys.append(f'<span class="key k-other other-{i}"></span>{_esc(model)}')
    if _observed(record) if observed is None else observed:
        keys.append('<span class="key k-truth"></span>what happened')
    return '<p class="legend">' + " ".join(keys) + "</p>"


def _horizon_table(record):
    """Your medians beside the hub's, week by week."""
    ends = record.get("target_end_dates") or {}
    medians = record.get("medians") or {}
    if not ends:
        return ""
    others = (record.get("score") or {}).get("hub_medians") or {}
    observed = ((record.get("score") or {}).get("observed") or {})
    hub_models = sorted(others)

    head = ["horizon", "week ending", "model median"]
    head += [m.replace("FluSight-", "") for m in hub_models]
    if any(v is not None for v in observed.values()):
        head.append("observed")

    rows = []
    for horizon in sorted(ends, key=int):
        mine = medians.get(horizon)
        cells = [f"+{horizon}" if int(horizon) > 0 else str(horizon),
                 _esc(ends[horizon]), f"<b>{_n(mine)}</b>"]
        for model in hub_models:
            theirs = others[model].get(horizon)
            cell = _n(theirs)
            if theirs and mine:
                cell += f' <span class="muted">({mine / theirs:.2f}x)</span>'
            cells.append(cell)
        if "observed" in head:
            value = observed.get(horizon)
            cells.append(f"<b>{_n(value)}</b>" if value is not None
                         else '<span class="muted">pending</span>')
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    header = "".join(f"<th>{_esc(h)}</th>" for h in head)
    return (f'<div class="scroll"><table class="grid">'
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _rationale(record, short=False):
    text = (record.get("rationale") or "").strip()
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if short and len(paragraphs) > 1:
        paragraphs = paragraphs[:1] + ["&hellip;"]
    body = "".join(f"<p>{_esc(p) if p != '&hellip;' else p}</p>"
                   for p in paragraphs)
    return (f'<div class="why"><h3>Why this forecast</h3>{body}</div>')


def _track_section(found):
    done = scored(found)
    rows = []
    for record in found:
        score = record.get("score") or {}
        you = score.get("you") or {}
        rank = (f"{you['rank']} / {you['field'] + 1}"
                if you.get("rank") else '<span class="muted">pending</span>')
        rel = you.get("rel_wis")
        rel_cell = ('<span class="muted">&mdash;</span>' if rel is None else
                    f'<span class="{"good" if rel < 1 else "bad"}">'
                    f'{rel:.2f}x</span>')
        versus = score.get("vs_ensemble")
        rows.append(
            "<tr>"
            f'<td><a href="f/{_esc(record["slug"])}.html">'
            f'{_esc(record["reference_date"])}</a></td>'
            f"<td>{_esc(record.get('location_name') or record['location'])}</td>"
            f"<td>{rank}</td>"
            f"<td>{rel_cell}</td>"
            f"<td>{'&mdash;' if versus is None else f'{versus:.2f}x'}</td>"
            f"<td>{_n(you.get('mean_wis'), 1)}</td>"
            f"<td>{_pct(you.get('coverage_95'))}</td>"
            "</tr>")

    chart = charts.track([
        {"rel_wis": r["score"]["you"]["rel_wis"],
         "label": f"{r['reference_date']} {r['location']}",
         "short": r["reference_date"][5:]}
        for r in reversed(done)])

    return f"""
<section class="card">
  <div class="card-head"><h2>Track record</h2></div>
  <p class="muted">Every forecast in the archive, scored the moment its week's
     truth is published. Relative WIS is this forecast's score divided by
     {_esc(BASELINE)}'s for the same weeks: below 1.0 beat doing nothing.</p>
  {chart}
  <div class="scroll"><table class="grid">
    <thead><tr><th>week of</th><th>location</th><th>rank</th>
      <th>vs. baseline</th><th>vs. ensemble</th><th>WIS</th>
      <th>95% coverage</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


def _glossary():
    return """
<section class="card note">
  <h3>How to read this</h3>
  <p><b>WIS</b> (the weighted interval score) is what FluSight ranks on. It
     rewards intervals that are narrow and punishes ones that miss, and for a
     forecast with no uncertainty it is just the absolute error &mdash; so it
     compares models that express very different amounts of doubt. Lower is
     better.</p>
  <p><b>Relative WIS</b> divides that by the flat baseline's score for the same
     weeks. The absolute number mostly measures the week: a quiet week is easy
     for everyone and a turning point is hard for everyone. The ratio is what
     transfers.</p>
  <p><b>Rank</b> is against every team that submitted to the hub for the same
     reference date and covered every horizon with published truth. A team that
     covered half the horizons is answering an easier question and is left
     out rather than allowed to win on it.</p>
  <p><b>Coverage</b> is how often the truth fell inside the stated interval.
     A well calibrated 95% interval contains the answer 95% of the time; much
     more than that means the forecast is hedging.</p>
</section>"""


# -- the preliminary NHSN view -----------------------------------------------

def _tabs(hub, nhsn):
    """Two views of the same forecasts, one per source of truth. Without
    JavaScript both are shown, one after the other."""
    def pane(name, parts):
        inner = "\n".join(p for p in parts if p)
        return f'<div class="pane" id="pane-{name}" data-pane="{name}">{inner}</div>'
    return f"""
<nav class="tabs" role="tablist">
  <a class="tab" role="tab" href="#hub" data-tab="hub">FluSight hub truth</a>
  <a class="tab" role="tab" href="#nhsn" data-tab="nhsn">NHSN (preliminary)</a>
</nav>
{pane("hub", hub)}
{pane("nhsn", nhsn)}"""


def _nhsn_caveat(found):
    last = max(((r.get("nhsn") or {}).get("truth_last_date") or ""
                for r in found), default="")
    return f"""
<p class="status pending">Testing numbers. Scored against the latest NHSN
   release, published through <b>{_esc(last or "nothing yet")}</b>, which the
   CDC still revises for recent weeks. Out of season there is no FluSight field
   to rank against, so &ldquo;vs. flat&rdquo; is a flat baseline built here
   from the last NHSN week before each reference date &mdash; not
   {_esc(BASELINE)}. The hub tab is the official record.</p>"""


def _nhsn_status(record):
    nhsn = record.get("nhsn") or {}
    if not nhsn.get("n_scored"):
        weeks = ", ".join(sorted((record.get("target_end_dates") or {}).values()))
        return ("pending",
                f"NHSN has published through "
                f"<b>{_esc(nhsn.get('truth_last_date') or 'nothing yet')}</b>; "
                f"this forecast covers {_esc(weeks)}.")
    bits = [f"<b>{nhsn['n_scored']} of {nhsn['n_total']}</b> horizons have "
            f"NHSN data",
            f"WIS <b>{nhsn['mean_wis']:,.1f}</b>"]
    rel = nhsn.get("rel_wis")
    if rel is not None:
        verdict = "better" if rel < 1 else "worse"
        bits.append(f"<b>{rel:.2f}x</b> the flat baseline's WIS ({verdict})")
    bits.append(f"95% interval coverage {_pct(nhsn.get('coverage_95'))}")
    tone = "pending" if rel is None else ("good" if rel < 1 else "bad")
    return (tone, " &middot; ".join(bits) + ".")


def _nhsn_observed(record):
    return {r["target_end_date"]: r["observed"]
            for r in (record.get("nhsn") or {}).get("rows", [])
            if r["observed"] is not None}


def _nhsn_card(record, history):
    tone, status = _nhsn_status(record)
    observed = _nhsn_observed(record)
    return f"""
<section class="card">
  <div class="card-head">
    <h2>{_esc(record.get('location_name') or record['location'])}
        <span class="muted">week of {_esc(record['reference_date'])}</span>
        <span class="tag">latest forecast</span></h2>
    <a class="more" href="f/{_esc(record['slug'])}.html#nhsn">full detail &rarr;</a>
  </div>
  <p class="status {tone}">{status}</p>
  {_chart(record, history, observed)}
  {_legend(record, observed)}
  {_nhsn_table(record)}
</section>"""


def _nhsn_table(record):
    rows = (record.get("nhsn") or {}).get("rows") or []
    if not rows:
        return ""
    body = []
    for r in rows:
        h = r["horizon"]
        pending = r["observed"] is None
        cells = [f"+{h}" if h > 0 else str(h), _esc(r["target_end_date"]),
                 f"<b>{_n(r['median'])}</b>",
                 '<span class="muted">pending</span>' if pending
                 else f"<b>{_n(r['observed'])}</b>",
                 _n(r["wis"], 1), _n(r["abs_error"]),
                 _yes(r["cover_50"]), _yes(r["cover_95"])]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = "".join(f"<th>{h}</th>" for h in
                   ["horizon", "week ending", "model median", "NHSN",
                    "WIS", "abs error", "in 50%", "in 95%"])
    return (f'<div class="scroll"><table class="grid">'
            f"<thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _yes(value):
    if value is None:
        return "&mdash;"
    return '<span class="good">yes</span>' if value else '<span class="bad">no</span>'


def _nhsn_strip(totals):
    if not totals.get("n_scored"):
        return (f'<div class="strip"><div class="stat"><b>'
                f'{totals["n_forecasts"]}</b><span>forecasts made</span></div>'
                f'<div class="stat"><b>0</b><span>horizons with NHSN data'
                f'</span></div></div>')
    rel = totals.get("mean_rel_wis")
    cells = [
        (f"{totals['n_scored']}/{totals['n_total']}", "horizons with NHSN data"),
        (f"{totals['mean_wis']:,.1f}", "mean WIS"),
        ("&mdash;" if rel is None else f"{rel:.2f}x", "mean WIS vs. flat"),
        (f"{totals.get('beat_baseline', 0)}/{totals.get('n_rel', 0)}",
         "beat the flat baseline"),
        (_pct(totals.get("coverage_50")), "50% coverage"),
        (_pct(totals.get("coverage_95")), "95% coverage"),
    ]
    stats = "".join(f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
                    for value, label in cells)
    return f'<div class="strip">{stats}</div>'


def _nhsn_track(found):
    rows = []
    for record in found:
        nhsn = record.get("nhsn") or {}
        rel = nhsn.get("rel_wis")
        rel_cell = ('<span class="muted">&mdash;</span>' if rel is None else
                    f'<span class="{"good" if rel < 1 else "bad"}">'
                    f'{rel:.2f}x</span>')
        rows.append(
            "<tr>"
            f'<td><a href="f/{_esc(record["slug"])}.html#nhsn">'
            f'{_esc(record["reference_date"])}</a></td>'
            f"<td>{_esc(record.get('location_name') or record['location'])}</td>"
            f"<td>{nhsn.get('n_scored', 0)} / {nhsn.get('n_total', 0)}</td>"
            f"<td>{_n(nhsn.get('mean_wis'), 1)}</td>"
            f"<td>{rel_cell}</td>"
            f"<td>{_n(nhsn.get('mean_abs_error'))}</td>"
            f"<td>{_pct(nhsn.get('coverage_50'))}</td>"
            f"<td>{_pct(nhsn.get('coverage_95'))}</td>"
            "</tr>")
    return f"""
<section class="card">
  <div class="card-head"><h2>Track record (preliminary)</h2></div>
  <p class="muted">Every forecast in the archive, scored on whichever horizons
     NHSN has already published. Recomputed on every build, never cached.</p>
  <div class="scroll"><table class="grid">
    <thead><tr><th>week of</th><th>location</th><th>horizons</th>
      <th>WIS</th><th>vs. flat</th><th>MAE</th><th>50%</th>
      <th>95%</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


# -- the detail page ----------------------------------------------------------

def _detail(record, history):
    tone, status = _status(record)
    score = record.get("score") or {}
    title = (f"{record.get('location_name') or record['location']} "
             f"&mdash; {record['reference_date']}")
    hub = [f'<p class="status {tone}">{status}</p>',
           _chart(record, history), _legend(record),
           _horizon_table(record), _leaderboard(score)]
    nhsn_tone, nhsn_status = _nhsn_status(record)
    observed = _nhsn_observed(record)
    nhsn = [_nhsn_caveat([record]),
            f'<p class="status {nhsn_tone}">{nhsn_status}</p>',
            _chart(record, history, observed), _legend(record, observed),
            _nhsn_table(record)]
    body = [_tabs(hub, nhsn),
            _rationale(record),
            _quantile_table(record),
            _provenance(record)]
    return _page(title, _detail_subtitle(record), "\n".join(b for b in body if b),
                 depth=1)


def _detail_subtitle(record):
    run = record.get("run") or {}
    bits = [f"Forecast made {_esc(record.get('created_at', 'unknown'))}",
            f"by <code>{_esc(record.get('model_id', 'unknown'))}</code>"]
    if run.get("tool_calls"):
        bits.append(f"after {run['tool_calls']} tool calls")
    return ", ".join(bits) + "."


def _quantile_table(record):
    quantiles = archive.quantiles(record)
    if not quantiles:
        return ""
    ends = record.get("target_end_dates") or {}
    levels = [0.025, 0.25, 0.5, 0.75, 0.975]
    rows = []
    for horizon in sorted(quantiles):
        q = quantiles[horizon]
        cells = [f"+{horizon}" if horizon > 0 else str(horizon),
                 _esc(ends.get(str(horizon), ""))]
        cells += [_n(q.get(level)) for level in levels]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    header = "".join(f"<th>{h}</th>" for h in
                     ["horizon", "week ending", "2.5%", "25%", "median",
                      "75%", "97.5%"])
    return f"""
<section class="card">
  <div class="card-head"><h2>The submission</h2></div>
  <p class="muted">The model states all 23 FluSight quantile levels itself, so
     the distribution below is its own and not an interpolation of a few
     anchors. Five of the 23 are shown; the rest are in the CSV.</p>
  <div class="scroll"><table class="grid">
    <thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody>
  </table></div>
</section>"""


def _leaderboard(score):
    entries = score.get("entries") or []
    if not entries:
        return ""
    horizons = score.get("scored_horizons") or []
    rows = []
    for entry in entries:
        klass = ("you" if entry.get("is_you") else
                 "hub" if entry["model"] in (ENSEMBLE, BASELINE) else "")
        rel = entry.get("rel_wis")
        rows.append(
            f'<tr class="{klass}">'
            f"<td>{entry.get('rank')}</td>"
            f"<td>{_esc(entry['model'])}</td>"
            f"<td>{_n(entry.get('mean_wis'), 1)}</td>"
            f"<td>{'&mdash;' if rel is None else f'{rel:.2f}'}</td>"
            f"<td>{_n(entry.get('mean_abs_error'), 1)}</td>"
            f"<td>{_pct(entry.get('coverage_50'))}</td>"
            f"<td>{_pct(entry.get('coverage_95'))}</td>"
            "</tr>")
    weeks = score.get("target_end_dates") or {}
    covered = ", ".join(weeks[str(h)] for h in horizons if str(h) in weeks)
    left_out = ""
    if score.get("n_partial") or score.get("n_missing"):
        left_out = (f' {score.get("n_missing", 0)} team(s) did not submit for '
                    f'this week and {score.get("n_partial", 0)} covered only '
                    f'some horizons; both are left out of the ranking.')
    return f"""
<section class="card">
  <div class="card-head"><h2>Against the field</h2></div>
  <p class="muted">Every FluSight team scored on the same weeks
     ({_esc(covered)}) with the same truth.{left_out}</p>
  <div class="scroll"><table class="grid board">
    <thead><tr><th>rank</th><th>model</th><th>WIS</th><th>rel</th>
      <th>MAE</th><th>50%</th><th>95%</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
</section>"""


def _provenance(record):
    run = record.get("run") or {}
    tools = run.get("tools") or []
    counted = {}
    for name in tools:
        counted[name] = counted.get(name, 0) + 1
    used = ", ".join(f"{name} &times;{n}" if n > 1 else name
                     for name, n in sorted(counted.items()))
    csv_link = (f"{REPO_URL}/blob/main/forecasts/"
                f"{record['reference_date']}/{record['location']}.csv")
    rows = [("made", record.get("created_at")),
            ("model", record.get("model_id")),
            ("stopped", run.get("stopped")),
            ("tool calls", run.get("tool_calls")),
            ("wall clock", f"{run['seconds']:.0f}s" if run.get("seconds")
             else None),
            ("tokens", f"{run.get('prompt_tokens', 0):,} prompt / "
                       f"{run.get('completion_tokens', 0):,} completion"
             if run else None),
            ("tools used", used or None)]
    body = ""
    for key, value in rows:
        if not value:
            continue
        cell = value if key == "tools used" else _esc(value)
        body += f"<tr><th>{_esc(key)}</th><td>{cell}</td></tr>"
    return f"""
<section class="card">
  <div class="card-head"><h2>Provenance</h2></div>
  <table class="grid meta"><tbody>{body}</tbody></table>
  <p class="muted"><a href="{_esc(csv_link)}">the submission CSV</a> as it was
     written, in the repository.</p>
</section>"""


# -- the shell ----------------------------------------------------------------

EMPTY = """
<section class="card">
  <h2>Nothing forecast yet</h2>
  <p>Run <code>flufc weekly</code> on a machine with the model up, and this
     page fills in.</p>
</section>"""


def _page(title, subtitle, body, depth=0):
    up = "../" * depth
    head = _strip(title)
    if "forecast" not in head.lower():
        head += " &middot; flu forecasts"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{head}</title>
<link rel="stylesheet" href="{up}style.css">
<script>{TABS_JS}</script>
</head>
<body>
<header class="top">
  <div class="wrap">
    <p class="crumb"><a href="{up}index.html">flu-forecasting-llm</a>
       &middot; <a href="{REPO_URL}">source</a></p>
    <h1>{title}</h1>
    <p class="lede">{subtitle}</p>
  </div>
</header>
<main class="wrap">
{body}
</main>
<footer class="wrap">
  <p class="muted">Forecasts are made by a locally served Qwen with tools, and
     scored with the weighted interval score against the FluSight hub's
     published truth. Everything here is generated from the CSVs in
     <code>forecasts/</code> by <code>flufc site</code>.</p>
</footer>
</body>
</html>
"""


# Picks a pane from the URL hash, else the last one chosen on any page, else
# the hub. Storage can be unavailable; the page works without it.
TABS_JS = """
document.documentElement.classList.add("js");
document.addEventListener("DOMContentLoaded", function () {
  var KEY = "flufc-source", names = ["hub", "nhsn"];
  function show(name) {
    if (names.indexOf(name) < 0) name = "hub";
    document.querySelectorAll("[data-pane]").forEach(function (el) {
      el.classList.toggle("on", el.dataset.pane === name);
    });
    document.querySelectorAll("[data-tab]").forEach(function (el) {
      var on = el.dataset.tab === name;
      el.classList.toggle("on", on);
      el.setAttribute("aria-selected", on);
    });
    try { localStorage.setItem(KEY, name); } catch (e) {}
    document.querySelectorAll("a[href^='f/'], a[href^='../index.html']")
      .forEach(function (a) { a.hash = name; });
  }
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  show(location.hash.slice(1) || saved);
  document.querySelectorAll("[data-tab]").forEach(function (el) {
    el.addEventListener("click", function (ev) {
      ev.preventDefault();
      history.replaceState(null, "", "#" + el.dataset.tab);
      show(el.dataset.tab);
    });
  });
});
"""


def _strip(text):
    return (text.replace("&mdash;", "-").replace("&middot;", "-")
            .replace("&nbsp;", " "))


STYLE = """
:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1c1b19; --muted: #6b6862;
  --line: #e5e1da; --accent: #1f6feb; --band: rgba(31,111,235,0.16);
  --band-in: rgba(31,111,235,0.30); --good: #17794b; --bad: #b3261e;
  --other0: #b06f00; --other1: #7a56c2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14151a; --panel: #1c1e25; --ink: #e9e7e2; --muted: #9b978f;
    --line: #2d3038; --accent: #6aa3ff; --band: rgba(106,163,255,0.18);
    --band-in: rgba(106,163,255,0.34); --good: #4ec98a; --bad: #ff8a80;
    --other0: #e0a33c; --other1: #b39ce8;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 0 20px; }
.top { border-bottom: 1px solid var(--line); padding: 28px 0 24px;
       margin-bottom: 26px; }
.crumb { margin: 0 0 10px; font-size: 13px; color: var(--muted); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { margin: 0 0 8px; font-size: 27px; letter-spacing: -0.02em; }
h2 { margin: 0; font-size: 19px; letter-spacing: -0.01em; }
h3 { margin: 22px 0 8px; font-size: 15px; text-transform: uppercase;
     letter-spacing: 0.07em; color: var(--muted); }
.lede { margin: 0; color: var(--muted); max-width: 62ch; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 22px;
}
.card-head { display: flex; align-items: baseline; justify-content: space-between;
             gap: 12px; margin-bottom: 12px; }
.card-head .muted { font-weight: 400; font-size: 15px; }
.more { font-size: 14px; white-space: nowrap; }
.tag { font-size: 11px; font-weight: 600; text-transform: uppercase;
       letter-spacing: 0.07em; color: var(--muted); border: 1px solid var(--line);
       border-radius: 999px; padding: 2px 9px; vertical-align: 2px;
       white-space: nowrap; }
.muted { color: var(--muted); }
.good { color: var(--good); }
.bad { color: var(--bad); }
.status {
  margin: 0 0 16px; padding: 11px 14px; border-radius: 8px;
  border-left: 3px solid var(--muted); background: rgba(127,127,127,0.07);
  font-size: 15px;
}
.status.good { border-left-color: var(--good); }
.status.bad { border-left-color: var(--bad); }
.status.pending { border-left-color: var(--muted); }
.strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; margin-bottom: 22px;
}
.stat { background: var(--panel); padding: 14px 16px; }
.stat b { display: block; font-size: 22px; letter-spacing: -0.02em; }
.stat span { font-size: 12.5px; color: var(--muted); }
.chart { width: 100%; height: auto; display: block; margin: 6px 0 4px; }
.grid, .grid-strong { stroke: var(--line); stroke-width: 1; }
.grid-strong { stroke: var(--muted); stroke-dasharray: 3 3; }
.axis { fill: var(--muted); font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.band-95 { fill: var(--band); }
.band-50 { fill: var(--band-in); }
.line-history { fill: none; stroke: var(--ink); stroke-width: 2;
                stroke-linejoin: round; }
.line-median { fill: none; stroke: var(--accent); stroke-width: 2.2;
               stroke-linejoin: round; }
.dot-median { fill: var(--accent); }
.dot-observed { fill: var(--bad); stroke: var(--panel); stroke-width: 1.5; }
.line-other { fill: none; stroke-width: 1.8; stroke-dasharray: 5 4; }
.other-0 { stroke: var(--other0); }
.other-1 { stroke: var(--other1); }
.bar-good { fill: var(--good); }
.bar-bad { fill: var(--bad); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px;
          color: var(--muted); margin: 2px 0 18px; align-items: center; }
.key { display: inline-block; width: 15px; height: 3px; margin-right: 6px;
       vertical-align: middle; border-radius: 2px; }
.k-history { background: var(--ink); }
.k-median { background: var(--accent); }
.k-band { background: var(--band-in); height: 10px; border-radius: 2px; }
.k-truth { background: var(--bad); height: 9px; width: 9px;
           border-radius: 50%; }
.k-other.other-0 { background: var(--other0); }
.k-other.other-1 { background: var(--other1); }
.scroll { overflow-x: auto; margin: 4px -4px 0; padding: 0 4px; }
table.grid { border-collapse: collapse; width: 100%; font-size: 14px;
             font-variant-numeric: tabular-nums; }
table.grid th, table.grid td { text-align: right; padding: 7px 10px;
                               border-bottom: 1px solid var(--line);
                               white-space: nowrap; }
table.grid th:first-child, table.grid td:first-child,
table.grid th:nth-child(2), table.grid td:nth-child(2) { text-align: left; }
table.grid thead th { font-size: 12px; text-transform: uppercase;
                      letter-spacing: 0.05em; color: var(--muted);
                      font-weight: 600; }
table.board tbody tr.you { background: rgba(31,111,235,0.10);
                           font-weight: 600; }
table.board tbody tr.hub td { color: var(--muted); }
table.meta th { text-align: left; color: var(--muted); font-weight: 500;
                width: 34%; }
table.meta td { text-align: left; white-space: normal; }
.why p { margin: 0 0 10px; max-width: 66ch; }
.note p { max-width: 68ch; font-size: 15px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
       font-size: 0.9em; background: rgba(127,127,127,0.12);
       padding: 1px 5px; border-radius: 4px; }
footer { padding: 8px 20px 48px; }
footer p { max-width: 68ch; font-size: 13.5px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line);
        margin: 0 0 22px; }
.tab { padding: 8px 14px; border: 1px solid transparent; border-bottom: 0;
       border-radius: 8px 8px 0 0; color: var(--muted); font-size: 14px;
       font-weight: 600; margin-bottom: -1px; }
.tab:hover { text-decoration: none; color: var(--ink); }
.tab.on { background: var(--panel); border-color: var(--line);
          color: var(--ink); }
html:not(.js) .tabs { display: none; }
html.js .pane:not(.on) { display: none; }
@media (max-width: 620px) {
  .card { padding: 16px; } h1 { font-size: 23px; }
  .card-head { flex-direction: column; gap: 4px; }
}
"""
