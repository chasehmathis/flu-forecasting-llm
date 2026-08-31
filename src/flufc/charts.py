"""Charts, drawn as SVG by hand.

The rest of the harness is stdlib-only so that it runs on a bare login node,
and a plotting library would be the one thing that broke that. It turns out
not to cost much: a fan chart is a filled polygon and two polylines, and
writing it out directly means the result is a few kilobytes of markup that
scales, prints, and inherits the page's colours in dark mode.

Everything here returns a string of SVG. Colours come from CSS custom
properties defined by the page, so the same chart works on either theme.
"""

import datetime as dt
from statistics import NormalDist

WIDTH, HEIGHT = 760, 320
PAD = {"top": 16, "right": 14, "bottom": 34, "left": 54}

# Roughly the width of "28 Feb" plus air, in viewBox units.
MIN_LABEL_GAP = 66


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class Scale:
    """A linear map from data to pixels, with the y axis flipped."""

    def __init__(self, lo, hi, out_lo, out_hi):
        if hi == lo:
            hi = lo + 1.0
        self.lo, self.hi, self.out_lo, self.out_hi = lo, hi, out_lo, out_hi

    def __call__(self, value):
        t = (value - self.lo) / (self.hi - self.lo)
        return self.out_lo + t * (self.out_hi - self.out_lo)


def _ticks(lo, hi, count=5):
    """Round numbers spanning [lo, hi] -- 1, 2, 2.5 or 5 times a power of ten."""
    span = (hi - lo) or 1.0
    raw = span / max(1, count)
    magnitude = 10 ** int(_floor_log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if step >= raw:
            break
    start = step * int(lo / step)
    out, value = [], start
    while value <= hi + step * 0.5:
        if value >= lo - 1e-9:
            out.append(value)
        value += step
    return out


def _floor_log10(x):
    import math
    return math.floor(math.log10(x)) if x > 0 else 0


def _fmt(value):
    if value >= 10000:
        return f"{value / 1000:.0f}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def _day(iso):
    """Days since the epoch, so weeks sit on a real timeline."""
    try:
        return dt.date.fromisoformat(iso).toordinal()
    except ValueError:
        return 0


def _short_date(iso):
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    return d.strftime("%d %b").lstrip("0")


def fan(history, forecast, *, observed=None, others=None, title=""):
    """A forecast in context: recent history, then the quantile fan.

    history   [(week ending, admissions)], oldest first
    forecast  {target_end_date: {quantile level: value}}
    observed  {target_end_date: admissions} -- truth for the forecast weeks
    others    {label: {target_end_date: median}} -- e.g. the hub ensemble
    """
    observed = observed or {}
    others = others or {}
    weeks = sorted(set([d for d, _ in history] + list(forecast)))
    if len(weeks) < 2:
        return ""
    # Weeks are placed by their date, not by their position in the list: a
    # missing week in the surveillance series should show as a gap, not be
    # quietly closed up.
    index = {week: _day(week) for week in weeks}

    values = [v for _, v in history]
    for quantiles in forecast.values():
        values += [quantiles.get(0.025, 0), quantiles.get(0.975, 0)]
    values += [v for v in observed.values() if v is not None]
    for series in others.values():
        values += [v for v in series.values() if v is not None]
    top = max(values or [1]) * 1.12

    x = Scale(index[weeks[0]], index[weeks[-1]],
              PAD["left"], WIDTH - PAD["right"])
    y = Scale(0, top, HEIGHT - PAD["bottom"], PAD["top"])

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="chart" '
             f'role="img" aria-label="{_esc(title or "forecast")}">']
    parts += _grid(x, y, weeks, index, top)

    # The fan hangs off the last observed week, so the bands start where the
    # data stops rather than floating away from it.
    anchor = history[-1] if history else None
    horizons = sorted(forecast)

    for lo, hi, klass in ((0.025, 0.975, "band-95"), (0.25, 0.75, "band-50")):
        points = []
        if anchor:
            points.append((x(index[anchor[0]]), y(anchor[1])))
        points += [(x(index[d]), y(forecast[d][hi])) for d in horizons
                   if hi in forecast[d]]
        back = [(x(index[d]), y(forecast[d][lo])) for d in reversed(horizons)
                if lo in forecast[d]]
        if anchor:
            back.append((x(index[anchor[0]]), y(anchor[1])))
        if len(points) + len(back) > 3:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points + back)
            parts.append(f'<polygon class="{klass}" points="{path}"/>')

    if history:
        parts.append(_line([(x(index[d]), y(v)) for d, v in history],
                           "line-history"))

    for i, (label, series) in enumerate(sorted(others.items())):
        pts = ([(x(index[anchor[0]]), y(anchor[1]))] if anchor else [])
        pts += [(x(index[d]), y(v)) for d, v in sorted(series.items())
                if d in index and v is not None]
        if len(pts) > 1:
            parts.append(_line(pts, f"line-other other-{i}"))

    medians = [(x(index[d]), y(forecast[d][0.5])) for d in horizons
               if 0.5 in forecast[d]]
    if medians:
        pts = ([(x(index[anchor[0]]), y(anchor[1]))] if anchor else []) + medians
        parts.append(_line(pts, "line-median"))
        parts += [f'<circle class="dot-median" cx="{px:.1f}" cy="{py:.1f}" '
                  f'r="3.5"/>' for px, py in medians]

    for date, value in sorted(observed.items()):
        if value is None or date not in index:
            continue
        parts.append(f'<circle class="dot-observed" cx="{x(index[date]):.1f}" '
                     f'cy="{y(value):.1f}" r="4"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def _line(points, klass):
    # fill="none" is a presentation attribute rather than a rule in the
    # stylesheet, so a polyline degrades to a line and not a filled blob if the
    # CSS is ever missing.
    path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    return f'<polyline class="{klass}" fill="none" points="{path}"/>'


def _grid(x, y, weeks, index, top):
    parts = []
    for value in _ticks(0, top):
        py = y(value)
        parts.append(f'<line class="grid" x1="{PAD["left"]}" y1="{py:.1f}" '
                     f'x2="{WIDTH - PAD["right"]}" y2="{py:.1f}"/>')
        parts.append(f'<text class="axis" x="{PAD["left"] - 8}" '
                     f'y="{py + 4:.1f}" text-anchor="end">{_fmt(value)}</text>')
    # Labels are spaced by how far apart they land, not by how many weeks sit
    # between them: the axis is a date scale, so counting list positions puts
    # two labels on top of each other wherever the series has a gap.
    last = None
    for week in weeks:
        px = x(index[week])
        if last is not None and px - last < MIN_LABEL_GAP:
            continue
        last = px
        # Pull the end labels inside the frame instead of letting them hang
        # off the edge of the viewBox.
        anchor = ("start" if px <= PAD["left"] + 2 else
                  "end" if px >= WIDTH - PAD["right"] - 2 else "middle")
        parts.append(f'<text class="axis" x="{px:.1f}" '
                     f'y="{HEIGHT - PAD["bottom"] + 18:.1f}" '
                     f'text-anchor="{anchor}">{_short_date(week)}</text>')
    return parts


def track(points, *, height=190):
    """Relative WIS over time: one bar per scored forecast, 1.0 is the baseline.

    Below the line is a forecast that beat doing nothing. The scale is
    logarithmic because "half the baseline" and "twice the baseline" are the
    same size of achievement in opposite directions, and a linear axis hides
    the first one.
    """
    import math

    points = [p for p in points if p.get("rel_wis")]
    if not points:
        return ""
    width, pad = WIDTH, {"top": 14, "right": 14, "bottom": 32, "left": 54}
    rels = [p["rel_wis"] for p in points]
    span = max(0.35, max(abs(math.log10(r)) for r in rels) * 1.25)
    y = Scale(-span, span, height - pad["bottom"], pad["top"])
    step = (width - pad["left"] - pad["right"]) / max(1, len(points))
    bar = min(38, step * 0.62)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
             f'aria-label="relative WIS by forecast">']
    for value in (0.5, 1.0, 2.0):
        if abs(math.log10(value)) > span:
            continue
        py = y(math.log10(value))
        klass = "grid-strong" if value == 1.0 else "grid"
        parts.append(f'<line class="{klass}" x1="{pad["left"]}" y1="{py:.1f}" '
                     f'x2="{width - pad["right"]}" y2="{py:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad["left"] - 8}" y="{py + 4:.1f}"'
                     f' text-anchor="end">{value:g}x</text>')

    zero = y(0.0)
    for i, point in enumerate(points):
        cx = pad["left"] + step * (i + 0.5)
        py = y(math.log10(point["rel_wis"]))
        top, bottom = min(py, zero), max(py, zero)
        klass = "bar-good" if point["rel_wis"] < 1 else "bar-bad"
        parts.append(f'<rect class="{klass}" x="{cx - bar / 2:.1f}" '
                     f'y="{top:.1f}" width="{bar:.1f}" '
                     f'height="{max(1.0, bottom - top):.1f}" rx="2">'
                     f'<title>{_esc(point["label"])}: '
                     f'{point["rel_wis"]:.2f}x baseline</title></rect>')
        if len(points) <= 14 or i % 2 == 0:
            parts.append(f'<text class="axis" x="{cx:.1f}" '
                         f'y="{height - pad["bottom"] + 18:.1f}" '
                         f'text-anchor="middle">'
                         f'{_esc(point["short"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def sparkline(values, *, width=120, height=26):
    """A tiny inline trend, for a table cell."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return ""
    x = Scale(0, len(values) - 1, 1, width - 1)
    y = Scale(min(values), max(values), height - 2, 2)
    path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 {width} {height}" class="spark" '
            f'aria-hidden="true"><polyline points="{path}"/></svg>')


def z(level):
    return NormalDist().inv_cdf(level)
