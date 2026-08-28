"""Influenza surveillance data.

Two independent signals, which is the point: NHSN hospital admissions are the
quantity FluSight scores, and ILINet outpatient visits are a leading-ish
indicator with a much longer history. Disagreement between them is information.

Sources
    NHSN weekly hospital respiratory data  data.cdc.gov resource mpgq-jmmr
    ILINet via Delphi Epidata              api.delphi.cmu.edu/epidata/fluview
    Location table                         cdcepi/FluSight-forecast-hub
"""

import csv
import datetime as dt
import io
import urllib.parse

from . import ToolError, tool
from ..cache import FetchError, fetch, fetch_json
from ..config import active

NHSN_URL = "https://data.cdc.gov/resource/mpgq-jmmr.json"
DELPHI_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
LOCATIONS_URL = ("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub"
                 "/main/auxiliary-data/locations.csv")

# NHSN calls the nation "USA"; FluSight calls it "US".
_NHSN_NATIONAL = "USA"

_locations_cache = None


def locations():
    """The FluSight location table, keyed every way a model might refer to it."""
    global _locations_cache
    if _locations_cache is None:
        body = fetch(LOCATIONS_URL, timeout=30, ttl=7 * 86400,
                     cache_dir=active().cache_dir)
        rows = list(csv.DictReader(io.StringIO(body)))
        _locations_cache = [
            {"abbreviation": r["abbreviation"], "location": r["location"],
             "location_name": r["location_name"],
             "population": int(r["population"])}
            for r in rows]
    return _locations_cache


def resolve_location(text):
    """Map 'US', 'CA', 'California' or '06' to one location record."""
    needle = (text or "").strip().lower()
    if not needle:
        raise ToolError("location is required (e.g. 'US', 'CA', 'California')")
    for row in locations():
        if needle in (row["abbreviation"].lower(), row["location"].lower(),
                      row["location_name"].lower()):
            return row
    # FIPS codes are zero-padded; be forgiving about "6" vs "06".
    if needle.isdigit():
        padded = needle.zfill(2)
        for row in locations():
            if row["location"] == padded:
                return row
    raise ToolError(
        f"unknown location {text!r}. Use list_locations to see valid values.")


@tool
def list_locations(query: str = "") -> str:
    """List FluSight locations with their codes and populations.

    Args:
        query: Optional substring to filter names or codes.
    """
    needle = query.strip().lower()
    rows = [r for r in locations()
            if not needle or needle in r["location_name"].lower()
            or needle == r["abbreviation"].lower()
            or needle == r["location"].lower()]
    if not rows:
        return f"No locations match {query!r}."
    lines = [f"{'code':>5}  {'abbr':>4}  {'name':<22} population",
             *(f"{r['location']:>5}  {r['abbreviation']:>4}  "
               f"{r['location_name']:<22} {r['population']:,}" for r in rows)]
    return "\n".join(lines)


@tool
def get_flu_hospitalizations(location: str, weeks: int = 12) -> str:
    """Weekly confirmed influenza hospital admissions (NHSN) for one location.

    This is the quantity FluSight forecasts. The series is weekly, dated by the
    Saturday ending each week. The most recent week or two can revise upward as
    late reports arrive, so treat the final point as provisional.

    Args:
        location: 'US', a state abbreviation, a state name, or a FIPS code.
        weeks: How many recent weeks to return (1-260).
    """
    row = resolve_location(location)
    weeks = max(1, min(int(weeks), 260))
    jurisdiction = _NHSN_NATIONAL if row["abbreviation"] == "US" \
        else row["abbreviation"]

    query = {
        "$select": "weekendingdate,totalconfflunewadm",
        "$where": f"jurisdiction='{jurisdiction}'",
        "$order": "weekendingdate DESC",
        "$limit": str(weeks),
    }
    try:
        data = fetch_json(NHSN_URL + "?" + urllib.parse.urlencode(query),
                          timeout=45, ttl=active().cache_ttl,
                          cache_dir=active().cache_dir)
    except FetchError as e:
        raise ToolError(f"NHSN query failed: {e}")
    if not data:
        raise ToolError(f"NHSN returned no rows for {row['location_name']}")

    series = []
    for item in reversed(data):                     # oldest first
        value = item.get("totalconfflunewadm")
        if value is None:
            continue
        series.append((item["weekendingdate"][:10], float(value)))

    per_100k = 100_000 / row["population"]
    lines = [f"Weekly influenza hospital admissions -- {row['location_name']} "
             f"({row['location']}), source NHSN",
             f"population {row['population']:,}",
             "",
             f"{'week ending':<12} {'admissions':>10} {'per 100k':>9} {'wk/wk':>8}"]
    previous = None
    for date, value in series:
        change = "" if previous in (None, 0) else f"{(value / previous - 1):+.0%}"
        lines.append(f"{date:<12} {value:>10.0f} {value * per_100k:>9.2f} "
                     f"{change:>8}")
        previous = value
    lines.append("")
    lines.append(f"latest week ending {series[-1][0]} "
                 "(provisional; may revise upward)")
    return "\n".join(lines)


@tool
def get_ili(region: str = "nat", weeks: int = 12) -> str:
    """Weekly outpatient influenza-like illness (ILINet) via Delphi Epidata.

    Reports weighted %ILI: the share of outpatient visits for influenza-like
    illness. Available for the nation ('nat'), HHS regions ('hhs1'..'hhs10')
    and states (two-letter code). Independent of the hospitalization signal and
    usually available for the same week.

    Args:
        region: 'nat', 'hhs1'-'hhs10', or a state abbreviation such as 'ca'.
        weeks: How many recent weeks to return (1-104).
    """
    region = (region or "nat").strip().lower()
    weeks = max(1, min(int(weeks), 104))
    if region in ("us", "national", "usa"):
        region = "nat"

    # Ask for a wide epiweek window and keep the tail: avoids reimplementing
    # MMWR week arithmetic just to build the query.
    year = dt.date.today().year
    span = f"{year - 3}01-{year + 1}53"
    url = DELPHI_URL + "?" + urllib.parse.urlencode(
        {"regions": region, "epiweeks": span})
    try:
        payload = fetch_json(url, timeout=45, ttl=active().cache_ttl,
                             cache_dir=active().cache_dir)
    except FetchError as e:
        raise ToolError(f"Delphi Epidata query failed: {e}")

    if payload.get("result") != 1 or not payload.get("epidata"):
        raise ToolError(
            f"no ILINet data for region {region!r} "
            f"({payload.get('message', 'unknown error')}). "
            "Valid: nat, hhs1-hhs10, or a two-letter state code.")

    rows = sorted(payload["epidata"], key=lambda r: r["epiweek"])[-weeks:]
    lines = [f"Weighted %ILI -- region {region}, source ILINet via Delphi",
             "",
             f"{'epiweek':>8} {'week ending':<12} {'%ILI':>7} "
             f"{'providers':>9} {'wk/wk':>8}"]
    previous = None
    for r in rows:
        wili = r.get("wili")
        if wili is None:
            continue
        change = "" if previous in (None, 0) else f"{(wili / previous - 1):+.0%}"
        lines.append(f"{r['epiweek']:>8} {epiweek_saturday(r['epiweek']):<12} "
                     f"{wili:>7.2f} {r.get('num_providers') or 0:>9} "
                     f"{change:>8}")
        previous = wili
    return "\n".join(lines)


# -- MMWR week helpers --------------------------------------------------------

def _epiweek_start(year):
    """Sunday that begins MMWR week 1 of `year`."""
    jan1 = dt.date(year, 1, 1)
    weekday = (jan1.weekday() + 1) % 7          # 0 = Sunday
    start = jan1 - dt.timedelta(days=weekday)
    if weekday > 3:            # week 1 is the first with >= 4 days in January
        start += dt.timedelta(days=7)
    return start


def epiweek_saturday(epiweek):
    """Date of the Saturday ending an MMWR epiweek like 202432."""
    year, week = divmod(int(epiweek), 100)
    return (_epiweek_start(year)
            + dt.timedelta(days=7 * (week - 1) + 6)).isoformat()


def date_epiweek(date):
    """MMWR epiweek containing `date`."""
    year = date.year + 1
    while date < _epiweek_start(year):
        year -= 1
    return year * 100 + (date - _epiweek_start(year)).days // 7 + 1
