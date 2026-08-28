"""The published literature, for the questions data cannot settle.

Whether a clade shift matters, how much a vaccine mismatch moved things last
time, what the reporting artefacts in NHSN look like -- these are answered in
papers, not in a time series. PubMed is the index; the agent decides from
titles and dates what is worth opening, which is the part that makes it
research rather than retrieval.

Source: NCBI E-utilities (esearch, esummary, efetch) over PubMed. No key
needed; be polite about the request rate, which the HTTP cache mostly handles.
"""

import datetime as dt
import re
import urllib.parse

from ..cache import FetchError, fetch, fetch_json
from ..config import active
from . import ToolError, tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL_ID = {"tool": "flufc"}          # NCBI asks callers to identify themselves


def _eutils(endpoint, params, parse_json=True):
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode({**TOOL_ID, **params})
    try:
        if parse_json:
            return fetch_json(url, timeout=45, ttl=active().cache_ttl,
                              cache_dir=active().cache_dir)
        return fetch(url, timeout=45, ttl=active().cache_ttl,
                     cache_dir=active().cache_dir)
    except FetchError as e:
        raise ToolError(f"PubMed request failed: {e}")


@tool
def search_literature(query: str, max_results: int = 8, years: int = 5,
                      sort: str = "relevance") -> str:
    """Search PubMed and return titles, journals, dates and PMIDs.

    For questions the surveillance data cannot answer: how a subtype or clade
    has behaved before, what a vaccine mismatch did to a season, how well
    particular forecasting methods have scored, known artefacts in the
    reporting. Judge from the titles and dates what is worth reading, then open
    the two or three that matter with read_paper -- reading everything is not
    the same as knowing more.

    Args:
        query: Search terms, e.g. 'influenza H3N2 subclade K severity'.
        max_results: How many records to return (1-25).
        years: Only return work published within this many years (1-40).
        sort: 'relevance' or 'date'.
    """
    query = (query or "").strip()
    if not query:
        raise ToolError("query is required")
    max_results = max(1, min(int(max_results), 25))
    years = max(1, min(int(years), 40))
    sort = "pub_date" if str(sort).strip().lower().startswith("date") \
        else "relevance"

    this_year = dt.date.today().year
    found = _eutils("esearch.fcgi", {
        "db": "pubmed", "term": query, "retmode": "json",
        "retmax": str(max_results), "sort": sort, "datetype": "pdat",
        "mindate": str(this_year - years), "maxdate": str(this_year + 1)})
    result = found.get("esearchresult", {})
    ids = result.get("idlist") or []
    if not ids:
        return (f"No PubMed records for {query!r} in the last {years} years. "
                "Try broader terms, more years, or search_web for grey "
                "literature and CDC reports.")

    summaries = _eutils("esummary.fcgi", {
        "db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    records = summaries.get("result", {})

    lines = [f"PubMed: {result.get('count', len(ids))} record(s) match "
             f"{query!r} (last {years} years, sorted by {sort}); "
             f"showing {len(ids)}", ""]
    for pmid in ids:
        item = records.get(pmid) or {}
        authors = item.get("authors") or []
        first = authors[0].get("name", "") if authors else ""
        et_al = " et al." if len(authors) > 1 else ""
        journal = item.get("source", "")
        date = item.get("pubdate", "")
        lines.append(f"PMID {pmid}  {item.get('title', '(no title)')}")
        lines.append(f"    {first}{et_al} -- {journal}, {date}")
        doi = next((x.get("value") for x in item.get("articleids", [])
                    if x.get("idtype") == "doi"), None)
        if doi:
            lines.append(f"    doi:{doi}")
    lines += ["", "read_paper(pmid) returns the abstract. Prefer recent work "
                  "and primary analyses; a preprint is evidence, not proof."]
    return "\n".join(lines)


@tool
def read_paper(pmid: str, max_chars: int = 4000) -> str:
    """Fetch a PubMed record's abstract by PMID.

    Args:
        pmid: The numeric PubMed ID, as printed by search_literature.
        max_chars: Truncate the abstract to this many characters.
    """
    pmid = str(pmid).strip().lstrip("PMIDpmid: ")
    if not re.fullmatch(r"\d+", pmid):
        raise ToolError(f"{pmid!r} is not a PMID; use the number "
                        "search_literature printed")
    text = _eutils("efetch.fcgi",
                   {"db": "pubmed", "id": pmid, "rettype": "abstract",
                    "retmode": "text"}, parse_json=False).strip()
    if not text:
        raise ToolError(f"PubMed returned nothing for PMID {pmid}")
    if "cannot get document summary" in text.lower():
        raise ToolError(f"PubMed has no record with PMID {pmid}")
    max_chars = max(200, min(int(max_chars), 20000))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated; record has {len(text)} chars]"
    return text
