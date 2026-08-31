"""Command line entry point.

    flufc status                       is the Qwen server up, what tools exist
    flufc ask "..."                    research agent, free-form question
    flufc forecast --location US       full forecasting run -> CSV + transcript
    flufc weekly                       the scheduled run: forecast, then rebuild
    flufc site                         regenerate docs/ from the archive
    flufc score outputs/x.csv          weighted interval score vs. published truth
    flufc compete outputs/x.csv        the same score, ranked against every hub team
    flufc data hosp --location CA      call one data tool directly
    flufc data compare --location CA   what the statistical models say
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import archive, config, site, standings, tools
from .agent import Agent, save_transcript
from .compete import CompetitionError, ENSEMBLE, leaderboard
from .config import Settings
from .evaluate import score_submission
from .forecast import HORIZONS, most_recent_saturday
from .llm import LLMError, QwenClient
from .prompts import FORECAST_SYSTEM, RESEARCH_SYSTEM, forecast_task
from .tools.surveillance import resolve_location

DIM, BOLD, CYAN, YELLOW, RESET = (
    ("\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m")
    if sys.stdout.isatty() else ("", "", "", "", ""))


def _reporter(verbose):
    """Print the agent's progress as it happens."""
    def report(event, **kw):
        if event == "step":
            print(f"{DIM}--- step {kw['step'].index + 1}{RESET}", flush=True)
        elif event == "call":
            print(f"  {CYAN}call{RESET} {kw['call'].describe()}", flush=True)
        elif event == "result":
            r = kw["result"]
            mark = "ok" if r.ok else "ERR"
            first = r.output.splitlines()[0][:100] if r.output else ""
            print(f"  {DIM}{mark:>4} {r.seconds:5.1f}s  {first}{RESET}",
                  flush=True)
            if verbose:
                print("\n".join("       " + l
                                for l in r.output.splitlines()[:40]))
        elif event == "answer":
            print(f"\n{BOLD}answer{RESET}\n{kw['text']}", flush=True)
    return report


def _client_settings(args):
    """Settings with the CLI's model flags applied, and no health check.

    Separate from _client because not every command that takes those flags
    needs the server: a weekly run with nothing left to forecast should
    rebuild the site rather than demand a GPU.
    """
    return config.use(Settings.load(       # so the tools see the same overrides
        endpoint=getattr(args, "endpoint", None),
        model=getattr(args, "model", None),
        temperature=getattr(args, "temperature", None),
        max_steps=getattr(args, "max_steps", None),
    ))


def _client(args):
    settings = _client_settings(args)
    client = QwenClient(settings)
    ok, message = client.health()
    if not ok:
        print(f"{YELLOW}{message}{RESET}", file=sys.stderr)
        raise SystemExit(2)
    return client, settings


# -- commands -----------------------------------------------------------------

def cmd_status(args):
    settings = Settings.load(endpoint=args.endpoint)
    client = QwenClient(settings)
    ok, message = client.health()
    print(f"server:    {message}")
    if not ok:
        return 2
    print(f"tool mode: {'native' if client.supports_native_tools() else 'prompt'} "
          f"(configured: {settings.tool_mode})")
    print(f"tools:     {', '.join(sorted(tools.REGISTRY))}")
    print(f"cache:     {settings.cache_dir}")
    print(f"outputs:   {settings.output_dir}")
    return 0


def cmd_tools(args):
    for name in sorted(tools.REGISTRY):
        entry = tools.REGISTRY[name]
        flag = " [terminal]" if entry.terminal else ""
        print(f"{BOLD}{name}{RESET}{flag}")
        print(f"  {entry.description.splitlines()[0]}")
        for pname, schema in entry.parameters["properties"].items():
            required = "" if pname in entry.parameters["required"] else " (optional)"
            print(f"    {pname}: {schema['type']}{required}"
                  f" -- {schema.get('description', '')}")
        print()
    return 0


def cmd_ask(args):
    client, settings = _client(args)
    agent = Agent(client, RESEARCH_SYSTEM, "research", settings,
                  on_event=_reporter(args.verbose))
    result = agent.run(args.question)
    _finish(result, settings, args, kind="ask")
    return 0


def _forecast_once(client, settings, args, location, reference_date, horizons,
                   archive_it=True, out=None):
    """One location's forecast: run the agent, write the CSV, archive it.

    Returns (forecast or None, agent result). Shared by `forecast` and
    `weekly`, so a scheduled batch and a hand-run single forecast leave behind
    exactly the same artifacts.
    """
    task = forecast_task(location["location_name"], location["location"],
                         reference_date, horizons)
    agent = Agent(client, FORECAST_SYSTEM, args.toolset, settings,
                  on_event=_reporter(args.verbose))
    print(f"{BOLD}forecasting{RESET} {location['location_name']} "
          f"@ {reference_date}  horizons {horizons}\n")
    result = agent.run(task)

    forecast = result.value
    if forecast is None:
        print(f"\n{YELLOW}No forecast submitted (stopped: {result.stopped}). "
              f"Try --max-steps higher, or raise max_tokens.{RESET}")
        _finish(result, settings, args, kind="forecast")
        return None, result

    forecast.model_id = f"qwen-agent/{settings.model}"
    print(f"\n{BOLD}forecast{RESET}\n{forecast.summary()}")
    print(f"\n{DIM}rationale:{RESET} {forecast.rationale}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out or (settings.output_dir /
                  f"{forecast.location}-{reference_date}-{stamp}.csv")
    forecast.write_csv(out)
    print(f"\nwrote {out}")

    transcript = _finish(result, settings, args, kind="forecast",
                         extra={"forecast": forecast.to_dict()})
    if archive_it:
        record = archive.save(settings.archive_dir, forecast,
                              location_name=location["location_name"],
                              run=result, transcript=transcript)
        print(f"{DIM}archived: {record['meta_path']}{RESET}")
    return forecast, result


def cmd_forecast(args):
    client, settings = _client(args)
    location = resolve_location(args.location)
    reference_date = args.reference_date or most_recent_saturday()
    horizons = args.horizons or [0, 1, 2, 3]

    forecast, _ = _forecast_once(client, settings, args, location,
                                 reference_date, horizons,
                                 archive_it=not args.no_archive, out=args.out)
    return 0 if forecast is not None else 1


def cmd_weekly(args):
    """The scheduled run: forecast every location, then rebuild the site.

    Written to be safe to run twice. A location already in the archive for this
    reference date is left alone unless --force, so a cron job that fires again
    after a partial failure finishes the job instead of overwriting it.
    """
    settings = _client_settings(args)
    if args.archive:
        settings.archive_dir = Path(args.archive)
    reference_date = args.reference_date or most_recent_saturday()
    horizons = args.horizons or [0, 1, 2, 3]
    locations = [resolve_location(name) for name in args.locations]

    print(f"{BOLD}weekly run{RESET}  reference_date {reference_date}  "
          f"{len(locations)} location(s)\n")
    todo, skipped = [], []
    for location in locations:
        if archive.has(settings.archive_dir, reference_date,
                       location["location"]) and not args.force:
            print(f"{DIM}skip {location['location_name']}: already in the "
                  f"archive for {reference_date}{RESET}")
            skipped.append(location["location"])
        else:
            todo.append(location)

    # Only wake the model if there is something to forecast. Re-running after
    # everything has landed should refresh the site, not demand a GPU.
    client = _client(args)[0] if todo else None

    made, failed = [], []
    for location in todo:
        code = location["location"]
        try:
            forecast, _ = _forecast_once(client, settings, args, location,
                                         reference_date, horizons)
        except (LLMError, tools.ToolError) as e:
            print(f"{YELLOW}{location['location_name']}: {e}{RESET}")
            failed.append(code)
            continue
        (made if forecast is not None else failed).append(code)
        print()

    print(f"{BOLD}made{RESET} {len(made)}   skipped {len(skipped)}   "
          f"failed {len(failed)}")
    if failed:
        print(f"{YELLOW}failed: {', '.join(failed)}{RESET}")

    if not args.no_site and (made or skipped):
        print()
        _build_site(settings, args, out=args.site_out)
    return 1 if failed else 0



def cmd_site(args):
    settings = Settings.load()
    if args.archive:
        settings.archive_dir = Path(args.archive)
    config.use(settings)              # the tools fetch with these settings
    return _build_site(settings, args, rescore=args.rescore, out=args.out)


def _build_site(settings, args, rescore=False, out=None):
    """Score whatever is in the archive and write the pages."""
    root = settings.archive_dir
    out = Path(out) if out else settings.site_dir

    if not archive.records(root):
        print(f"{YELLOW}no forecasts in {root} yet.{RESET} Run "
              f"`flufc weekly` first; the site needs something to show.")
    report = _site_progress(args.verbose)

    # Score once and hand the result to the builder, rather than letting it
    # refresh again: the leaderboards are the expensive part.
    records = standings.refresh(root, cache_dir=settings.cache_dir,
                                force=rescore, on_event=report)
    print(f"{BOLD}site{RESET}  {len(records)} forecast(s) in {root}")
    written = site.build(root, out, records=records, on_event=report)
    totals = standings.summary(records)

    print(f"\n{BOLD}wrote{RESET} {len(written)} file(s) to {out}")
    print(f"{totals['n_scored']} scored, {totals['n_pending']} waiting on truth")
    if totals.get("mean_rel_wis"):
        print(f"mean WIS {totals['mean_rel_wis']:.2f}x the flat baseline, "
              f"mean rank {totals['mean_rank']:.0f} of "
              f"~{totals['mean_field'] + 1:.0f}")
    print(f"{DIM}open {out / 'index.html'}{RESET}")
    return 0


def _site_progress(verbose):
    def report(event, **kw):
        record = kw.get("record")
        name = f"{record['reference_date']} {record['location']}" \
            if record else kw.get("location", "")
        if event == "scoring":
            print(f"  {DIM}scoring {name}...{RESET}", flush=True)
        elif event == "failed":
            print(f"  {YELLOW}{name}: {kw.get('error')}{RESET}", flush=True)
        elif event == "history":
            print(f"  {DIM}history for {name}{RESET}", flush=True)
        elif event == "settled" and verbose:
            print(f"  {DIM}{name}: settled, cached{RESET}", flush=True)
    return report


def cmd_score(args):
    settings = Settings.load()
    rows, totals = score_submission(args.path, cache_dir=settings.cache_dir)
    print(f"{'loc':>4} {'horizon':>7} {'week ending':>12} {'median':>9} "
          f"{'observed':>9} {'WIS':>9} {'50%':>4} {'95%':>4}")
    for r in rows:
        observed = "-" if r["observed"] is None else f"{r['observed']:.0f}"
        score = "-" if r["wis"] is None else f"{r['wis']:.2f}"
        c50 = "-" if r["cover_50"] is None else ("yes" if r["cover_50"] else "no")
        c95 = "-" if r["cover_95"] is None else ("yes" if r["cover_95"] else "no")
        print(f"{r['location']:>4} {r['horizon']:>7} {r['target_end_date']:>12} "
              f"{r['median']:>9.1f} {observed:>9} {score:>9} {c50:>4} {c95:>4}")
    print()
    if totals.get("n_scored"):
        print(f"scored {totals['n_scored']}/{totals['n_total']} horizons  "
              f"mean WIS {totals['mean_wis']}  MAE {totals['mean_abs_error']}  "
              f"50% coverage {totals['coverage_50']:.0%}  "
              f"95% coverage {totals['coverage_95']:.0%}")
    else:
        print("nothing scored yet: the truth for these weeks is not published.")
    return 0


def cmd_compete(args):
    settings = Settings.load()
    board = leaderboard(args.path, models=_split(args.models),
                        cache_dir=settings.cache_dir,
                        on_event=_hub_progress())

    horizons = board["scored_horizons"]
    print(f"{BOLD}compete{RESET}  {board['location']} @ "
          f"{board['reference_date']}   {args.path}")
    if not horizons:
        weeks = ", ".join(f"h{h} -> {d}" for h, d in
                          board["target_end_dates"].items())
        print(f"\n{YELLOW}Nothing to score yet.{RESET} The hub's published "
              f"truth for {board['location']} ends at "
              f"{board['truth_last_date']}, and this forecast covers {weeks}.")
        print("A competition needs weeks that have already happened: forecast "
              "an in-season reference date the other teams also submitted for, "
              "or come back when the truth catches up.")
        return 1

    ends = board["target_end_dates"]
    print(f"scored horizons {', '.join(str(h) for h in horizons)}  "
          f"(weeks ending {ends[horizons[0]]} to {ends[horizons[-1]]})  "
          f"{len(board['entries']) - 1} other teams")
    if board["unscored_horizons"]:
        print(f"{DIM}not scored (no truth yet): horizons "
              f"{', '.join(str(h) for h in board['unscored_horizons'])}{RESET}")
    if _retrospective(board["reference_date"]):
        print(f"{YELLOW}note{RESET} this is a past week, and the harness has "
              f"no as-of replay: if the forecast was produced after "
              f"{board['reference_date']}, its tools saw data the other teams "
              f"did not, and the rank flatters it.")
    print()

    print(f"{'rank':>5}  {'model':<32} {'WIS':>9} {'rel':>6} {'MAE':>9} "
          f"{'50%':>5} {'95%':>5}")
    for entry in board["entries"]:
        rel = "-" if entry["rel_wis"] is None else f"{entry['rel_wis']:.2f}"
        line = (f"{entry['rank']:>5}  {entry['model'][:32]:<32} "
                f"{entry['mean_wis']:>9.1f} {rel:>6} "
                f"{entry['mean_abs_error']:>9.1f} "
                f"{entry['coverage_50']:>5.0%} {entry['coverage_95']:>5.0%}")
        if entry["is_you"]:
            print(f"{BOLD}{line}{RESET}")
        elif entry["model"] in (ENSEMBLE, "FluSight-baseline"):
            print(f"{CYAN}{line}{RESET}")
        else:
            print(line)

    you = board["you"]
    field = you["field"]
    share = f" ({you['beat'] / field:.0%})" if field else ""
    print(f"\n{BOLD}rank {you['rank']} of {field + 1}{RESET} -- beat "
          f"{you['beat']} of {field} teams{share}")
    if you["rel_wis"] is not None:
        verdict = "better than" if you["rel_wis"] < 1 else "worse than"
        print(f"WIS {you['rel_wis']:.2f}x the flat baseline "
              f"({verdict} doing nothing)")
    ens = next((e for e in board["entries"] if e["model"] == ENSEMBLE), None)
    if ens:
        gap = you["mean_wis"] / ens["mean_wis"]
        print(f"the hub ensemble placed {ens['rank']} with WIS "
              f"{ens['mean_wis']:.1f}; you are {gap:.2f}x that")

    if len(horizons) > 1:
        print(f"\n{DIM}WIS by horizon{RESET}")
        print(f"{'model':<32} " + " ".join(f"{'h' + str(h):>9}"
                                           for h in horizons))
        for entry in board["entries"][:3] + ([you] if you["rank"] > 3 else []):
            cells = " ".join(f"{entry['wis_by_horizon'][h]:>9.1f}"
                             for h in horizons)
            print(f"{entry['model'][:32]:<32} {cells}")

    skipped = board["partial"] + board["missing"]
    if skipped:
        print(f"\n{DIM}{len(board['missing'])} team(s) did not submit for "
              f"{board['reference_date']}; {len(board['partial'])} covered only "
              f"some horizons{RESET}")
        if args.verbose:
            for item in skipped:
                print(f"{DIM}  {item['model']}: {item['reason']}{RESET}")
    return 0


def _retrospective(reference_date, today=None, days=21):
    """Was this week over well before now? Then the comparison needs a caveat."""
    try:
        reference = dt.date.fromisoformat(reference_date)
    except ValueError:
        return False
    return ((today or dt.date.today()) - reference).days > days


def _hub_progress():
    """A one-line counter while the hub submissions come down."""
    def report(event, **kw):
        if event == "start":
            print(f"{DIM}reading {kw['models']} hub submissions for "
                  f"{kw['reference_date']}...{RESET}", flush=True)
        elif event == "fetched" and sys.stdout.isatty():
            print(f"\r{DIM}  {kw['done']}/{kw['total']}{RESET}",
                  end="", flush=True)
            if kw["done"] == kw["total"]:
                print("\r" + " " * 24 + "\r", end="", flush=True)
    return report


def _split(text):
    if not text:
        return None
    return [part for part in text.replace(",", " ").split() if part]


def cmd_data(args):
    mapping = {
        "hosp": ("get_flu_hospitalizations",
                 {"location": args.location, "weeks": args.weeks}),
        "ili": ("get_ili", {"region": args.location, "weeks": args.weeks}),
        "locations": ("list_locations", {"query": args.location or ""}),
        "model": ("fit_forecast_model",
                  {"location": args.location, "method": args.method,
                   "reference_date": args.reference_date or "",
                   "horizons": args.horizons}),
        "compare": ("compare_forecast_models",
                    {"location": args.location,
                     "reference_date": args.reference_date or "",
                     "horizons": args.horizons}),
        "hub": ("get_hub_forecasts",
                {"location": args.location,
                 "reference_date": args.reference_date or "",
                 "models": args.models or ""}),
        "papers": ("search_literature",
                   {"query": args.query or args.location,
                    "max_results": args.weeks}),
    }
    name, call_args = mapping[args.what]
    result = tools.call(name, call_args, max_chars=100000)
    print(result.output)
    return 0 if result.ok else 1


def _finish(result, settings, args, kind, extra=None):
    """Save the transcript and print the run's cost."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_transcript(result, settings.run_dir(f"{kind}-{stamp}") /
                           "transcript.json", extra=extra)
    print(f"{DIM}{result.tool_calls} tool calls, {result.seconds:.0f}s, "
          f"{result.prompt_tokens} prompt / {result.completion_tokens} "
          f"completion tokens, stopped: {result.stopped}{RESET}")
    print(f"{DIM}transcript: {path}{RESET}")
    return path


# -- argument parsing ---------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="flufc", description="LLM influenza forecasting harness")
    parser.add_argument("--endpoint", help="vLLM base URL (default: discover)")
    parser.add_argument("--model")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print full tool output as it arrives")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="check the server and list tools"
                   ).set_defaults(fn=cmd_status)
    sub.add_parser("tools", help="show tool schemas").set_defaults(fn=cmd_tools)

    ask = sub.add_parser("ask", help="ask the research agent a question")
    ask.add_argument("question")
    ask.add_argument("--max-steps", type=int)
    ask.add_argument("--temperature", type=float)
    ask.set_defaults(fn=cmd_ask)

    fc = sub.add_parser("forecast", help="produce a FluSight-style forecast")
    fc.add_argument("--location", default="US", help="US, a state code or name")
    fc.add_argument("--reference-date", help="Saturday, YYYY-MM-DD")
    fc.add_argument("--horizons", type=int, nargs="+", choices=HORIZONS)
    fc.add_argument("--out", help="where to write the submission CSV")
    fc.add_argument("--no-archive", action="store_true",
                    help="do not add this run to forecasts/ (the public record)")
    fc.add_argument("--toolset", default="forecast",
                    choices=sorted(tools.TOOLSETS),
                    help="forecast (everything) or quant (no reading)")
    fc.add_argument("--max-steps", type=int)
    fc.add_argument("--temperature", type=float)
    fc.set_defaults(fn=cmd_forecast)

    weekly = sub.add_parser(
        "weekly", help="forecast every location for this week, then rebuild")
    weekly.add_argument("--locations", nargs="+", default=["US"],
                        help="locations to forecast (default: US)")
    weekly.add_argument("--reference-date", help="Saturday, YYYY-MM-DD")
    weekly.add_argument("--horizons", type=int, nargs="+", choices=HORIZONS)
    weekly.add_argument("--toolset", default="forecast",
                        choices=sorted(tools.TOOLSETS))
    weekly.add_argument("--force", action="store_true",
                        help="re-forecast locations already in the archive")
    weekly.add_argument("--no-site", action="store_true",
                        help="skip the site rebuild")
    weekly.add_argument("--site-out", help="site directory (default: docs/)")
    weekly.add_argument("--archive", help="archive directory (default: "
                                          "forecasts/)")
    weekly.add_argument("--max-steps", type=int)
    weekly.add_argument("--temperature", type=float)
    weekly.set_defaults(fn=cmd_weekly, no_archive=False)

    pages = sub.add_parser(
        "site", help="rebuild the published site from the archive")
    pages.add_argument("--out", help="output directory (default: docs/)")
    pages.add_argument("--archive", help="archive directory (default: "
                                         "forecasts/)")
    pages.add_argument("--rescore", action="store_true",
                       help="recompute every leaderboard, ignoring the cache")
    pages.set_defaults(fn=cmd_site)

    score = sub.add_parser("score", help="score a submission against truth")
    score.add_argument("path")
    score.set_defaults(fn=cmd_score)

    compete = sub.add_parser(
        "compete", help="rank a submission against every FluSight team")
    compete.add_argument("path")
    compete.add_argument("--models", help="hub models to compete against, "
                                          "comma separated (default: all)")
    compete.set_defaults(fn=cmd_compete)

    data = sub.add_parser("data", help="call a data tool directly")
    data.add_argument("what", choices=["hosp", "ili", "locations", "model",
                                       "compare", "hub", "papers"])
    data.add_argument("--location", default="US")
    data.add_argument("--weeks", type=int, default=12,
                      help="weeks of history (or results, for papers)")
    data.add_argument("--method", default="loglinear",
                      help="statistical model for 'data model'")
    data.add_argument("--reference-date", help="Saturday, YYYY-MM-DD")
    data.add_argument("--horizons", default="0,1,2,3")
    data.add_argument("--models", help="hub models, comma separated")
    data.add_argument("--query", help="search terms for 'data papers'")
    data.set_defaults(fn=cmd_data)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (LLMError, tools.ToolError, CompetitionError) as e:
        print(f"{YELLOW}{e}{RESET}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
