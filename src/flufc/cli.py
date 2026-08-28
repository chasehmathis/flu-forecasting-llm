"""Command line entry point.

    flufc status                       is the Qwen server up, what tools exist
    flufc ask "..."                    research agent, free-form question
    flufc forecast --location US       full forecasting run -> CSV + transcript
    flufc score outputs/x.csv          weighted interval score vs. published truth
    flufc data hosp --location CA      call one data tool directly
"""

import argparse
import datetime as dt
import json
import sys

from . import config, tools
from .agent import Agent, save_transcript
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


def _client(args):
    settings = Settings.load(
        endpoint=getattr(args, "endpoint", None),
        model=getattr(args, "model", None),
        temperature=getattr(args, "temperature", None),
        max_steps=getattr(args, "max_steps", None),
    )
    config.use(settings)              # so the tools see the same overrides
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


def cmd_forecast(args):
    client, settings = _client(args)
    location = resolve_location(args.location)
    reference_date = args.reference_date or most_recent_saturday()
    horizons = args.horizons or [0, 1, 2, 3]

    task = forecast_task(location["location_name"], location["location"],
                         reference_date, horizons)
    agent = Agent(client, FORECAST_SYSTEM, "forecast", settings,
                  on_event=_reporter(args.verbose))
    print(f"{BOLD}forecasting{RESET} {location['location_name']} "
          f"@ {reference_date}  horizons {horizons}\n")
    result = agent.run(task)

    forecast = result.value
    if forecast is None:
        print(f"\n{YELLOW}No forecast submitted (stopped: {result.stopped}). "
              f"Try --max-steps higher, or raise max_tokens.{RESET}")
        _finish(result, settings, args, kind="forecast")
        return 1

    forecast.model_id = f"qwen-agent/{settings.model}"
    print(f"\n{BOLD}forecast{RESET}\n{forecast.summary()}")
    print(f"\n{DIM}rationale:{RESET} {forecast.rationale}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (settings.output_dir /
                       f"{forecast.location}-{reference_date}-{stamp}.csv")
    forecast.write_csv(out)
    print(f"\nwrote {out}")
    _finish(result, settings, args, kind="forecast",
            extra={"forecast": forecast.to_dict()})
    return 0


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


def cmd_data(args):
    mapping = {
        "hosp": ("get_flu_hospitalizations",
                 {"location": args.location, "weeks": args.weeks}),
        "ili": ("get_ili", {"region": args.location, "weeks": args.weeks}),
        "locations": ("list_locations", {"query": args.location or ""}),
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
    fc.add_argument("--max-steps", type=int)
    fc.add_argument("--temperature", type=float)
    fc.set_defaults(fn=cmd_forecast)

    score = sub.add_parser("score", help="score a submission against truth")
    score.add_argument("path")
    score.set_defaults(fn=cmd_score)

    data = sub.add_parser("data", help="call a data tool directly")
    data.add_argument("what", choices=["hosp", "ili", "locations"])
    data.add_argument("--location", default="US")
    data.add_argument("--weeks", type=int, default=12)
    data.set_defaults(fn=cmd_data)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (LLMError, tools.ToolError) as e:
        print(f"{YELLOW}{e}{RESET}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
