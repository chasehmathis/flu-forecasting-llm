# flu-forecasting-llm

A harness that lets a locally-served Qwen forecast US influenza hospital
admissions, with tools to look things up rather than recall them.

The model gets surveillance data, web search, and a calculator; it researches,
reasons, and submits a quantile forecast in the CDC FluSight format. Forecasts
can then be scored against the published truth with the weighted interval
score, so "is this any good?" has an answer.

```
$ flufc forecast --location US
forecasting US @ 2026-08-22  horizons [0, 1, 2, 3]

--- step 1
  call get_flu_hospitalizations(location='US', weeks='52')
    ok   0.4s  Weekly influenza hospital admissions -- US (US), source NHSN
  call get_ili(region='nat', weeks='12')
    ok   0.0s  Weighted %ILI -- region nat, source ILINet via Delphi
--- step 2
  call calculate(expression='795 / 677 - 1')
    ok   0.0s  795 / 677 - 1 = 0.174298
...
```

## Setup

There is nothing to install. The core is stdlib-only Python 3.10+, so it runs
on a bare login node:

```bash
qwen up                                  # start the vLLM server (~4 min)
cd ~/flu-forecasting-llm
export PYTHONPATH=$PWD/src
python3 -m flufc.cli status
```

Optionally `pip install -e .` to get a `flufc` command instead of
`python3 -m flufc.cli`. The examples below use `flufc`.

The harness finds the model by asking Slurm which node the `qwen-vllm` job is
on, so it works without configuration as long as `qwen up` has been run. Set
`QWEN_URL` to point somewhere else.

## Commands

```bash
flufc status                             # server, tool mode, tool list
flufc tools                              # tool schemas, as the model sees them

flufc ask "Is flu activity rising in California?"
flufc forecast --location US             # -> outputs/US-<date>-<stamp>.csv
flufc forecast --location CA --horizons 0 1 2 --max-steps 20
flufc score outputs/US-2026-08-22-*.csv  # WIS, MAE, interval coverage

flufc data hosp --location CA --weeks 8  # call a data tool directly
flufc data ili  --location nat
```

Every run writes a full transcript to `outputs/runs/<kind>-<stamp>/` -- every
message, tool call, result and timing. When a forecast comes out strange, that
file is the answer to "what did it actually look at?".

## How it fits together

```
cli.py         commands, progress display
  agent.py     the loop: think -> call tools -> observe -> submit
    llm.py     transport to vLLM's OpenAI-compatible endpoint
    protocol.py  Qwen's XML tool-call format: render specs, parse calls
    tools/     what the model can do
      web.py            search_web, fetch_url
      surveillance.py   get_flu_hospitalizations, get_ili, list_locations
      compute.py        calculate
      submit.py         submit_forecast  (terminal: ends the run)
  forecast.py  quantile expansion, FluSight CSV
  evaluate.py  weighted interval score against published truth
```

### Tool calling without server flags

vLLM only assembles OpenAI-style `tool_calls` when the server is started with
`--enable-auto-tool-choice --tool-call-parser qwen3_xml`. The current
`serve_qwen.sbatch` isn't, so by default the harness runs in **prompt mode**:
it puts the same tool block in the system prompt that the model's own chat
template would have produced, and parses the XML back out.

This is not a workaround so much as the same wire format from the other side --
the model sees exactly what it was trained on. `tool_mode: auto` probes the
server once and uses native calls when they're available, so adding those two
flags to the sbatch file changes nothing else.

### Quantiles

A language model cannot be talked into producing 23 coherent quantiles, so it
gives five anchors per horizon (2.5, 25, 50, 75, 97.5) and `forecast.py`
expands them onto the FluSight grid by interpolating linearly in
z = Phi^-1(q). That is monotone by construction and gives reasonable tails from
few anchors. Crossed quantiles are repaired rather than rejected -- a model
that puts the 25th above the median made a slip, not an unrecoverable error.

### Data

| what | source | tool |
|---|---|---|
| Weekly flu hospital admissions | NHSN via `data.cdc.gov` (`mpgq-jmmr`) | `get_flu_hospitalizations` |
| Outpatient %ILI | ILINet via Delphi Epidata | `get_ili` |
| Location codes and populations | FluSight hub `locations.csv` | `list_locations` |
| Scoring truth | FluSight hub `target-hospital-admissions.csv` | `flufc score` |

HTTP responses are cached under `data/cache/` for six hours, so re-runs are
fast and a run can be replayed against the data it actually saw.

## Configuration

`config/default.yaml` holds the defaults. Any key can be overridden by an
environment variable (`FLUFC_MAX_STEPS=24`) or, where it matters, a CLI flag.
Search uses DuckDuckGo with no API key; setting `TAVILY_API_KEY`,
`BRAVE_API_KEY` or `SERPER_API_KEY` and `search_backend` switches to a real
search API.

## Tests

```bash
python3 -m pytest tests/ -q
```

The agent tests drive the loop with a scripted model, so they are fast,
deterministic, and cover the failure modes that actually happen: a tool that
errors, a reply cut off by the token limit, a truncated tool call, context
overflow, and hitting the step limit.

## Known limits

- **One location per run.** A full FluSight submission is 53 locations; that
  is a loop over `cmd_forecast` plus a CSV concatenation, not written yet.
- **No backtesting.** Scoring works against published truth, but there is no
  as-of data replay, so a run today cannot honestly be evaluated on a past
  week -- the tools would see data that did not exist then. Delphi's `issues`
  parameter is the way in when this matters.
- **No baseline comparison in the CLI.** `evaluate.baseline_flat` exists;
  wiring it into `flufc score` as a reference column is the obvious next step.
- **The model is the weak link, not the harness.** It reasons at length, and
  its instinct is to write the forecast as prose rather than call the tool.
  The prompt pushes back on both.
