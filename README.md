# flu-forecasting-llm

A harness that lets a locally-served Qwen forecast US influenza hospital
admissions, with tools to look things up rather than recall them.

The model gets surveillance data, statistical models it can fit on demand, the
forecasts every other FluSight team submitted, web search, the published
literature, and a calculator. It researches, reasons, and submits a quantile
forecast in the CDC FluSight format. Forecasts can then be scored against the
published truth with the weighted interval score, so "is this any good?" has an
answer.

The method it is pushed towards is the one a superforecaster uses: anchor on
something mechanical, find out what everybody else thinks, then move off that
anchor only for a reason you can write down.

```
$ flufc forecast --location US
forecasting US @ 2026-08-22  horizons [0, 1, 2, 3]

--- step 1
  call get_flu_hospitalizations(location='US', weeks='52')
    ok   0.4s  Weekly influenza hospital admissions -- US (US), source NHSN
  call get_ili(region='nat', weeks='12')
    ok   0.0s  Weighted %ILI -- region nat, source ILINet via Delphi
--- step 2
  call compare_forecast_models(location='US')
    ok   2.1s  US (US) -- 316 weeks of NHSN history, last observed 2026-08-22
  call get_hub_forecasts(location='US')
    ok   0.6s  FluSight hub forecasts -- US (US), reference_date 2026-08-22
--- step 3
  call search_literature(query='H3N2 subclade K severity')
    ok   0.9s  PubMed: 9 record(s) match ...
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
flufc forecast --location US --toolset quant   # models and hub, no reading
flufc score outputs/US-2026-08-22-*.csv  # WIS, MAE, interval coverage

flufc data hosp --location CA --weeks 8  # call a data tool directly
flufc data ili  --location nat
flufc data compare --location US         # every statistical model, side by side
flufc data model --location US --method gbt --horizons 1,2,3
flufc data hub --location US --reference-date 2026-02-07   # other teams
flufc data papers --query "H3N2 subclade K severity"
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
      literature.py     search_literature, read_paper   (PubMed)
      surveillance.py   get_flu_hospitalizations, get_ili, list_locations
      models.py         fit_forecast_model, compare_forecast_models
      hub.py            get_hub_forecasts, list_hub_models
      compute.py        calculate
      submit.py         submit_forecast  (terminal: ends the run)
  stats.py     the statistical models and their walk-forward backtest
  forecast.py  quantile expansion, FluSight CSV
  evaluate.py  weighted interval score against published truth
```

### Anchors, not opinions

An LLM asked for a number produces one, and it will be a plausible number
arrived at by no method at all. So the harness gives it a method to start from
and asks it to argue with that instead.

**Its own models.** `compare_forecast_models` fits five statistical models to
the same NHSN series the agent can read, all in stdlib Python:

| method | what it is |
|---|---|
| `persistence` | last observed week carried forward -- the reference to beat |
| `linear` | OLS on the last 8 weeks |
| `loglinear` | constant growth rate in log space, rate-capped and damped with horizon |
| `seasonal` | the median trajectory over the same calendar weeks in the last 4 years |
| `gbt` | gradient-boosted regression trees, ~100 lines, no dependencies |

The trees are the interesting one. Predicting the *level* would cap every
forecast at the worst week in the training data, which is exactly wrong for an
epidemic, so they predict `log(y[t+h] / y[t])` -- the growth rate -- and leave
the level to the anchor.

Uncertainty does not come from the fit. Each method is re-fitted at a series of
earlier origins, seeing only what existed then, and the spread of those
walk-forward errors becomes the interval. A method whose recent errors were
large gets a wide band automatically, and the same backtest tells the model
which method to believe:

```
horizon       MAE  MAE flat  vs flat  log sd  95% hit
      1      31.9      42.2     +24%    0.08     100%
      2      36.4      51.5     +29%    0.11     100%
      3      60.0      75.2     +20%    0.14     100%
```

**Everybody else's models.** `get_hub_forecasts` reads what the other FluSight
teams submitted for the same week, straight out of the hub repository. The
ensemble is a strong prior and is hard to beat; the prompt asks the model to
land near it or say what it thinks the ensemble is missing.

**The literature.** `search_literature` and `read_paper` query PubMed, for the
questions surveillance cannot settle -- how a clade behaved elsewhere, what a
vaccine mismatch did last time, how these methods have scored before. The model
chooses what to open from titles and dates, which is the part that makes it
research rather than retrieval.

What comes back is a forecast with its reasoning attached:

> Anchored on gbt (lowest backtest MAE, 60 at h3): medians 798/824/977 for
> h1-h3. No hub submission (off-season), so no ensemble to beat. I nudged
> centres 1-3% above gbt because the last observed week is provisional and
> revises upward, and there is a small early uptick (+17% wk/wk; ILI
> 0.87 -> 0.94)... I widened the 95% tails to span the gbt-vs-seasonal
> disagreement (1.15-1.48x) and the inflection risk.

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

The model states the whole distribution: all 23 FluSight levels for every
horizon, which is what actually gets scored. `submit_forecast` takes them as a
list of 23 numbers in grid order -- around 500 characters for four horizons,
against roughly four times that for an object keyed by level, which matters
when the run is one truncated reply away from losing its forecast.

It used to give five anchors (2.5, 25, 50, 75, 97.5) that `expand_quantiles`
interpolated onto the grid linearly in z = Phi^-1(q). That is cheap and always
monotone, but it is a lognormal by construction, so the only distribution the
model could ever submit was the one the interpolator knew how to draw. Every
shape worth having an opinion about -- a hard floor under a long upper tail, a
fatter right side near an inflection, the bimodality of "either it has turned
or it has not" -- was flattened on the way to the CSV. Asking for the grid
directly costs the model a few dozen numbers and buys back the ability to say
something the interpolator cannot.

`repair_quantiles` then enforces only what a submission cannot violate:
non-negative, and non-decreasing across levels. Crossed quantiles are repaired
rather than rejected -- a model that puts the 25th above the median made a
slip, not an unrecoverable error -- but a missing level is an error, because
filling it in would be the interpolation this change exists to avoid.

The anchor path is still there and still used by the statistical models, where
a lognormal is exactly what is meant: their intervals come from a spread read
off backtest residuals, so five anchors and 23 levels describe the same curve.

What the prompt has to supply is the shape. Left to itself the model writes
evenly spaced numbers, which is a distribution no epidemic ever had, so
`FORECAST_SYSTEM` says explicitly that the steps between levels tighten around
the median and stretch into the tails, and that admissions are right-skewed.

### Data

| what | source | tool |
|---|---|---|
| Weekly flu hospital admissions | NHSN via `data.cdc.gov` (`mpgq-jmmr`) | `get_flu_hospitalizations` |
| Outpatient %ILI | ILINet via Delphi Epidata | `get_ili` |
| Location codes and populations | FluSight hub `locations.csv` | `list_locations` |
| Scoring truth | FluSight hub `target-hospital-admissions.csv` | `flufc score` |
| Other teams' forecasts | FluSight hub `model-output/` | `get_hub_forecasts`, `list_hub_models` |
| Published literature | PubMed via NCBI E-utilities | `search_literature`, `read_paper` |

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
- **No baseline comparison in the CLI.** `evaluate.baseline_flat` exists, and
  `flufc data compare` will tell you what the statistical models say, but
  `flufc score` still does not print a reference column next to the WIS.
- **The statistical intervals assume the recent past is representative.** They
  are read off backtest errors, so at a turn they are too narrow, not too wide.
  That is what the model's judgement is for, and the prompt says so.
- **The hub is empty out of season and stale in it.** Submissions land on the
  Wednesday they are due, so a forecast made early in the week sees last
  week's ensemble or nothing at all.
- **The model is the weak link, not the harness.** It reasons at length, and
  its instinct is to write the forecast as prose rather than call the tool.
  The prompt pushes back on both.
