#!/usr/bin/env bash
#
# The scheduled forecasting run, for cron on the cluster.
#
# Making a forecast needs the model, and the model needs a GPU, so this half of
# the pipeline has to run where Slurm is. It starts the vLLM server if it is
# not already up, forecasts every location for the current reference date,
# rebuilds the site, and pushes the result to GitHub. Scoring is somebody
# else's job: .github/workflows/refresh.yml re-ranks the archive every day
# from public data, with no GPU involved.
#
# Safe to run more than once in a week. A location already in the archive for
# this reference date is left alone, so a re-run after a partial failure
# finishes the job rather than overwriting a forecast that was made honestly
# before the week happened.
#
#   ./scripts/weekly.sh                 forecast, rebuild, commit, push
#   FLUFC_LOCATIONS="US CA NY" ./scripts/weekly.sh
#   FLUFC_PUSH=0 ./scripts/weekly.sh    do everything but publish
#
# Install (Wednesdays at 08:00, after the hub's submission deadline):
#   crontab -e
#   0 8 * * 3 /accounts/grad/$USER/flu-forecasting-llm/scripts/weekly.sh

set -uo pipefail

REPO=${FLUFC_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOCATIONS=${FLUFC_LOCATIONS:-US}
PUSH=${FLUFC_PUSH:-1}
STOP_MODEL=${FLUFC_STOP_MODEL:-0}
QWEN=${QWEN_BIN:-$HOME/.local/bin/qwen}

cd "$REPO" || { echo "no such directory: $REPO" >&2; exit 2; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p outputs/logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="outputs/logs/weekly-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }

# Only one run at a time, across every machine that shares this home
# directory. Cron is per-node here but $HOME is not, so the same entry can sit
# on more than one login node and fire twice at the same minute -- and the
# "already in the archive" check does not help when both runs start before
# either has written anything. mkdir is the one operation NFS makes atomic.
LOCK="$REPO/outputs/.weekly.lock"
if mkdir "$LOCK" 2>/dev/null; then
    echo "$(hostname -s) $$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK/owner"
    trap 'rm -rf "$LOCK"' EXIT
else
    OWNER=$(cat "$LOCK/owner" 2>/dev/null || echo "unknown")
    # A run killed mid-flight would otherwise block every later one, so a lock
    # older than six hours is treated as wreckage rather than a live run.
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +360 2>/dev/null)" ]; then
        say "breaking a stale lock held by $OWNER"
        rm -rf "$LOCK"
        mkdir "$LOCK" 2>/dev/null || { say "lost the race for the lock"; exit 0; }
        echo "$(hostname -s) $$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK/owner"
        trap 'rm -rf "$LOCK"' EXIT
    else
        say "another run holds the lock ($OWNER); nothing to do"
        exit 0
    fi
fi

say "weekly run in $REPO, locations: $LOCATIONS"

# Start from whatever is on the remote: the daily rescore writes to docs/ too,
# and rebasing first is cheaper than resolving a conflict after the fact.
if [ "$PUSH" = "1" ] && git remote get-url origin >/dev/null 2>&1; then
    git pull --rebase --autostash || say "warning: pull failed, continuing"
fi

# `qwen up` starts the Slurm job if it is not already running and blocks until
# the API answers, which takes about four minutes from cold.
if [ -x "$QWEN" ]; then
    say "bringing the model up"
    "$QWEN" up || { say "could not start the model"; exit 3; }
else
    say "no qwen helper at $QWEN; assuming a server is already reachable"
fi

say "forecasting"
python3 -m flufc.cli weekly --locations $LOCATIONS
STATUS=$?
say "flufc weekly exited $STATUS"

if [ "$STOP_MODEL" = "1" ] && [ -x "$QWEN" ]; then
    say "releasing the GPU"
    "$QWEN" down || true
fi

# Publish whatever was produced, even if one location failed: a partial week is
# still a week, and the missing location is visible on the page by its absence.
if [ "$PUSH" != "1" ]; then
    say "FLUFC_PUSH=0, leaving the working tree alone"
    exit $STATUS
fi

git add forecasts docs 2>/dev/null
if git diff --cached --quiet; then
    say "nothing new to publish"
    exit $STATUS
fi

NEW=$(git diff --cached --name-only -- forecasts \
      | grep -E 'forecasts/[0-9-]+/[A-Z0-9]+\.json$' \
      | sed -E 's|forecasts/([0-9-]+)/([A-Z0-9]+)\.json|\1 \2|' | tr '\n' ',' \
      | sed 's/,$//')
git commit -q -m "forecasts: ${NEW:-site rebuild}" \
           -m "Made by scripts/weekly.sh on $(hostname -s) at $STAMP." \
    || { say "commit failed"; exit 4; }

BRANCH=$(git rev-parse --abbrev-ref HEAD)
say "pushing $BRANCH"
git push origin "$BRANCH" || { say "push failed -- commit is local"; exit 5; }
say "published: https://github.com/chasehmathis/flu-forecasting-llm"
exit $STATUS
