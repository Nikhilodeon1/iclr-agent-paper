#!/usr/bin/env bash
# bash pod/launch.sh H          start one job in the background
# bash pod/launch.sh wave1      start H, A, B, E together (one process each)
# Never start the same job twice: both would append to the same CSV.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"; . "$here/jobs.sh"
command -v python >/dev/null || { echo "no python: run . pod/env.sh first"; exit 1; }
python "$here/preflight.py" || { echo "preflight failed: not launching (no job should run on CPU)"; exit 1; }
start() {
  local j="$1" var="JOB_$1"; local spec="${!var:-}"
  [ -n "$spec" ] || { echo "unknown job $j"; return 1; }
  local csv="${spec%%|*}" rest="${spec#*|}"; local args="${rest#*|}"
  if [ -f "$LOG_DIR/$j.pid" ] && kill -0 "$(cat "$LOG_DIR/$j.pid")" 2>/dev/null; then echo "job $j already running (pid $(cat "$LOG_DIR/$j.pid"))"; return 0; fi
  if [ -s "$EXP_DIR/$csv" ]; then echo "refusing: $csv already exists and is non-empty (move it or delete it first)"; return 1; fi
  ( cd "$EXP_DIR"; nohup python validate_transformer.py $args > "$LOG_DIR/$j.log" 2>&1 & echo $! > "$LOG_DIR/$j.pid" )   # $! must be python's pid
  echo "started job $j  log: $LOG_DIR/$j.log"
}
[ $# -ge 1 ] || { echo "usage: launch.sh wave1 | <letter>..."; exit 1; }
if [ "$1" = "wave1" ]; then for j in $WAVE1; do start "$j" || true; done; else for j in "$@"; do start "$j" || true; done; fi
