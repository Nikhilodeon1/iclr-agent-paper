#!/usr/bin/env bash
# Makes results_wave.tgz (CSVs + env versions only; no logs) in the home directory, then lists it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$here/env.sh"
cd "$EXP_DIR"
tar -czf "$HOME/results_wave.tgz" results_*.csv pod_env_versions.txt
tar -tzf "$HOME/results_wave.tgz"; ls -l "$HOME/results_wave.tgz"
