# Source this, do not run it:   . pod/env.sh
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$_here/.." && pwd)"
export POD_SCRATCH="${POD_SCRATCH:-/tmp/project_scratch}"      # large, wiped on restart/crash
export VENV="${VENV:-$POD_SCRATCH/venv}"
export UV_PYTHON_INSTALL_DIR="$POD_SCRATCH/uvpy"
export UV_CACHE_DIR="$POD_SCRATCH/uvcache"
export HF_HOME="${HF_HOME:-$POD_SCRATCH/hf}"                   # Hugging Face downloads (Job G only)
export LOG_DIR="${LOG_DIR:-$POD_SCRATCH/logs}"                 # logs stay out of the repo
export EXP_DIR="$REPO_ROOT/experiments/poisoning"              # result CSVs are written here (small)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"                 # one thread per process
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$POD_SCRATCH" "$LOG_DIR" "$HF_HOME"
if [ -f "$VENV/bin/activate" ]; then . "$VENV/bin/activate"; fi   # (an `&&` here would make a fresh pod's source fail under set -e)
