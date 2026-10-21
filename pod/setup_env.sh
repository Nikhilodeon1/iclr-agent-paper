#!/usr/bin/env bash
# Builds the Python environment in scratch. Run once per new pod (and again after a crash).
#   bash pod/setup_env.sh                 # default CUDA wheels
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash pod/setup_env.sh   # V100 fallback
#   WITH_HF=1 bash pod/setup_env.sh       # also install transformers+datasets (Job G only)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/env.sh"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PY_VERSION="${PY_VERSION:-3.12}"
command -v uv >/dev/null 2>&1 || { python3 -m pip install --user --quiet uv; export PATH="$HOME/.local/bin:$PATH"; }
uv python install "$PY_VERSION"
uv venv --python "$PY_VERSION" "$VENV"
. "$VENV/bin/activate"
uv pip install numpy pandas scipy matplotlib
uv pip install torch --index-url "$TORCH_INDEX"
if [ "${WITH_HF:-0}" = "1" ]; then uv pip install transformers datasets; fi
uv pip freeze > "$EXP_DIR/pod_env_versions.txt"
python "$here/preflight.py"
