#!/usr/bin/env bash
# Adds transformers + datasets to the EXISTING environment (Job G only). Safe to run while other jobs are running:
# it does not rebuild the venv. Do not use `WITH_HF=1 bash pod/setup_env.sh` for this; that recreates the venv.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$here/env.sh"
if python -c "import transformers, datasets" 2>/dev/null; then echo "transformers and datasets already installed"; exit 0; fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || python3 -m pip install --user --quiet uv
uv pip install --python "$VENV/bin/python" transformers datasets
python -c "import transformers, datasets; print('HF OK', transformers.__version__, datasets.__version__)"
