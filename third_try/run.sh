#!/usr/bin/env bash
# third_try: upstream-faithful reproduction + optional TensorFlow stress flags.
# Default: MALLOC_ARENA_MAX=2 (glibc malloc arena cap, same idea as second_try). Override: MALLOC_ARENA_MAX=8 ./run.sh
#
#   chmod +x run.sh && ./run.sh
# Heavier load: ./run.sh -- --opt1
#   ./run.sh -- --opt1 --opt2 --opt3
set -euo pipefail
: "${MALLOC_ARENA_MAX:=2}"
export MALLOC_ARENA_MAX
cd "$(dirname "$0")" || {
  echo "Cannot cd to script directory."
  exit 1
}

_py_is_310() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null
}

PY="${PYTHON:-python3.10}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10 is required. No interpreter at: $PY"
  echo "Ubuntu 22.04: sudo apt update && sudo apt install -y python3.10 python3.10-venv"
  exit 1
fi

if ! _py_is_310 "$PY"; then
  echo "Python 3.10 is required."
  exit 1
fi

if ! "$PY" -m venv .venv; then
  echo "venv failed: sudo apt install -y python3.10-venv"
  exit 1
fi

source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

set +e
python -c "import tensorflow as tf; print(tf.__version__)" >/tmp/tf_leak_chk.txt 2>&1
tf_ec=$?
set -e
if [ "$tf_ec" != "0" ]; then
  echo "TensorFlow failed to load. CPU may lack AVX."
  cat /tmp/tf_leak_chk.txt 2>/dev/null || true
  exit 1
fi

exec python app.py "$@"
