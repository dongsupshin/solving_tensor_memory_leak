#!/usr/bin/env bash
# Requires Python 3.10 only. TensorFlow stack pins: ai/source/ai/upstream/SETTING_SAMPLE/requirements_ubuntu.txt
# Interpreter: $PYTHON if set, else python3.10 on PATH.
#
# NUMPY_DEMO=1 still uses the same Python 3.10 venv (no TensorFlow).
#
# If "getcwd() failed" during sudo apt: run commands from cd ~
# VirtualBox without AVX: NUMPY_DEMO=1 ./run.sh
# Optional: ./run.sh -- --leak
# Heavier TF load (opt-in): e.g.
#   ./run.sh -- --spatial 128 --batch 32 --forwards 8 --clear-every 2000 --yield-forward 0.002
set -euo pipefail
cd "$(dirname "$0")" || {
  echo "Cannot cd to script directory. Open a new terminal: cd ~ && cd <this-folder>"
  exit 1
}

_py_is_310() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null
}

PY="${PYTHON:-python3.10}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10 is required. No interpreter at: $PY"
  echo ""
  echo "Ubuntu 22.04 (jammy) / WSL:"
  echo "  sudo apt update && sudo apt install -y python3.10 python3.10-venv"
  echo ""
  echo "Ubuntu 24.04 (noble): python3.10 is not in default apt. Pick one:"
  echo "  - WSL:  wsl --install -d Ubuntu-22.04   then use that distro + apt above"
  echo "  - Docker (host has Docker):"
  echo "      docker build -t ai-leak-tf210 ."
  echo "      docker run --rm -p 8765:8765 ai-leak-tf210"
  echo "  - pyenv: https://github.com/pyenv/pyenv#installation  then  pyenv install 3.10.14"
  echo "      export PYTHON=\"\$(pyenv root)/versions/3.10.14/bin/python\""
  exit 1
fi

if ! _py_is_310 "$PY"; then
  echo "Python 3.10 is required. PYTHON=$PY reports:"
  "$PY" -c 'import sys; print("%s" % sys.version)' 2>/dev/null || true
  exit 1
fi

if ! "$PY" -m venv .venv; then
  echo ""
  echo "venv failed. Install venv for Python 3.10 (from a valid cwd, e.g. cd ~):"
  echo "  sudo apt update && sudo apt install -y python3.10-venv"
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip wheel

if [ "${NUMPY_DEMO:-}" = "1" ]; then
  export NUMPY_DEMO=1
  python -m pip install -r requirements-demo.txt
else
  python -m pip install -r requirements.txt
  set +e
  python -c "import tensorflow as tf; print(tf.__version__)" >/tmp/tf_leak_chk.txt 2>&1
  tf_ec=$?
  set -e
  if [ "$tf_ec" != "0" ]; then
    echo "TensorFlow failed to load (exit $tf_ec). Typical: CPU has no AVX (many VirtualBox guests)."
    cat /tmp/tf_leak_chk.txt 2>/dev/null || true
    echo "Try: rm -rf .venv && NUMPY_DEMO=1 ./run.sh"
    exit 1
  fi
fi

exec python app.py "$@"
