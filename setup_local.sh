#!/usr/bin/env bash
set -euo pipefail

# Change to the directory of this script
cd "$(dirname "$0")"

PYTHON=python3

echo "============================================"
echo "Setting up local environment (venv + deps)"
echo "============================================"

echo "Checking Python venv support (ensurepip)..."
if ! $PYTHON -c "import ensurepip" >/dev/null 2>&1; then
  echo "Python ensurepip not available. Installing venv support..."
  if command -v apt >/dev/null 2>&1; then
    # Try both generic and version-specific packages (Ubuntu 24.04 uses Python 3.12)
    (sudo apt-get update -y || sudo apt update -y) && \
    sudo apt-get install -y python3-venv python3.12-venv || sudo apt install -y python3-venv python3.12-venv
  else
    echo "ERROR: Package manager not found to install python venv support." >&2
    echo "Please install your distro's python venv package and re-run." >&2
    exit 1
  fi
fi

echo "Creating virtual environment (.venv) if missing..."
if [ ! -d ".venv" ]; then
  $PYTHON -m venv .venv
fi

VENV_PY=".venv/bin/python"
PIP=".venv/bin/pip"
RAY=".venv/bin/ray"
SERVE=".venv/bin/serve"

echo "Ensuring pip exists inside venv..."
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  # Try seeding pip via ensurepip first
  if ! "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "ensurepip not available in venv, bootstrapping pip via get-pip.py ..."
    if command -v curl >/dev/null 2>&1; then
      curl -sSLo get-pip.py https://bootstrap.pypa.io/get-pip.py
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O get-pip.py https://bootstrap.pypa.io/get-pip.py
    else
      echo "ERROR: Neither curl nor wget is available to fetch get-pip.py" >&2
      echo "Please install curl or wget and re-run." >&2
      exit 1
    fi
    "$VENV_PY" get-pip.py
    rm -f get-pip.py
  fi
fi

echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  "$PIP" install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in $(pwd)" >&2
  exit 1
fi

echo "Checking Ray cluster status..."
if ! "$RAY" status >/dev/null 2>&1; then
  echo "Starting local Ray head node..."
  "$RAY" start --head --include-dashboard true --dashboard-host=0.0.0.0
fi

echo "Starting Ray Serve app: serve_app:app ..."
"$SERVE" run serve_app:app


