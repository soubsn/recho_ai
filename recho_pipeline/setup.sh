#!/usr/bin/env bash
set -euo pipefail

VENV_NAME="recho-dev"

# If venv already exists, just activate it
if [ -f "$VENV_NAME/bin/activate" ]; then
    source "$VENV_NAME/bin/activate"
    return 0 2>/dev/null || exit 0
fi

# Check uv is available
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH so uv is available immediately
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating uv venv: $VENV_NAME (Python 3.10)"
uv venv "$VENV_NAME" --python 3.10

echo "Installing requirements..."
uv pip install --python "$VENV_NAME/bin/python" -r requirements.txt

# Optional: tensorflow-model-optimization for QAT
echo ""
read -r -p "Install tensorflow-model-optimization for QAT support? [y/N] " tfmot
if [[ "$tfmot" =~ ^[Yy]$ ]]; then
    uv pip install --python "$VENV_NAME/bin/python" tensorflow-model-optimization
fi

echo ""
echo "Activating $VENV_NAME..."
source "$VENV_NAME/bin/activate"
