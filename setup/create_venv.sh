#!/usr/bin/env sh

set -e

# Detect the highest available Python version (3.x)
for v in 3.17 3.16 3.15 3.14 3.13 3.12 3.11 3.10 3.9 3.8; do
    if command -v "python$v" >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v "python$v")
        break
    fi
done

# Fallback if no versioned python found
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python)
    else
        echo "Error: Python is not installed."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Create virtual environment only if it doesn't exist
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Skipping creation."
else
    echo "Creating virtual environment 'venv'..."
    $PYTHON_BIN -m venv venv
fi

# Activate virtual environment (safe even if it already exists)
echo "Activating virtual environment..."
# shellcheck disable=SC1091
. venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements if file exists
if [ -f "./setup/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r ./setup/requirements.txt
else
    echo "No requirements.txt found, skipping dependency installation."
fi

echo "Virtual environment setup complete!"
echo "Activate later using: source venv/bin/activate"