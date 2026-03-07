#!/bin/bash

set -e

echo "Activating virtual environment..."
source venv/bin/activate

echo "Moving files..."
mv outputs/* old_outputs/ 2>/dev/null || true

echo "Running main.py in debug mode"
python main.py --mode debug