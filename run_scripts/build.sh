#!/bin/bash

set -e

echo "Activating virtual environment..."
source venv/bin/activate


echo "Cleanings up old files"
rm debug_logs/server.log || true
rm telemetry_logs/index.csv || true

echo "Moving files..."
mv outputs/imgs/* old_outputs/ 2>/dev/null || true
mv telemetry_logs/* old_telemetry_logs/ 2>/dev/null || true

echo "Starting servers..."
./run_scripts/server.sh >> debug_logs/server.log &

echo "Running main.py in production mode"
python main.py --mode prod