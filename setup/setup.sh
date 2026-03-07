#!/bin/bash
echo "Getting permissions"
chmod +x setup/create_venv.sh
chmod +x run_scripts/debug.sh
chmod +x run_scripts/build.sh

echo "Creating virtual env"

setup/create_venv.sh