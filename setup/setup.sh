#!/bin/bash
echo "Getting permissions"
chmod +x setup/create_venv.sh
chmod +x debug.sh
chmod +x build.sh

echo "Creating virtual env"

setup/create_venv.sh