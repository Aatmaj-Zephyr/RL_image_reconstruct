#!/usr/bin/env bash

set -e

IMG_DIR="outputs/imgs"
TELEMETRY_DIR="telemetry_logs"

# wait for telemetry csv
while true; do
    LATEST=$(ls -t "$TELEMETRY_DIR"/*.csv 2>/dev/null | head -n 1 || true)

    if [[ -n "$LATEST" ]]; then
        ln -sf "$(basename "$LATEST")" "$TELEMETRY_DIR/index.csv"
        break
    fi

    echo "Waiting for telemetry CSV..."
    sleep 1
done


# generate images.json continuously
generate_images_json() {
    while true; do
        ls "$IMG_DIR"/*.png 2>/dev/null | \
        xargs -n1 basename | \
        jq -R . | jq -s . > "$IMG_DIR/images.json"

        sleep 2
    done
}

echo "Starting image index generator..."
generate_images_json &
PID_INDEXER=$!


echo "Starting telemetry server..."
python -m http.server 8000 --directory "$TELEMETRY_DIR" &
PID_TELEMETRY=$!


echo "Starting image server..."
python -m http.server 8001 --directory "$IMG_DIR" &
PID_IMAGES=$!


echo "All services started"
echo "Telemetry: http://localhost:8000"
echo "Images: http://localhost:8001"
echo "Image API: http://localhost:8001/images.json"
echo ""


cleanup() {

    echo ""
    echo "Stopping all processes..."

    kill $PID_TELEMETRY $PID_IMAGES $PID_INDEXER 2>/dev/null || true
    wait

    echo "Shutdown complete"
}

trap cleanup SIGINT SIGTERM

wait