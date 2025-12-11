#!/bin/bash
set -e

# Activate virtual environment (created by uv sync in /app/inference-cloud)
if [ -f "/app/inference-cloud/.venv/bin/activate" ]; then
    source /app/inference-cloud/.venv/bin/activate
elif [ -f "/app/.venv/bin/activate" ]; then
    source /app/.venv/bin/activate
fi

# ============================================================================
# STEP 1: Download model checkpoint from GCS
# ============================================================================
download_model_from_gcs() {
    local gcs_path="${1}"
    local local_path="${2}"

    if [ -z "$gcs_path" ]; then
        echo "WARNING: MODEL_GCS_PATH not set, skipping model download"
        return 0
    fi

    # Create directory
    mkdir -p "$(dirname "$local_path")"

    echo "Downloading model from $gcs_path to $local_path..."

    # Use Python with google-cloud-storage
    python3 << 'EOF'
import os
import sys
from pathlib import Path
from google.cloud import storage

gcs_path = os.environ.get("MODEL_GCS_PATH", "")
local_path = os.environ.get("MODEL_CHECKPOINT_PATH", "")

if not gcs_path or not local_path:
    print("Missing MODEL_GCS_PATH or MODEL_CHECKPOINT_PATH")
    sys.exit(0)

# Parse GCS path: gs://bucket/key
if not gcs_path.startswith("gs://"):
    print(f"Invalid GCS path (must start with gs://): {gcs_path}")
    sys.exit(0)

parts = gcs_path[5:].split("/", 1)
if len(parts) != 2:
    print(f"Invalid GCS path format: {gcs_path}")
    sys.exit(0)

bucket_name, blob_path = parts

try:
    print(f"   Bucket: {bucket_name}")
    print(f"   Blob: {blob_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Check if blob exists
    if not blob.exists():
        print(f"Model not found at {gcs_path}")
        print("Proceeding with baseline model")
        sys.exit(0)

    # Get size safely
    blob.reload()
    size_mb = blob.size / (1024**2) if blob.size else 0
    print(f"   Size: {size_mb:.1f} MB")

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

    print(f"Model downloaded to {local_path}")

except Exception as e:
    print(f"ERROR: Failed to download model: {e}")
    print("Proceeding with baseline model")

EOF
}

# ============================================================================
# STEP 2: Parse command and run appropriate service
# ============================================================================

# Check if we're in development mode or running tests
if [ "$1" = "pytest" ]; then
    # Run pytest with all arguments passed after "pytest"
    shift
    exec pytest "$@"
elif [ "$1" = "bash" ]; then
    # Interactive shell for debugging
    exec /bin/bash
else
    # Determine service mode (default to "serve" for uvicorn)
    SERVICE_MODE="${1:-serve}"

    # Download model before starting the service
    if [ "$SERVICE_MODE" = "serve" ]; then
        download_model_from_gcs "$MODEL_GCS_PATH" "$MODEL_CHECKPOINT_PATH"
    fi

    # Start the appropriate service
    # Use uv run to automatically handle virtual environment
    if [ "$DEV" = "1" ]; then
        # Development mode - run server with reload
        exec uv run uvicorn main:app --host 0.0.0.0 --port ${PORT} --reload
    else
        # Production mode - run server without reload
        exec uv run uvicorn main:app --host 0.0.0.0 --port ${PORT}
    fi
fi
