#!/bin/bash
set -e

# Activate virtual environment
source /app/.venv/bin/activate

# ============================================================================
# STEP 1: Download model checkpoint from GCS if needed
# ============================================================================
download_model_from_gcs() {
    local gcs_path="${1}"
    local local_path="${2}"

    if [ -z "$gcs_path" ]; then
        echo "⚠️  MODEL_GCS_PATH not set, skipping model download"
        return 0
    fi

    # Check if model already exists locally
    if [ -f "$local_path" ]; then
        echo "✓ Model checkpoint already exists at $local_path"
        return 0
    fi

    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$local_path")"

    echo "📥 Downloading model from $gcs_path to $local_path..."

    # Use Python with google-cloud-storage for reliable download
    python3 << 'EOF'
import os
import sys
from pathlib import Path
from google.cloud import storage

gcs_path = os.environ.get("MODEL_GCS_PATH", "")
local_path = os.environ.get("MODEL_CHECKPOINT_PATH", "")

if not gcs_path or not local_path:
    print("⚠️  Missing GCS_PATH or CHECKPOINT_PATH, skipping download")
    sys.exit(0)

# Parse GCS path: gs://bucket/key
if not gcs_path.startswith("gs://"):
    print(f"❌ Invalid GCS path (must start with gs://): {gcs_path}")
    sys.exit(1)

parts = gcs_path[5:].split("/", 1)  # Remove gs:// and split on first /
if len(parts) != 2:
    print(f"❌ Invalid GCS path format (expected gs://bucket/key): {gcs_path}")
    sys.exit(1)

bucket_name, blob_path = parts

try:
    # Check for GCS credentials (optional - Cloud Run uses Workload Identity)
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_file and not os.path.exists(creds_file):
        print(f"⚠️  GCS credentials file specified but not found: {creds_file}")
        print("   Attempting to use Application Default Credentials (Workload Identity)...")
    elif creds_file:
        print(f"   Using GCS credentials from: {creds_file}")
    else:
        print("   Using Application Default Credentials (Workload Identity)...")

    # Initialize GCS client (will auto-detect credentials from env)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    print(f"  Bucket: {bucket_name}")
    print(f"  Blob: {blob_path}")
    print(f"  Size: {blob.size / (1024**2):.1f} MB")

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

    print(f"✓ Model downloaded successfully to {local_path}")

except Exception as e:
    print(f"❌ Failed to download model: {e}")
    print("⚠️  Proceeding with baseline model (random initialization)")
    print("   Predictions will be unreliable without a trained checkpoint")

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
        download_model_from_gcs
    fi

    # Start the appropriate service
    if [ "$DEV" = "1" ]; then
        # Development mode - run server with reload
        exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --reload
    else
        # Production mode - run server without reload
        exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
    fi
fi
