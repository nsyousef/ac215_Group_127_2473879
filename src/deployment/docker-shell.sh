#!/bin/bash

# Exit on error
set -e

# Secrets folder path: Project/secrets
# Structure: Project/secrets/ (at Project level, outside the project directory)
# Goes up 3 levels from script: src/deployment -> src -> project_root -> Project -> /secrets
# Can be overridden with SECRETS_PATH environment variable
export SECRETS_PATH=${SECRETS_PATH:-$(cd "$(dirname "$0")/../../.." && pwd)/secrets}
export BASE_DIR=$(pwd)
export GCP_PROJECT="level-scheme-471117-i9"
export GCP_REGION="us-east1"
export LLM_MODEL_SIZE="27b"
export MODAL_USERNAME="nsyousef"

# Get absolute paths to directories on host
INFERENCE_CLOUD_HOST=$(cd "$BASE_DIR/../inference-cloud" && pwd)
LLM_HOST=$(cd "$BASE_DIR/../llm" && pwd)
ML_WORKFLOW_HOST=$(cd "$BASE_DIR/../ml_workflow" && pwd)

# Run container as root to ensure access to host docker.sock (root:root on macOS)
USER_FLAG="-u root"
GROUP_FLAG=""

# Build the Docker image
echo "Building deployment container..."
docker build -t pibu-ai-deployment:latest .

# Run the container with interactive shell
echo "Starting deployment shell..."
echo "Using secrets from: $SECRETS_PATH"
echo "Mounting inference-cloud from: $INFERENCE_CLOUD_HOST"
echo "Mounting llm from: $LLM_HOST"
echo "Mounting ml_workflow from: $ML_WORKFLOW_HOST"

docker run --rm -it \
  -v "$BASE_DIR:/app" \
  -v "$BASE_DIR/..:/ac215_Group_127_2473879/src" \
  -v "$HOME/.config/gcloud:/root/.config/gcloud" \
  -v "$HOME/.modal.toml:/root/.modal.toml" \
  -v "$SECRETS_PATH:/app/secrets:ro" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  $GROUP_FLAG \
  -e GCP_PROJECT="$GCP_PROJECT" \
  -e GCP_REGION="$GCP_REGION" \
  -e LLM_MODEL_SIZE="$LLM_MODEL_SIZE" \
  -e MODAL_USERNAME="$MODAL_USERNAME" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/deployment.json" \
  -e PULUMI_BUCKET="gs://$GCP_PROJECT-pulumi-state" \
  --workdir /app \
  $USER_FLAG \
  pibu-ai-deployment:latest \
  /bin/bash
