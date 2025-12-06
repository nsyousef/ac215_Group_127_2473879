#!/bin/bash

# Exit on error
set -e

# Secrets folder path: Project/secrets
# Structure: Project/secrets/ (at Project level, outside the project directory)
# Goes up 3 levels from script: src/deployment -> src -> project_root -> Project -> /secrets
# Can be overridden with SECRETS_PATH environment variable
SECRETS_PATH=${SECRETS_PATH:-$(cd "$(dirname "$0")/../../.." && pwd)/secrets}

# Build the Docker image
echo "Building deployment container..."
docker build -t pibu-ai-deployment:latest .

# Run the container with interactive shell
echo "Starting deployment shell..."
echo "Using secrets from: $SECRETS_PATH"

docker run --rm -it \
  -v "$(pwd):/app" \
  -v "$HOME/.pulumi:/home/app/.pulumi" \
  -v "$HOME/.config/gcloud:/root/.config/gcloud" \
  -v "$HOME/.modal.toml:/root/.modal.toml" \
  -v "$SECRETS_PATH:/app/secrets:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/gcp-service.json" \
  -e MODAL_TOKEN_ID="$(cat "$SECRETS_PATH/modal-token-id.txt" 2>/dev/null || echo '')" \
  -e MODAL_TOKEN_SECRET="$(cat "$SECRETS_PATH/modal-token-secret.txt" 2>/dev/null || echo '')" \
  --workdir /app \
  pibu-ai-deployment:latest \
  /bin/bash
