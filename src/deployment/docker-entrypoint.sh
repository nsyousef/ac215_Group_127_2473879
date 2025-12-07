#!/bin/bash
set -e
echo "Container is running!!!"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

# Clean up any empty mount directories in /app that might shadow root-level mounts
echo "Cleaning up empty mount directories..."
rm -rf /app/inference-cloud /app/llm /app/ml_workflow 2>/dev/null || true

# Activate virtual environment (created by uv sync)
# This ensures Python can find Pulumi SDK packages (pulumi-gcp, pulumi-docker, etc.)
echo "Checking for environment..."
if [ -f "/home/app/.venv/bin/activate" ]; then
    source /home/app/.venv/bin/activate
    echo "Environment ready! Virtual environment activated."
fi

# Authenticate gcloud using service account
echo "Authenticating gcloud with service account..."
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project $GCP_PROJECT
# login to artifact-registry
gcloud auth configure-docker us-docker.pkg.dev --quiet
# Check if the bucket exists
if ! gsutil ls -b $PULUMI_BUCKET >/dev/null 2>&1; then
    echo "Bucket does not exist. Creating..."
    gsutil mb -p $GCP_PROJECT $PULUMI_BUCKET
    gsutil versioning set on $PULUMI_BUCKET
else
    echo "Bucket already exists. Skipping creation."
fi

echo "Logging into Pulumi using GCS bucket: $PULUMI_BUCKET"
pulumi login $PULUMI_BUCKET

# List available stacks
echo "Available Pulumi stacks in GCS:"
# gsutil cat $PULUMI_BUCKET/.pulumi/stacks/  || echo "No stacks found."
pulumi stack ls

# set up modal tokens
echo "Setting up Modal tokens..."
modal token set --token-id $(cat ./secrets/modal-token-id.txt) --token-secret $(cat ./secrets/modal-token-secret.txt)

# Initialize and configure Pulumi stack
echo "Configuring Pulumi stack..."

# Create dev stack if it doesn't exist
if ! pulumi stack select dev 2>/dev/null; then
    echo "Creating dev stack..."
    pulumi stack init dev
else
    echo "Dev stack already exists."
fi

# Check and set/update GCP project
CURRENT_PROJECT=$(pulumi config get gcp:project 2>/dev/null || echo "")
if [ "$CURRENT_PROJECT" != "$GCP_PROJECT" ]; then
    echo "Setting GCP project: $GCP_PROJECT (was: ${CURRENT_PROJECT:-not set})"
    pulumi config set gcp:project "$GCP_PROJECT"
else
    echo "GCP project already configured: $GCP_PROJECT"
fi

# Check and set/update GCP region
CURRENT_REGION=$(pulumi config get gcp:region 2>/dev/null || echo "")
if [ "$CURRENT_REGION" != "$GCP_REGION" ]; then
    echo "Setting GCP region: $GCP_REGION (was: ${CURRENT_REGION:-not set})"
    pulumi config set gcp:region "$GCP_REGION"
else
    echo "GCP region already configured: $GCP_REGION"
fi

# Check and set/update Modal username (only if provided)
if [ -n "$MODAL_USERNAME" ]; then
    CURRENT_USERNAME=$(pulumi config get pibu-ai-deployment:modal_username 2>/dev/null || echo "")
    if [ "$CURRENT_USERNAME" != "$MODAL_USERNAME" ]; then
        echo "Setting Modal username: $MODAL_USERNAME (was: ${CURRENT_USERNAME:-not set})"
        pulumi config set pibu-ai-deployment:modal_username "$MODAL_USERNAME"
    else
        echo "Modal username already configured: $MODAL_USERNAME"
    fi
fi

# Check and set/update LLM model size
CURRENT_MODEL_SIZE=$(pulumi config get pibu-ai-deployment:llm_model_size 2>/dev/null || echo "")
if [ "$CURRENT_MODEL_SIZE" != "$LLM_MODEL_SIZE" ]; then
    echo "Setting LLM model size: $LLM_MODEL_SIZE (was: ${CURRENT_MODEL_SIZE:-not set})"
    pulumi config set pibu-ai-deployment:llm_model_size "$LLM_MODEL_SIZE"
else
    echo "LLM model size already configured: $LLM_MODEL_SIZE"
fi

# Check and set Modal token ID secret (always update to ensure sync with secrets file)
NEW_TOKEN_ID=$(cat ./secrets/modal-token-id.txt)
CURRENT_TOKEN_ID=$(pulumi config get pibu-ai-deployment:modal_token_id 2>/dev/null || echo "")
if [ "$CURRENT_TOKEN_ID" != "$NEW_TOKEN_ID" ]; then
    echo "Setting Modal token ID secret... (updating)"
    pulumi config set --secret pibu-ai-deployment:modal_token_id "$NEW_TOKEN_ID"
else
    echo "Modal token ID already configured."
fi

# Check and set Modal token secret (always update to ensure sync with secrets file)
NEW_TOKEN_SECRET=$(cat ./secrets/modal-token-secret.txt)
CURRENT_TOKEN_SECRET=$(pulumi config get pibu-ai-deployment:modal_token_secret 2>/dev/null || echo "")
if [ "$CURRENT_TOKEN_SECRET" != "$NEW_TOKEN_SECRET" ]; then
    echo "Setting Modal token secret... (updating)"
    pulumi config set --secret pibu-ai-deployment:modal_token_secret "$NEW_TOKEN_SECRET"
else
    echo "Modal token secret already configured."
fi

echo "Pulumi configuration complete!"

# Execute command
exec "$@"
