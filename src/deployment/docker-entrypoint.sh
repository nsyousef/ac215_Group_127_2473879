#!/bin/bash
set -e
echo "Container is running!!!"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

# Ensure Pulumi CLI is on PATH (installed under /home/app/.pulumi/bin)
export PATH="$PATH:/home/app/.pulumi/bin"

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
# Configure Docker authentication for Artifact Registry
echo "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet
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

MODAL_TOKEN_ID_VALUE="${MODAL_TOKEN_ID:-}"
MODAL_TOKEN_SECRET_VALUE="${MODAL_TOKEN_SECRET:-}"

# Fallback to secrets directory if env vars not set
if { [ -z "$MODAL_TOKEN_ID_VALUE" ] || [ -z "$MODAL_TOKEN_SECRET_VALUE" ]; } && \
   [ -f "./secrets/modal-token-id.txt" ] && [ -f "./secrets/modal-token-secret.txt" ]; then
    MODAL_TOKEN_ID_VALUE="$(cat ./secrets/modal-token-id.txt)"
    MODAL_TOKEN_SECRET_VALUE="$(cat ./secrets/modal-token-secret.txt)"
fi

if [ -n "$MODAL_TOKEN_ID_VALUE" ] && [ -n "$MODAL_TOKEN_SECRET_VALUE" ]; then
    echo "Configuring Modal CLI tokens..."
    modal token set --token-id "$MODAL_TOKEN_ID_VALUE" --token-secret "$MODAL_TOKEN_SECRET_VALUE"
else
    echo "Modal tokens not provided; skipping Modal CLI token setup."
fi

# Initialize and configure Pulumi stack
echo "Configuring Pulumi stack..."

# Ensure required stacks exist
for STACK_NAME in dev prod; do
    if ! pulumi stack select "$STACK_NAME" >/dev/null 2>&1; then
        echo "Creating $STACK_NAME stack..."
        pulumi stack init "$STACK_NAME"
    else
        echo "$STACK_NAME stack already exists."
    fi
done

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
if [ -n "$MODAL_TOKEN_ID_VALUE" ] && [ -n "$MODAL_TOKEN_SECRET_VALUE" ]; then
    # Check and set Modal token ID secret (always update to ensure sync with provided values)
    NEW_TOKEN_ID="$MODAL_TOKEN_ID_VALUE"
    CURRENT_TOKEN_ID=$(pulumi config get pibu-ai-deployment:modal_token_id 2>/dev/null || echo "")
    if [ "$CURRENT_TOKEN_ID" != "$NEW_TOKEN_ID" ]; then
        echo "Setting Modal token ID secret... (updating)"
        pulumi config set --secret pibu-ai-deployment:modal_token_id "$NEW_TOKEN_ID"
    else
        echo "Modal token ID already configured."
    fi

    # Check and set Modal token secret
    NEW_TOKEN_SECRET="$MODAL_TOKEN_SECRET_VALUE"
    CURRENT_TOKEN_SECRET=$(pulumi config get pibu-ai-deployment:modal_token_secret 2>/dev/null || echo "")
    if [ "$CURRENT_TOKEN_SECRET" != "$NEW_TOKEN_SECRET" ]; then
        echo "Setting Modal token secret... (updating)"
        pulumi config set --secret pibu-ai-deployment:modal_token_secret "$NEW_TOKEN_SECRET"
    else
        echo "Modal token secret already configured."
    fi
else
    echo "Skipping Modal token Pulumi config updates; tokens not provided."
fi

echo "Pulumi configuration complete!"

# Execute command
exec "$@"
