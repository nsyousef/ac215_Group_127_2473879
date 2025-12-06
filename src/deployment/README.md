# Pibu.AI Deployment

Automated deployment of pibu.ai infrastructure using Pulumi.

## Overview

This deployment automates provisioning of:
- **Inference API**: Cloud Run service for disease classification
- **LLM Service**: Modal deployment for medical explanations (MedGemma)
- **Electron App**: Automated DMG build with deployed service URLs

## Prerequisites

1. **GCP Account** with a project
3. **Modal Account** with API tokens
4. **Docker** (for deployment container)
5. **GitHub Account** (for CI/CD)

## Setup

### 1. Create Secrets Folder

Create a `secret` folder **outside the project** (at the same level as the project directory) to store all credentials:

**Project Structure**:
```
Project/
  ├── ac215_Group_127_2473879/    # Project root
  │   └── src/deployment/          # Deployment code
  └── secrets/                     # Secrets folder (Project/secrets)
      ├── gcp-service.json
      ├── modal-token-id.txt
      └── modal-token-secret.txt
```

**⚠️ Important**: The secrets folder is outside the project and should never be committed to git.

### 2. Configure GCP

**Create Service Account**:
```bash
# Set your project ID
export GCP_PROJECT="your-project-id"

# Create service account
gcloud iam service-accounts create pulumi-deployment \
  --display-name="Pulumi Deployment Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:pulumi-deployment@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:pulumi-deployment@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:pulumi-deployment@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create and download key to secrets folder (Project/secrets)
# Adjust path to match your secrets folder location
gcloud iam service-accounts keys create ../../../secrets/gcp-service.json \
  --iam-account=pulumi-deployment@$GCP_PROJECT.iam.gserviceaccount.com
```

**Enable Required APIs**:
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 4. Configure Pulumi Backend (GCS - Recommended)

**Use GCS Backend** - Store state in a GCS bucket using your existing GCP credentials (no Pulumi Cloud account needed):

```bash
cd src/deployment

# Set your GCP project
export GCP_PROJECT="your-project-id"

# Create GCS bucket for Pulumi state
gsutil mb -p $GCP_PROJECT gs://$GCP_PROJECT-pulumi-state

# Enable versioning (recommended for state backups)
gsutil versioning set on gs://$GCP_PROJECT-pulumi-state

# Initialize Pulumi with GCS backend
pulumi login gs://$GCP_PROJECT-pulumi-state

# Verify backend
pulumi whoami
# Should show: gs://your-project-id-pulumi-state
```


### 5. Configure Modal

**Get Modal Tokens**:
1. Go to https://modal.com/settings
2. Navigate to "API Tokens"
3. Create a new token or use existing
4. Save tokens to secrets folder (outside project):
```bash
# Adjust path to match your secrets folder location (Project/secrets)
echo "YOUR_MODAL_TOKEN_ID" > ../../../secrets/modal-token-id.txt
echo "YOUR_MODAL_TOKEN_SECRET" > ../../../secrets/modal-token-secret.txt
```

**Set Modal credentials** (optional, for CLI usage):
```bash
# Adjust path to match your secrets folder location (Project/secrets)
modal token set --token-id $(cat ../../../secrets/modal-token-id.txt) --token-secret $(cat ../../../secrets/modal-token-secret.txt)
```

### 6. Verify Secrets Folder Structure

Your `secrets/` folder (outside project) should contain:
```
/secrets/                   # Outside project directory
├── gcp-service.json        # GCP service account JSON key (required)
├── modal-token-id.txt      # Modal API token ID (required)
└── modal-token-secret.txt  # Modal API token secret (required)
```

**Verify secrets are set**:
```bash
# From deployment directory
cd src/deployment
ls -la ../../../secrets/
# Or if using absolute path:
ls -la /path/to/Project/secrets/
# Should show: gcp-service.json, modal-token-id.txt, modal-token-secret.txt
```

**Note**: The `docker-shell.sh` script automatically detects the secrets folder at `Project/secrets/` (3 levels up from deployment folder). You can override this by setting the `SECRETS_PATH` environment variable:
```bash
export SECRETS_PATH=/absolute/path/to/secrets
./docker-shell.sh
```

### 7. Configure Pulumi Stack

**Initialize stack**:
```bash
cd src/deployment

# Create dev stack
pulumi stack init dev

# Set GCP project
pulumi config set gcp:project YOUR_GCP_PROJECT_ID

# Set region
pulumi config set gcp:region us-east4

# Set Modal username (if different from default)
pulumi config set pibu-ai-deployment:modal_username YOUR_MODAL_USERNAME

# Set LLM model size (optional, defaults to 27b)
pulumi config set pibu-ai-deployment:llm_model_size "27b"

# Set secrets (encrypted in Pulumi) - read from secrets folder (Project/secrets)
# Adjust path to match your secrets folder location
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
pulumi config set --secret pibu-ai-deployment:modal_token_id $(cat "$SECRETS_PATH/modal-token-id.txt")
pulumi config set --secret pibu-ai-deployment:modal_token_secret $(cat "$SECRETS_PATH/modal-token-secret.txt")
```

**Create prod stack** (optional):
```bash
pulumi stack init prod
pulumi config set gcp:project YOUR_GCP_PROJECT_ID
pulumi config set gcp:region us-east4
# ... same config as dev
```

## Local Testing

### Test Deployment (Local)

**Option 1: Using Docker Container** (Recommended)
```bash
cd src/deployment

# Build container
./docker-build.sh

# Run interactive shell (secrets folder is automatically mounted)
./docker-shell.sh

# Inside container:
pulumi stack select dev
pulumi preview
pulumi up
```

**Option 2: Direct Local Execution**
```bash
cd src/deployment

# Install dependencies
uv sync

# Activate venv
source .venv/bin/activate

# Set environment variables from secrets folder (Project/secrets)
# Adjust path to match your secrets folder location
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
export GOOGLE_APPLICATION_CREDENTIALS="$SECRETS_PATH/gcp-service.json"
export MODAL_TOKEN_ID=$(cat "$SECRETS_PATH/modal-token-id.txt")
export MODAL_TOKEN_SECRET=$(cat "$SECRETS_PATH/modal-token-secret.txt")

# Preview changes
pulumi preview --stack dev

# Deploy
pulumi up --stack dev
```

### Test Export Config

After deployment, export URLs for frontend:
```bash
cd src/deployment

# Using Docker container (secrets folder is mounted automatically from Project/secrets/)
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
docker run --rm \
  -v "$(pwd):/app" \
  -v "$HOME/.pulumi:/home/app/.pulumi" \
  -v "$SECRETS_PATH:/app/secrets:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/gcp-service.json" \
  --workdir /app \
  pibu-ai-deployment:latest \
  bash -c "source .venv/bin/activate && pulumi stack select dev && pulumi stack output frontend_config --json > frontend-config.json"

# Copy to frontend
cp frontend-config.json ../frontend-react/.pulumi-config.json
```

### Test Full Build Flow

```bash
# 1. Deploy infrastructure
cd src/deployment
pulumi up --stack dev

# 2. Export config
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
docker run --rm \
  -v "$(pwd):/app" \
  -v "$HOME/.pulumi:/home/app/.pulumi" \
  -v "$SECRETS_PATH:/app/secrets:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/gcp-service.json" \
  --workdir /app \
  pibu-ai-deployment:latest \
  bash -c "source .venv/bin/activate && pulumi stack select dev && pulumi stack output frontend_config --json > frontend-config.json"

cp frontend-config.json ../frontend-react/.pulumi-config.json

# 3. Build Electron app
cd ../frontend-react
npm install
npm run bundle-python
npm run build
npm run make-dmg

# 4. Test DMG
open dist/pibu_ai.dmg
```

## CI/CD Deployment

### GitHub Actions Workflow

The workflow (`.github/workflows/deploy-and-build.yml`) automatically:
1. Builds deployment container
2. Deploys infrastructure via Pulumi
3. Exports frontend config
4. Builds Electron app with deployed URLs
5. Creates DMG and uploads as artifact

**Trigger manually**:
1. Go to GitHub → Actions → "Deploy Infrastructure & Build Electron App"
2. Click "Run workflow"
3. Select stack (dev/prod)
4. Click "Run workflow"

**Required GitHub Secrets** (for CI/CD only - local uses `secrets/` folder):
- `GCP_SERVICE_ACCOUNT_KEY` - Contents of `secrets/gcp-service.json` (required)
- `MODAL_TOKEN_ID` - Contents of `secrets/modal-token-id.txt` (required)
- `MODAL_TOKEN_SECRET` - Contents of `secrets/modal-token-secret.txt` (required)

**To set GitHub secrets from local secrets folder**:
```bash
# Copy secrets to GitHub (one-time setup)
# Adjust path to match your secrets folder location (Project/secrets)
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
gh secret set GCP_SERVICE_ACCOUNT_KEY < "$SECRETS_PATH/gcp-service.json"
gh secret set MODAL_TOKEN_ID < "$SECRETS_PATH/modal-token-id.txt"
gh secret set MODAL_TOKEN_SECRET < "$SECRETS_PATH/modal-token-secret.txt"
```

## Deployment Commands Reference

### Using Docker Container

```bash
# Build container
./docker-build.sh

# Interactive shell
./docker-shell.sh

# Inside container, you can run:
pulumi stack select dev
pulumi preview
pulumi up
pulumi stack output inference_url
pulumi destroy  # ⚠️ Destroys all resources
```

### Direct Commands

```bash
# Preview changes
pulumi preview --stack dev

# Deploy
pulumi up --stack dev

# View outputs
pulumi stack output --stack dev
pulumi stack output inference_url --stack dev
pulumi stack output frontend_config --json --stack dev

# Destroy (careful!)
pulumi destroy --stack dev
```

## Configuration Files

- **`Pulumi.yaml`**: Project definition
- **`Pulumi.dev.yaml`**: Dev stack configuration
- **`Pulumi.prod.yaml`**: Prod stack configuration (create if needed)
- **`__main__.py`**: Main Pulumi program
- **`modules/`**: Deployment modules (inference, modal_llm)
- **`../secrets/`**: Secrets folder (outside project, not committed to git)
  - `gcp-service.json`: GCP service account key (required)
  - `modal-token-id.txt`: Modal API token ID (required)
  - `modal-token-secret.txt`: Modal API token secret (required)

## Troubleshooting

### Pulumi Authentication Issues

**GCS Backend** (default):
```bash
# Verify GCS backend is set
pulumi whoami
# Should show: gs://your-project-id-pulumi-state

# If not set, login to GCS backend
pulumi login gs://your-project-id-pulumi-state

# Verify GCP credentials
gcloud auth application-default print-access-token
```

### GCP Authentication Issues
```bash
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config get-value project
```

### Modal Authentication Issues
```bash
# Set token
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET

# Verify
modal token list
```

### Container Build Issues
```bash
# Rebuild from scratch
docker build --no-cache -t pibu-ai-deployment:latest .
```

### Pulumi State Issues
```bash
# Refresh state
pulumi refresh --stack dev

# Export state (backup)
pulumi stack export > backup.json
```

## Outputs

After deployment, get service URLs:

```bash
# All outputs
pulumi stack output --stack dev

# Specific outputs
pulumi stack output inference_url --stack dev
pulumi stack output llm_explain_url --stack dev
pulumi stack output llm_followup_url --stack dev

# Frontend config (JSON)
pulumi stack output frontend_config --json --stack dev
```

## Cleanup

### Destroy Infrastructure
```bash
pulumi destroy --stack dev
```

### Remove Stack
```bash
pulumi stack rm dev
```

## Cost Estimate

**Development (minimal usage)**:
- Cloud Run: ~$0 (free tier + scale-to-zero)
- Modal LLM: ~$0.50/hour when active
- GCS/Monitoring: ~$1/month

**Production (moderate usage)**:
- Cloud Run: ~$30-50/month (1 min instance)
- Modal LLM: ~$100/month (depends on usage)
- GCS/Monitoring: ~$5/month

## Next Steps

After deployment:
1. Test inference API: `curl $(pulumi stack output inference_url)/health`
2. Export config: Copy `frontend-config.json` to `../frontend-react/.pulumi-config.json`
3. Build Electron app: Follow `BUILD_MACOS.md` in `src/frontend-react/`
4. Test DMG: Open and install the app

## References

- [Pulumi GCP Documentation](https://www.pulumi.com/docs/clouds/gcp/)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Modal Documentation](https://modal.com/docs)
- [Project Architecture](../../docs/architecture.pdf)
