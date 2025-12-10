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

Create a `secrets` folder **outside the project** (at the same level as the project directory) to store all credentials:

**Project Structure**:
```
Project/
  ├── ac215_Group_127_2473879/    # Project root
  │   └── src/deployment/          # Deployment code
  └── secrets/                     # Secrets folder (Project/secrets)
      ├── gcp-service.json
      ├── deployment.json
      ├── modal-token-id.txt
      └── modal-token-secret.txt
```

**⚠️ Important**: The secrets folder is outside the project and should never be committed to git.

### 2. Configure GCP

**Create Service Accounts via GCP Console**:
1. In the GCP Console, go to **IAM & Admin > Service accounts** and click **Create service account** named `deployment`.
2. Assign roles:
   - Compute Admin
   - Compute OS Login
   - Artifact Registry Administrator
   - Kubernetes Engine Admin
   - Service Account User
   - Storage Admin
   Then click **Done**.
3. In the service accounts list, open the **Actions (⋮)** menu for `deployment` → **Create key** → select **JSON** → **Create**. Move the downloaded key into your `secrets` folder (three levels up from `src/deployment`) and rename it to `deployment.json`.

4. Repeat: create another service account named `gcp-service`.
5. Assign roles:
   - Storage Object Viewer
   - Vertex AI Administrator
   - Artifact Registry Reader
   Then click **Done**.
6. From the **Actions (⋮)** menu for `gcp-service`, choose **Create key** → **JSON** → **Create**. Move the downloaded key into the same `secrets` folder and rename it to `gcp-service.json`.

**Enable Required APIs**:
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 3. Configure Modal

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

### 4. Verify Secrets Folder Structure

Your `secrets/` folder (outside project) should contain:
```
/secrets/                   # Outside project directory
├── deployment.json         # GCP service account JSON key (required)
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

## Local Testing

### Test Deployment (Local)

**Option 1: Using Docker Container** (Recommended)
```bash
cd src/deployment

# Run interactive shell (secrets folder is automatically mounted)
sudo sh docker-shell.sh

# Inside container:
pulumi stack select dev
pulumi preview
pulumi up
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
open dist/Pibu.dmg
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

## How to delete EVERYTHING and start over

```bash
# 1. Delete GKE Node Pool first (this is causing your Pulumi errors)
gcloud container node-pools delete pibu-ai-dev-pool \
  --cluster=pibu-ai-dev \
  --region=us-east1 \
  --project=level-scheme-471117-i9 \
  --quiet

# 2. Delete GKE Cluster
gcloud container clusters delete pibu-ai-dev \
  --region=us-east1 \
  --project=level-scheme-471117-i9 \
  --quiet

# 3. Delete Service Account
gcloud iam service-accounts delete inference-dev@level-scheme-471117-i9.iam.gserviceaccount.com \
  --project=level-scheme-471117-i9 \
  --quiet

# 4. Delete Subnetwork
gcloud compute networks subnets delete pibu-ai-dev-subnet \
  --region=us-east1 \
  --project=level-scheme-471117-i9 \
  --quiet

# 5. Delete Network (must be last - no dependencies)
gcloud compute networks delete pibu-ai-dev-network \
  --project=level-scheme-471117-i9 \
  --quiet

# 6. delete router
gcloud compute routers delete pibu-ai-dev-router --region=us-east1 --project=level-scheme-471117-i9 --quiet

# 7. delete vpc
gcloud compute networks delete pibu-ai-dev-vpc --project=level-scheme-471117-i9 --quiet

# 8. delete artifact registry
gcloud artifacts repositories delete pibu-ai-images --location=us-east1 --project=level-scheme-471117-i9 --quiet
```
