# Pibu.AI Deployment

Automated deployment of infrastructure using Pulumi: GKE (Google Kubernetes Engine) for inference API, Modal for LLM service, and Electron app builds.

## Prerequisites

- GCP project with billing enabled
- Modal account with API tokens
- Docker installed
- GitHub account (for CI/CD)

## Setup

### 1. Secrets Folder

Create `secrets/` folder **outside the project** (at same level as project directory):

```
Project/
  ├── ac215_Group_127_2473879/
  └── secrets/                    # Outside project
      ├── gcp-service.json
      ├── deployment.json
      ├── modal-token-id.txt
      └── modal-token-secret.txt
```

### 2. GCP Service Accounts

**Create `deployment` service account** with roles:
- Compute Admin, Artifact Registry Administrator, Kubernetes Engine Admin, Service Account User, Storage Admin

**Create `gcp-service` service account** with roles:
- Storage Object Viewer, Artifact Registry Reader

Download JSON keys for both and save to `secrets/` folder.

### 3. Modal Tokens

1. Go to https://modal.com/settings → API Tokens
2. Create token and save to `secrets/`

## Local Deployment

```bash
cd src/deployment

# Build and run container
./docker-build.sh
./docker-shell.sh

# Inside container:
pulumi stack select dev
pulumi preview
pulumi up
```

**Export frontend config**:
```bash
pulumi stack output frontend_config --json > frontend-config.json
cp frontend-config.json ../frontend-react/.pulumi-config.json
```

## CI/CD

GitHub Actions workflow (`.github/workflows/deploy-and-build.yml`) automatically:
1. Builds deployment container
2. Deploys infrastructure
3. Exports frontend config
4. Builds Electron app
5. Creates DMG artifact

**Set GitHub secrets** (one-time):
```bash
SECRETS_PATH=${SECRETS_PATH:-$(cd ../../.. && pwd)/secrets}
gh secret set GCP_SERVICE_ACCOUNT_KEY < "$SECRETS_PATH/gcp-service.json"
gh secret set MODAL_TOKEN_ID < "$SECRETS_PATH/modal-token-id.txt"
gh secret set MODAL_TOKEN_SECRET < "$SECRETS_PATH/modal-token-secret.txt"
```

Trigger: GitHub → Actions → "Deploy Infrastructure & Build Electron App" → Run workflow

## Files

- `Pulumi.yaml` / `Pulumi.dev.yaml`: Stack configuration
- `__main__.py`: Main Pulumi program
- `modules/`: Deployment modules (inference, modal_llm)
- `secrets/`: Credentials (outside project, not in git)
