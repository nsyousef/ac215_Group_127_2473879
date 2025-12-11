# Inference Cloud Service

FastAPI service for skin disease classification using multimodal vision and text embeddings.

## Overview

Provides ML inference endpoints for:
- **Text embedding**: Generate embeddings from patient descriptions
- **Disease prediction**: Classify skin conditions from vision + text embeddings
- **Model info**: Get available disease classes

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health with model info
- `POST /embed-text` - Generate text embedding
- `POST /predict` - Predict skin condition from embeddings
- `GET /classes` - List all available disease classes

## Quick Start

### Local Development

```bash
# Build and run
docker build -f Dockerfile -t inference-cloud:latest ../..
docker run -p 8080:8080 inference-cloud:latest
```

### Environment Variables

```bash
export PORT=8080
export MODEL_CHECKPOINT_PATH=/tmp/models/test_best.pth
export MODEL_GCS_PATH=gs://apcomp215-datasets/test_best.pth
export DEVICE=cpu  # or cuda
```

### Run Tests

```bash
# Unit tests
pytest tests/test_unit.py -v

# Integration tests
pytest tests/test_integration.py -v

# System tests (requires deployed service)
pytest tests/test_system.py -v
```

## Architecture

- **FastAPI** application with async endpoints
- **MultimodalClassifier** from `ml_workflow` for predictions
- **Text encoder** (transformers) for text embeddings
- **Model loading** from GCS on startup (via `docker-entrypoint.sh`)

## Deployment

Deployed to GKE via Pulumi (see `src/deployment/modules/gke_inference.py`).

Service automatically downloads model checkpoint from GCS on startup if `MODEL_GCS_PATH` is set.
