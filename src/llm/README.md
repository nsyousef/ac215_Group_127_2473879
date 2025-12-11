# LLM Service

Modal-deployed dermatology assistant using MedGemma models for medical explanations and follow-up questions.

## Overview

Provides LLM-powered endpoints for:
- **Initial explanations**: Generate medical explanations from disease predictions
- **Follow-up questions**: Answer user questions with conversation context
- **Time tracking summaries**: Analyze progression of skin conditions over time
- **Streaming responses**: Real-time token streaming for better UX

## Models

- **MedGemma-4b**: Smaller, faster model (50 tokens)
- **MedGemma-27b**: Larger, more capable model (400-700 tokens)

## API Endpoints

- `POST /explain` - Generate explanation from predictions and metadata
- `POST /ask_followup` - Answer follow-up question with history
- `POST /ask_followup_stream` - Streaming follow-up (JSONL)
- `POST /explain_stream` - Streaming explanation (JSONL)
- `POST /time_tracking_summary` - Generate time progression summary

## Quick Start

### Deploy to Modal

```bash
# Manual deployment
export MODAL_MODEL_NAME="medgemma-27b"
export MODAL_MAX_TOKENS="700"
export MODAL_GPU="H200"
modal deploy llm_modal.py

# Or use helper script
./deploy.sh 27b H200
```

### Environment Variables

```bash
export MODAL_MODEL_NAME="medgemma-27b"  # or medgemma-4b
export MODAL_MAX_TOKENS="700"
export MODAL_GPU="H200"  # GPU type for Modal
```

## Architecture

- **llm.py**: Core LLM wrapper with MedGemma integration
- **llm_modal.py**: Modal deployment configuration
- **prompts.py**: Prompt templates for different use cases
- **Modal App**: Auto-scales with GPU support, caches model in volume

## Testing

```bash
# Unit tests
./run_unit_tests.sh

# Integration tests (requires Modal deployment)
./run_integration_tests.sh

# System tests (requires deployed endpoints)
./run_system_tests.sh
```

## Deployment

Deployed via Pulumi (see `src/deployment/modules/modal_llm.py`) or manually using `deploy.sh`.

Service auto-scales (0-1 containers) with 1-hour scale-down window to minimize costs.
