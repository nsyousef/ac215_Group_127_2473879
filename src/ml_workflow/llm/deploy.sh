#!/bin/bash

# Simple deployment script for dermatology LLM assistant
# Usage: ./deploy.sh [4b|27b] [gpu_type]

# Configuration
MODEL=${1:-27b}  # Default to 27b
GPU=${2:-H200}   # Default to H200

# Map model names
if [ "$MODEL" = "4b" ]; then
    MODEL_NAME="medgemma-4b"
    MAX_TOKENS=500
    GPU_DEFAULT="A100"
elif [ "$MODEL" = "27b" ]; then
    MODEL_NAME="medgemma-27b"
    MAX_TOKENS=700
    GPU_DEFAULT="H200"
else
    echo "❌ Invalid model. Use '4b' or '27b'"
    echo "Usage: ./deploy.sh [4b|27b] [gpu_type]"
    exit 1
fi

# Use provided GPU or default
GPU=${2:-$GPU_DEFAULT}

echo "================================"
echo "Deploying: $MODEL_NAME"
echo "GPU: $GPU"
echo "Max Tokens: $MAX_TOKENS"
echo "================================"

# Set environment variables for Modal
export MODAL_MODEL_NAME="$MODEL_NAME"
export MODAL_MAX_TOKENS="$MAX_TOKENS"
export MODAL_GPU="$GPU"

# Deploy
modal serve llm_modal.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "Model: $MODEL_NAME"
    echo "GPU: $GPU"
    echo ""
    echo "Endpoints:"
    echo "  /explain      - Generate initial analysis"
    echo "  /ask_followup - Answer follow-up questions"
else
    echo ""
    echo "❌ Deployment failed!"
    exit 1
fi
