#!/bin/bash

# Script to run integration tests for APIManager
# Assumes cloud APIs and Modal LLM API are already deployed

set -e

echo "================================================"
echo "Running APIManager Integration Tests"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ==================== API Configuration ====================
echo "Configuring API URLs..."
echo "------------------------------------------------"

# Set defaults (can be overridden via environment variables)
MODAL_USER="${MODAL_USER:-tanushkmr2001}"
MODEL_SUFFIX="${MODEL_SUFFIX:-4b}"
APP_NAME="dermatology-llm-${MODEL_SUFFIX}"

# Cloud ML APIs
export BASE_URL="${BASE_URL:-https://inference-cloud-469023639150.us-east4.run.app}"
export TEXT_EMBEDDING_URL="${TEXT_EMBEDDING_URL:-${BASE_URL}/embed-text}"
export PREDICTION_URL="${PREDICTION_URL:-${BASE_URL}/predict}"

# Modal LLM APIs
export LLM_EXPLAIN_URL="${LLM_EXPLAIN_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-explain.modal.run}"
export LLM_FOLLOWUP_URL="${LLM_FOLLOWUP_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-ask-followup.modal.run}"

echo "API Endpoints:"
echo "  Cloud ML Base: $BASE_URL"
echo "  Text Embedding: $TEXT_EMBEDDING_URL"
echo "  Prediction: $PREDICTION_URL"
echo "  LLM Explain: $LLM_EXPLAIN_URL"
echo "  LLM Followup: $LLM_FOLLOWUP_URL"
echo ""

# ==================== Run Tests ====================
echo "Running integration tests..."
echo "------------------------------------------------"
echo "Note: These tests make real HTTP calls"
echo ""

pytest tests/integration.py \
    -v \
    --tb=short \
    --color=yes \
    -m "integration" \
    --durations=10

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ Integration Tests Passed!"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "❌ Integration Tests Failed!"
    echo "================================================"
    exit 1
fi
