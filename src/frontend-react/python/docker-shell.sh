#!/bin/bash

set -e

IMAGE_NAME="api-manager-tests"

echo "Building Docker image..."
docker build -t $IMAGE_NAME . > /dev/null 2>&1
echo "✓ Build complete"
echo ""

echo "Running all tests (unit + integration + system)..."
echo "Note: System tests require deployed APIs"
echo ""

docker run --rm \
  -e BASE_URL="${BASE_URL:-}" \
  -e TEXT_EMBEDDING_URL="${TEXT_EMBEDDING_URL:-}" \
  -e PREDICTION_URL="${PREDICTION_URL:-}" \
  -e LLM_EXPLAIN_URL="${LLM_EXPLAIN_URL:-}" \
  -e LLM_FOLLOWUP_URL="${LLM_FOLLOWUP_URL:-}" \
  -e MODAL_USER="${MODAL_USER:-tanushkmr2001}" \
  -e MODEL_SUFFIX="${MODEL_SUFFIX:-4b}" \
  $IMAGE_NAME bash -c "./run_unit_tests.sh && ./run_integration_tests.sh && ./run_system_tests.sh"
