#!/bin/bash

set -e

IMAGE_NAME="llm-tests"

echo "Building Docker image..."
docker build -t $IMAGE_NAME . > /dev/null 2>&1
echo "✓ Build complete"
echo ""

echo "Running all tests (unit + integration + system)..."
echo "Note: System tests require deployed Modal app"
echo ""

docker run --rm \
  -e MODAL_USER="${MODAL_USER:-tanushkmr2001}" \
  -e MODEL_SUFFIX="${MODEL_SUFFIX:-4b}" \
  -e MODAL_API_EXPLAIN_URL="${MODAL_API_EXPLAIN_URL:-}" \
  -e MODAL_API_ASK_FOLLOWUP_URL="${MODAL_API_ASK_FOLLOWUP_URL:-}" \
  $IMAGE_NAME bash -c "./run_unit_tests.sh && ./run_integration_tests.sh && ./run_system_tests.sh"
