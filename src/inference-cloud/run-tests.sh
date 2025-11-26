#!/bin/bash
set -e

echo "Building Docker image..."
docker build -f Dockerfile -t inference-cloud:test ..

echo ""
echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="
docker run --rm inference-cloud:test pytest tests/test_unit.py -v

echo ""
echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
docker run --rm inference-cloud:test pytest tests/test_integration.py -v

echo ""
echo "=========================================="
echo "Running System Tests (Cloud Run)"
echo "=========================================="
docker run --rm inference-cloud:test pytest tests/test_system.py -v

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
