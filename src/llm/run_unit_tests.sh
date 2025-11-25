#!/bin/bash

# Run unit tests only (excludes integration tests)
# This script runs all tests except those marked with @pytest.mark.integration

set -e  # Exit on error

echo "================================"
echo "Running LLM Unit Tests"
echo "================================"
echo ""

# Run tests excluding integration tests
pytest tests/test_llm.py -v

echo ""
echo "================================"
echo "Unit tests completed successfully!"
echo "================================"
