#!/bin/bash
set -e

# Build the Docker image for deployment
# Usage: ./docker-build.sh

echo "Building deployment container..."
docker build -t pibu-ai-deployment:latest .

echo "✅ Container built successfully"
