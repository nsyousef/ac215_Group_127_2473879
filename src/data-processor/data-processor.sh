#!/bin/bash
set -e  # Exit on any error

echo "Building data-processor Docker container..."
docker build -t data-processor -f Dockerfile ../../..

echo "Running data-processor..."
docker run --rm -ti data-processor

echo "Done."
