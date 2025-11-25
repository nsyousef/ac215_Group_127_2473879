#!/bin/bash
set -e  # Exit on any error

# Build the container
docker build -f Dockerfile -t inference-cloud:latest ..

# Run the container
docker run -p 8080:8080 --rm inference-cloud:latest
