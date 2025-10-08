# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m app
USER app
WORKDIR /app

# Copy only dependency file first (for caching)
COPY --chown=app:app requirements.txt .

# Install Python dependencies directly (no virtualenv)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Start interactive bash by default
ENTRYPOINT ["/bin/bash"]