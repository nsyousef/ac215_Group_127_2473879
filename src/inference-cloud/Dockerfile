FROM python:3.11-slim-bookworm

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app

USER app
WORKDIR /app

# Copy ml_workflow first (from parent context)
COPY --chown=app:app ml_workflow/ ./ml_workflow/

# Copy inference-cloud contents
COPY --chown=app:app inference-cloud/ ./

# Install dependencies
RUN uv sync

# Set environment variables
ENV PORT=8080
ENV MODEL_CHECKPOINT_PATH=/app/models/test_best.pth
ENV DEVICE=cpu

# Expose port
EXPOSE 8080

# Start FastAPI server automatically
CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
