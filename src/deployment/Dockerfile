FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv
ENV PULUMI_CONFIG_PASSPHRASE=""

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gnupg \
    ca-certificates \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# Install gcloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app

USER app
WORKDIR /app

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Install Pulumi (as app user)
RUN curl -fsSL https://get.pulumi.com | sh
ENV PATH="${PATH}:/home/app/.pulumi/bin"

# Copy application code
COPY --chown=app:app . ./

USER root
RUN chmod +x docker-entrypoint.sh || true

USER app

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
CMD ["/bin/bash"]
