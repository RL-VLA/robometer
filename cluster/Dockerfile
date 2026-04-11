FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Install deps into /opt/venv (outside /workspace so the bind mount doesn't hide it)
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv && \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-dev --no-install-project

# Copy code
COPY robometer/ robometer/
COPY dataset_upload/ dataset_upload/
COPY train.py setup.py README.md ./

# Install robometer into /opt/venv so it survives the /workspace bind mount at runtime
RUN VIRTUAL_ENV=/opt/venv uv pip install . --no-deps

ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/workspace"
