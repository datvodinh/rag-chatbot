FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /code

# Bring in a minimal set of system packages needed for builds and runtime
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget gcc libgl1 libglib2.0-0 libpython3-dev

# Configure runtime defaults and disable telemetry
ENV UV_CACHE_DIR=/root/.cache/uv UV_COMPILE_BYTECODE=0 VIRTUAL_ENV=/opt/conda UV_LINK_MODE=copy

# Bundle the application source into the image
COPY . /code

# Sync the Python environment using uv
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,id=kubox-serve,target=/root/.cache/uv \
    uv sync --active --locked --no-dev

CMD ["python", "-m", "rag_chatbot", "--host", "host.docker.internal"]