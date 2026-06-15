FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .

COPY app ./app
COPY configs ./configs

RUN groupadd --system cslr \
    && useradd --system --gid cslr --create-home cslr \
    && mkdir -p /workspace/artifacts/exports /workspace/data \
    && chown -R cslr:cslr /workspace

USER cslr

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

USER root
COPY requirements-dev.txt ./
RUN python -m pip install -r requirements-dev.txt
COPY tests ./tests
COPY scripts ./scripts
RUN chown -R cslr:cslr /workspace
USER cslr

FROM base AS runtime
