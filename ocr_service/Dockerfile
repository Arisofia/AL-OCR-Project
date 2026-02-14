# Stage 1: build Python deps in an isolated prefix
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY ocr_reconstruct ./ocr_reconstruct

RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 && \
    pip install --prefix=/install --prefer-binary --no-cache-dir ./ocr_reconstruct && \
    pip install --prefix=/install --prefer-binary --no-cache-dir -r ./requirements.txt && \
    pip install --prefix=/install --no-cache-dir awslambdaric

# Stage 2: Lambda runtime image on Debian with real tesseract packages
FROM python:3.11-slim-bookworm

WORKDIR /var/task

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m appuser && chown -R appuser:appuser /var/task

COPY --chown=appuser:appuser ocr_service/ /var/task/ocr_service/
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct/

ENV PYTHONPATH=/var/task
ENV TESSERACT_CMD=/usr/bin/tesseract

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD [ "echo", "ok" ]

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["ocr_service.lambda_handler.handler"]

USER appuser
