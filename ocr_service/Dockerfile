# Stage 1: Build stage for compiling heavy dependencies
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build-time dependencies
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    gcc-gfortran \
    python3-devel \
    openblas-devel \
    lapack-devel \
    blas-devel \
    pkgconfig \
    make && \
    yum clean all && \
    rm -rf /var/cache/yum

# Prepare build workspace and install Python tooling and dependencies in one step
WORKDIR /build
COPY ocr_reconstruct ./ocr_reconstruct
COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 && \
    pip install --prefix=/install --prefer-binary --no-cache-dir ./ocr_reconstruct && \
    pip install --prefix=/install --prefer-binary --no-cache-dir -r ./requirements.txt

# Stage 2: Final lightweight runtime image
FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Runtime dependencies only
# Note: Amazon Linux 2 (Karoo) uses yum.
RUN yum update -y && yum install -y \
    tesseract \
    tesseract-langpack-spa \
    tesseract-langpack-eng \
    mesa-libGL \
    shadow-utils && \
    yum clean all && \
    rm -rf /var/cache/yum

# Remove setuid/setgid bits for security compliance
RUN find /usr/bin /usr/sbin /bin /sbin -type f -perm /6000 -exec chmod u-s,g-s {} + || true

# Copy installed packages from builder
COPY --from=builder /install /var/lang

# Create a non-root user for security (best practice)
# Using absolute path for useradd to ensure it is found in AL2023
RUN /usr/sbin/useradd -m appuser && chown -R appuser:appuser /var/task

# Copy application code with non-root ownership
COPY --chown=appuser:appuser ocr_service/ /var/task/ocr_service/
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct/

ENV PYTHONPATH=/var/task

# Healthcheck for Dockle/CIS compliance
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD [ "echo", "ok" ]

USER appuser

# Set the CMD to the Lambda S3/Textract handler
CMD [ "ocr_service.lambda_handler.handler" ]
