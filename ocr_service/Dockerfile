# Stage 1: Build stage for compiling heavy dependencies
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build-time dependencies
RUN yum install -y --no-install-recommends \
    gcc \
    gcc-c++ \
    gcc-gfortran \
    python3-devel \
    openblas-devel \
    lapack-devel \
    blas-devel \
    pkgconfig \
    make

RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies to a temporary directory
WORKDIR /build
COPY ocr_reconstruct ./ocr_reconstruct
COPY ocr_service/requirements.txt ./ocr_service/requirements.txt

RUN pip install --prefix=/install --prefer-binary --no-cache-dir ./ocr_reconstruct && \
    pip install --prefix=/install --prefer-binary --no-cache-dir -r ./ocr_service/requirements.txt

# Stage 2: Final lightweight runtime image
FROM public.ecr.aws/lambda/python:3.11

# Runtime dependencies only
RUN yum install -y --no-install-recommends \
    tesseract \
    tesseract-langpack-spa \
    tesseract-langpack-eng \
    mesa-libGL \
    shadow-utils && \
    yum clean all


# Copy installed packages from builder
COPY --from=builder /install /var/lang

# Create a non-root user for security (best practice)
RUN useradd -m appuser && chown -R appuser:appuser /var/task

# Copy application code with non-root ownership
COPY --chown=appuser:appuser ocr_service/ /var/task
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct

ENV PYTHONPATH=/var/task

USER appuser

# Set the CMD to the Lambda S3/Textract handler
CMD [ "lambda_handler.handler" ]
