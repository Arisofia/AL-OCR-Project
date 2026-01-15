# Use the AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install Tesseract OCR and its dependencies
RUN yum install -y \
    tesseract \
    tesseract-langpack-spa \
    tesseract-langpack-eng \
    mesa-libGL

# Upgrade pip and wheel so we prefer binary wheels when available
RUN python -m pip install --upgrade pip setuptools wheel

# Install build tools and libraries required to compile packages
RUN yum install -y \
    gcc \
    gcc-c++ \
    gcc-gfortran \
    python3-devel \
    openblas-devel \
    lapack-devel \
    blas-devel \
    pkgconfig \
    make

# Copy ocr_reconstruct first as it's a dependency
COPY ocr_reconstruct ./ocr_reconstruct
RUN pip install --prefer-binary --no-cache-dir -e ./ocr_reconstruct

# Create requirements.txt and install dependencies
COPY ocr-service/requirements.txt .
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /var/task

# Copy the application code with correct ownership
COPY --chown=appuser:appuser ocr-service/ .

USER appuser

# Set the CMD to the Lambda S3/Textract handler
CMD [ "lambda_handler.handler" ]
