# Use the AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install Tesseract OCR and its dependencies
RUN yum install -y \
    tesseract \
    tesseract-langpack-spa \
    tesseract-langpack-eng \
    mesa-libGL

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Upgrade pip and wheel so we prefer binary wheels when available
RUN python -m pip install --upgrade pip setuptools wheel

# Install build tools and libraries required to compile packages like NumPy/SciPy if wheels are unavailable
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

# Install Python dependencies (prefer binary wheels to avoid source builds)
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy the application code
COPY . .

# Set the CMD to the Lambda S3/Textract handler
CMD [ "lambda_handler.handler" ]
