# Stage 1: Build stage for compiling heavy dependencies
FROM public.ecr.aws/lambda/python:3.11 AS builder

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

WORKDIR /build
COPY ocr_reconstruct ./ocr_reconstruct
COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 && \
    pip install --prefix=/install --prefer-binary --no-cache-dir ./ocr_reconstruct && \
    pip install --prefix=/install --prefer-binary --no-cache-dir -r ./requirements.txt

# Stage 2: Final runtime image on official Lambda base
FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

RUN yum update -y && \
    yum install -y python2 amazon-linux-extras && \
    ln -sf /usr/bin/python2 /usr/bin/python && \
    amazon-linux-extras install -y epel && \
    yum install -y \
    tesseract \
    tesseract-langpack-eng \
    tesseract-langpack-spa \
    mesa-libGL \
    shadow-utils && \
    /usr/bin/tesseract --version && \
    yum clean all && \
    rm -rf /var/cache/yum

# Remove setuid/setgid bits for security compliance
RUN find /usr/bin /usr/sbin /bin /sbin -type f -perm /6000 -exec chmod u-s,g-s {} + || true

COPY --from=builder /install /var/lang

RUN /usr/sbin/useradd -m appuser && chown -R appuser:appuser /var/task

COPY --chown=appuser:appuser ocr_service/ /var/task/ocr_service/
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct/

ENV PYTHONPATH=/var/task
ENV TESSERACT_CMD=/usr/bin/tesseract

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD [ "echo", "ok" ]

USER appuser

CMD [ "ocr_service.lambda_handler.handler" ]
