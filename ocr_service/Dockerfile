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

RUN yum update -y && yum install -y \
    mesa-libGL \
    shadow-utils \
    curl \
    libjpeg \
    libpng \
    libtiff \
    zlib \
    libwebp \
    giflib \
    openjpeg2 \
    libcurl \
    libarchive && \
    curl -fsSL -o /tmp/tesseract.rpm \
      "https://api.opensuse.org/public/source/home:chethanuk/tesseract5/tesseract-ocr-5.3.1-1.amzn2023.leptonica1.80.0.x86_64.rpm" && \
    yum localinstall -y /tmp/tesseract.rpm && \
    mkdir -p /usr/local/share/tessdata && \
    curl -fsSL -o /usr/local/share/tessdata/spa.traineddata \
      "https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata" && \
    /usr/local/bin/tesseract --version && \
    rm -f /tmp/tesseract.rpm && \
    yum clean all && \
    rm -rf /var/cache/yum

# Remove setuid/setgid bits for security compliance
RUN find /usr/bin /usr/sbin /bin /sbin -type f -perm /6000 -exec chmod u-s,g-s {} + || true

COPY --from=builder /install /var/lang

RUN /usr/sbin/useradd -m appuser && chown -R appuser:appuser /var/task

COPY --chown=appuser:appuser ocr_service/ /var/task/ocr_service/
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct/

ENV PYTHONPATH=/var/task
ENV PATH=/usr/local/bin:${PATH}
ENV TESSERACT_CMD=/usr/local/bin/tesseract
ENV TESSDATA_PREFIX=/usr/local/share/tessdata

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD [ "echo", "ok" ]

USER appuser

CMD [ "ocr_service.lambda_handler.handler" ]
