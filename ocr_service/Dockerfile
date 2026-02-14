# Stage 0: Build a portable Tesseract runtime on Amazon Linux 2
FROM amazonlinux:2 AS tesseract

RUN yum update -y && \
    yum install -y yum-utils amazon-linux-extras && \
    amazon-linux-extras install -y epel && \
    yum install -y \
      tesseract \
      tesseract-langpack-eng \
      tesseract-langpack-spa && \
    /usr/bin/tesseract --version && \
    yum clean all && \
    rm -rf /var/cache/yum

RUN set -eux; \
    mkdir -p /opt/tesseract-root/usr/bin /opt/tesseract-root/usr/share/tessdata; \
    cp -a /usr/bin/tesseract /opt/tesseract-root/usr/bin/; \
    TESSDATA_SRC=""; \
    for p in \
      /usr/share/tessdata \
      /usr/share/tesseract/tessdata \
      /usr/share/tesseract-ocr/tessdata \
      /usr/share/tesseract/4.00/tessdata \
      /usr/share/tesseract-ocr/4.00/tessdata; do \
      if [ -d "$p" ]; then \
        TESSDATA_SRC="$p"; \
        break; \
      fi; \
    done; \
    if [ -z "$TESSDATA_SRC" ]; then \
      echo "Could not locate tessdata directory" >&2; \
      find /usr/share -maxdepth 5 -type d -name tessdata -print >&2; \
      exit 1; \
    fi; \
    cp -a "$TESSDATA_SRC/." /opt/tesseract-root/usr/share/tessdata/; \
    ldd /usr/bin/tesseract | awk '/=> \\/(usr|lib64|lib)\\// {print $3} /^\\/lib64\\// {print $1} /^\\/lib\\// {print $1}' | sort -u | \
    while read -r lib; do \
      [ -n "$lib" ] || continue; \
      [ -e "$lib" ] || continue; \
      mkdir -p "/opt/tesseract-root$(dirname "$lib")"; \
      cp -a "$lib" "/opt/tesseract-root$lib"; \
    done

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
    yum install -y \
    mesa-libGL \
    shadow-utils && \
    yum clean all && \
    rm -rf /var/cache/yum

# Remove setuid/setgid bits for security compliance
RUN find /usr/bin /usr/sbin /bin /sbin -type f -perm /6000 -exec chmod u-s,g-s {} + || true

COPY --from=tesseract /opt/tesseract-root/ /

COPY --from=builder /install /var/lang

RUN /usr/sbin/useradd -m appuser && chown -R appuser:appuser /var/task

COPY --chown=appuser:appuser ocr_service/ /var/task/ocr_service/
COPY --chown=appuser:appuser ocr_reconstruct/ /var/task/ocr_reconstruct/

RUN /usr/bin/tesseract --version

ENV PYTHONPATH=/var/task
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tessdata

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD [ "echo", "ok" ]

USER appuser

CMD [ "ocr_service.lambda_handler.handler" ]
