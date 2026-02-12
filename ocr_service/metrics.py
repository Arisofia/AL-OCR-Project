from prometheus_client import Counter, Histogram

# Histograms for latency
OCR_REQUEST_LATENCY = Histogram(
    "ocr_request_latency_seconds", "Latency of OCR requests", ["method", "status"]
)
OCR_PROCESS_BYTES_LATENCY = Histogram(
    "ocr_process_bytes_latency_seconds", "Latency of process_bytes method", ["status"]
)
OCR_ENGINE_PROCESS_IMAGE_LATENCY = Histogram(
    "ocr_engine_process_image_latency_seconds",
    "Latency of IterativeOCREngine.process_image method",
    ["status"],
)
OCR_ENGINE_PROCESS_IMAGE_ADVANCED_LATENCY = Histogram(
    "ocr_engine_process_image_advanced_latency_seconds",
    "Latency of IterativeOCREngine.process_image_advanced method",
    ["status"],
)
OCR_RECONSTRUCTION_LATENCY = Histogram(
    "ocr_reconstruction_latency_seconds",
    "Latency of reconstruction pipeline",
    ["status"],
)
OCR_EXTRACTION_LATENCY = Histogram(
    "ocr_extraction_latency_seconds", "Latency of text extraction", ["method", "status"]
)


# Counters for total requests and errors
OCR_REQUEST_COUNT = Counter(
    "ocr_request_total", "Total number of OCR requests", ["method", "status"]
)
OCR_ERROR_COUNT = Counter(
    "ocr_error_total", "Total number of OCR errors", ["phase", "error_type"]
)
OCR_ITERATION_COUNT = Counter(
    "ocr_iteration_total", "Total number of OCR iterations run"
)
OCR_IDEMPOTENCY_HIT_COUNT = Counter(
    "ocr_idempotency_hit_total", "Number of times idempotency cache was hit"
)
OCR_IDEMPOTENCY_MISS_COUNT = Counter(
    "ocr_idempotency_miss_total", "Number of times idempotency cache was missed"
)

# Redis-related error counts impacting idempotency operations (get/set/delete)
OCR_IDEMPOTENCY_REDIS_ERROR_COUNT = Counter(
    "ocr_idempotency_redis_errors_total",
    "Redis errors impacting idempotency operations",
    ["operation"],
)
