from typing import Optional

from opentelemetry import trace


def get_current_trace_id() -> Optional[str]:
    """Return current trace ID as a hex string, or None if unavailable."""
    span = trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context() if span is not None else None
    if ctx is not None and getattr(ctx, "trace_id", None) is not None:
        return format(ctx.trace_id, "x")
    return None
