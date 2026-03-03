"""Tracing utility functions."""

import sys
from types import ModuleType
from typing import Any, Optional, cast


def _ensure_trace_module() -> ModuleType:
    """Return a trace module object, creating a lightweight fallback if needed."""
    opentelemetry_mod = sys.modules.get("opentelemetry")
    if opentelemetry_mod is None:
        opentelemetry_mod = ModuleType("opentelemetry")
        sys.modules["opentelemetry"] = opentelemetry_mod

    trace_mod = getattr(opentelemetry_mod, "trace", None)
    if not isinstance(trace_mod, ModuleType):
        trace_mod = ModuleType("opentelemetry.trace")
        cast(Any, opentelemetry_mod).trace = trace_mod
        sys.modules["opentelemetry.trace"] = trace_mod

    if not hasattr(trace_mod, "get_current_span"):
        cast(Any, trace_mod).get_current_span = lambda: None

    return trace_mod


try:
    from opentelemetry import trace as _trace
except ImportError:
    _trace = _ensure_trace_module()
trace: Any = _trace


def get_current_trace_id() -> Optional[str]:
    """Return current trace ID as a hex string, or None if unavailable."""
    get_current_span = getattr(trace, "get_current_span", None)
    if not callable(get_current_span):
        return None
    span = get_current_span()
    if span is None:
        return None
    get_span_context = getattr(span, "get_span_context", None)
    if not callable(get_span_context):
        return None
    ctx = get_span_context()
    trace_id = getattr(ctx, "trace_id", None)
    if isinstance(trace_id, int):
        return format(trace_id, "x")
    return None
