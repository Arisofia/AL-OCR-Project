"""Tracing utility functions."""

import sys
from types import ModuleType
from typing import Optional


def _ensure_trace_module() -> ModuleType:
    """Return a trace module object, creating a lightweight fallback if needed."""
    opentelemetry_mod = sys.modules.get("opentelemetry")
    if opentelemetry_mod is None:
        opentelemetry_mod = ModuleType("opentelemetry")
        sys.modules["opentelemetry"] = opentelemetry_mod

    trace_mod = getattr(opentelemetry_mod, "trace", None)
    if not isinstance(trace_mod, ModuleType):
        trace_mod = ModuleType("opentelemetry.trace")
        opentelemetry_mod.trace = trace_mod
        sys.modules["opentelemetry.trace"] = trace_mod

    if not hasattr(trace_mod, "get_current_span"):
        trace_mod.get_current_span = lambda: None  # type: ignore[attr-defined]

    return trace_mod


try:
    from opentelemetry import trace as _trace
except ImportError:
    _trace = _ensure_trace_module()
trace = _trace


def get_current_trace_id() -> Optional[str]:
    """Return current trace ID as a hex string, or None if unavailable."""
    span = trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context() if span is not None else None
    if ctx is not None and getattr(ctx, "trace_id", None) is not None:
        return format(ctx.trace_id, "x")
    return None
