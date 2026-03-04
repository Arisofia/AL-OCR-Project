"""Unit tests for trace id helper functions."""

from unittest.mock import MagicMock

from ocr_service.utils.tracing import get_current_trace_id


def test_get_current_trace_id_none(monkeypatch):
    span = MagicMock()
    span.get_span_context = MagicMock(return_value=MagicMock(trace_id=None))
    monkeypatch.setattr("opentelemetry.trace.get_current_span", lambda: span)

    assert get_current_trace_id() is None


def test_get_current_trace_id_hex(monkeypatch):
    span = MagicMock()
    span.get_span_context = MagicMock(return_value=MagicMock(trace_id=0xABCDEF))
    monkeypatch.setattr("opentelemetry.trace.get_current_span", lambda: span)

    assert get_current_trace_id() == format(0xABCDEF, "x")
