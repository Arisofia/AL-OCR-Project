"""Smoke test for exception handler guard script entrypoint."""

import importlib


def test_exception_guard_script_runs():
    mod = importlib.import_module("ocr_service.scripts.check_exception_handlers")
    assert hasattr(mod, "main")
    assert mod.main() == 0
