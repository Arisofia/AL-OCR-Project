import importlib


def test_exception_guard_script_runs():
    mod = importlib.import_module("ocr_service.scripts.check_exception_handlers")
    assert hasattr(mod, "main")
    # main should return 0 (no violations) for current codebase
    assert mod.main() == 0
