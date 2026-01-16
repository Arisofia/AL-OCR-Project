import logging
from ocr_service.utils import limiter as limiter_mod


def test_init_limiter_success():
    """Return a Limiter instance when initialization succeeds.

    This ensures the happy-path initialization is functional.
    """
    limiter = limiter_mod.init_limiter()
    assert limiter is not None


def test_init_limiter_failure(monkeypatch, caplog):
    """Return None and log an error when the Limiter constructor raises.

    This verifies the failure path (and logging) for init_limiter.
    """
    caplog.set_level(logging.ERROR)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(limiter_mod, "Limiter", _raise)

    res = limiter_mod.init_limiter()
    assert res is None
    assert any("Failed to initialize Limiter" in r.message for r in caplog.records)
