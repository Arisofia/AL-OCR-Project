"""Scan for exception handlers without logging or re-raise."""

import ast
import logging
import sys
from pathlib import Path

logger = logging.getLogger("exception-guard")


def _has_logging_or_raise(node: ast.ExceptHandler, source_text: str) -> bool:
    body_src = "".join(
        ast.get_source_segment(source_text, stmt) or "" for stmt in node.body
    )
    return ("logger." in body_src) or ("raise" in body_src)


def main() -> int:
    root = Path("ocr_service")
    failures = []

    for py in root.rglob("*.py"):
        try:
            src = py.read_text()
            tree = ast.parse(src)
        except Exception as e:
            logger.debug("Failed to parse %s: %s", py, e)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    handler_name = getattr(handler.type, "id", None)
                    cond = handler.type is None or handler_name == "Exception"
                    if cond and not _has_logging_or_raise(handler, src):
                        failures.append(str(py))
                        break

    if failures:
        logger.error(
            "Found exception handlers without logging or re-raise in: %s",
            sorted(set(failures)),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
