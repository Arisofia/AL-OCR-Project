from collections.abc import Mapping
from typing import Any


def get_request_id_from_scope(scope: Mapping[str, Any]) -> str:
    """Extracts AWS Request ID from Mangum/Lambda scope or defaults to local."""
    if "aws.context" in scope:
        return str(getattr(scope["aws.context"], "aws_request_id", "local-development"))
    return "local-development"
