"""Error responses the frontend can tell apart.

Every error is sent as ``{"detail": {"code": ..., "message": ...}}``. The
frontend picks what to show from `code` (see frontend/src/constants/errors.ts,
the codes must match); `message` is English, for logs and /docs.
"""
from enum import Enum

from fastapi import HTTPException


class ErrorCode(str, Enum):
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPT_FILE = "CORRUPT_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    DURATION_TOO_SHORT = "DURATION_TOO_SHORT"
    DURATION_TOO_LONG = "DURATION_TOO_LONG"
    SILENT_AUDIO = "SILENT_AUDIO"
    UNPROCESSABLE_AUDIO = "UNPROCESSABLE_AUDIO"
    # ReccoBeats (the audio-feature service) could not be used
    FEATURE_SERVICE_UNAVAILABLE = "FEATURE_SERVICE_UNAVAILABLE"   # down, timed out, 5xx
    FEATURE_SERVICE_BUSY = "FEATURE_SERVICE_BUSY"                 # 429 rate limit
    FEATURE_SERVICE_FAILED = "FEATURE_SERVICE_FAILED"             # rejected the clip or sent a bad answer
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def error_detail(code: ErrorCode, message: str) -> dict:
    return {"code": code.value, "message": message}


def api_error(status_code: int, code: ErrorCode, message: str) -> HTTPException:
    """An HTTPException whose body is ``{"detail": {"code", "message"}}``."""
    return HTTPException(status_code=status_code, detail=error_detail(code, message))
