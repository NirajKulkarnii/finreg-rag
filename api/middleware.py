"""
Middleware
----------
1. Structlog configuration  — call configure_logging() once at startup.
2. RequestLoggingMiddleware — logs every request/response with timing and
   updates Prometheus HTTP counters/histograms.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output with timestamps and log level."""
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs each HTTP request with:
      - request_id  (UUID, attached to context so downstream logs inherit it)
      - method, path, status_code
      - duration_ms

    Also updates Prometheus HTTP counters and latency histograms.
    Skips /health and /metrics paths to avoid noise.
    """

    SKIP_PATHS = {"/health", "/metrics"}

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t0 = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - t0

        status_code = response.status_code
        method = request.method
        path = request.url.path

        logger.info(
            "http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 1),
        )

        HTTP_REQUESTS_TOTAL.labels(
            method=method, path=path, status_code=str(status_code)
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)

        response.headers["X-Request-Id"] = request_id
        return response
