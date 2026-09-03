import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("api.security")


def parse_allowed_origins(raw):
    """'http://a, http://b' -> ['http://a', 'http://b']; '' or None -> []."""
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class RateLimiter:
    """In-memory fixed-window rate limiter: at most `limit` requests per
    `window_seconds` per client key. In-memory state only works correctly
    for a single-process server (true here: local uvicorn, one worker) --
    a multi-worker/production deployment would need a shared store like
    Redis instead, noted in docs/SECURITY.md."""

    def __init__(self, limit=20, window_seconds=60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)

    def is_allowed(self, client_id):
        now = time.time()
        window = self._requests[client_id]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose Content-Length exceeds max_bytes BEFORE the
    body is read into memory. Pydantic's field constraints (Phase 18) only
    apply after the JSON body has already been parsed -- too late to
    prevent memory pressure from a maliciously huge payload.

    path_overrides lets specific routes use a different limit than the
    default -- e.g. an image upload endpoint legitimately needs a much
    higher cap than a short JSON prompt, without loosening the limit for
    every other route."""

    def __init__(self, app, max_bytes=10_000, path_overrides=None):
        super().__init__(app)
        self.max_bytes = max_bytes
        self.path_overrides = path_overrides or {}

    def _limit_for(self, path):
        for prefix, limit in self.path_overrides.items():
            if path.startswith(prefix):
                return limit
        return self.max_bytes

    async def dispatch(self, request: Request, call_next):
        limit = self._limit_for(request.url.path)
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > limit:
            logger.warning(f"rejected oversized request: {content_length} bytes from {request.client}")
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        return await call_next(request)


async def require_api_key(request: Request, x_api_key: str = Header(default=None)):
    """Fails closed: if the server itself has no API_KEY configured, every
    request is rejected rather than silently allowed through."""
    configured_key = getattr(request.app.state, "api_key", None)
    if not configured_key:
        raise HTTPException(status_code=503, detail="Server API key is not configured.")
    if x_api_key != configured_key:
        logger.warning(f"rejected request with invalid API key from {request.client}")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


async def enforce_rate_limit(request: Request):
    limiter = request.app.state.rate_limiter
    client_id = request.client.host if request.client else "unknown"
    if not limiter.is_allowed(client_id):
        logger.warning(f"rate limit exceeded for {client_id}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
