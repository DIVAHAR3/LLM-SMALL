# API Security Plan (Phase 19)

**Status: local-only (`127.0.0.1`). No public exposure has happened or is planned without this document being explicitly reviewed and approved first — see CLAUDE.md hard rule 7.**

This document covers every topic Phase 19 requires: what it is, why it matters, and its current status in this project — implemented now, or planned for a later phase.

## Implemented now

### Authentication (API keys)

A single shared secret, generated with `secrets.token_urlsafe(32)` (cryptographically random, stdlib, no dependency), required in the `X-API-Key` header on `POST /generate`. Checked in `api/security.py:require_api_key`.

Why a shared API key rather than OAuth2/JWT/user accounts: this project has exactly one user (whoever runs it) and no multi-tenant concept. A shared secret is the correct amount of complexity for that — anything more would be solving a problem this project doesn't have.

**Fails closed**: if the server has no `API_KEY` configured at all, every request to `/generate` is rejected (503), not silently allowed through. A misconfiguration should never be equivalent to "no auth required."

`GET /health` is deliberately left unauthenticated — health checks need to be reachable by monitoring tooling without a key, and the endpoint leaks nothing sensitive (checkpoint path, param count).

### Secrets management

`API_KEY` lives in `.env` (gitignored, never committed — verified with `git status --ignored`). `.env.example` documents the expected variable names with placeholder values and instructions for generating a real key. Loaded automatically at server startup via `python-dotenv`.

### Rate limiting

In-memory fixed-window limiter (`api/security.py:RateLimiter`), default 20 requests per 60 seconds per client IP, configurable via `RATE_LIMIT_REQUESTS`/`RATE_LIMIT_WINDOW_SECONDS`. Exceeding it returns `429 Too Many Requests`.

**Known limitation, stated plainly**: this is in-process memory, so it only works correctly for a single-worker server — which is exactly what this project runs (`uvicorn` with default settings, one process). A multi-worker or multi-instance deployment would need a shared store (Redis is the standard choice) instead, since each worker would otherwise track its own independent counters. Not needed until there's an actual multi-worker deployment; noted here so it isn't forgotten.

### Request-size limits

`api/security.py:RequestSizeLimitMiddleware` rejects any request whose `Content-Length` exceeds 10,000 bytes with `413 Request Entity Too Large`, **before** the body is read into memory. This matters because Pydantic's field constraints (e.g. `prompt` capped at 2000 characters, from Phase 18) only apply *after* the JSON has already been parsed — a large-enough payload could still cause memory pressure during parsing itself, which is a real concern given how little free RAM this machine typically has (Phase 1: often under 1GB free).

`POST /analyze/image` (added outside the numbered phase sequence — see `docs/IMAGE_ANALYSIS.md`) needs a much higher limit than a short text prompt, since it accepts real image files. Rather than loosening the default for every route, the middleware takes a `path_overrides` dict so this one route gets its own 5 MB cap while everything else keeps the original 10 KB. The endpoint also re-checks the actual decoded size after reading the upload, as defense in depth against a missing/absent `Content-Length` header (e.g. chunked transfer encoding), which the middleware alone can't catch.

### Input validation

Already built in Phase 18: Pydantic field constraints on every `/generate` parameter (non-empty/length-bounded prompt, bounded `max_new_tokens`, positive `temperature`, valid `top_k`/`top_p` ranges). Verified this phase to still be correctly enforced (see `tests/test_api.py`).

### CORS

`CORSMiddleware`, configured from the `ALLOWED_ORIGINS` env var (comma-separated list, parsed by `api/security.py:parse_allowed_origins`). **Default is empty — no browser origin is allowed until explicitly configured.** This will be set to the Vite dev server's origin (`http://localhost:5173`) when Phase 20 builds the frontend, and nothing else. CORS only restricts browser-enforced requests; it has no bearing on `curl`/server-to-server calls, which was verified live in this phase's testing.

### Logging

Phase 18 already logs every request (prompt length, generation params) to console and `logs/api.log`. This phase adds logging specifically for security-relevant events — failed API key checks and rate-limit rejections — at `WARNING` level, so abuse attempts are visible in the logs rather than indistinguishable from normal traffic.

## Planned, not yet implemented (deployment-dependent)

These genuinely cannot be meaningfully implemented against `127.0.0.1` with no real domain — they're deployment-time concerns. Phase 31 (`docs/DEPLOYMENT_PLAN.md`) has since written the full operational plan (process management, restarts, logs, health checks, resource limits) around the HTTPS/reverse-proxy path sketched below; Phase 32 (Docker) is where an actual deployable target gets built.

### HTTPS

Encrypts traffic between client and server, preventing eavesdropping or tampering in transit. Requires a certificate tied to a real domain (e.g. via Let's Encrypt) — meaningless to set up against `localhost`. **Plan**: terminate TLS at the reverse proxy (below), not inside the FastAPI/Uvicorn process itself — this is the standard pattern and keeps certificate management out of the application code entirely.

### Reverse proxy

A dedicated process (nginx or Caddy, most likely) sitting in front of Uvicorn in any real deployment. **Plan**: it will (1) terminate HTTPS, (2) be the only process actually reachable from the internet — Uvicorn stays bound to `127.0.0.1` even in production, reachable only via the proxy on the same machine, (3) can enforce its own coarse-grained size/rate limits as a second layer in front of this project's own (belt-and-suspenders, not redundant — the proxy layer protects against things this project's own middleware can't, like slow-loris style connection abuse).

## Abuse prevention — how the above compose

No single control here is sufficient alone; together they form layered defense: auth prevents anonymous use entirely → rate limiting caps what an authenticated client can do → the request-size limit caps the cost of any single request → input validation rejects malformed requests before they reach the model → logging makes any attempted abuse visible after the fact. The reverse proxy (when it exists) adds a layer in front of all of this.

## Explicit stop condition

Per CLAUDE.md hard rule 7 and this phase's own stop condition: **this document must be reviewed before any public exposure.** As of this phase, the server remains bound to `127.0.0.1` only — nothing here changes that; it prepares the ground for when a deployment phase (31/32) might.
