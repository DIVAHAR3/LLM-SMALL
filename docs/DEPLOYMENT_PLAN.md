# Phase 31 — Deployment Plan

**Status: local-only (`127.0.0.1`), exactly as `docs/SECURITY.md` already states. This document is a plan, not an action — nothing here has been deployed, and nothing will be until this plan and `docs/SECURITY.md` are both explicitly reviewed and approved (CLAUDE.md hard rule 7).**

## Why "it runs locally" isn't "it's deployed"

Running `uvicorn api.main:app` in a terminal works fine for development: if it crashes, you see the traceback and restart it by hand; if the terminal closes, so does the server; logs scroll past and vanish once the window fills. None of that is acceptable for anything meant to stay up and be reachable by someone else. Deployment means answering four questions development doesn't have to: what keeps the process running when *you're* not watching it (process management), what happens when it crashes or gets updated (restarts), how do you find out what happened after the fact (logs), and how does anything external know whether it's healthy (health checks) — plus, since "reachable by someone else" implies the public internet, how the connection itself gets secured (HTTPS) and fronted (reverse proxy). This document plans all of it; Phase 32 (Docker) builds the actual process-management mechanism.

## The full request path

```
Internet
   |
   v
DNS  (a real domain name -- not needed for this plan itself, but the
   |   starting point for anyone actually reaching this service)
   v
HTTPS/TLS termination  (at the reverse proxy -- see below)
   |
   v
Reverse proxy (nginx or Caddy)
   |  - the ONLY process actually reachable from the internet
   |  - terminates TLS (certificate management stays out of this
   |    project's application code entirely)
   |  - can enforce its own coarse-grained size/connection limits as a
   |    second layer in front of this project's own (protects against
   |    things app-level middleware can't, e.g. slow-loris connection abuse)
   v
Uvicorn (FastAPI app, api/main.py)
   |  - stays bound to 127.0.0.1 EVEN IN PRODUCTION -- reachable only
   |    via the reverse proxy on the same machine, never directly
   |  - X-API-Key auth, rate limiting, request-size limits, input
   |    validation (all Phase 18/19, docs/SECURITY.md)
   v
The loaded GPT (or OCR) model, in process memory
   |
   v
Response, back out the same path
```

This is the same path `docs/SECURITY.md` sketched (HTTPS + reverse proxy); this document is where it gets made concrete and operational.

## Process management

**Plan: containerize (Phase 32), and let the container runtime's restart policy be the process manager** — `docker run --restart unless-stopped` or the equivalent `restart:` key in `docker-compose.yml`. This is the modern, portable default and matches the very next roadmap phase, so this plan deliberately doesn't duplicate that mechanism ahead of time.

**Documented alternative, if not containerized**: a `systemd` unit file (Linux) with `Restart=on-failure` — the traditional way to keep a process running on a bare Linux server, included here so the plan doesn't silently assume Docker is the only path, per this project's own "present the choice, don't silently pick one" convention. Not written out in full here since Phase 32 is expected to make it moot.

Either way, the process must **never** be started by manually running `uvicorn` in a terminal session in production — that's development-only.

## Restarts

Two different situations need different handling:

- **Crash recovery**: the restart policy above (`unless-stopped` / `Restart=on-failure`) handles this automatically — the process manager notices the process died and starts a new one.
- **Deploy-time restarts** (shipping a code change): Uvicorn supports graceful shutdown natively (SIGTERM lets in-flight requests finish before the process exits) — the reverse proxy stops routing new requests to it during that window. At this project's actual scale (a single instance, no load-balanced replicas), a deploy honestly means a brief window of unavailability while the new process starts — not zero-downtime blue-green deployment. Claiming otherwise for a single-box educational project would be exactly the kind of overclaiming this project has avoided throughout (see e.g. `docs/INSTRUCTION_TUNING.md`'s honesty about what fine-tuning did and didn't prove).

## Logs

Already logging every request to `logs/api.log` (Phase 18) plus console, and security-relevant events at `WARNING` (Phase 19). **A real gap this phase found and fixed**: the file handler was a plain `logging.FileHandler` — no rotation, so a long-running deployed server would grow that file forever. At the time of writing this document, this project's own local `logs/api.log` (from routine dev/test use alone, not even a real deployment) had already reached ~580KB — concrete evidence of exactly the problem being described, not a hypothetical one. Fixed in `api/main.py`: switched to `logging.handlers.RotatingFileHandler` (stdlib, no new dependency), default 5MB per file with 5 backups kept (30MB total cap), both configurable via `LOG_MAX_BYTES`/`LOG_BACKUP_COUNT` env vars. Verified via `tests/test_api.py`'s `TestLogRotation`.

**Not planned for this project's scale**: centralized/shipped logging (e.g. to a log aggregation service). That solves a multi-instance problem this single-server project doesn't have; local rotated files are the right amount of complexity here, matching the reasoning `docs/SECURITY.md` already used for choosing a shared API key over OAuth2.

## Health checks

`GET /health` (Phase 18, deliberately unauthenticated) already returns `{"status": "ok", "checkpoint": ..., "params": ...}`. In a deployment, three different things would consume it:

1. **The reverse proxy** — before routing a request to Uvicorn, it can check `/health` to confirm the backend is actually up (avoiding routing traffic into a process still starting up or already dead).
2. **A container orchestrator's liveness/readiness probes** — Docker's `HEALTHCHECK` directive (or Kubernetes-style probes, if ever scaled that far, which is not currently planned) would hit this same endpoint on an interval; Phase 32 is where this gets wired into the actual `Dockerfile`/`docker-compose.yml`.
3. **External uptime monitoring** — a simple periodic ping from outside the deployment, so an outage is noticed even if internal health checks somehow all pass.

No changes needed to `/health` itself — its existing response shape already serves all three.

## Resource limits

**Plan: enforce at the container/OS level, not in application code.** Docker's `--memory`/`--cpus` flags (or the equivalent `docker-compose.yml` `deploy.resources.limits`) are the standard, correct layer for this — again, Phase 32's actual mechanism. This matters more than usual for this project specifically: Phase 1's hardware inspection found this machine often runs with under 1GB free RAM, and that constraint has shown up concretely multiple times since (Phase 30's monitoring work, and an aborted local-vision-model attempt that found only ~980MB free even after closing what could be closed). A container memory limit is what stops one runaway or malicious request from taking down the whole host, not just this one service.

**Already in place at the application level** (the complementary layer, not a substitute): `max_new_tokens` bounded to ≤500 (Phase 18), request bodies capped at 10KB for `/generate` and 5MB for `/analyze/image` (Phase 19, extended for the image-analysis feature), rate limiting at 20 requests/60s by default. These bound the cost of any single request; container limits bound the cost of the process as a whole.

## No public exposure without approval

Per CLAUDE.md hard rule 7 and `docs/SECURITY.md`'s own stop condition, reiterated here as this phase's actual gate: **this plan, together with `docs/SECURITY.md`, must be explicitly reviewed and approved before any public exposure happens.** As of this phase, the server remains bound to `127.0.0.1` only. Writing this plan does not itself authorize deployment — Phase 32 (Docker) is the next concrete step, and even that produces a deployable artifact, not a deployed one.
