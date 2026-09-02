# GPT-from-Scratch — Frontend

A minimal React + Vite chat UI for the local FastAPI inference server (`../api/main.py`).

## Setup

```
npm install
cp .env.example .env   # then fill in VITE_API_KEY (must match ../.env's API_KEY)
npm run dev
```

Requires the backend running first (`.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000` from the project root), and the backend's `ALLOWED_ORIGINS` to include `http://localhost:5173` (Vite's default dev port).

## Node version note

`npm install` needs Node **22.12+** (or 20.19+). On Node 22.11.x, npm silently skips the platform-specific native bindings that Vite 8's bundler (rolldown) and its linter (oxlint) need, producing confusing `Cannot find module './<binding>.win32-x64-msvc.node'` errors at build/lint time — even though the bindings work fine once present. `package.json` pins `@rolldown/binding-win32-x64-msvc` and `@oxlint/binding-win32-x64-msvc` as explicit devDependencies to force them to always install regardless of Node's exact patch version, so `npm install` here should work reproducibly even on 22.11.x. If you hit the same error on a different OS/architecture, the equivalent `@<package>/binding-<platform>` package needs the same explicit pin.

## Security note

`VITE_API_KEY` gets bundled into the built client-side JS — anyone loading the page can read it (see the comment in `src/api.js`). Acceptable only because this project is local-only and single-user; see `../docs/SECURITY.md`.
