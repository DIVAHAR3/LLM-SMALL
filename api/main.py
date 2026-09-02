"""Local-only FastAPI inference server. Run from the project root:

    .venv\\Scripts\\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Binds to 127.0.0.1 only, deliberately -- public exposure requires the
security review in Phase 19 first (see CLAUDE.md hard rule 7, and
docs/SECURITY.md for the full written plan).
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # must happen before any os.environ reads below

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse  # noqa: E402
from api.security import (  # noqa: E402
    RateLimiter,
    RequestSizeLimitMiddleware,
    enforce_rate_limit,
    parse_allowed_origins,
    require_api_key,
)
from inference.generate import generate_text  # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log")],
)
logger = logging.getLogger("api")

DEFAULT_CHECKPOINT = str(ROOT / "checkpoints" / "phase13_run.pt")
DEFAULT_TOKENIZER = str(ROOT / "tokenizer" / "vocab.json")

# CORS origins are read once at module/middleware-registration time (Starlette
# builds its middleware stack when the app object is constructed, not per
# request), so this can't be deferred into lifespan the way model config is.
_allowed_origins = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", ""))


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = os.environ.get("DEVICE", "cpu")
    if device != "cpu":
        # This project deliberately runs CPU-only (see Phase 2); GPU support
        # is explicitly Phase 23's job, not silently ignored here.
        raise RuntimeError(f"DEVICE={device!r} is not supported yet -- only 'cpu' is implemented (see Phase 23).")

    checkpoint_path = os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
    tokenizer_path = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER)

    logger.info(f"Loading model from {checkpoint_path} (device={device})")
    model, _ = load_for_inference(checkpoint_path)
    tokenizer = CharTokenizer.load(tokenizer_path)

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.checkpoint_path = checkpoint_path
    app.state.param_count = model.num_parameters()
    app.state.api_key = os.environ.get("API_KEY")
    app.state.rate_limiter = RateLimiter(
        limit=int(os.environ.get("RATE_LIMIT_REQUESTS", "20")),
        window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")),
    )
    logger.info(f"Model loaded: {app.state.param_count:,} params from {checkpoint_path}")
    if not app.state.api_key:
        logger.warning("API_KEY is not set -- /generate will reject all requests until it is configured.")

    yield

    logger.info("Server shutting down")


app = FastAPI(title="GPT-from-Scratch Inference API", lifespan=lifespan)

app.add_middleware(RequestSizeLimitMiddleware, max_bytes=10_000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    # Deliberately unauthenticated: health checks need to be reachable by
    # monitoring tools without a key, and leak no sensitive information.
    return HealthResponse(
        status="ok",
        checkpoint=app.state.checkpoint_path,
        params=app.state.param_count,
    )


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def generate(request: GenerateRequest):
    logger.info(
        f"generate: prompt_len={len(request.prompt)} max_new_tokens={request.max_new_tokens} "
        f"temperature={request.temperature} top_k={request.top_k} top_p={request.top_p} greedy={request.greedy}"
    )
    try:
        text = generate_text(
            app.state.model,
            app.state.tokenizer,
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
        )
    except Exception:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail="Generation failed. See server logs for details.")
    return GenerateResponse(text=text)
