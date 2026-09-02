"""Local-only FastAPI inference server. Run from the project root:

    .venv\\Scripts\\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Binds to 127.0.0.1 only, deliberately -- public exposure requires the
security review in Phase 19 first (see CLAUDE.md hard rule 7).
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from inference.generate import generate_text
from tokenizer.char_tokenizer import CharTokenizer
from training.checkpoint import load_for_inference

ROOT = Path(__file__).resolve().parent.parent
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
    logger.info(f"Model loaded: {app.state.param_count:,} params from {checkpoint_path}")

    yield

    logger.info("Server shutting down")


app = FastAPI(title="GPT-from-Scratch Inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        checkpoint=app.state.checkpoint_path,
        params=app.state.param_count,
    )


@app.post("/generate", response_model=GenerateResponse)
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
