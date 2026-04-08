import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.batching import DynamicBatcher
from src.caching import build_cache, make_cache_key
from src.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Global inference state
# ─────────────────────────────────────────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_device: str = "cpu"


def _load_model() -> None:
    """Load tokenizer and model into globals. Raises on any failure."""
    global _tokenizer, _model, _device

    model_name = settings.model_name
    logger.info("Loading model: %s", model_name)

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        raise

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Inference device: %s", _device)

    try:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # float32 for CPU / MPS safety
        )
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

    # Move to device after loading (avoids meta-tensor issues on some versions)
    try:
        _model = _model.to(_device)
    except Exception as e:
        logger.warning("Could not move model to %s, staying on cpu: %s", _device, e)
        _device = "cpu"

    _model.eval()
    logger.info("Model loaded successfully on %s", _device)


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference function
# ─────────────────────────────────────────────────────────────────────────────
async def _batch_inference(
    prompts: List[str],
    max_tokens_list: List[int],
    temperature: float,
) -> List[str]:
    """Run a batch of prompts through the loaded model."""
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded — check startup logs")

    max_new = min(max(max_tokens_list), settings.max_output_tokens)
    loop = asyncio.get_event_loop()

    def _sync_generate() -> List[str]:
        inputs = _tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
        ).to(_device)

        gen_kwargs: dict = dict(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=_tokenizer.eos_token_id,
            do_sample=(temperature > 0),
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = _model.generate(**gen_kwargs)

        results = []
        input_len = inputs["input_ids"].shape[1]
        for out in output_ids:
            new_tokens = out[input_len:]
            results.append(_tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results

    return await loop.run_in_executor(None, _sync_generate)


# ─────────────────────────────────────────────────────────────────────────────
# Cache + Batcher (module-level singletons)
# ─────────────────────────────────────────────────────────────────────────────
cache = build_cache(
    backend=settings.cache_backend,
    redis_url=settings.redis_url,
    ttl_seconds=settings.cache_ttl_seconds,
    max_entries=settings.cache_max_entries,
)
batcher = DynamicBatcher(
    inference_fn=_batch_inference,
    max_batch_size=settings.max_batch_size,
    batch_timeout_ms=settings.batch_timeout_ms,
)


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting up — loading model...")
    try:
        _load_model()
    except Exception as e:
        logger.error("FATAL: model failed to load: %s", e)
        raise   # Let uvicorn surface the error and exit cleanly

    await batcher.start()
    logger.info("Server ready")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    await batcher.stop()
    logger.info("Server shutdown complete")


app = FastAPI(
    title="LLM Inference Server — Milestone 5",
    description=(
        "Production-ready LLM serving with dynamic request batching "
        "and privacy-safe response caching."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    bypass_cache: bool = Field(default=False)


class GenerateResponse(BaseModel):
    generated_text: str
    cached: bool
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_loaded: bool


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        device=_device,
        model_loaded=(_model is not None),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest):
    """
    Generate text for a given prompt.
    - temperature=0 responses are cache-eligible.
    - All requests go through the dynamic batcher.
    - Cache keys are SHA-256 hashes; no plaintext prompts stored.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — check server startup logs",
        )

    t_start = time.perf_counter()

    # ── 1. Cache lookup ───────────────────────────────────────────────────────
    use_cache = (req.temperature == 0.0) and (not req.bypass_cache)
    cache_key = make_cache_key(
        prompt=req.prompt,
        model=settings.model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    if use_cache:
        cached_value = await cache.get(cache_key)
        if cached_value is not None:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.debug("Cache HIT (%.1f ms)", latency_ms)
            return GenerateResponse(
                generated_text=cached_value,
                cached=True,
                latency_ms=round(latency_ms, 2),
                model=settings.model_name,
            )

    # ── 2. Dynamic batcher ────────────────────────────────────────────────────
    try:
        result = await asyncio.wait_for(
            batcher.submit(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Inference timed out after {settings.request_timeout_seconds}s",
        )

    # ── 3. Store in cache ─────────────────────────────────────────────────────
    if use_cache:
        await cache.set(cache_key, result)

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.debug("Inference complete (%.1f ms)", latency_ms)

    return GenerateResponse(
        generated_text=result,
        cached=False,
        latency_ms=round(latency_ms, 2),
        model=settings.model_name,
    )


@app.get("/metrics", tags=["ops"])
async def metrics():
    return {
        "batching": batcher.stats(),
        "caching": cache.stats(),
    }


@app.post("/cache/clear", tags=["ops"])
async def clear_cache():
    await cache.clear()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
    )