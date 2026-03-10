"""
finreg-rag FastAPI Application
-------------------------------
Production-ready RAG API with:
  - Single shared RAGGenerator loaded at startup via lifespan
  - Structured JSON logging (structlog)
  - Prometheus metrics
  - CORS + GZip compression
  - Per-request logging with X-Request-Id header
  - Interactive docs at /docs and /redoc

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1

workers=1 is required because the RAGGenerator holds non-picklable state
(ML models, async client). For true horizontal scaling, deploy multiple
containers behind a load balancer.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Make project root importable when api/ is inside it
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Load .env BEFORE config.settings is imported — settings reads os.getenv() at class definition time
load_dotenv(_project_root / ".env", override=False)

from config.settings import settings
from ingestion import Embedder, VectorStore
from retrieval import HybridRetriever
from generation.generator import RAGGenerator

from .dependencies import StatsBuffer
from .metrics import APP_INFO
from .middleware import RequestLoggingMiddleware, configure_logging
from .routers import evaluate, generate, health, stats

APP_VERSION = "1.0.0"

logger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG pipeline once on startup; tear it down on shutdown."""
    configure_logging(settings.log_level)
    logger.info("startup", version=APP_VERSION, model=settings.vllm_model)

    embedder = Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    store = VectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
        persistent_path=settings.chroma_persistent_path,
    )
    retriever = HybridRetriever(
        vector_store=store,
        embedder=embedder,
        reranker_model=settings.reranker_model,
        device=settings.embedding_device,
    )
    generator = RAGGenerator(
        retriever=retriever,
        vllm_base_url=settings.vllm_base_url,
        model=settings.vllm_model,
    )

    # ── Warm-start: load models now so the first request isn't slow ───────
    logger.info("warmup_start", embedding_model=settings.embedding_model, reranker_model=settings.reranker_model)
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, lambda: embedder.model),    # loads BGE-M3
        loop.run_in_executor(None, lambda: retriever._reranker.model),  # loads cross-encoder
    )
    logger.info("warmup_complete")

    app.state.generator    = generator
    app.state.stats_buffer = StatsBuffer(maxlen=1000)

    APP_INFO.labels(version=APP_VERSION, model=settings.vllm_model).set(1)

    logger.info("pipeline_ready", embedding_model=settings.embedding_model)
    yield

    logger.info("shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="finreg-rag API",
        description=(
            "Production RAG pipeline for financial regulation queries.\n\n"
            "**Pipeline**: BGE-M3 embedder → ChromaDB (dense) + BM25 (lexical) "
            "→ cross-encoder reranker → vLLM (Qwen2.5).\n\n"
            "**Endpoints**:\n"
            "- `POST /v1/generate` — structured answer with per-phase timings\n"
            "- `POST /v1/generate/stream` — SSE streaming\n"
            "- `POST /v1/evaluate` — ROUGE scoring\n"
            "- `POST /v1/evaluate/batch` — batch ROUGE scoring (up to 20 items)\n"
            "- `GET  /v1/stats` — rolling latency/quality stats\n"
            "- `GET  /health` — liveness probe\n"
            "- `GET  /ready` — readiness probe\n"
            "- `GET  /metrics` — Prometheus text format\n"
        ),
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters: outer → inner) ──────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(evaluate.router)
    app.include_router(stats.router)

    return app


app = create_app()
