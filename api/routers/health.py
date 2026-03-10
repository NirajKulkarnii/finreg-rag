"""
Health & Observability Endpoints
---------------------------------
GET /health   — liveness probe (always 200 if the process is running)
GET /ready    — readiness probe (checks vLLM + ChromaDB connectivity)
GET /metrics  — Prometheus text exposition format
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request) -> dict:
    """
    Checks that downstream dependencies (vLLM and ChromaDB) are reachable.
    Returns 200 when ready, 503 when not.
    """
    from fastapi import HTTPException
    from config.settings import settings

    failures: list[str] = []

    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.vllm_base_url}/models")
            r.raise_for_status()
    except Exception as exc:
        failures.append(f"vllm: {exc}")

    # Check ChromaDB
    try:
        if settings.chroma_persistent_path:
            import chromadb
            chromadb.PersistentClient(path=settings.chroma_persistent_path)
        else:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat"
                )
                r.raise_for_status()
    except Exception as exc:
        failures.append(f"chromadb: {exc}")

    if failures:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "failures": failures})

    return {"status": "ready"}


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
