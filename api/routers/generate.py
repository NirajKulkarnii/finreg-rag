"""
Generation Endpoints
---------------------
POST /v1/generate        — full RAG pipeline, returns structured JSON with timings
POST /v1/generate/stream — SSE streaming: token chunks then a meta frame
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.generator import RAGGenerator
from generation.models import GenerationRequest, GenerationResponse
from ..dependencies import StatsBuffer, get_generator, get_stats_buffer
from ..metrics import (
    RAG_CLASSIFY_RETRIEVE_MS,
    RAG_GENERATE_MS,
    RAG_REQUESTS_TOTAL,
    RAG_RETRIEVAL_HITS,
    RAG_TOP_SCORE,
    RAG_TOTAL_MS,
    RAG_WEB_SEARCH_MS,
    RAG_WEB_SEARCH_TOTAL,
)

router = APIRouter(prefix="/v1", tags=["generation"])
logger = structlog.get_logger(__name__)


# ── Response models ───────────────────────────────────────────────────────────

class PipelineTimings(BaseModel):
    classify_retrieve_ms: float = 0.0
    web_search_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    retrieval_hits: int = 0
    top_score: float = 0.0


class TimedGenerationResponse(GenerationResponse):
    request_id: str
    timings: PipelineTimings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _record_metrics(trace: dict[str, Any], resp: GenerationResponse) -> None:
    """Update Prometheus metrics from a completed pipeline trace."""
    RAG_REQUESTS_TOTAL.labels(
        intent=resp.intent.value,
        confidence=resp.confidence.value,
    ).inc()

    if "classify_retrieve_ms" in trace:
        RAG_CLASSIFY_RETRIEVE_MS.observe(trace["classify_retrieve_ms"])
    if trace.get("web_search_ms", 0) > 0:
        RAG_WEB_SEARCH_MS.observe(trace["web_search_ms"])
    if "generate_ms" in trace:
        RAG_GENERATE_MS.observe(trace["generate_ms"])
    if "total_ms" in trace:
        RAG_TOTAL_MS.observe(trace["total_ms"])
    if "retrieval_hits" in trace:
        RAG_RETRIEVAL_HITS.observe(trace["retrieval_hits"])
    if "top_score" in trace:
        RAG_TOP_SCORE.observe(trace["top_score"])
    if resp.used_web_search:
        RAG_WEB_SEARCH_TOTAL.inc()


async def _push_stats(
    stats_buffer: StatsBuffer,
    trace: dict[str, Any],
    resp: GenerationResponse,
) -> None:
    await stats_buffer.push({
        "intent":               resp.intent.value,
        "confidence":           resp.confidence.value,
        "used_web_search":      resp.used_web_search,
        "retrieval_used":       resp.retrieval_used,
        **{k: trace.get(k, 0) for k in (
            "classify_retrieve_ms", "web_search_ms",
            "generate_ms", "total_ms", "retrieval_hits", "top_score",
        )},
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=TimedGenerationResponse,
    summary="Full RAG generation (non-streaming)",
)
async def generate(
    request: GenerationRequest,
    generator: Annotated[RAGGenerator, Depends(get_generator)],
    stats_buffer: Annotated[StatsBuffer, Depends(get_stats_buffer)],
) -> TimedGenerationResponse:
    request_id = str(uuid.uuid4())
    trace: dict[str, Any] = {}

    logger.info("generate_request", request_id=request_id, query=request.query[:120])

    resp = await generator.generate(request, _trace=trace)

    _record_metrics(trace, resp)
    await _push_stats(stats_buffer, trace, resp)

    logger.info(
        "generate_complete",
        request_id=request_id,
        intent=resp.intent.value,
        total_ms=round(trace.get("total_ms", 0), 1),
    )

    return TimedGenerationResponse(
        **resp.model_dump(),
        request_id=request_id,
        timings=PipelineTimings(**{k: trace.get(k, 0) for k in PipelineTimings.model_fields}),
    )


@router.post(
    "/generate/stream",
    summary="Streaming RAG generation (SSE)",
    response_class=StreamingResponse,
)
async def generate_stream(
    request: GenerationRequest,
    generator: Annotated[RAGGenerator, Depends(get_generator)],
    stats_buffer: Annotated[StatsBuffer, Depends(get_stats_buffer)],
) -> StreamingResponse:
    """
    Server-Sent Events stream.

    Frame format:
      data: {"type": "token",  "content": "<token text>"}
      data: {"type": "meta",   "request_id": "...", "intent": "...", ...}
      data: [DONE]
    """
    request_id = str(uuid.uuid4())
    logger.info("stream_request", request_id=request_id, query=request.query[:120])

    async def event_generator():
        accumulated = []
        meta_payload: dict[str, Any] | None = None

        async for chunk in generator.generate_stream(request):
            if chunk.startswith("\n__META__:"):
                meta_payload = json.loads(chunk[len("\n__META__:"):])
            else:
                accumulated.append(chunk)
                frame = json.dumps({"type": "token", "content": chunk})
                yield f"data: {frame}\n\n"

        # Build meta frame
        if meta_payload:
            trace: dict[str, Any] = {
                "intent":          meta_payload.get("intent", ""),
                "used_web_search": meta_payload.get("used_web_search", False),
                "retrieval_used":  meta_payload.get("retrieval_used", False),
            }
            meta_frame = json.dumps({
                "type":             "meta",
                "request_id":       request_id,
                "intent":           meta_payload.get("intent"),
                "used_web_search":  meta_payload.get("used_web_search"),
                "web_search_query": meta_payload.get("web_search_query"),
                "retrieval_used":   meta_payload.get("retrieval_used"),
                "sources":          meta_payload.get("sources", []),
            })
            yield f"data: {meta_frame}\n\n"

            # Push a minimal stats record (no per-phase timing in stream mode)
            await stats_buffer.push({
                "intent":          meta_payload.get("intent", "unknown"),
                "confidence":      "unknown",
                "used_web_search": meta_payload.get("used_web_search", False),
                "retrieval_used":  meta_payload.get("retrieval_used", False),
                "total_ms":        0,
                "classify_retrieve_ms": 0,
                "web_search_ms":   0,
                "generate_ms":     0,
                "retrieval_hits":  len(meta_payload.get("sources", [])),
                "top_score":       0,
            })

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )
