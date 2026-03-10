"""
Stats Endpoint
--------------
GET /v1/stats — aggregated pipeline performance stats from the rolling StatsBuffer
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ..dependencies import StatsBuffer, get_stats_buffer

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get(
    "/stats",
    summary="Aggregated pipeline stats (rolling window of last 1000 requests)",
)
async def stats(
    buffer: Annotated[StatsBuffer, Depends(get_stats_buffer)],
) -> dict[str, Any]:
    """
    Returns aggregated statistics over the rolling request window:

    - count                     : total requests in window
    - total_latency_ms          : p50 / p95 / p99 / mean end-to-end latency
    - classify_retrieve_ms      : p50 / p95 / p99 / mean retrieval phase latency
    - web_search_ms             : p50 / p95 / p99 / mean web search latency (when used)
    - generate_ms               : p50 / p95 / p99 / mean LLM generation latency
    - avg_retrieval_hits        : mean number of retrieval results per request
    - avg_top_score             : mean top cross-encoder score per request
    - web_search_rate           : fraction of requests that triggered web search
    - retrieval_rate            : fraction of requests that used retrieval context
    - intent_distribution       : {intent: count}
    - confidence_distribution   : {confidence: count}
    """
    return await buffer.aggregate()
