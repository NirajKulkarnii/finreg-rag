"""
FastAPI Dependencies
--------------------
Provides:
  - get_generator()    — yields the shared RAGGenerator from app.state
  - get_stats_buffer() — yields the shared StatsBuffer from app.state
  - StatsBuffer        — asyncio-safe rolling deque with percentile helpers
"""

from __future__ import annotations

import asyncio
import statistics
from collections import deque
from typing import Any

from fastapi import Request

from generation.generator import RAGGenerator


# ── StatsBuffer ───────────────────────────────────────────────────────────────

class StatsBuffer:
    """
    Thread-safe rolling buffer of the last N RAG request records.

    Each record is a dict with keys:
      intent, confidence, retrieval_hits, top_score,
      classify_retrieve_ms, web_search_ms, generate_ms, total_ms,
      used_web_search, retrieval_used
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def push(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._buf.append(record)

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._buf)

    @staticmethod
    def _percentiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.95, 0.99)) -> dict[str, float]:
        if not values:
            return {f"p{int(q*100)}": 0.0 for q in qs}
        s = sorted(values)
        n = len(s)
        result = {}
        for q in qs:
            idx = min(int(q * n), n - 1)
            result[f"p{int(q*100)}"] = round(s[idx], 1)
        return result

    async def aggregate(self) -> dict[str, Any]:
        """Return aggregated stats over the rolling window."""
        records = await self.snapshot()
        if not records:
            return {"count": 0}

        n = len(records)

        total_ms_vals      = [r["total_ms"]             for r in records if "total_ms"             in r]
        cr_ms_vals         = [r["classify_retrieve_ms"] for r in records if "classify_retrieve_ms" in r]
        ws_ms_vals         = [r["web_search_ms"]        for r in records if r.get("web_search_ms", 0) > 0]
        gen_ms_vals        = [r["generate_ms"]          for r in records if "generate_ms"          in r]
        retrieval_hits     = [r["retrieval_hits"]       for r in records if "retrieval_hits"        in r]
        top_scores         = [r["top_score"]            for r in records if "top_score"             in r]

        # Intent distribution
        from collections import Counter
        intent_dist      = dict(Counter(r.get("intent") for r in records))
        confidence_dist  = dict(Counter(r.get("confidence") for r in records))

        web_search_count = sum(1 for r in records if r.get("used_web_search"))
        retrieval_count  = sum(1 for r in records if r.get("retrieval_used"))

        return {
            "count": n,
            "total_latency_ms":           {**self._percentiles(total_ms_vals), "mean": round(statistics.mean(total_ms_vals), 1) if total_ms_vals else 0.0},
            "classify_retrieve_ms":       {**self._percentiles(cr_ms_vals),    "mean": round(statistics.mean(cr_ms_vals),    1) if cr_ms_vals    else 0.0},
            "web_search_ms":              {**self._percentiles(ws_ms_vals),    "mean": round(statistics.mean(ws_ms_vals),    1) if ws_ms_vals    else 0.0},
            "generate_ms":                {**self._percentiles(gen_ms_vals),   "mean": round(statistics.mean(gen_ms_vals),   1) if gen_ms_vals   else 0.0},
            "avg_retrieval_hits":         round(statistics.mean(retrieval_hits), 2) if retrieval_hits else 0.0,
            "avg_top_score":              round(statistics.mean(top_scores),    4) if top_scores     else 0.0,
            "web_search_rate":            round(web_search_count / n, 4),
            "retrieval_rate":             round(retrieval_count  / n, 4),
            "intent_distribution":        intent_dist,
            "confidence_distribution":    confidence_dist,
        }


# ── FastAPI dependency functions ──────────────────────────────────────────────

def get_generator(request: Request) -> RAGGenerator:
    return request.app.state.generator


def get_stats_buffer(request: Request) -> StatsBuffer:
    return request.app.state.stats_buffer
