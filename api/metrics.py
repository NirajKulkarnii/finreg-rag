"""
Prometheus Metrics
------------------
All application-level metrics are defined here and imported by routers/middleware.
Using prometheus_client directly (not the FastAPI instrumentator) so we get
fine-grained, per-phase RAG histograms alongside the standard HTTP metrics.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── HTTP ──────────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "finreg_http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "finreg_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ── RAG pipeline phase latencies ──────────────────────────────────────────────

RAG_CLASSIFY_RETRIEVE_MS = Histogram(
    "finreg_rag_classify_retrieve_ms",
    "Intent classification + retrieval latency (ms)",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
)

RAG_WEB_SEARCH_MS = Histogram(
    "finreg_rag_web_search_ms",
    "Web search fallback latency (ms, only when triggered)",
    buckets=[100, 250, 500, 1000, 2500, 5000],
)

RAG_GENERATE_MS = Histogram(
    "finreg_rag_generate_ms",
    "LLM generation latency (ms)",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

RAG_TOTAL_MS = Histogram(
    "finreg_rag_total_ms",
    "Total RAG pipeline latency (ms)",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

# ── RAG semantic metrics ───────────────────────────────────────────────────────

RAG_REQUESTS_TOTAL = Counter(
    "finreg_rag_requests_total",
    "Total RAG generation requests",
    ["intent", "confidence"],
)

RAG_WEB_SEARCH_TOTAL = Counter(
    "finreg_rag_web_search_total",
    "Number of requests that triggered web search fallback",
)

RAG_RETRIEVAL_HITS = Histogram(
    "finreg_rag_retrieval_hits",
    "Number of retrieval results returned",
    buckets=[0, 1, 2, 3, 5, 10, 20],
)

RAG_TOP_SCORE = Histogram(
    "finreg_rag_top_score",
    "Top cross-encoder score from retrieval (0-1)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Application state ─────────────────────────────────────────────────────────

APP_INFO = Gauge(
    "finreg_app_info",
    "Application metadata",
    ["version", "model"],
)
