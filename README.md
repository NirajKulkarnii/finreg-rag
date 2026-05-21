# finreg-rag — Production RAG Pipeline for Financial Regulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-15%20files%20%7C%20unit%20%2B%20integration%20%2B%20security-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade Retrieval-Augmented Generation (RAG) system** purpose-built for financial regulation queries. Given a question about FCA rules, EU directives, or financial data, the system retrieves the most relevant regulatory documents, reranks them with a cross-encoder, and generates a structured, cited answer via a locally-served LLM.

Built to demonstrate end-to-end ML engineering skills across data ingestion, retrieval, generation, API design, observability, and evaluation — the same concerns that arise in real production deployments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                             │
│  POST /v1/generate       POST /v1/generate/stream (SSE)                 │
│  POST /v1/evaluate       POST /v1/evaluate/batch                        │
│  GET  /v1/stats          GET  /health   GET /ready   GET /metrics       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    RAGGenerator        │
                    │  (intent → retrieve    │
                    │   → web? → generate)  │
                    └─────────┬─────────────┘
           ┌──────────────────┼──────────────────────┐
           │                  │                       │
  ┌────────▼────────┐ ┌───────▼──────────┐ ┌────────▼────────┐
  │ IntentClassifier│ │ HybridRetriever  │ │   vLLM Server   │
  │ (regex + LLM    │ │                  │ │ Qwen2.5-7B-AWQ  │
  │  fallback)      │ │ ┌──────────────┐ │ │ OpenAI-compat.  │
  └─────────────────┘ │ │Dense(BGE-M3) │ │ │ API             │
                      │ │+ BM25 Lexical│ │ └─────────────────┘
                      │ │+ CrossEncoder│ │
                      │ │  Reranker    │ │
                      │ └──────────────┘ │
                      │   ChromaDB       │
                      │ (PersistentClient│
                      │  or HTTP)        │
                      └──────────────────┘
```

### Pipeline (per request)

| Stage | Component | Detail |
|-------|-----------|--------|
| **1. Intent** | `IntentClassifier` | Regex keywords → 4-way classification; LLM fallback for ambiguous queries |
| **2. Dense retrieval** | `DenseRetriever` + BGE-M3 | 1024-dim embeddings, cosine ANN via ChromaDB HNSW |
| **3. Lexical retrieval** | `BM25Retriever` | TF-IDF-style keyword matching over all stored chunks |
| **4. Merge & dedup** | `HybridRetriever` | Union of both result sets keyed by `chunk_id` |
| **5. Reranking** | `CrossEncoderReranker` | `ms-marco-MiniLM-L-6-v2`; sigmoid-normalised logits → (0, 1) |
| **6. Web fallback** | `web_search` | DuckDuckGo via `asyncio.to_thread` when top reranker score < threshold |
| **7. Generation** | `RAGGenerator` → vLLM | Qwen2.5 via OpenAI-compatible API; structured JSON output |

---

## Key Engineering Decisions

**Hybrid retrieval over pure dense search** — BM25 catches exact regulatory references (article numbers, acronyms like "EMIR", "CRR") that dense embeddings can underweight. The cross-encoder reranker then re-scores the merged set with full query–passage attention.

**Sigmoid normalisation on cross-encoder scores** — The `ms-marco` model produces unbounded logits. Hard-clamping to `[0, 1]` collapses all high-confidence results to 1.0. Sigmoid `1 / (1 + exp(-x))` preserves relative ordering while satisfying the API's `relevance_score ≤ 1.0` constraint.

**Per-request `_trace` dict** — Timing is captured into an optional `dict` passed to `generator.generate()`. No allocation overhead when unused (streaming, scripts), full per-phase breakdown when called from the API.

**Lazy model loading + explicit warm-start** — Models initialise via `@property` on first access, but the FastAPI lifespan calls `.model` on both the embedder and reranker in parallel via `asyncio.gather + run_in_executor` so the first real request is never slow.

**`workers=1` with async I/O** — ML models are not picklable. Rather than spawning multiple processes (which would each reload multi-GB models), the app runs one worker with async endpoints. True horizontal scaling is done at the container level.

---

## Data Sources

| Source | Content | Format |
|--------|---------|--------|
| **FCA Handbook** | UK regulatory rules, guidance, policy statements | HTML scraping + PDF extraction |
| **EUR-Lex** | EU financial directives (MiFID II, EMIR, SFDR, AIFMD, CRR, GDPR) | HuggingFace `multi_eurlex` dataset |
| **FinanceBench** | Public company financial reports (10-Ks, earnings) | HuggingFace `nlpaueb/financebench` dataset |

Documents are chunked at 512 tokens with 64-token overlap and stored with rich metadata (source, jurisdiction, date, URL) for filtered retrieval.

---

## Project Structure

```
finreg-rag/
├── api/                        # FastAPI application
│   ├── main.py                 #   App factory, lifespan, middleware registration
│   ├── metrics.py              #   Prometheus counters/histograms
│   ├── middleware.py           #   Structlog JSON logging + HTTP timing middleware
│   ├── dependencies.py         #   StatsBuffer (rolling deque) + FastAPI deps
│   └── routers/
│       ├── generate.py         #   POST /v1/generate, /v1/generate/stream (SSE)
│       ├── evaluate.py         #   POST /v1/evaluate, /v1/evaluate/batch
│       ├── stats.py            #   GET  /v1/stats
│       └── health.py           #   GET  /health, /ready, /metrics
│
├── generation/                 # LLM generation pipeline
│   ├── generator.py            #   RAGGenerator — orchestrates all stages
│   ├── intent_classifier.py    #   4-way intent: regulatory/financial/general/OOS
│   ├── context_builder.py      #   Token-budgeted context assembly + source ranking
│   ├── web_search.py           #   DuckDuckGo fallback (asyncio.to_thread)
│   ├── prompts.py              #   Versioned prompt templates (v1, …)
│   └── models.py               #   GenerationRequest / GenerationResponse / Source
│
├── retrieval/                  # Retrieval pipeline
│   ├── retriever.py            #   HybridRetriever — dense + BM25 + reranker
│   ├── dense_retriever.py      #   ChromaDB ANN search
│   ├── bm25_retriever.py       #   BM25 keyword search
│   └── reranker.py             #   CrossEncoderReranker (sigmoid normalisation)
│
├── ingestion/                  # Ingestion pipeline
│   ├── ingest.py               #   End-to-end ingest script
│   ├── embedder.py             #   BGE-M3 embedder (lazy load, batch encode)
│   ├── chunker.py              #   Sliding-window chunker with overlap
│   └── vector_store.py         #   ChromaDB wrapper (PersistentClient or HTTP)
│
├── data/                       # Data loading
│   ├── loaders/
│   │   ├── fca_loader.py       #   FCA website scraper (HTML + PDF)
│   │   ├── eurlex_loader.py    #   EUR-Lex via HuggingFace datasets
│   │   └── financebench_loader.py
│   └── pipeline.py             #   Unified multi-source loading pipeline
│
├── config/settings.py          # Environment-variable configuration (singleton)
├── scripts/query_test.py       # CLI smoke-test against live pipeline
└── tests/
    ├── unit/                   # 13 unit test files (all components mocked)
    ├── rag/                    # RAG integration tests (end-to-end)
    └── security/               # Prompt injection resistance tests
```

---

## API Reference

### `POST /v1/generate`

Full RAG pipeline. Returns a structured response with per-phase timing.

**Request**
```json
{
  "query": "What are the key obligations under FCA Consumer Duty?",
  "source_filter": "fca",
  "jurisdiction_filter": "UK"
}
```

**Response**
```json
{
  "request_id": "a3f2...",
  "intent": "regulatory_query",
  "answer": "Under the FCA Consumer Duty (PS22/9), firms must...",
  "sources": [
    {
      "title": "Consumer Duty — Final Rules",
      "url": "https://www.fca.org.uk/...",
      "source": "fca",
      "jurisdiction": "UK",
      "date_published": "2022-07-27",
      "relevance_score": 0.9134
    }
  ],
  "follow_up_questions": ["What are the four Consumer Duty outcomes?", "..."],
  "confidence": "high",
  "used_web_search": false,
  "retrieval_used": true,
  "timings": {
    "classify_retrieve_ms": 312.4,
    "web_search_ms": 0.0,
    "generate_ms": 4821.7,
    "total_ms": 5134.1,
    "retrieval_hits": 5,
    "top_score": 0.9134
  }
}
```

### `POST /v1/generate/stream`

SSE streaming. Frames:
```
data: {"type": "token",  "content": "Under"}
data: {"type": "token",  "content": " the"}
...
data: {"type": "meta",   "intent": "regulatory_query", "sources": [...], ...}
data: [DONE]
```

### `POST /v1/evaluate`

ROUGE-1/2/L + exact match scoring for a single answer.

```json
{
  "query": "What is Consumer Duty?",
  "generated_answer": "Consumer Duty is an FCA regulation...",
  "expected_answer":  "Consumer Duty requires firms to deliver good outcomes..."
}
```

### `POST /v1/evaluate/batch`

Up to 20 items. Set `"run_pipeline": true` to call the live RAG pipeline per item.

### `GET /v1/stats`

Rolling stats over the last 1000 requests:

```json
{
  "count": 42,
  "total_latency_ms":      {"p50": 3200, "p95": 8400, "p99": 11200, "mean": 4100},
  "classify_retrieve_ms":  {"p50": 290,  "p95": 620,  "p99": 890,   "mean": 320},
  "generate_ms":           {"p50": 2800, "p95": 7500, "p99": 10400, "mean": 3700},
  "web_search_rate": 0.14,
  "retrieval_rate":  0.88,
  "avg_top_score":   0.847,
  "intent_distribution": {
    "regulatory_query": 28, "financial_data": 10,
    "general_finreg": 3,    "out_of_scope": 1
  }
}
```

### `GET /health` · `GET /ready` · `GET /metrics`

Standard Kubernetes liveness / readiness probes. `/metrics` serves Prometheus text format.

---

## Observability Stack

| Tool | What's measured |
|------|----------------|
| **Prometheus** (`prometheus_client`) | HTTP request count + latency histogram (by method/path/status), per-phase RAG histograms (`classify_retrieve_ms`, `web_search_ms`, `generate_ms`, `total_ms`), retrieval hit count, top reranker score, web search trigger count |
| **structlog** | JSON-structured request logs with `request_id`, method, path, status, duration |
| **StatsBuffer** | In-process asyncio-safe rolling deque (last 1000 requests), queryable via `/v1/stats` — p50/p95/p99 latency, intent & confidence distributions |
| **X-Request-Id** | UUID injected into every response header; propagated through all log lines |

---

## Setup & Running

### Prerequisites

- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) serving any OpenAI-compatible model (Qwen2.5, Gemma, etc.)
- Docker (recommended for ChromaDB) — or use in-process persistent mode (no Docker needed)

### 1. Install

```bash
git clone https://github.com/<your-username>/finreg-rag.git
cd finreg-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit VLLM_BASE_URL, VLLM_MODEL, etc.
```

---

### 2. Start ChromaDB

**Option A — Docker (recommended)**

Starts ChromaDB as a persistent container on port 8001. Data survives restarts via a named Docker volume.

```bash
docker compose up chromadb -d

# Verify it's healthy
docker compose ps
curl http://localhost:8001/api/v1/heartbeat
# → {"nanosecond heartbeat": ...}
```

Make sure `.env` has (already set in `.env.example`):
```
CHROMA_HOST=localhost
CHROMA_PORT=8001
```
And that `CHROMA_PERSISTENT_PATH` is **not** set.

**Option B — No Docker (in-process persistent)**

Add this line to your `.env` — ChromaDB runs inside the Python process, no server needed:
```
CHROMA_PERSISTENT_PATH=/absolute/path/to/finreg-rag/chroma_db
```

---

### 3. Download Raw Documents

Fetches documents from all three sources (FCA, EUR-Lex, FinanceBench) and caches them as JSONL files in `data/raw/`. Only needs to run once.

```bash
python -m data.pipeline
# Creates: data/raw/fca.jsonl, data/raw/eurlex.jsonl, data/raw/financebench.jsonl
```

To download a single source:
```bash
python -c "
from data.pipeline import DataPipeline
p = DataPipeline()
p.run(sources=['eurlex'], eurlex_max=100)
"
```

> **Note:** FCA document downloads may fail with 403 errors due to Cloudflare bot protection on their CDN. EUR-Lex (HuggingFace dataset) and FinanceBench are unaffected. The pipeline logs warnings for failed docs and continues.

---

### 4. Ingest into ChromaDB

Chunks, embeds (BGE-M3), and upserts all downloaded documents into ChromaDB. This is the slow step — BGE-M3 on CPU takes ~1–2 min per 100 chunks.

```bash
# All sources (uses CHROMA_* settings from .env)
python -m ingestion.ingest

# Single source (faster for a quick test)
python -m ingestion.ingest --source eurlex

# If using Option B (no Docker), pass --local
python -m ingestion.ingest --local --local-path chroma_db

# Wipe collection and re-ingest from scratch
python -m ingestion.ingest --reset

# Lower batch size on machines with <8 GB RAM
python -m ingestion.ingest --batch-size 8
```

After ingestion, confirm the chunk count:
```bash
python -c "
from config.settings import settings
from ingestion.vector_store import VectorStore
s = VectorStore(host=settings.chroma_host, port=settings.chroma_port,
                persistent_path=settings.chroma_persistent_path)
print('Chunks indexed:', s.count())
"
```

---

### 5. Start vLLM (separate terminal)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Any OpenAI-compatible server works — set `VLLM_BASE_URL` and `VLLM_MODEL` in `.env` accordingly.

---

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1
```

> `workers=1` is required — ML models are not picklable across processes.

Interactive docs: http://localhost:8080/docs

---

### 7. (Optional) Start the Demo Frontend

A split-screen demo UI with a mobile chat interface on the left and an animated RAG pipeline diagram on the right. Designed for live presentations.

```bash
cd frontend
npm install       # first time only
npm run dev
```

Then open **`http://<your-server-ip>:3000`** in your local browser. The dev server binds to `0.0.0.0:3000` so it's reachable from any machine on the network.

The API URL field in the header defaults to `http://localhost:8080` — change it to `http://<your-server-ip>:8080` if the browser and API are on different machines.

---

### CLI Smoke Test

```bash
# Run default 4 queries (regulatory, financial, general, out-of-scope)
python scripts/query_test.py

# Single query in streaming mode
python scripts/query_test.py --stream --query "Explain SFDR disclosure obligations"
```

---

### Full Stack with Docker Compose

To run ChromaDB + the API together (vLLM must still run separately):

```bash
# Edit .env first — set VLLM_BASE_URL to your vLLM server
docker compose up chromadb regiq -d

# View logs
docker compose logs -f regiq

# Stop everything
docker compose down
```

---

## Testing

```bash
pytest                        # all tests
pytest tests/unit/            # unit tests only (no live services needed)
pytest tests/rag/             # end-to-end integration tests (requires vLLM + ChromaDB)
pytest tests/security/        # prompt injection resistance
pytest --tb=short -q          # compact output
```

**Test coverage across 15 files:**

| Suite | Files | What's tested |
|-------|-------|---------------|
| `tests/unit/` | 13 files | Every component in isolation — embedder, chunker, vector store, BM25, dense retriever, hybrid retriever, reranker, intent classifier, context builder, prompts, generator (sync + stream), web search |
| `tests/rag/` | 1 file | Full pipeline from query → intent → retrieval → reranking → generation |
| `tests/security/` | 1 file | Prompt injection patterns — instruction override, jailbreak, context poisoning |

All unit tests use `unittest.mock` / `pytest-asyncio` — no GPU, no network, no running services required.

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env`:

```bash
# LLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# Embeddings — use cpu when GPU is occupied by vLLM
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

# ChromaDB — use persistent local directory (no Docker needed)
CHROMA_PERSISTENT_PATH=./chroma_db

# Retrieval
RETRIEVAL_TOP_K=20      # candidates fetched before reranking
RERANK_TOP_K=5          # final results after reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Models Used

| Model | Role | Why |
|-------|------|-----|
| **BAAI/bge-m3** | Query + document embeddings | 1024-dim, 8192-token context, strong multilingual + domain performance; handles long EU directives |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Reranking | Fast (6-layer), trained on MS MARCO passage retrieval, strong passage-level relevance signal |
| **Qwen2.5-7B-Instruct-AWQ** | Answer generation | AWQ 4-bit quantisation fits in 21 GB GPU alongside the rest of the stack; follows instruction and JSON output format reliably |

---

## Skills Demonstrated

| Area | Detail |
|------|--------|
| **RAG architecture** | Multi-stage pipeline: hybrid retrieval (dense + BM25), cross-encoder reranking, dynamic web fallback, intent-conditioned prompting |
| **LLM serving** | vLLM integration via OpenAI-compatible API; streaming token generation with SSE; structured JSON output parsing |
| **Embeddings** | BGE-M3 (1024-dim, 8192-token context); batch encoding; lazy load + parallel warm-start |
| **Vector databases** | ChromaDB HNSW ANN; both `PersistentClient` (local) and `HttpClient` (server) modes; metadata filtering |
| **API design** | FastAPI with async endpoints, SSE streaming, Pydantic v2 models, dependency injection, lifespan management |
| **Observability** | Prometheus metrics (per-phase histograms), structlog JSON logging, rolling percentile stats (p50/p95/p99) |
| **Evaluation** | ROUGE-1/2/L scoring, exact match, batch evaluation with live pipeline option |
| **Testing** | 15 test files, unit + integration + security; pytest-asyncio for async mocking; no test requires live services |
| **Production concerns** | Warm-start on boot, CORS + GZip middleware, `X-Request-Id` propagation, env-based config, GPU memory management (CPU offload for embedder when GPU is saturated) |
| **Data engineering** | Multi-source loaders (web scraping, PDF extraction, HuggingFace datasets); sliding-window chunking; metadata-enriched upsert |

---

## Potential Extensions

- **Re-ranking with larger model** — swap `ms-marco-MiniLM` for `bge-reranker-v2-m3` for higher accuracy at the cost of latency
- **Query expansion** — HyDE (hypothetical document embeddings) or LLM-generated sub-queries to improve recall
- **Feedback loop** — log user ratings back to a database to fine-tune retrieval thresholds
- **Multi-turn conversation** — thread conversation history into context builder for follow-up questions
- **Grafana dashboard** — connect Prometheus metrics to a pre-built LLM observability dashboard

---

## License

MIT
