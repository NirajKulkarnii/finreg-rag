#!/usr/bin/env python3
"""
End-to-end RAG pipeline smoke test.

Usage:
    python scripts/query_test.py                        # run default queries
    python scripts/query_test.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/query_test.py --query "Your question here"
    python scripts/query_test.py --stream               # streaming mode

Requires:
    - vLLM running at VLLM_BASE_URL (default: http://localhost:8000/v1)
    - ChromaDB running at CHROMA_HOST:CHROMA_PORT (default: localhost:8001)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Make project root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from config.settings import settings
from ingestion import Embedder, VectorStore
from retrieval import HybridRetriever
from generation.generator import RAGGenerator
from generation.models import GenerationRequest


# ── Colour helpers ────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"


def hdr(text: str) -> str:
    return f"{BOLD}{CYAN}{text}{RESET}"

def ok(text: str) -> str:
    return f"{GREEN}{text}{RESET}"

def warn(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"

def err(text: str) -> str:
    return f"{RED}{text}{RESET}"

def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


# ── Default test queries ──────────────────────────────────────────────────────

DEFAULT_QUERIES = [
    # regulatory_query — should use retrieval
    "What are the FCA Consumer Duty obligations for retail firms?",
    # financial_data — may fall back to web
    "What is the CET1 capital ratio requirement under CRR?",
    # general_finreg — LLM classifies, broad answer
    "How does financial regulation work?",
    # out_of_scope — should decline politely
    "What is the best recipe for pasta?",
]


# ── Pre-flight: check vLLM and ChromaDB ──────────────────────────────────────

def check_vllm(base_url: str) -> list[str]:
    """Returns list of available model IDs, or empty list on failure."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=5)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        print(ok(f"  vLLM OK — available models: {models}"))
        return models
    except Exception as exc:
        print(err(f"  vLLM not reachable at {base_url}: {exc}"))
        return []


def check_chroma(persistent_path: str | None, host: str, port: int) -> bool:
    if persistent_path:
        import chromadb
        try:
            client = chromadb.PersistentClient(path=persistent_path)
            colls = [c.name for c in client.list_collections()]
            print(ok(f"  ChromaDB OK — local path '{persistent_path}'  collections: {colls}"))
            return True
        except Exception as exc:
            print(warn(f"  ChromaDB local path '{persistent_path}' error: {exc}"))
            print(warn("  Retrieval will return empty results (web-search fallback may engage)."))
            return False
    else:
        try:
            r = httpx.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=5)
            r.raise_for_status()
            print(ok(f"  ChromaDB OK — {host}:{port}"))
            return True
        except Exception as exc:
            print(warn(f"  ChromaDB not reachable at {host}:{port}: {exc}"))
            print(warn("  Retrieval will return empty results (web-search fallback may engage)."))
            return False


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_pipeline(model: str, device: str = "cpu") -> RAGGenerator:
    embedder = Embedder(
        model_name=settings.embedding_model,
        device=device,
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
        device=device,
    )
    generator = RAGGenerator(
        retriever=retriever,
        vllm_base_url=settings.vllm_base_url,
        model=model,
    )
    return generator


# ── Query runners ─────────────────────────────────────────────────────────────

async def run_query(generator: RAGGenerator, query: str) -> None:
    print(f"\n{hdr('Q:')} {query}")
    print(dim("─" * 72))

    t0 = time.perf_counter()
    try:
        resp = await generator.generate(GenerationRequest(query=query))
    except Exception as exc:
        print(err(f"  ERROR: {exc}"))
        return
    elapsed = time.perf_counter() - t0

    # Intent
    print(f"  {BOLD}Intent:{RESET}     {resp.intent.value}")
    print(f"  {BOLD}Confidence:{RESET} {resp.confidence.value}")
    print(f"  {BOLD}Retrieval:{RESET}  {'yes' if resp.retrieval_used else 'no'}"
          f"  |  Web search: {'yes' if resp.used_web_search else 'no'}"
          + (f"  ({resp.web_search_query})" if resp.web_search_query else ""))

    # Sources
    if resp.sources:
        print(f"  {BOLD}Sources ({len(resp.sources)}):{RESET}")
        for s in resp.sources[:3]:
            print(f"    [{s.relevance_score:.3f}] {s.title[:70]}  [{s.source}/{s.jurisdiction}]")

    # Answer
    print(f"\n  {BOLD}Answer:{RESET}")
    for line in resp.answer.strip().splitlines():
        print(f"    {line}")

    # Follow-ups
    if resp.follow_up_questions:
        print(f"\n  {BOLD}Follow-ups:{RESET}")
        for fq in resp.follow_up_questions:
            print(f"    - {fq}")

    print(dim(f"\n  [{elapsed:.2f}s]"))


async def run_stream(generator: RAGGenerator, query: str) -> None:
    print(f"\n{hdr('Q:')} {query}")
    print(dim("─" * 72))
    print(f"  {BOLD}Answer (streaming):{RESET}")
    print("  ", end="", flush=True)

    t0 = time.perf_counter()
    meta = None
    try:
        async for chunk in generator.generate_stream(GenerationRequest(query=query)):
            if chunk.startswith("\n__META__:"):
                meta = json.loads(chunk[len("\n__META__:"):])
            else:
                print(chunk, end="", flush=True)
    except Exception as exc:
        print(err(f"\n  ERROR: {exc}"))
        return

    elapsed = time.perf_counter() - t0
    print()  # newline after streamed answer
    if meta:
        print(f"\n  {BOLD}Intent:{RESET}     {meta.get('intent')}")
        print(f"  {BOLD}Retrieval:{RESET}  {'yes' if meta.get('retrieval_used') else 'no'}"
              f"  |  Web: {'yes' if meta.get('used_web_search') else 'no'}")
    print(dim(f"\n  [{elapsed:.2f}s]"))


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline smoke test")
    parser.add_argument("--model", default=None,
                        help="vLLM model name (auto-detected from vLLM if not set)")
    parser.add_argument("--query", default=None,
                        help="Run a single custom query instead of defaults")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode (generate_stream)")
    parser.add_argument("--device", default="cpu",
                        help="Device for embedder/reranker: cpu or cuda (default: cpu, "
                             "use cpu when vLLM is already using the GPU)")
    args = parser.parse_args()

    # Pre-flight checks
    print(hdr("Pre-flight checks:"))
    served_models = check_vllm(settings.vllm_base_url)
    chroma_ok = check_chroma(settings.chroma_persistent_path, settings.chroma_host, settings.chroma_port)

    if not served_models:
        print(err("\nvLLM is required. Start it first, then re-run."))
        sys.exit(1)

    # Resolve model name: CLI arg → .env/settings → first model served by vLLM
    model = args.model or settings.vllm_model
    if model not in served_models:
        if served_models:
            print(warn(f"  Configured model '{model}' not found in vLLM. "
                       f"Auto-selecting '{served_models[0]}'."))
            model = served_models[0]
        else:
            print(err(f"  Model '{model}' not available and no fallback found."))
            sys.exit(1)

    print(hdr("\n=== finreg-rag end-to-end test ==="))
    print(f"  vLLM:     {settings.vllm_base_url}")
    print(f"  Model:    {model}")
    chroma_loc = settings.chroma_persistent_path or f"{settings.chroma_host}:{settings.chroma_port}"
    print(f"  ChromaDB: {chroma_loc}  collection={settings.chroma_collection}")
    print(f"  Device:   {args.device} (embedder + reranker)")
    print()

    print(hdr("Building pipeline..."))
    try:
        generator = build_pipeline(model, device=args.device)
        print(ok("  Pipeline ready."))
    except Exception as exc:
        print(err(f"  Failed to build pipeline: {exc}"))
        sys.exit(1)

    queries = [args.query] if args.query else DEFAULT_QUERIES
    runner = run_stream if args.stream else run_query

    print(hdr(f"\nRunning {len(queries)} quer{'y' if len(queries)==1 else 'ies'}...\n"))
    for q in queries:
        await runner(generator, q)

    print(f"\n{hdr('Done.')}\n")


if __name__ == "__main__":
    asyncio.run(main())
