---
name: project-demo-frontend
description: FinReg RAG demo frontend — split-screen mobile chat + animated pipeline diagram for AI bootcamp presentation
metadata:
  type: project
---

Demo frontend built for AI Bootcamp session on "AI for Customer Support & Automation". Audience: non-technical one-person business founders.

**Location:** `frontend/index.html` — single self-contained HTML file, zero build step.

**Layout:** Two-panel split screen
- Left: iPhone-style mobile chat UI (chat with FinReg bot, example chips, follow-up questions, sources)
- Right: Animated RAG pipeline flow diagram that lights up step-by-step as each phase completes

**Flow nodes (in order):** Your Question → [Intent Classifier + Document Search (parallel)] → Smart Reranker → Context Builder → Web Search Fallback (conditional) → AI Generation → Answer Delivered

**Animation:** Driven by `timings` from `POST /v1/generate` (non-streaming endpoint). Animations are time-compressed (35% of real timing, min 500ms per step) so the diagram feels live without being too slow.

**How to run:**
```
# Backend
uvicorn api.main:app --port 8089

# Frontend (any of these)
python frontend/serve.py          # opens http://localhost:3000
python -m http.server 3000 --directory frontend
# or just open frontend/index.html directly in a browser
```

**Why:** Uses this as demo at bootcamp to show how RAG = "smart document search + AI answer". Non-jargon labels used throughout ("Smart Reranker" not "cross-encoder", "Document Search" not "vector store").
