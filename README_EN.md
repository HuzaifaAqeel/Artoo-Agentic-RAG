<div align="center">

# 🤖 Artoo Agentic RAG

**Let the knowledge base retrieve itself — a ReAct-agent-driven agentic RAG framework.**

A ReAct agent autonomously orchestrates keyword search, hybrid retrieval, deep reading,
web search, and MCP tools — delivering an "evidence-first, then answer" traceable Q&A
experience over your documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00a1ea.svg)](https://milvus.io/)

</div>

---

## Quick start

```bash
cp backend/.env.example backend/.env   # point at your LLM / embedding / rerank APIs
docker compose up -d --build           # backend + Milvus + PostgreSQL + worker
```

Then open the web UI, create a knowledge base, upload documents (or paste a web
article link), and ask questions. The agent's thoughts, tool calls, citations, and
token usage stream to the UI in real time.

### Offline demo (no services, no keys)

```bash
cd backend && python examples/offline_demo.py
```

Runs the real `HierarchicalChunker` plus a real hybrid-retrieval + RRF fusion pass
over a sample doc, then walks the Think → Act → Observe loop with scripted agent
decisions so you can see the evidence-first flow end to end.

---

## How it works

### 1. A real ReAct agent — not a fixed pipeline

The LLM decides the retrieval strategy itself inside a Think → Act → Observe loop:
search the KB, grep chunks, deep-read a promising chunk's parent, search the web,
load a skill — then decide whether to keep digging or submit a cited answer.
An "evidence-first" system prompt enforces *search first, deep-read, then answer* —
no fabrication from parametric memory.

### 2. Three-way hybrid retrieval (+ optional graph route)

Dense semantic + sparse vector + BM25 full-text recall run in parallel, then:

```
RRF fusion → rerank → composite scoring → MMR de-duplication → parent-chunk expansion
```

With the knowledge graph enabled, an event-centric GraphRAG joins as a fourth route:
events (subject–verb–object + time/place) are extracted per chunk, seeded via event
vector recall + entity bridging, expanded multi-hop, then back-fetch related chunks —
built for cross-document multi-hop QA.

### 3. Structure-aware document processing

Documents split along logical structure into parent/child chunk hierarchies with
breadcrumb headers; mixed text-and-image documents get concurrent OCR; web/WeChat
article links are fetched and saved as KB documents with the source URL retained.

### 4. Three-tier progressive context management

BPE token estimation + API usage delta tracking + LLM summary consolidation +
group-based truncation keep long agent conversations inside the window.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Chat API (OpenAI-compatible · SSE)  │  Admin API (REST)      │
│  MCP Server (exposes KB capabilities as tools)               │
├─────────────────────────────────────────────────────────────┤
│  ReAct Agent Engine                                          │
│  Think(stream LLM) → Analyze(stop) → Act(tools) → Observe    │
│  EventBus │ 3-tier context mgmt │ Agent Skills                │
├─────────────────────────────────────────────────────────────┤
│  Tools: knowledge_search │ grep_chunks │ deep_read │ web_search│
│         attachment_read │ thinking │ skill_load │ MCP tools   │
├─────────────────────────────────────────────────────────────┤
│  Retrieval: Dense + Sparse + BM25 (+ Graph)                  │
│             → RRF → Rerank → MMR → parent expansion           │
├─────────────────────────────────────────────────────────────┤
│  Index:  Milvus (dense+sparse) │ PostgreSQL (metadata)       │
│          Neo4j (knowledge graph, optional)                   │
├─────────────────────────────────────────────────────────────┤
│  Pipeline: Loader → OCR → Chunker → Embedder → Indexer       │
│            (async worker over Redis Stream)                   │
├─────────────────────────────────────────────────────────────┤
│  Models (external HTTP): LLM │ Embedding │ Rerank │ OCR       │
└─────────────────────────────────────────────────────────────┘
```

All AI inference runs through HTTP calls to external model services, so the backend
stays lightweight and self-hostable with full data sovereignty.

---

## Configuration

Everything is environment variables — see `backend/.env.example`:

| Variable | What it controls |
|---|---|
| `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` | Chat model for the ReAct agent |
| `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` | Dense/sparse embedding service |
| `RERANK_API_BASE` / `RERANK_API_KEY` | Reranker (hot-swappable) |
| `MILVUS_URI` | Vector index |
| `DATABASE_URL` | PostgreSQL metadata |
| `NEO4J_URI` | Knowledge graph (optional) |
| `TAVILY_API_KEY` | Web search tool |

---

## Project structure

```
backend/
├── app/agent/        # ReAct engine, tools, prompts, skills, memory
├── app/retrieval/    # hybrid (dense+sparse+BM25), RRF, rerank, MMR, graph
├── app/pipeline/     # loader → OCR → chunker → embedder → indexer + worker
├── app/models/       # LLM / embedding / rerank provider abstractions
├── app/api/          # chat (SSE), KB admin, OpenAPI docs
├── app/storage/      # Milvus / Postgres / Neo4j clients
└── examples/
    └── offline_demo.py
frontend/             # React chat + KB admin + graph explorer
docs/                 # deeper documentation
```

---

## License

MIT — see [LICENSE](LICENSE).
