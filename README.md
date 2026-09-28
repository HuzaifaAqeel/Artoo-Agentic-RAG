<div align="center">

# 🤖 Artoo Agentic RAG

**Let the knowledge base retrieve itself — a ReAct-agent-driven agentic RAG framework.**

A ReAct agent autonomously orchestrates keyword search, hybrid retrieval, deep reading,
web search, and MCP tools — delivering an "evidence-first, then answer" traceable Q&A
experience over your documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**Full documentation: [README_EN.md](README_EN.md)**

</div>

---

## Quick start

```bash
cp backend/.env.example backend/.env   # point at your LLM / embedding / rerank APIs
docker compose up -d --build
```

Offline demo (no services, no keys):

```bash
cd backend && python examples/offline_demo.py
```

See [README_EN.md](README_EN.md) for architecture, configuration, and project structure.

## License

MIT — see [LICENSE](LICENSE).
