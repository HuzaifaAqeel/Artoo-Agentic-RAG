"""Offline demo — Artoo's agentic-RAG retrieval loop with no services running.

What is REAL here:
  * `HierarchicalChunker` — the actual structure-aware parent/child chunker
  * hybrid retrieval — dense (mock embeddings) + BM25 routes fused with the same
    RRF formula the production `HybridRetriever` uses: 1/(k+rank+1), k=60
  * parent-chunk expansion via the real parent_child_map

What is SCRIPTED (labeled below): the ReAct agent's Think/Act decisions and the
final synthesis — a stand-in for the LLM that drives the Think → Act → Observe
loop in production.

Usage:
    cd backend && python examples/offline_demo.py
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.chunker import HierarchicalChunker  # noqa: E402  (real chunker)

DOC = """# Vector Database Index Guide

## HNSW Index
Hierarchical Navigable Small World (HNSW) builds a multi-layer graph where each
vector links to its nearest neighbors. Search starts at the top layer and greedily
descends, giving logarithmic query time. HNSW delivers excellent recall — typically
95%+ at ef_search=64 — but the graph is memory hungry: expect roughly 1.2x the raw
vector bytes in overhead. Build time is slower than IVF because every insert walks
the graph. HNSW shines for latency-sensitive serving with fewer than ~100M vectors.

## IVF Index
Inverted File Index (IVF) clusters vectors with k-means into nlist partitions and
only scans the nprobe nearest partitions at query time. Memory overhead is tiny —
just the centroids plus posting lists. Recall is controlled by nprobe: higher nprobe
means better recall but slower queries. IVF trains fast and scales to billions of
vectors, but queries suffer when the data distribution drifts from the training
centroids. IVF with product quantization (IVF-PQ) compresses vectors further for
large-scale deployments.

## Choosing Between Them
Pick HNSW when p99 latency matters most and the dataset fits in RAM. Pick IVF or
IVF-PQ when the dataset is huge, memory is tight, or you need fast index builds.
A common production pattern is IVF for the cold corpus plus HNSW for the hot set.
"""

QUESTION = "What are the trade-offs between HNSW and IVF indexes?"


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25:
    """Minimal pure-python BM25 (the keyword route)."""

    def __init__(self, docs: list[str]):
        self.docs = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1

    def scores(self, query: str) -> list[float]:
        out = []
        for d in self.docs:
            s, dl = 0.0, len(d)
            for term in tokenize(query):
                if term not in self.df:
                    continue
                f = d.count(term)
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * (f * 2.2) / (f + 1.2 * (1 - 0.75 + 0.75 * dl / self.avgdl))
            out.append(s)
        return out


def mock_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic hash-bag embedding — MOCK stand-in for the embedding API."""
    vec = [0.0] * dim
    for tok in set(tokenize(text)):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Same RRF formula as HybridRetriever._rrf_fusion: score = Σ 1/(k+rank+1)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def main() -> None:
    print("=" * 72)
    print("Artoo Agentic RAG — offline demo (real chunker + real RRF, scripted agent)")
    print("=" * 72)

    # ---- 1. structure-aware chunking (REAL) ----
    chunker = HierarchicalChunker(parent_size=900, child_size=280, overlap=40)
    res = chunker.chunk(DOC)
    print(f"\n[ingest] {len(res.parent_chunks)} parent chunks, "
          f"{len(res.child_chunks)} child chunks")
    for pi, children in res.parent_child_map.items():
        print(f"         parent[{pi}] -> children {children}  "
              f"({res.context_headers[children[0]] if children else ''})")

    children = res.child_chunks
    child_vecs = [mock_embed(c) for c in children]  # MOCK embeddings
    q_vec = mock_embed(QUESTION)

    def hybrid_search(query: str, top_k: int = 4):
        dense_rank = sorted(range(len(children)),
                            key=lambda i: cosine(q_vec, child_vecs[i]), reverse=True)
        bm25_rank = sorted(range(len(children)),
                           key=lambda i: BM25(children).scores(query)[i], reverse=True)
        return dense_rank, bm25_rank, rrf_fusion([dense_rank, bm25_rank])

    # ---- 2. ReAct loop (agent decisions SCRIPTED, retrieval REAL) ----
    print(f"\n[question] {QUESTION}")
    print("\n[think] I need to compare HNSW vs IVF — search the knowledge base first.")
    print("[act]   knowledge_search(\"HNSW vs IVF trade-offs\")  → hybrid retrieval")

    dense_rank, bm25_rank, fused = hybrid_search(QUESTION)
    print(f"[observe] dense route top-3: {dense_rank[:3]} | "
          f"bm25 route top-3: {bm25_rank[:3]}")
    print("[observe] RRF fused ranking:")
    for idx, score in fused[:4]:
        print(f"          child[{idx}] rrf={score:.4f} :: {children[idx][:90]}…")

    top_child = fused[0][0]
    parent_idx = next(pi for pi, chs in res.parent_child_map.items() if top_child in chs)
    print(f"\n[think] child[{top_child}] has the recall/latency numbers — "
          "deep-read its parent for full context.")
    print(f"[act]   deep_read(parent[{parent_idx}])")
    print(f"[observe] parent[{parent_idx}] ({len(res.parent_chunks[parent_idx])} chars):")
    print("          " + res.parent_chunks[parent_idx][:220].replace("\n", " ") + "…")

    # ---- 3. evidence-first answer ----
    ev1, ev2 = res.parent_chunks[parent_idx], res.parent_chunks[1 - parent_idx]
    print("\n[final_answer] (synthesis scripted for the offline demo)")
    print("  HNSW gives 95%+ recall at logarithmic query time but costs ~1.2x vector "
          "bytes in graph overhead [1]. IVF has tiny memory overhead (centroids + "
          "posting lists) and scales to billions of vectors, but recall depends on "
          "nprobe and drifts hurt [2]. Rule of thumb: HNSW when p99 latency matters "
          "and data fits in RAM; IVF/IVF-PQ for huge corpora or tight memory [1][2].")
    print(f"\n  [1] parent[{parent_idx}] — HNSW section")
    print(f"  [2] parent[{1 - parent_idx}] — IVF section")
    print("\nevidence-first: every claim traces to a retrieved chunk — nothing from")
    print("parametric memory. In production the Think/Act steps call a real LLM,")
    print("Milvus (dense+sparse+BM25), a reranker, and optionally the event graph.")
    print("=" * 72)


if __name__ == "__main__":
    main()
