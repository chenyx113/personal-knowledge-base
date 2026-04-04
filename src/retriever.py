"""
retriever.py - Hybrid search engine combining vector search, FTS, and RRF fusion.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .database import Database
from .embedder import (
    OllamaEmbedder, bytes_to_vector, cosine_similarity, vector_to_bytes
)


class SearchResult:
    """A single search result with scoring metadata."""

    def __init__(self, l1_id: int, content: str, score: float,
                 tier: int = 1, source: str = "hybrid",
                 doc_path: str = "", title: str = "",
                 chunk_index: int = 0, content_hash: str = "",
                 summary: str = "", overview: str = "",
                 duplicate_group_id: int = None,
                 start_line: int = 0, end_line: int = 0):
        self.l1_id = l1_id
        self.content = content
        self.score = score
        self.tier = tier
        self.source = source
        self.doc_path = doc_path
        self.title = title
        self.chunk_index = chunk_index
        self.content_hash = content_hash
        self.summary = summary
        self.overview = overview
        self.duplicate_group_id = duplicate_group_id
        self.start_line = start_line
        self.end_line = end_line

    def to_dict(self) -> Dict:
        return {
            "id": self.l1_id,
            "content": self.content,
            "score": round(self.score, 4),
            "tier": self.tier,
            "source": self.source,
            "doc_path": self.doc_path,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "overview": self.overview,
            "is_duplicate": self.duplicate_group_id is not None,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


class HybridRetriever:
    """Hybrid retriever: vector search + full-text search + RRF fusion."""

    def __init__(self, db: Database, embedder: OllamaEmbedder, config: Dict):
        self.db = db
        self.embedder = embedder
        self.config = config.get("search", {})
        self.rrf_k = self.config.get("rrf_k", 60)
        self.auto_fallback = self.config.get("auto_tier_fallback", True)

    def search(self, query: str, top_k: int = None,
               tier: int = None, tag_filter: str = None) -> List[SearchResult]:
        """Execute hybrid search across memory tiers.

        Args:
            query: Search query (natural language).
            top_k: Number of results to return.
            tier: Restrict to specific tier (1, 2, or 3). None = all.
            tag_filter: Filter by tag name.

        Returns:
            List of SearchResult sorted by relevance.
        """
        top_k = top_k or self.config.get("default_top_k", 10)

        # 1. Vector search
        vec_results = self._vector_search(query, top_k * 2, tier)

        # 2. Full-text search
        fts_results = self._fts_search(query, top_k * 2)

        # 3. RRF fusion
        combined = self._reciprocal_rank_fusion(vec_results, fts_results)

        # 4. Tag filter
        if tag_filter:
            combined = self._filter_by_tag(combined, tag_filter)

        # 5. Auto tier fallback
        if self.auto_fallback and len(combined) < top_k and tier == 1:
            l2_results = self._vector_search(query, top_k, tier=2)
            for r in l2_results:
                if r.l1_id not in {c.l1_id for c in combined}:
                    combined.append(r)

        # 6. Update access stats for returned results
        results = combined[:top_k]
        for r in results:
            self.db.update_l1_access(r.l1_id)

        # Log search
        self.db.log_search(query, len(results))

        return results

    def _vector_search(self, query: str, top_k: int,
                       tier: int = None) -> List[SearchResult]:
        """Search by vector similarity."""
        query_vec = self.embedder.embed(query)

        # Get all records (for small-medium KBs; for large, use FAISS)
        records = self.db.get_all_l1(tier=tier)

        scored = []
        for rec in records:
            if not rec.get("vector"):
                continue
            rec_vec = bytes_to_vector(rec["vector"])
            sim = cosine_similarity(query_vec, rec_vec)
            scored.append((rec, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rec, sim in scored[:top_k]:
            doc = self.db.get_document(rec.get("doc_id", 0)) or {}
            results.append(SearchResult(
                l1_id=rec["id"],
                content=rec.get("raw_content", ""),
                score=sim,
                tier=rec.get("memory_tier", 1),
                source="vector",
                doc_path=doc.get("file_path", ""),
                title=doc.get("title", ""),
                chunk_index=rec.get("chunk_index", 0),
                content_hash=rec.get("content_hash", ""),
                summary=rec.get("summary_content", ""),
                overview=rec.get("overview_content", ""),
                duplicate_group_id=rec.get("duplicate_group_id"),
                start_line=rec.get("start_line", 0),
                end_line=rec.get("end_line", 0),
            ))

        return results

    def _fts_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Full-text search using FTS5."""
        fts_rows = self.db.fts_search(query, top_k)

        results = []
        for i, row in enumerate(fts_rows):
            rowid = row.get("rowid", 0)
            rec = self.db.get_l1(rowid)
            if not rec:
                continue
            doc = self.db.get_document(rec.get("doc_id", 0)) or {}

            # FTS rank is negative (closer to 0 = better)
            fts_rank = abs(row.get("fts_rank", 0))
            score = 1.0 / (1.0 + fts_rank) if fts_rank else 0.5

            results.append(SearchResult(
                l1_id=rec["id"],
                content=rec.get("raw_content", ""),
                score=score,
                tier=rec.get("memory_tier", 1),
                source="fts",
                doc_path=doc.get("file_path", ""),
                title=doc.get("title", ""),
                chunk_index=rec.get("chunk_index", 0),
                content_hash=rec.get("content_hash", ""),
                summary=rec.get("summary_content", ""),
                overview=rec.get("overview_content", ""),
                duplicate_group_id=rec.get("duplicate_group_id"),
                start_line=rec.get("start_line", 0),
                end_line=rec.get("end_line", 0),
            ))

        return results

    def _reciprocal_rank_fusion(self, vec_results: List[SearchResult],
                                 fts_results: List[SearchResult]) -> List[SearchResult]:
        """Combine vector and FTS results using Reciprocal Rank Fusion."""
        scores: Dict[int, float] = {}
        results_map: Dict[int, SearchResult] = {}

        for rank, r in enumerate(vec_results):
            scores[r.l1_id] = scores.get(r.l1_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            results_map[r.l1_id] = r

        for rank, r in enumerate(fts_results):
            scores[r.l1_id] = scores.get(r.l1_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            if r.l1_id not in results_map:
                results_map[r.l1_id] = r

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        combined = []
        for lid in sorted_ids:
            r = results_map[lid]
            r.score = scores[lid]
            r.source = "hybrid"
            combined.append(r)

        return combined

    def _filter_by_tag(self, results: List[SearchResult],
                       tag: str) -> List[SearchResult]:
        """Filter results by document tag."""
        import json
        filtered = []
        for r in results:
            doc = self.db.get_document_by_path(r.doc_path)
            if doc:
                tags = json.loads(doc.get("tags", "[]"))
                if tag.lower() in [t.lower() for t in tags]:
                    filtered.append(r)
        return filtered

    def search_by_tier(self, query: str, tier: int,
                       top_k: int = 5) -> List[SearchResult]:
        """Search within a specific memory tier only."""
        return self._vector_search(query, top_k, tier=tier)
