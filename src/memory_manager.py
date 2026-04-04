"""
memory_manager.py - Tiered memory management (L1/L2/L3).
Handles ingestion, tier promotion/demotion, and lifecycle maintenance.
"""

import json
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .database import Database
from .chunker import (
    chunk_text, compute_content_hash, extract_text_from_file, ChunkResult
)
from .embedder import (
    OllamaEmbedder, OllamaGenerator, vector_to_bytes, bytes_to_vector
)
from .dedup import DeduplicationEngine


class MemoryManager:
    """Manages three-tier memory: L1 (working), L2 (short-term), L3 (long-term)."""

    def __init__(self, db: Database, config: Dict):
        ollama_cfg = config.get("ollama", {})
        host = ollama_cfg.get("host", "http://localhost:11434")
        models = ollama_cfg.get("models", {})

        self.db = db
        self.config = config
        self.embedder = OllamaEmbedder(
            host=host,
            model=models.get("embedding", "nomic-embed-text"),
            timeout=ollama_cfg.get("timeout", 120),
        )
        self.generator = OllamaGenerator(
            host=host,
            model=models.get("summary", "phi3:mini"),
            timeout=ollama_cfg.get("timeout", 120),
        )
        self.dedup = DeduplicationEngine(
            simhash_threshold=config.get("memory", {}).get("deduplication", {}).get("simhash_threshold", 3),
            vector_threshold=config.get("memory", {}).get("deduplication", {}).get("vector_threshold", 0.92),
        )
        self.chunk_config = config.get("chunking", {})
        self.memory_config = config.get("memory", {}).get("tiers", {})

    def ingest_file(self, file_path: str, tags: List[str] = None,
                    title: str = "") -> Dict:
        """Ingest a file into the knowledge base.

        Returns dict with status, doc_id, chunks_added, duplicates_found.
        """
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Check if already indexed with same hash
        text = extract_text_from_file(file_path)
        if not text.strip():
            return {"status": "error", "message": "Empty or unreadable file"}

        content_hash = compute_content_hash(text)
        existing = self.db.get_document_by_path(str(p.resolve()))
        if existing and existing.get("content_hash") == content_hash:
            return {"status": "unchanged", "doc_id": existing["id"],
                    "message": "File unchanged since last index"}

        # Parse and chunk
        file_type = p.suffix.lstrip(".")
        chunks = chunk_text(
            text, file_type,
            chunk_size=self.chunk_config.get("default_chunk_size", 500),
            overlap=self.chunk_config.get("chunk_overlap", 50),
        )

        if not chunks:
            return {"status": "error", "message": "No chunks extracted"}

        # Insert/update document record
        if not title:
            title = p.stem.replace("_", " ").replace("-", " ").title()

        doc_id = self.db.insert_document(
            file_path=str(p.resolve()),
            file_type=file_type,
            title=title,
            tags=tags or [],
            content_hash=content_hash,
        )

        # Generate summary
        summary = self.generator.generate_summary(text)
        if summary:
            self.db.conn.execute(
                "UPDATE documents SET summary = ? WHERE id = ?",
                (summary, doc_id)
            )
            self.db.conn.commit()

        # Process chunks
        chunks_added = 0
        duplicates_found = 0
        existing_records = self.db.get_all_l1()

        for chunk in chunks:
            result = self._ingest_chunk(doc_id, chunk, existing_records)
            if result.get("added"):
                chunks_added += 1
                # Add to existing records for subsequent dedup checks
                if result.get("record"):
                    existing_records.append(result["record"])
            if result.get("duplicate"):
                duplicates_found += 1

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_added": chunks_added,
            "duplicates_found": duplicates_found,
            "total_chunks": len(chunks),
            "title": title,
        }

    def _ingest_chunk(self, doc_id: int, chunk: ChunkResult,
                      existing_records: List[Dict]) -> Dict:
        """Process a single chunk: embed, dedup, store."""
        content = chunk.content
        if not content.strip():
            return {"added": False}

        # Generate embedding
        vec = self.embedder.embed(content)
        vec_bytes = vector_to_bytes(vec)

        # Check duplicates
        content_hash = compute_content_hash(content)
        dedup_result = self.dedup.check_duplicate(
            content, vec, existing_records
        )

        if dedup_result["is_duplicate"]:
            # Handle duplicate: update cluster
            match_id = dedup_result["match_id"]
            cluster = self.db.get_duplicate_cluster_for(content_hash)
            if not cluster:
                match_rec = self.db.get_l1(match_id)
                if match_rec:
                    cid = self.db.insert_duplicate_cluster(
                        match_rec["content_hash"],
                        self.dedup.recommend_merge_strategy(dedup_result["similarity"])
                    )
                    self.db.add_duplicate_member(
                        cid, match_rec["content_hash"],
                        1.0, is_canonical=True
                    )
                    self.db.add_duplicate_member(
                        cid, content_hash,
                        dedup_result["similarity"]
                    )

            auto_merge = self.config.get("memory", {}).get(
                "deduplication", {}
            ).get("auto_merge", False)

            if not auto_merge:
                return {"added": False, "duplicate": True,
                        "match_id": match_id,
                        "similarity": dedup_result["similarity"]}

        # Generate tiered content
        semantic_hash = self.dedup.compute_semantic_hash(content)
        summary = self.generator.generate_summary(content)
        overview = self.generator.generate_overview(content)

        # Store in L1
        l1_id = self.db.insert_l1(
            doc_id=doc_id,
            chunk_index=chunk.index,
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            raw_content=content,
            summary_content=summary,
            overview_content=overview,
            vector=vec_bytes,
            vector_dim=len(vec),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

        record = {
            "id": l1_id,
            "content_hash": content_hash,
            "semantic_hash": semantic_hash,
            "vector": vec_bytes,
            "raw_content": content,
        }

        return {"added": True, "l1_id": l1_id, "record": record}

    def ingest_directory(self, dir_path: str, recursive: bool = True,
                         tags: List[str] = None,
                         extensions: List[str] = None) -> Dict:
        """Ingest all supported files from a directory."""
        p = Path(dir_path)
        if not p.is_dir():
            return {"status": "error", "message": f"Not a directory: {dir_path}"}

        supported = extensions or [
            ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
            ".toml", ".sql", ".sh", ".css", ".html", ".xml", ".ipynb",
            ".java", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".php",
        ]

        pattern = "**/*" if recursive else "*"
        files = [f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in supported]

        results = {"total_files": len(files), "success": 0, "failed": 0,
                    "unchanged": 0, "details": []}

        for f in sorted(files):
            r = self.ingest_file(str(f), tags=tags)
            results["details"].append({"file": str(f), "result": r})
            if r["status"] == "success":
                results["success"] += 1
            elif r["status"] == "unchanged":
                results["unchanged"] += 1
            else:
                results["failed"] += 1

        return results

    # ---- Tier Management ----

    def demote_to_l2(self, l1_id: int, reason: str = "decay") -> bool:
        """Move a chunk from L1 to L2 (compress and summarize)."""
        record = self.db.get_l1(l1_id)
        if not record:
            return False

        # Compress content
        summary = record.get("summary_content") or record["raw_content"][:200]
        compressed = zlib.compress(summary.encode("utf-8"))

        # Compress vector (keep as-is for L2, quantize for L3)
        vec_bytes = record.get("vector", b"")

        self.db.insert_l2(
            l1_id=l1_id,
            content_hash=record["content_hash"],
            compressed_content=compressed,
            compressed_vector=vec_bytes,
            summary=summary,
        )

        # Update tier
        self.db.update_l1_tier(l1_id, 2)
        self.db.log_transition(record["content_hash"], 1, 2, reason)

        return True

    def demote_to_l3(self, l1_id: int, reason: str = "decay") -> bool:
        """Move from L2 to L3 (archive with minimal data)."""
        record = self.db.get_l1(l1_id)
        if not record:
            return False

        overview = record.get("overview_content") or record.get("summary_content", "")[:100]
        key_concepts = self.dedup.compute_semantic_hash(record["raw_content"])

        # Quantize vector to int8 for L3
        vec_bytes = record.get("vector", b"")
        if vec_bytes:
            vec = bytes_to_vector(vec_bytes)
            quantized = np.clip(vec * 127, -128, 127).astype(np.int8)
            vec_bytes = quantized.tobytes()

        archive_path = self.config.get("memory", {}).get("tiers", {}).get(
            "l3", {}
        ).get("archive_path", "./archives/")
        Path(archive_path).mkdir(parents=True, exist_ok=True)
        full_path = str(Path(archive_path) / f"{record['content_hash'][:16]}.zlib")

        # Archive full content
        compressed_full = zlib.compress(record["raw_content"].encode("utf-8"))
        with open(full_path, "wb") as f:
            f.write(compressed_full)

        self.db.insert_l3(
            content_hash=record["content_hash"],
            key_concepts=key_concepts,
            overview=overview,
            archive_path=full_path,
            compressed_vector=vec_bytes,
        )

        self.db.update_l1_tier(l1_id, 3)
        self.db.log_transition(record["content_hash"], 2, 3, reason)

        return True

    def promote_to_l1(self, content_hash: str) -> Optional[Dict]:
        """Restore archived content back to L1."""
        # Find in L3
        rows = self.db.conn.execute(
            "SELECT * FROM l3_long_term WHERE content_hash = ?",
            (content_hash,)
        ).fetchall()

        if not rows:
            return None

        row = dict(rows[0])

        # Decompress
        if row.get("archive_path"):
            try:
                with open(row["archive_path"], "rb") as f:
                    data = f.read()
                full_content = zlib.decompress(data).decode("utf-8")
            except Exception:
                full_content = row.get("overview", "")
        else:
            full_content = row.get("overview", "")

        # Re-embed
        vec = self.embedder.embed(full_content)
        vec_bytes = vector_to_bytes(vec)

        # Update L1 record
        l1_rec = self.db.get_l1_by_hash(content_hash)
        if l1_rec:
            self.db.update_l1_tier(l1_rec["id"], 1)
            self.db.update_l1_access(l1_rec["id"])
            self.db.log_transition(content_hash, 3, 1, "promoted")
            return self.db.get_l1(l1_rec["id"])

        return None

    def run_maintenance(self) -> Dict:
        """Run periodic maintenance: decay, demote, archive."""
        l1_config = self.memory_config.get("l1", {})
        l2_config = self.memory_config.get("l2", {})

        l1_retention = l1_config.get("retention_days", 7)
        l2_retention = l2_config.get("retention_days", 30)

        stats = {"demoted_l1_to_l2": 0, "demoted_l2_to_l3": 0}

        # Demote stale L1 to L2
        stale_l1 = self.db.get_decayed_l1(l1_retention)
        for rec in stale_l1:
            if self.demote_to_l2(rec["id"]):
                stats["demoted_l1_to_l2"] += 1

        # Demote stale L2 to L3
        stale_l2 = self.db.get_decayed_l2(l2_retention)
        for rec in stale_l2:
            l1_rec = self.db.get_l1(rec.get("l1_id", 0))
            if l1_rec and self.demote_to_l3(l1_rec["id"]):
                stats["demoted_l2_to_l3"] += 1

        return stats

    def get_stats(self) -> Dict:
        """Get memory tier statistics."""
        db_stats = self.db.get_stats()
        db_stats["ollama_available"] = self.embedder.is_available()
        db_stats["embedding_model"] = self.embedder.model
        return db_stats
