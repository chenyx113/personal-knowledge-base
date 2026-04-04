"""
database.py - SQLite database layer for the knowledge base.
Handles schema creation, CRUD operations, and FTS5 full-text search.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


DB_SCHEMA = """
-- Documents table: stores original file metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_type TEXT,
    title TEXT,
    author TEXT,
    tags TEXT DEFAULT '[]',
    summary TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_indexed REAL,
    content_hash TEXT
);

-- L1 Working Memory (hot data, full content)
CREATE TABLE IF NOT EXISTS l1_working_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER DEFAULT 0,
    content_hash TEXT NOT NULL,
    semantic_hash TEXT,
    content_type TEXT DEFAULT 'chunk',

    raw_content TEXT NOT NULL,
    summary_content TEXT,
    overview_content TEXT,

    vector BLOB,
    vector_dim INTEGER,

    access_count INTEGER DEFAULT 0,
    last_accessed REAL,
    created_at REAL NOT NULL,

    duplicate_group_id INTEGER,
    similarity_score REAL,

    memory_tier INTEGER DEFAULT 1,
    decay_score REAL DEFAULT 1.0,
    start_line INTEGER,
    end_line INTEGER
);

-- L2 Short-term Memory (warm data, summaries)
CREATE TABLE IF NOT EXISTS l2_short_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    l1_id INTEGER,
    content_hash TEXT NOT NULL,
    compressed_content BLOB,
    compressed_vector BLOB,
    summary TEXT,
    created_at REAL NOT NULL,
    last_accessed REAL
);

-- L3 Long-term Memory (cold data, overviews only)
CREATE TABLE IF NOT EXISTS l3_long_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE,
    key_concepts TEXT,
    overview TEXT,
    archive_path TEXT,
    compressed_vector BLOB,
    last_activated REAL,
    activation_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

-- Duplicate clusters for deduplication
CREATE TABLE IF NOT EXISTS duplicate_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_hash TEXT NOT NULL,
    cluster_size INTEGER DEFAULT 1,
    merge_strategy TEXT DEFAULT 'keep_newest',
    created_at REAL NOT NULL
);

-- Duplicate cluster members
CREATE TABLE IF NOT EXISTS duplicate_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER REFERENCES duplicate_clusters(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    similarity REAL,
    is_canonical INTEGER DEFAULT 0
);

-- Memory transition log
CREATE TABLE IF NOT EXISTS memory_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    from_tier INTEGER,
    to_tier INTEGER,
    reason TEXT,
    transitioned_at REAL NOT NULL
);

-- Search history
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    clicked_doc_id INTEGER,
    timestamp REAL NOT NULL
);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    color TEXT DEFAULT '#6366f1',
    description TEXT
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
    content,
    content_hash,
    tokenize='unicode61'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_l1_content_hash ON l1_working_memory(content_hash);
CREATE INDEX IF NOT EXISTS idx_l1_semantic_hash ON l1_working_memory(semantic_hash);
CREATE INDEX IF NOT EXISTS idx_l1_memory_tier ON l1_working_memory(memory_tier);
CREATE INDEX IF NOT EXISTS idx_l1_last_accessed ON l1_working_memory(last_accessed);
CREATE INDEX IF NOT EXISTS idx_l1_doc_id ON l1_working_memory(doc_id);
CREATE INDEX IF NOT EXISTS idx_l2_content_hash ON l2_short_term(content_hash);
CREATE INDEX IF NOT EXISTS idx_l3_content_hash ON l3_long_term(content_hash);
CREATE INDEX IF NOT EXISTS idx_docs_file_path ON documents(file_path);
CREATE INDEX IF NOT EXISTS idx_dup_members_cluster ON duplicate_members(cluster_id);
"""


class Database:
    """SQLite database manager for the knowledge base."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- Document CRUD ----

    def insert_document(self, file_path: str, file_type: str, title: str = "",
                        author: str = "", tags: List[str] = None,
                        summary: str = "", content_hash: str = "") -> int:
        now = time.time()
        tags_json = json.dumps(tags or [])
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO documents
               (file_path, file_type, title, author, tags, summary,
                created_at, updated_at, last_indexed, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_path, file_type, title, author, tags_json, summary,
             now, now, now, content_hash)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_document(self, doc_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_path(self, path: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int):
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()

    def count_documents(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # ---- L1 Memory CRUD ----

    def insert_l1(self, doc_id: int, chunk_index: int, content_hash: str,
                  semantic_hash: str, raw_content: str, summary_content: str,
                  overview_content: str, vector: bytes, vector_dim: int,
                  start_line: int = 0, end_line: int = 0) -> int:
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO l1_working_memory
               (doc_id, chunk_index, content_hash, semantic_hash, content_type,
                raw_content, summary_content, overview_content,
                vector, vector_dim, access_count, last_accessed, created_at,
                memory_tier, decay_score, start_line, end_line)
               VALUES (?, ?, ?, ?, 'chunk', ?, ?, ?, ?, ?, 0, ?, ?, 1, 1.0, ?, ?)""",
            (doc_id, chunk_index, content_hash, semantic_hash,
             raw_content, summary_content, overview_content,
             vector, vector_dim, now, now, start_line, end_line)
        )
        # Insert into FTS
        self.conn.execute(
            "INSERT INTO fts_content (rowid, content, content_hash) VALUES (?, ?, ?)",
            (cur.lastrowid, raw_content, content_hash)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_l1(self, l1_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM l1_working_memory WHERE id = ?", (l1_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_l1_by_hash(self, content_hash: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM l1_working_memory WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_l1(self, tier: int = None) -> List[Dict]:
        if tier:
            rows = self.conn.execute(
                "SELECT * FROM l1_working_memory WHERE memory_tier = ?", (tier,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM l1_working_memory"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_l1_access(self, l1_id: int):
        self.conn.execute(
            """UPDATE l1_working_memory
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (time.time(), l1_id)
        )
        self.conn.commit()

    def update_l1_tier(self, l1_id: int, new_tier: int):
        self.conn.execute(
            "UPDATE l1_working_memory SET memory_tier = ? WHERE id = ?",
            (new_tier, l1_id)
        )
        self.conn.commit()

    def delete_l1(self, l1_id: int):
        self.conn.execute(
            "DELETE FROM fts_content WHERE rowid = ?", (l1_id,)
        )
        self.conn.execute(
            "DELETE FROM l1_working_memory WHERE id = ?", (l1_id,)
        )
        self.conn.commit()

    def count_l1(self, tier: int = None) -> int:
        if tier:
            return self.conn.execute(
                "SELECT COUNT(*) FROM l1_working_memory WHERE memory_tier = ?",
                (tier,)
            ).fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM l1_working_memory"
        ).fetchone()[0]

    def get_decayed_l1(self, retention_days: int) -> List[Dict]:
        threshold = time.time() - (retention_days * 86400)
        rows = self.conn.execute(
            """SELECT * FROM l1_working_memory
               WHERE memory_tier = 1 AND last_accessed < ?""",
            (threshold,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- L2 Memory ----

    def insert_l2(self, l1_id: int, content_hash: str,
                  compressed_content: bytes, compressed_vector: bytes,
                  summary: str) -> int:
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO l2_short_term
               (l1_id, content_hash, compressed_content, compressed_vector,
                summary, created_at, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (l1_id, content_hash, compressed_content, compressed_vector,
             summary, now, now)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_l2(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM l2_short_term").fetchall()
        return [dict(r) for r in rows]

    def get_decayed_l2(self, retention_days: int) -> List[Dict]:
        threshold = time.time() - (retention_days * 86400)
        rows = self.conn.execute(
            "SELECT * FROM l2_short_term WHERE last_accessed < ?",
            (threshold,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_l2(self, l2_id: int):
        self.conn.execute("DELETE FROM l2_short_term WHERE id = ?", (l2_id,))
        self.conn.commit()

    def count_l2(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM l2_short_term").fetchone()[0]

    # ---- L3 Memory ----

    def insert_l3(self, content_hash: str, key_concepts: str,
                  overview: str, archive_path: str,
                  compressed_vector: bytes) -> int:
        now = time.time()
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO l3_long_term
               (content_hash, key_concepts, overview, archive_path,
                compressed_vector, last_activated, activation_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (content_hash, key_concepts, overview, archive_path,
             compressed_vector, now, now)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_l3(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM l3_long_term").fetchall()
        return [dict(r) for r in rows]

    def count_l3(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM l3_long_term").fetchone()[0]

    # ---- FTS Search ----

    def fts_search(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search using FTS5."""
        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')
        try:
            rows = self.conn.execute(
                """SELECT rowid, content, content_hash,
                          rank as fts_rank
                   FROM fts_content
                   WHERE fts_content MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback: search with LIKE
            rows = self.conn.execute(
                """SELECT rowid, content, content_hash, 0 as fts_rank
                   FROM fts_content
                   WHERE content LIKE ?
                   LIMIT ?""",
                (f"%{query}%", limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Duplicate Clusters ----

    def insert_duplicate_cluster(self, canonical_hash: str,
                                  merge_strategy: str = "keep_newest") -> int:
        cur = self.conn.execute(
            """INSERT INTO duplicate_clusters
               (canonical_hash, cluster_size, merge_strategy, created_at)
               VALUES (?, 1, ?, ?)""",
            (canonical_hash, merge_strategy, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def add_duplicate_member(self, cluster_id: int, content_hash: str,
                              similarity: float, is_canonical: bool = False):
        self.conn.execute(
            """INSERT INTO duplicate_members
               (cluster_id, content_hash, similarity, is_canonical)
               VALUES (?, ?, ?, ?)""",
            (cluster_id, content_hash, similarity, 1 if is_canonical else 0)
        )
        self.conn.execute(
            "UPDATE duplicate_clusters SET cluster_size = cluster_size + 1 WHERE id = ?",
            (cluster_id,)
        )
        self.conn.commit()

    def get_duplicate_cluster_for(self, content_hash: str) -> Optional[Dict]:
        row = self.conn.execute(
            """SELECT dc.* FROM duplicate_clusters dc
               JOIN duplicate_members dm ON dc.id = dm.cluster_id
               WHERE dm.content_hash = ?""",
            (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM duplicate_members WHERE cluster_id = ?",
            (cluster_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Memory Transitions ----

    def log_transition(self, content_hash: str, from_tier: int,
                       to_tier: int, reason: str):
        self.conn.execute(
            """INSERT INTO memory_transitions
               (content_hash, from_tier, to_tier, reason, transitioned_at)
               VALUES (?, ?, ?, ?, ?)""",
            (content_hash, from_tier, to_tier, reason, time.time())
        )
        self.conn.commit()

    # ---- Search Logs ----

    def log_search(self, query: str, results_count: int, clicked_id: int = None):
        self.conn.execute(
            """INSERT INTO search_logs (query, results_count, clicked_doc_id, timestamp)
               VALUES (?, ?, ?, ?)""",
            (query, results_count, clicked_id, time.time())
        )
        self.conn.commit()

    # ---- Tags ----

    def insert_tag(self, name: str, color: str = "#6366f1",
                   description: str = "") -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO tags (name, color, description) VALUES (?, ?, ?)",
            (name, color, description)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_tags(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM tags ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # ---- Stats ----

    def get_stats(self) -> Dict:
        return {
            "documents": self.count_documents(),
            "l1_chunks": self.count_l1(tier=1),
            "l2_chunks": self.count_l2(),
            "l3_chunks": self.count_l3(),
            "total_chunks": self.count_l1(),
            "tags": self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0],
            "searches": self.conn.execute("SELECT COUNT(*) FROM search_logs").fetchone()[0],
        }
