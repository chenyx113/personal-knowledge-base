"""
test_all.py - Comprehensive tests for the Knowledge Base system.
Tests all core components: database, chunker, embedder, dedup, memory, retriever, web API.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database
from src.chunker import (
    chunk_text, chunk_markdown, chunk_python, chunk_sliding_window,
    compute_content_hash, ChunkResult
)
from src.embedder import (
    OllamaEmbedder, OllamaGenerator, vector_to_bytes, bytes_to_vector,
    cosine_similarity
)
from src.dedup import SimHash, SemanticHasher, DeduplicationEngine
from src.memory_manager import MemoryManager
from src.retriever import HybridRetriever, SearchResult
import numpy as np


class TestDatabase(unittest.TestCase):
    """Test SQLite database operations."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_init_schema(self):
        """Schema tables exist after init."""
        tables = self.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t["name"] for t in tables}
        self.assertIn("documents", names)
        self.assertIn("l1_working_memory", names)
        self.assertIn("l2_short_term", names)
        self.assertIn("l3_long_term", names)
        self.assertIn("fts_content", names)

    def test_insert_and_get_document(self):
        doc_id = self.db.insert_document(
            file_path="/test/doc.md", file_type="md",
            title="Test Doc", tags=["python", "ml"]
        )
        self.assertGreater(doc_id, 0)
        doc = self.db.get_document(doc_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["title"], "Test Doc")
        self.assertEqual(doc["file_type"], "md")
        tags = json.loads(doc["tags"])
        self.assertIn("python", tags)

    def test_get_document_by_path(self):
        self.db.insert_document("/test/x.py", "py", title="X")
        doc = self.db.get_document_by_path("/test/x.py")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["title"], "X")

    def test_list_documents(self):
        self.db.insert_document("/a.md", "md", title="A")
        self.db.insert_document("/b.py", "py", title="B")
        docs = self.db.list_documents()
        self.assertEqual(len(docs), 2)

    def test_delete_document(self):
        did = self.db.insert_document("/del.md", "md")
        self.db.delete_document(did)
        self.assertIsNone(self.db.get_document(did))

    def test_count_documents(self):
        self.assertEqual(self.db.count_documents(), 0)
        self.db.insert_document("/c.md", "md")
        self.assertEqual(self.db.count_documents(), 1)

    def test_insert_and_get_l1(self):
        did = self.db.insert_document("/test.md", "md")
        vec = np.random.randn(768).astype(np.float32)
        l1_id = self.db.insert_l1(
            doc_id=did, chunk_index=0,
            content_hash="abc123", semantic_hash="sem456",
            raw_content="Hello world test content",
            summary_content="Hello summary",
            overview_content="Hello overview",
            vector=vector_to_bytes(vec), vector_dim=768
        )
        self.assertGreater(l1_id, 0)
        rec = self.db.get_l1(l1_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["raw_content"], "Hello world test content")
        self.assertEqual(rec["memory_tier"], 1)

    def test_l1_access_update(self):
        did = self.db.insert_document("/t.md", "md")
        vec = np.zeros(10, dtype=np.float32)
        l1_id = self.db.insert_l1(did, 0, "h", "s", "content", "", "", vector_to_bytes(vec), 10)
        self.db.update_l1_access(l1_id)
        rec = self.db.get_l1(l1_id)
        self.assertEqual(rec["access_count"], 1)

    def test_fts_search(self):
        did = self.db.insert_document("/fts.md", "md")
        vec = np.zeros(10, dtype=np.float32)
        self.db.insert_l1(did, 0, "h1", "s", "machine learning is great", "", "",
                          vector_to_bytes(vec), 10)
        self.db.insert_l1(did, 1, "h2", "s", "deep learning neural networks", "", "",
                          vector_to_bytes(vec), 10)
        results = self.db.fts_search("machine learning")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("machine", results[0]["content"])

    def test_stats(self):
        stats = self.db.get_stats()
        self.assertIn("documents", stats)
        self.assertIn("l1_chunks", stats)
        self.assertIn("total_chunks", stats)

    def test_tags(self):
        self.db.insert_tag("python", "#3776AB")
        self.db.insert_tag("ml", "#FF6F00")
        tags = self.db.get_all_tags()
        self.assertEqual(len(tags), 2)

    def test_duplicate_clusters(self):
        cid = self.db.insert_duplicate_cluster("canon_hash", "keep_newest")
        self.db.add_duplicate_member(cid, "canon_hash", 1.0, is_canonical=True)
        self.db.add_duplicate_member(cid, "dup_hash", 0.95)
        members = self.db.get_cluster_members(cid)
        self.assertEqual(len(members), 2)
        cluster = self.db.get_duplicate_cluster_for("dup_hash")
        self.assertIsNotNone(cluster)


class TestChunker(unittest.TestCase):
    """Test document chunking strategies."""

    def test_markdown_chunking(self):
        md = """# Title
Some intro text here.

## Section One
Content of section one with details.

## Section Two
Content of section two with more details.

### Subsection
Deep content here.
"""
        chunks = chunk_markdown(md, max_chunk_size=200)
        self.assertGreater(len(chunks), 0)
        # All chunks should have content
        for c in chunks:
            self.assertTrue(c.content.strip())
            self.assertEqual(c.chunk_type, "markdown")

    def test_python_chunking(self):
        py_code = '''
def hello():
    """Say hello."""
    print("Hello, world!")

class MyClass:
    """A test class."""
    def method(self):
        return 42

def another_func(x, y):
    return x + y
'''
        chunks = chunk_python(py_code, max_chunk_size=500)
        self.assertGreater(len(chunks), 0)
        # Should find functions and class
        types = [c.chunk_type for c in chunks]
        self.assertTrue(any("function" in t for t in types) or any("class" in t for t in types))

    def test_sliding_window(self):
        text = "Line one\n" * 100
        chunks = chunk_sliding_window(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Every chunk has content
        for c in chunks:
            self.assertTrue(c.content.strip())

    def test_empty_text(self):
        self.assertEqual(chunk_text("", "md"), [])
        self.assertEqual(chunk_text("   \n  ", "py"), [])

    def test_chunk_text_routing(self):
        md_chunks = chunk_text("# Test\nContent", "md", chunk_size=1000)
        self.assertGreater(len(md_chunks), 0)

        py_chunks = chunk_text("def f(): pass", "py", chunk_size=1000)
        self.assertGreater(len(py_chunks), 0)

        txt_chunks = chunk_text("Hello world test", "txt", chunk_size=1000)
        self.assertGreater(len(txt_chunks), 0)

    def test_content_hash(self):
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        h3 = compute_content_hash("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_chunk_result_to_dict(self):
        cr = ChunkResult("test content", 0, 1, 5, "text", "heading")
        d = cr.to_dict()
        self.assertEqual(d["content"], "test content")
        self.assertEqual(d["index"], 0)
        self.assertEqual(d["heading"], "heading")

    def test_large_markdown_splitting(self):
        """Large sections should be split by sliding window."""
        md = "# Big Section\n" + ("word " * 200) + "\n# Another\nSmall."
        chunks = chunk_markdown(md, max_chunk_size=200)
        self.assertGreater(len(chunks), 1)


class TestEmbedder(unittest.TestCase):
    """Test embedding generation (fallback mode, no Ollama needed)."""

    def setUp(self):
        self.embedder = OllamaEmbedder(host="http://localhost:99999", model="test")
        # Force fallback mode
        self.embedder._ollama_available = False

    def test_fallback_embed(self):
        vec = self.embedder.embed("Hello world test content")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(len(vec), 768)

    def test_fallback_consistency(self):
        vec1 = self.embedder.embed("Same text here")
        vec2 = self.embedder.embed("Same text here")
        np.testing.assert_array_equal(vec1, vec2)

    def test_different_texts_different_vectors(self):
        vec1 = self.embedder.embed("Machine learning algorithms")
        vec2 = self.embedder.embed("Cooking recipes for dinner")
        # Should not be identical
        self.assertFalse(np.array_equal(vec1, vec2))

    def test_empty_text(self):
        vec = self.embedder.embed("")
        self.assertEqual(len(vec), 768)
        self.assertTrue(np.allclose(vec, 0))

    def test_batch_embed(self):
        texts = ["Hello", "World", "Test"]
        vecs = self.embedder.embed_batch(texts)
        self.assertEqual(len(vecs), 3)
        for v in vecs:
            self.assertEqual(len(v), 768)

    def test_vector_serialization(self):
        vec = np.random.randn(768).astype(np.float32)
        b = vector_to_bytes(vec)
        recovered = bytes_to_vector(b)
        np.testing.assert_array_almost_equal(vec, recovered)

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0)

    def test_cosine_similarity_edge_cases(self):
        self.assertEqual(cosine_similarity(None, None), 0.0)
        zero = np.zeros(3, dtype=np.float32)
        self.assertEqual(cosine_similarity(zero, zero), 0.0)

    def test_generator_fallback(self):
        gen = OllamaGenerator(host="http://localhost:99999")
        gen._available = False
        summary = gen.generate_summary("This is a test document about machine learning.")
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)


class TestDedup(unittest.TestCase):
    """Test deduplication engine."""

    def test_simhash_identical(self):
        sh = SimHash(bits=64)
        h1 = sh.compute("The quick brown fox jumps over the lazy dog")
        h2 = sh.compute("The quick brown fox jumps over the lazy dog")
        self.assertEqual(h1, h2)

    def test_simhash_similar(self):
        sh = SimHash(bits=64)
        h1 = sh.compute("The quick brown fox jumps over the lazy dog")
        h2 = sh.compute("The quick brown fox leaps over the lazy dog")
        distance = sh.hamming_distance(h1, h2)
        self.assertLess(distance, 10)  # Should be fairly close

    def test_simhash_different(self):
        sh = SimHash(bits=64)
        h1 = sh.compute("Machine learning neural networks deep learning")
        h2 = sh.compute("Cooking pasta recipes Italian food")
        distance = sh.hamming_distance(h1, h2)
        self.assertGreater(distance, 5)

    def test_semantic_hasher(self):
        hasher = SemanticHasher()
        h1 = hasher.compute("Python machine learning deep neural networks")
        h2 = hasher.compute("Python machine learning deep neural networks")
        self.assertEqual(h1, h2)

    def test_dedup_engine_exact_match(self):
        engine = DeduplicationEngine()
        existing = [{
            "id": 1,
            "content_hash": engine.compute_content_hash("test content"),
            "semantic_hash": "",
            "vector": vector_to_bytes(np.random.randn(768).astype(np.float32)),
            "raw_content": "test content",
        }]
        new_vec = np.random.randn(768).astype(np.float32)
        result = engine.check_duplicate("test content", new_vec, existing)
        self.assertTrue(result["is_duplicate"])
        self.assertEqual(result["level"], "exact")

    def test_dedup_engine_no_match(self):
        engine = DeduplicationEngine()
        existing = [{
            "id": 1,
            "content_hash": engine.compute_content_hash("existing content about cooking"),
            "semantic_hash": "",
            "vector": vector_to_bytes(np.random.randn(768).astype(np.float32)),
            "raw_content": "existing content about cooking",
        }]
        new_vec = np.random.randn(768).astype(np.float32)
        result = engine.check_duplicate("completely different topic xyz", new_vec, existing)
        self.assertFalse(result["is_duplicate"])

    def test_merge_strategy_recommendations(self):
        engine = DeduplicationEngine()
        self.assertEqual(engine.recommend_merge_strategy(0.99), "exact_duplicate")
        self.assertEqual(engine.recommend_merge_strategy(0.96), "incremental_update")
        self.assertEqual(engine.recommend_merge_strategy(0.91), "soft_reference")
        self.assertEqual(engine.recommend_merge_strategy(0.80), "distinct")

    def test_find_similar_pairs(self):
        engine = DeduplicationEngine()
        vec = np.ones(10, dtype=np.float32)
        records = [
            {"id": 1, "vector": vector_to_bytes(vec)},
            {"id": 2, "vector": vector_to_bytes(vec * 0.99)},
            {"id": 3, "vector": vector_to_bytes(-vec)},
        ]
        pairs = engine.find_similar_pairs(records, threshold=0.9)
        self.assertGreater(len(pairs), 0)
        self.assertEqual(pairs[0][0], 1)
        self.assertEqual(pairs[0][1], 2)


class TestMemoryManager(unittest.TestCase):
    """Test tiered memory management."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)
        self.config = {
            "ollama": {"host": "http://localhost:99999", "models": {}},
            "memory": {
                "tiers": {
                    "l1": {"retention_days": 7},
                    "l2": {"retention_days": 30},
                    "l3": {"archive_path": tempfile.mkdtemp()},
                },
                "deduplication": {
                    "simhash_threshold": 3,
                    "vector_threshold": 0.92,
                    "auto_merge": False,
                },
            },
            "chunking": {"default_chunk_size": 200, "chunk_overlap": 20},
            "search": {"default_top_k": 10, "rrf_k": 60},
        }
        self.mm = MemoryManager(self.db, self.config)
        # Force fallback mode
        self.mm.embedder._ollama_available = False
        self.mm.generator._available = False

        # Create test files
        self.test_dir = tempfile.mkdtemp()
        self.test_md = os.path.join(self.test_dir, "test.md")
        with open(self.test_md, "w") as f:
            f.write("""# Test Document

## Introduction
This is a test document about machine learning and neural networks.
It contains multiple sections for testing the chunking system.

## Methods
We use gradient descent optimization with batch normalization.
The learning rate is 0.001 with Adam optimizer.

## Results
Our model achieves 95% accuracy on the test set.
The training converged in 50 epochs.

## Conclusion
Machine learning is powerful for pattern recognition tasks.
""")

        self.test_py = os.path.join(self.test_dir, "example.py")
        with open(self.test_py, "w") as f:
            f.write('''"""Example module for testing."""

def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

class Calculator:
    """A simple calculator."""
    def __init__(self):
        self.history = []

    def compute(self, op, a, b):
        if op == "add":
            result = add(a, b)
        else:
            result = multiply(a, b)
        self.history.append(result)
        return result
''')

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        archive_path = self.config["memory"]["tiers"]["l3"]["archive_path"]
        shutil.rmtree(archive_path, ignore_errors=True)

    def test_ingest_markdown_file(self):
        result = self.mm.ingest_file(self.test_md, tags=["test", "ml"])
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["chunks_added"], 0)
        self.assertGreater(result["doc_id"], 0)

        doc = self.db.get_document(result["doc_id"])
        self.assertIsNotNone(doc)
        tags = json.loads(doc["tags"])
        self.assertIn("test", tags)

    def test_ingest_python_file(self):
        result = self.mm.ingest_file(self.test_py)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["chunks_added"], 0)

    def test_ingest_nonexistent_file(self):
        result = self.mm.ingest_file("/nonexistent/file.md")
        self.assertEqual(result["status"], "error")

    def test_ingest_unchanged_file(self):
        self.mm.ingest_file(self.test_md)
        result = self.mm.ingest_file(self.test_md)
        self.assertEqual(result["status"], "unchanged")

    def test_ingest_directory(self):
        result = self.mm.ingest_directory(self.test_dir)
        self.assertGreater(result["success"], 0)
        self.assertEqual(result["failed"], 0)

    def test_demote_to_l2(self):
        self.mm.ingest_file(self.test_md)
        l1_records = self.db.get_all_l1(tier=1)
        self.assertGreater(len(l1_records), 0)

        l1_id = l1_records[0]["id"]
        success = self.mm.demote_to_l2(l1_id)
        self.assertTrue(success)

        rec = self.db.get_l1(l1_id)
        self.assertEqual(rec["memory_tier"], 2)
        self.assertGreater(self.db.count_l2(), 0)

    def test_demote_to_l3(self):
        self.mm.ingest_file(self.test_md)
        l1_records = self.db.get_all_l1(tier=1)
        l1_id = l1_records[0]["id"]

        self.mm.demote_to_l2(l1_id)
        success = self.mm.demote_to_l3(l1_id)
        self.assertTrue(success)

        rec = self.db.get_l1(l1_id)
        self.assertEqual(rec["memory_tier"], 3)

    def test_get_stats(self):
        self.mm.ingest_file(self.test_md)
        stats = self.mm.get_stats()
        self.assertGreater(stats["documents"], 0)
        self.assertGreater(stats["l1_chunks"], 0)
        self.assertIn("ollama_available", stats)

    def test_run_maintenance(self):
        result = self.mm.run_maintenance()
        self.assertIn("demoted_l1_to_l2", result)
        self.assertIn("demoted_l2_to_l3", result)


class TestRetriever(unittest.TestCase):
    """Test hybrid search and retrieval."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)
        self.config = {
            "ollama": {"host": "http://localhost:99999", "models": {}},
            "memory": {
                "tiers": {"l1": {"retention_days": 7}, "l2": {"retention_days": 30},
                          "l3": {"archive_path": tempfile.mkdtemp()}},
                "deduplication": {"simhash_threshold": 3, "vector_threshold": 0.92, "auto_merge": False},
            },
            "chunking": {"default_chunk_size": 200, "chunk_overlap": 20},
            "search": {"default_top_k": 10, "rrf_k": 60, "auto_tier_fallback": True},
        }
        self.mm = MemoryManager(self.db, self.config)
        self.mm.embedder._ollama_available = False
        self.mm.generator._available = False
        self.retriever = HybridRetriever(self.db, self.mm.embedder, self.config)

        # Create and ingest test content
        self.test_dir = tempfile.mkdtemp()

        f1 = os.path.join(self.test_dir, "ml_notes.md")
        with open(f1, "w") as f:
            f.write("""# Machine Learning Notes

## Supervised Learning
Classification and regression are supervised learning tasks.
Decision trees, random forests, and SVMs are classic algorithms.

## Deep Learning
Neural networks with multiple layers can learn complex patterns.
Convolutional networks excel at image recognition.
Recurrent networks are good for sequence data.
""")

        f2 = os.path.join(self.test_dir, "python_guide.md")
        with open(f2, "w") as f:
            f.write("""# Python Guide

## Functions
Functions are defined with the def keyword.
They can take arguments and return values.

## Classes
Classes define objects with attributes and methods.
Inheritance allows code reuse.
""")

        self.mm.ingest_file(f1, tags=["ml"])
        self.mm.ingest_file(f2, tags=["python"])

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_vector_search(self):
        results = self.retriever.search("neural networks deep learning", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], SearchResult)

    def test_hybrid_search(self):
        results = self.retriever.search("supervised learning classification", top_k=5)
        self.assertGreater(len(results), 0)
        # Check result structure
        rd = results[0].to_dict()
        self.assertIn("content", rd)
        self.assertIn("score", rd)
        self.assertIn("tier", rd)

    def test_empty_query_results(self):
        results = self.retriever.search("xyznonexistenttopic12345", top_k=5)
        # Should still return results (vector search always returns something)
        # But scores should be low
        if results:
            self.assertIsInstance(results[0].score, float)

    def test_tier_specific_search(self):
        results = self.retriever.search("python functions", top_k=5, tier=1)
        for r in results:
            self.assertEqual(r.tier, 1)

    def test_search_result_to_dict(self):
        r = SearchResult(
            l1_id=1, content="test", score=0.85, tier=1,
            source="hybrid", doc_path="/test.md", title="Test"
        )
        d = r.to_dict()
        self.assertEqual(d["id"], 1)
        self.assertEqual(d["score"], 0.85)
        self.assertEqual(d["title"], "Test")

    def test_search_updates_access(self):
        # Get initial access counts
        results = self.retriever.search("deep learning", top_k=3)
        if results:
            rec = self.db.get_l1(results[0].l1_id)
            self.assertGreater(rec["access_count"], 0)

    def test_search_logging(self):
        self.retriever.search("test query")
        logs = self.db.conn.execute(
            "SELECT * FROM search_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(logs)
        self.assertEqual(dict(logs)["query"], "test query")


class TestWebApp(unittest.TestCase):
    """Test FastAPI web application."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.config = {
            "ollama": {"host": "http://localhost:99999", "models": {}},
            "memory": {
                "tiers": {"l1": {"retention_days": 7}, "l2": {"retention_days": 30},
                          "l3": {"archive_path": tempfile.mkdtemp()}},
                "deduplication": {"simhash_threshold": 3, "vector_threshold": 0.92, "auto_merge": False},
            },
            "chunking": {"default_chunk_size": 200, "chunk_overlap": 20},
            "search": {"default_top_k": 10, "rrf_k": 60, "auto_tier_fallback": True},
        }

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_create_app(self):
        from src.web_app import create_app
        app = create_app(self.tmp.name, self.config)
        self.assertIsNotNone(app)

    def test_api_routes_exist(self):
        from src.web_app import create_app
        app = create_app(self.tmp.name, self.config)
        routes = [r.path for r in app.routes]
        self.assertIn("/", routes)
        self.assertIn("/api/stats", routes)
        self.assertIn("/api/search", routes)
        self.assertIn("/api/documents", routes)

    def test_frontend_html(self):
        from src.web_app import get_frontend_html
        html = get_frontend_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Knowledge Base", html)
        self.assertIn("search", html.lower())


class TestCLI(unittest.TestCase):
    """Test CLI module imports and configuration."""

    def test_load_config_missing_file(self):
        from src.cli import load_config
        config = load_config("/nonexistent/config.yaml")
        self.assertEqual(config, {})

    def test_load_config_valid(self):
        from src.cli import load_config
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
        tmp.write("ollama:\n  host: 'http://localhost:11434'\n")
        tmp.close()
        config = load_config(tmp.name)
        self.assertIn("ollama", config)
        os.unlink(tmp.name)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)
        self.archive_dir = tempfile.mkdtemp()
        self.config = {
            "ollama": {"host": "http://localhost:99999", "models": {}},
            "memory": {
                "tiers": {"l1": {"retention_days": 7}, "l2": {"retention_days": 30},
                          "l3": {"archive_path": self.archive_dir}},
                "deduplication": {"simhash_threshold": 3, "vector_threshold": 0.92, "auto_merge": False},
            },
            "chunking": {"default_chunk_size": 300, "chunk_overlap": 30},
            "search": {"default_top_k": 10, "rrf_k": 60, "auto_tier_fallback": True},
        }
        self.mm = MemoryManager(self.db, self.config)
        self.mm.embedder._ollama_available = False
        self.mm.generator._available = False
        self.retriever = HybridRetriever(self.db, self.mm.embedder, self.config)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.archive_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test: ingest -> search -> access -> demote -> search again."""
        # 1. Create test file
        f = os.path.join(self.test_dir, "notes.md")
        with open(f, "w") as fh:
            fh.write("""# Deep Learning Notes

## Transformer Architecture
The transformer uses self-attention mechanisms.
Multi-head attention allows learning different representation subspaces.

## Training Tips
Use learning rate warmup and cosine decay.
Gradient clipping prevents exploding gradients.
""")

        # 2. Ingest
        result = self.mm.ingest_file(f, tags=["deep-learning", "transformers"])
        self.assertEqual(result["status"], "success")
        doc_id = result["doc_id"]

        # 3. Search
        results = self.retriever.search("transformer attention mechanism", top_k=5)
        self.assertGreater(len(results), 0)

        # 4. Verify access count increased
        l1_id = results[0].l1_id
        rec = self.db.get_l1(l1_id)
        self.assertGreater(rec["access_count"], 0)

        # 5. Demote to L2
        success = self.mm.demote_to_l2(l1_id)
        self.assertTrue(success)
        rec = self.db.get_l1(l1_id)
        self.assertEqual(rec["memory_tier"], 2)

        # 6. Search still works (auto fallback)
        results2 = self.retriever.search("transformer attention", top_k=5)
        self.assertGreater(len(results2), 0)

        # 7. Stats
        stats = self.mm.get_stats()
        self.assertGreater(stats["documents"], 0)

    def test_multi_file_ingest_and_search(self):
        """Ingest multiple files, search across them."""
        files = {
            "python.md": "# Python\nPython is a programming language.\nUsed for ML and web dev.",
            "rust.md": "# Rust\nRust is a systems programming language.\nMemory safety without GC.",
            "go.md": "# Go\nGo is for concurrent programming.\nGoroutines and channels.",
        }
        for name, content in files.items():
            with open(os.path.join(self.test_dir, name), "w") as f:
                f.write(content)

        result = self.mm.ingest_directory(self.test_dir)
        self.assertEqual(result["success"], 3)

        # Search for programming language
        results = self.retriever.search("programming language", top_k=10)
        self.assertGreater(len(results), 0)

        # Search for specific topics
        results_ml = self.retriever.search("machine learning", top_k=5)
        results_memory = self.retriever.search("memory safety", top_k=5)
        self.assertGreater(len(results_ml), 0)
        self.assertGreater(len(results_memory), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
