#!/usr/bin/env python3
"""
repair_l2_l3.py - 为现有的L1记录创建缺失的L2和L3记录。

此脚本用于修复旧数据库中只有L1记录而缺少L2/L3记录的问题。
"""

import sys
import time
import zlib
from pathlib import Path

from src.database import Database
from src.memory_manager import MemoryManager
from src.chunker import compute_content_hash
from src.embedder import OllamaEmbedder, OllamaGenerator, vector_to_bytes, bytes_to_vector
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        return {}


def repair_l2_l3(db_path: str = "kb.db", config_path: str = "config.yaml", dry_run: bool = False):
    """
    Repair missing L2 and L3 records for existing L1 entries.
    
    Args:
        db_path: Path to the SQLite database
        config_path: Path to the configuration file
        dry_run: If True, only show what would be done without making changes
    """
    config = load_config(config_path)
    db = Database(db_path)
    
    # Get all L1 records that don't have corresponding L2 records
    l1_records = db.conn.execute("""
        SELECT l1.* FROM l1_working_memory l1
        LEFT JOIN l2_short_term l2 ON l1.content_hash = l2.content_hash
        WHERE l2.id IS NULL
    """).fetchall()
    l1_records = [dict(r) for r in l1_records]
    
    print(f"Found {len(l1_records)} L1 records missing L2/L3 entries")
    
    if not l1_records:
        print("No repair needed. All L1 records have L2/L3 entries.")
        db.close()
        return
    
    if dry_run:
        print("\n--- DRY RUN MODE ---")
        print("The following records would be created:")
        for rec in l1_records[:10]:
            print(f"  L1 ID {rec['id']}: content_hash={rec['content_hash'][:16]}...")
        if len(l1_records) > 10:
            print(f"  ... and {len(l1_records) - 10} more")
        db.close()
        return
    
    # Get configuration
    ollama_cfg = config.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    models = ollama_cfg.get("models", {})
    
    # Initialize embedder and generator
    embedder = OllamaEmbedder(
        host=host,
        model=models.get("embedding", "nomic-embed-text"),
        timeout=ollama_cfg.get("timeout", 120),
    )
    generator = OllamaGenerator(
        host=host,
        model=models.get("summary", "phi3:mini"),
        timeout=ollama_cfg.get("timeout", 120),
    )
    
    # Get archive path
    archive_path = config.get("memory", {}).get("tiers", {}).get(
        "l3", {}
    ).get("archive_path", "./archives/")
    Path(archive_path).mkdir(parents=True, exist_ok=True)
    
    # Process each L1 record
    created_l2 = 0
    created_l3 = 0
    errors = 0
    
    for rec in l1_records:
        try:
            content_hash = rec["content_hash"]
            raw_content = rec["raw_content"]
            summary_content = rec.get("summary_content") or ""
            overview_content = rec.get("overview_content") or ""
            vector_bytes = rec.get("vector", b"")
            
            # Create L2 record
            l2_summary = summary_content or raw_content[:200]
            compressed_l2 = zlib.compress(l2_summary.encode("utf-8"))
            
            db.insert_l2(
                l1_id=rec["id"],
                content_hash=content_hash,
                compressed_content=compressed_l2,
                compressed_vector=vector_bytes,
                summary=l2_summary,
            )
            created_l2 += 1
            
            # Create L3 record
            l3_overview = overview_content or (summary_content[:100] if summary_content else raw_content[:100])
            
            # Compute semantic hash for key concepts
            from src.dedup import DeduplicationEngine
            dedup = DeduplicationEngine()
            key_concepts = dedup.compute_semantic_hash(raw_content)
            
            # Quantize vector to int8 for L3
            if vector_bytes:
                vec = bytes_to_vector(vector_bytes)
                quantized = np.clip(vec * 127, -128, 127).astype(np.int8)
                quantized_vec_bytes = quantized.tobytes()
            else:
                quantized_vec_bytes = b""
            
            # Create archive file
            archive_file = str(Path(archive_path) / f"{content_hash[:16]}.zlib")
            compressed_full = zlib.compress(raw_content.encode("utf-8"))
            with open(archive_file, "wb") as f:
                f.write(compressed_full)
            
            db.insert_l3(
                content_hash=content_hash,
                key_concepts=key_concepts,
                overview=l3_overview,
                archive_path=archive_file,
                compressed_vector=quantized_vec_bytes,
            )
            created_l3 += 1
            
        except Exception as e:
            errors += 1
            print(f"Error processing L1 ID {rec['id']}: {e}")
    
    print(f"\n--- Repair Complete ---")
    print(f"Created L2 records: {created_l2}")
    print(f"Created L3 records: {created_l3}")
    print(f"Errors: {errors}")
    
    # Print updated stats
    stats = db.get_stats()
    print(f"\nUpdated stats:")
    print(f"  L1 chunks: {stats['l1_chunks']}")
    print(f"  L2 chunks: {stats['l2_chunks']}")
    print(f"  L3 chunks: {stats['l3_chunks']}")
    
    db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Repair missing L2/L3 records for existing L1 entries")
    parser.add_argument("--db", default="kb.db", help="Path to the SQLite database")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    repair_l2_l3(args.db, args.config, args.dry_run)