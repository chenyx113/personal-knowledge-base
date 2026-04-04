"""
cli.py - Command-line interface for the knowledge base.
Usage: python -m src.cli <command> [options]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .database import Database
from .memory_manager import MemoryManager
from .retriever import HybridRetriever


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_kb(args) -> tuple:
    """Initialize KB components from args."""
    config = load_config(args.config)
    db = Database(args.db)
    mm = MemoryManager(db, config)
    retriever = HybridRetriever(db, mm.embedder, config)
    return db, mm, retriever, config


def cmd_init(args):
    """Initialize a new knowledge base."""
    db = Database(args.db)
    print(f"Knowledge base initialized at: {args.db}")
    # Create default config if not exists
    if not Path(args.config).exists():
        default_config = {
            "ollama": {
                "host": "http://localhost:11434",
                "models": {
                    "embedding": "nomic-embed-text",
                    "summary": "phi3:mini",
                    "chat": "qwen2.5:7b",
                },
            },
            "memory": {
                "tiers": {
                    "l1": {"retention_days": 7},
                    "l2": {"retention_days": 30},
                    "l3": {"archive_path": "./archives/"},
                },
                "deduplication": {
                    "simhash_threshold": 3,
                    "vector_threshold": 0.92,
                    "auto_merge": False,
                },
            },
            "chunking": {
                "default_chunk_size": 500,
                "chunk_overlap": 50,
            },
            "search": {
                "default_top_k": 10,
                "rrf_k": 60,
                "auto_tier_fallback": True,
            },
        }
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
        print(f"Default config created at: {args.config}")
    db.close()


def cmd_add(args):
    """Add a file or directory to the knowledge base."""
    db, mm, _, _ = get_kb(args)

    path = Path(args.path)
    tags = args.tags.split(",") if args.tags else []

    if path.is_dir():
        result = mm.ingest_directory(
            str(path), recursive=args.recursive, tags=tags
        )
        print(f"Directory: {path}")
        print(f"  Files processed: {result['total_files']}")
        print(f"  Successful: {result['success']}")
        print(f"  Unchanged: {result['unchanged']}")
        print(f"  Failed: {result['failed']}")
    elif path.is_file():
        result = mm.ingest_file(str(path), tags=tags, title=args.title or "")
        print(f"File: {path}")
        print(f"  Status: {result['status']}")
        if result["status"] == "success":
            print(f"  Chunks added: {result['chunks_added']}")
            print(f"  Duplicates found: {result['duplicates_found']}")
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    db.close()


def cmd_search(args):
    """Search the knowledge base."""
    db, mm, retriever, _ = get_kb(args)

    query = " ".join(args.query)
    tier = args.tier
    top_k = args.top

    results = retriever.search(query, top_k=top_k, tier=tier,
                                tag_filter=args.filter)

    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results for: \"{query}\"\n")
        for i, r in enumerate(results, 1):
            rd = r.to_dict()
            print(f"--- [{i}] Score: {rd['score']:.4f} | Tier: L{rd['tier']} | Source: {rd['source']} ---")
            if rd["title"]:
                print(f"    Title: {rd['title']}")
            if rd["doc_path"]:
                print(f"    File: {rd['doc_path']}")
            if rd["start_line"]:
                print(f"    Lines: {rd['start_line']}-{rd['end_line']}")
            # Show content preview
            content = rd["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"    {content}\n")

    db.close()


def cmd_list(args):
    """List documents in the knowledge base."""
    db, _, _, _ = get_kb(args)

    docs = db.list_documents(limit=args.recent)
    if not docs:
        print("No documents in the knowledge base.")
    else:
        print(f"{'ID':>4} | {'Type':>4} | {'Title':<40} | {'Tags':<20} | {'Path'}")
        print("-" * 100)
        for doc in docs:
            tags = json.loads(doc.get("tags", "[]"))
            tag_str = ", ".join(tags) if tags else "-"
            title = (doc.get("title") or "")[:40]
            print(f"{doc['id']:>4} | {doc.get('file_type', '?'):>4} | {title:<40} | {tag_str:<20} | {doc['file_path']}")

    db.close()


def cmd_stats(args):
    """Show knowledge base statistics."""
    db, mm, _, _ = get_kb(args)
    stats = mm.get_stats()

    print("=== Knowledge Base Statistics ===")
    print(f"  Documents:       {stats['documents']}")
    print(f"  L1 Chunks (hot): {stats['l1_chunks']}")
    print(f"  L2 Chunks (warm):{stats['l2_chunks']}")
    print(f"  L3 Chunks (cold):{stats['l3_chunks']}")
    print(f"  Total Chunks:    {stats['total_chunks']}")
    print(f"  Tags:            {stats['tags']}")
    print(f"  Searches logged: {stats['searches']}")
    print(f"  Ollama available:{stats['ollama_available']}")
    print(f"  Embedding model: {stats['embedding_model']}")

    db.close()


def cmd_maintain(args):
    """Run maintenance: decay, demote, archive."""
    db, mm, _, _ = get_kb(args)
    result = mm.run_maintenance()
    print("Maintenance complete:")
    print(f"  L1 → L2 demoted: {result['demoted_l1_to_l2']}")
    print(f"  L2 → L3 archived: {result['demoted_l2_to_l3']}")
    db.close()


def cmd_delete(args):
    """Delete a document and its chunks."""
    db, _, _, _ = get_kb(args)
    doc = db.get_document(args.doc_id)
    if not doc:
        print(f"Document {args.doc_id} not found.", file=sys.stderr)
        sys.exit(1)

    # Delete associated chunks
    chunks = db.conn.execute(
        "SELECT id FROM l1_working_memory WHERE doc_id = ?", (args.doc_id,)
    ).fetchall()
    for (chunk_id,) in [(r["id"],) for r in chunks]:
        db.delete_l1(chunk_id)

    db.delete_document(args.doc_id)
    print(f"Deleted document {args.doc_id}: {doc['file_path']}")
    print(f"  Removed {len(chunks)} chunks")
    db.close()


def cmd_export(args):
    """Export knowledge base as markdown."""
    db, _, _, _ = get_kb(args)
    docs = db.list_documents(limit=9999)

    output = []
    output.append("# Knowledge Base Export\n")
    output.append(f"Exported at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.append(f"Total documents: {len(docs)}\n\n")

    for doc in docs:
        output.append(f"## {doc.get('title', 'Untitled')}\n")
        output.append(f"- **File:** {doc['file_path']}")
        output.append(f"- **Type:** {doc.get('file_type', '?')}")
        tags = json.loads(doc.get("tags", "[]"))
        if tags:
            output.append(f"- **Tags:** {', '.join(tags)}")
        if doc.get("summary"):
            output.append(f"- **Summary:** {doc['summary']}")
        output.append("")

        # Get chunks
        chunks = db.conn.execute(
            """SELECT raw_content, chunk_index, start_line, end_line
               FROM l1_working_memory WHERE doc_id = ? ORDER BY chunk_index""",
            (doc["id"],)
        ).fetchall()

        for chunk in chunks:
            c = dict(chunk)
            output.append(f"### Chunk {c['chunk_index']} (lines {c['start_line']}-{c['end_line']})")
            output.append(f"```\n{c['raw_content']}\n```\n")

    export_path = args.output or "kb_export.md"
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print(f"Exported to: {export_path}")
    db.close()


def cmd_serve(args):
    """Start the web server."""
    from .web_app import create_app
    import uvicorn

    config = load_config(args.config)
    app = create_app(args.db, config)

    port = args.port or config.get("web", {}).get("port", 3000)
    host = args.host or config.get("web", {}).get("host", "0.0.0.0")

    print(f"Starting Knowledge Base Web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Base - Three-tier Memory System",
        prog="kb",
    )
    parser.add_argument("--db", default="kb.db", help="Database path (default: kb.db)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sub.add_parser("init", help="Initialize a new knowledge base")

    # add
    p_add = sub.add_parser("add", help="Add file or directory")
    p_add.add_argument("path", help="File or directory path")
    p_add.add_argument("--tags", help="Comma-separated tags")
    p_add.add_argument("--title", help="Document title")
    p_add.add_argument("--recursive", "-r", action="store_true", default=True,
                       help="Recurse into subdirectories (default: true)")

    # search
    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--top", type=int, default=10, help="Number of results")
    p_search.add_argument("--tier", type=int, choices=[1, 2, 3],
                          help="Search specific tier")
    p_search.add_argument("--filter", help="Filter by tag")

    # list
    p_list = sub.add_parser("list", help="List documents")
    p_list.add_argument("--recent", type=int, default=20, help="Number of recent docs")

    # stats
    sub.add_parser("stats", help="Show statistics")

    # maintain
    sub.add_parser("maintain", help="Run maintenance (decay/demote/archive)")

    # delete
    p_del = sub.add_parser("delete", help="Delete a document")
    p_del.add_argument("doc_id", type=int, help="Document ID to delete")

    # export
    p_exp = sub.add_parser("export", help="Export as markdown")
    p_exp.add_argument("--output", "-o", help="Output file path")

    # serve
    p_serve = sub.add_parser("serve", help="Start web UI")
    p_serve.add_argument("--port", type=int, help="Port number")
    p_serve.add_argument("--host", help="Host address")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "search": cmd_search,
        "list": cmd_list,
        "stats": cmd_stats,
        "maintain": cmd_maintain,
        "delete": cmd_delete,
        "export": cmd_export,
        "serve": cmd_serve,
    }

    func = commands.get(args.command)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
