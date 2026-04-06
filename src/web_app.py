"""
web_app.py - FastAPI web interface for the knowledge base.
Serves both API endpoints and the single-page frontend.
"""

import json
import time
import asyncio
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict
from queue import Queue

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .database import Database
from .memory_manager import MemoryManager
from .retriever import HybridRetriever
from .chunker import extract_text_from_file


# Background processing state
background_tasks: Dict[str, dict] = {}
processing_queue = Queue()
processing_lock = threading.Lock()

def _background_processor(db_path: str, config: dict):
    """Background thread that processes pending ingestion tasks."""
    mm = MemoryManager(Database(db_path), config)
    while True:
        task = processing_queue.get()
        if task is None:  # Sentinel to stop
            break
        task_id = task["task_id"]
        start_time = time.time()
        try:
            background_tasks[task_id]["status"] = "processing"
            background_tasks[task_id]["progress"] = 0
            background_tasks[task_id]["start_time"] = start_time
            
            if task["type"] == "file":
                # For single file, estimate progress based on time
                background_tasks[task_id]["progress"] = 10  # Start processing
                result = mm.ingest_file(task["file_path"], task.get("tags", []), task.get("title", ""))
                background_tasks[task_id]["result"] = result
                background_tasks[task_id]["progress"] = 100
            elif task["type"] == "directory":
                # For directory, we can track file count
                upload_dir = Path(task["dir_path"])
                if upload_dir.exists():
                    supported = [".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
                                ".toml", ".sql", ".sh", ".css", ".html", ".xml", ".ipynb",
                                ".java", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".php"]
                    pattern = "**/*" if task.get("recursive", True) else "*"
                    files = [f for f in upload_dir.glob(pattern) if f.is_file() and f.suffix.lower() in supported]
                    total_files = len(files)
                    background_tasks[task_id]["total_files"] = total_files
                    
                    if total_files > 0:
                        # Process files one by one and update progress
                        processed = 0
                        results = {"total_files": total_files, "success": 0, "failed": 0, "unchanged": 0, "details": []}
                        
                        for f in sorted(files):
                            try:
                                r = mm.ingest_file(str(f), tags=task.get("tags", []))
                                processed += 1
                                background_tasks[task_id]["progress"] = int((processed / total_files) * 100)
                                background_tasks[task_id]["current_file"] = f.name
                                
                                if r["status"] == "success":
                                    results["success"] += 1
                                elif r["status"] == "unchanged":
                                    results["unchanged"] += 1
                                else:
                                    results["failed"] += 1
                                results["details"].append({"file": str(f), "result": r})
                            except Exception as e:
                                processed += 1
                                results["failed"] += 1
                                results["details"].append({"file": str(f), "error": str(e)})
                        
                        background_tasks[task_id]["result"] = results
                        background_tasks[task_id]["progress"] = 100
                    else:
                        background_tasks[task_id]["result"] = {"total_files": 0, "success": 0, "message": "No supported files found"}
                        background_tasks[task_id]["progress"] = 100
                else:
                    background_tasks[task_id]["result"] = {"status": "error", "message": "Directory not found"}
                    background_tasks[task_id]["progress"] = 100
                    
            elif task["type"] == "text":
                background_tasks[task_id]["progress"] = 10
                result = mm.ingest_file(task["file_path"], task.get("tags", []), task.get("title", ""))
                background_tasks[task_id]["result"] = result
                background_tasks[task_id]["progress"] = 100
                
            background_tasks[task_id]["end_time"] = time.time()
            background_tasks[task_id]["status"] = "completed"
        except Exception as e:
            background_tasks[task_id]["status"] = "failed"
            background_tasks[task_id]["error"] = str(e)
            background_tasks[task_id]["end_time"] = time.time()
        
        processing_queue.task_done()

def create_app(db_path: str = "kb.db", config: dict = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or {}
    app = FastAPI(title="Personal Knowledge Base", version="1.0.0")

    db = Database(db_path)
    mm = MemoryManager(db, config)
    retriever = HybridRetriever(db, mm.embedder, config)
    
    # Start background processor thread
    processor_thread = threading.Thread(target=_background_processor, args=(db_path, config), daemon=True)
    processor_thread.start()

    # --- API Routes ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return get_frontend_html()

    @app.get("/api/stats")
    async def api_stats():
        return mm.get_stats()

    @app.get("/api/search")
    async def api_search(
        q: str = Query(..., min_length=1),
        tier: Optional[int] = Query(None, ge=1, le=3),
        top_k: int = Query(10, ge=1, le=100),
        tag: Optional[str] = None,
    ):
        results = await asyncio.to_thread(
            retriever.search, q, top_k, tier, tag
        )
        return {
            "query": q,
            "tier_searched": tier,
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

    @app.post("/api/upload")
    async def api_upload(
        file: UploadFile = File(...),
    ):
        """Upload a single file only (fast). Parsing happens in background."""
        upload_dir = Path("files/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        dest = upload_dir / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        
        return {"status": "uploaded", "file": str(dest)}

    @app.post("/api/ingest")
    async def api_ingest(
        file: UploadFile = File(...),
        tags: str = Form(""),
        title: str = Form(""),
        async_mode: bool = Form(False),
    ):
        """Ingest a file. If async_mode=True, returns task_id immediately."""
        # Save uploaded file temporarily
        upload_dir = Path("files/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle subdirectory paths in filename (from folder upload)
        dest = upload_dir / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        
        if async_mode:
            # Queue for background processing
            task_id = str(uuid.uuid4())
            background_tasks[task_id] = {
                "status": "queued",
                "progress": 0,
                "type": "file",
                "file_path": str(dest),
                "tags": tag_list,
                "title": title,
            }
            processing_queue.put({
                "task_id": task_id,
                "type": "file",
                "file_path": str(dest),
                "tags": tag_list,
                "title": title,
            })
            return {"status": "queued", "task_id": task_id}
        else:
            result = await asyncio.to_thread(
                mm.ingest_file, str(dest), tag_list, title
            )
            return result

    @app.post("/api/ingest-directory")
    async def api_ingest_directory(
        tags: str = Form(""),
        recursive: bool = Form(True),
        async_mode: bool = Form(False),
    ):
        """Ingest all supported files from the uploads directory."""
        upload_dir = Path("files/uploads")
        if not upload_dir.exists():
            return {"status": "error", "message": "Upload directory not found"}

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        
        if async_mode:
            task_id = str(uuid.uuid4())
            background_tasks[task_id] = {
                "status": "queued",
                "progress": 0,
                "type": "directory",
                "dir_path": str(upload_dir),
                "recursive": recursive,
                "tags": tag_list,
            }
            processing_queue.put({
                "task_id": task_id,
                "type": "directory",
                "dir_path": str(upload_dir),
                "recursive": recursive,
                "tags": tag_list,
            })
            return {"status": "queued", "task_id": task_id}
        else:
            result = await asyncio.to_thread(
                mm.ingest_directory, str(upload_dir), recursive, tag_list
            )
            return result

    @app.post("/api/ingest-text")
    async def api_ingest_text(
        text: str = Form(...),
        title: str = Form(""),
        tags: str = Form(""),
    ):
        """Ingest raw text directly."""
        upload_dir = Path("files/notes")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save as markdown
        safe_title = title or f"note_{int(time.time())}"
        safe_title = "".join(c for c in safe_title if c.isalnum() or c in " _-").strip()
        dest = upload_dir / f"{safe_title}.md"

        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        result = await asyncio.to_thread(
            mm.ingest_file, str(dest), tag_list, title
        )
        return result

    @app.get("/api/task/{task_id}")
    async def api_task_status(task_id: str):
        """Get status of a background task."""
        if task_id not in background_tasks:
            raise HTTPException(404, "Task not found")
        task = background_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 0),
            "result": task.get("result"),
            "error": task.get("error"),
        }

    @app.get("/api/tasks")
    async def api_tasks():
        """Get all background tasks."""
        return {"tasks": background_tasks}

    @app.get("/api/documents")
    async def api_documents(limit: int = 50, offset: int = 0):
        docs = db.list_documents(limit=limit, offset=offset)
        for d in docs:
            d["tags"] = json.loads(d.get("tags", "[]"))
        return {"documents": docs, "total": db.count_documents()}

    @app.get("/api/documents/{doc_id}")
    async def api_document_detail(doc_id: int):
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        doc["tags"] = json.loads(doc.get("tags", "[]"))

        chunks = db.conn.execute(
            """SELECT id, chunk_index, raw_content, summary_content,
                      overview_content, memory_tier, access_count,
                      start_line, end_line, content_hash
               FROM l1_working_memory WHERE doc_id = ? ORDER BY chunk_index""",
            (doc_id,)
        ).fetchall()

        return {
            "document": doc,
            "chunks": [dict(c) for c in chunks],
        }

    @app.delete("/api/documents/{doc_id}")
    async def api_delete_document(doc_id: int):
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")

        chunks = db.conn.execute(
            "SELECT id FROM l1_working_memory WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        for c in chunks:
            db.delete_l1(c["id"])
        db.delete_document(doc_id)

        return {"status": "deleted", "doc_id": doc_id}

    @app.get("/api/memory/{l1_id}")
    async def api_memory_detail(l1_id: int):
        rec = db.get_l1(l1_id)
        if not rec:
            raise HTTPException(404, "Memory record not found")

        result = dict(rec)
        result.pop("vector", None)  # Don't send binary data

        # Get duplicate info
        cluster = db.get_duplicate_cluster_for(rec["content_hash"])
        if cluster:
            members = db.get_cluster_members(cluster["id"])
            result["duplicate_cluster"] = {
                "id": cluster["id"],
                "strategy": cluster["merge_strategy"],
                "members": members,
            }

        return result

    @app.post("/api/memory/{l1_id}/promote")
    async def api_promote(l1_id: int):
        rec = db.get_l1(l1_id)
        if not rec:
            raise HTTPException(404, "Record not found")
        result = await asyncio.to_thread(
            mm.promote_to_l1, rec["content_hash"]
        )
        if result:
            return {"status": "promoted", "tier": 1}
        return {"status": "already_l1"}

    @app.post("/api/maintain")
    async def api_maintain():
        result = await asyncio.to_thread(mm.run_maintenance)
        return result

    @app.get("/api/tags")
    async def api_tags():
        return {"tags": db.get_all_tags()}

    @app.post("/api/tags")
    async def api_create_tag(
        name: str = Form(...),
        color: str = Form("#6366f1"),
        description: str = Form(""),
    ):
        tid = db.insert_tag(name, color, description)
        return {"id": tid, "name": name}

    @app.get("/api/duplicates")
    async def api_duplicates():
        clusters = db.conn.execute(
            "SELECT * FROM duplicate_clusters ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for c in clusters:
            cd = dict(c)
            cd["members"] = db.get_cluster_members(cd["id"])
            result.append(cd)
        return {"clusters": result}

    return app


def get_frontend_html() -> str:
    """Return the single-page frontend HTML."""
    return FRONTEND_HTML


# ---- Frontend HTML (inline single-page app) ----

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Personal Knowledge Base - Three-Tier Memory</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0f1117;--surface:#1a1d27;--surface2:#242836;--border:#2d3148;
--text:#e4e4e7;--text2:#a1a1aa;--accent:#6366f1;--accent2:#818cf8;
--green:#22c55e;--yellow:#eab308;--red:#ef4444;--cyan:#06b6d4;
--tier1:#ef4444;--tier2:#06b6d4;--tier3:#6366f1}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
background:var(--bg);color:var(--text);min-height:100vh}
.app{display:flex;flex-direction:column;height:100vh}
header{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 24px;
display:flex;align-items:center;justify-content:space-between;gap:16px;flex-shrink:0}
header h1{font-size:18px;font-weight:600;white-space:nowrap}
header h1 span{color:var(--accent);font-weight:700}
.tier-pills{display:flex;gap:6px}
.tier-pill{padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;
border:1px solid var(--border);background:transparent;color:var(--text2);cursor:pointer;
transition:all .2s}
.tier-pill.active,.tier-pill:hover{color:#fff}
.tier-pill[data-tier="1"].active{background:var(--tier1);border-color:var(--tier1)}
.tier-pill[data-tier="2"].active{background:var(--tier2);border-color:var(--tier2)}
.tier-pill[data-tier="3"].active{background:var(--tier3);border-color:var(--tier3)}
.tier-pill[data-tier="0"].active{background:var(--accent);border-color:var(--accent)}
.header-actions{display:flex;gap:8px;align-items:center}
.header-actions button{padding:6px 14px;border-radius:6px;border:1px solid var(--border);
background:var(--surface2);color:var(--text);font-size:13px;cursor:pointer;transition:all .2s}
.header-actions button:hover{background:var(--accent);border-color:var(--accent);color:#fff}
.search-bar{padding:16px 24px;background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}
.search-input-wrap{display:flex;gap:8px;max-width:900px;margin:0 auto}
.search-input-wrap input{flex:1;padding:10px 16px;border-radius:8px;border:1px solid var(--border);
background:var(--bg);color:var(--text);font-size:15px;outline:none;transition:border .2s}
.search-input-wrap input:focus{border-color:var(--accent)}
.search-input-wrap button{padding:10px 20px;border-radius:8px;border:none;
background:var(--accent);color:#fff;font-weight:600;cursor:pointer;font-size:14px;transition:all .2s}
.search-input-wrap button:hover{background:var(--accent2)}
.main{display:flex;flex:1;overflow:hidden}
.panel{overflow-y:auto;padding:16px}
.panel::-webkit-scrollbar{width:6px}
.panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.results-panel{width:400px;border-right:1px solid var(--border);flex-shrink:0}
.detail-panel{flex:1;padding:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:14px;margin-bottom:10px;cursor:pointer;transition:all .2s}
.card:hover{border-color:var(--accent);transform:translateX(2px)}
.card.active{border-color:var(--accent);background:var(--surface2)}
.card-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.tier-badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;color:#fff}
.tier-badge.t1{background:var(--tier1)}.tier-badge.t2{background:var(--tier2)}.tier-badge.t3{background:var(--tier3)}
.score-badge{font-size:11px;color:var(--green);font-weight:600}
.dup-badge{font-size:10px;color:var(--yellow);border:1px solid var(--yellow);
padding:1px 6px;border-radius:8px}
.card h4{font-size:14px;font-weight:600;margin-bottom:4px;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card .preview{font-size:12px;color:var(--text2);line-height:1.5;
display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.card-meta{margin-top:8px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--text2)}
.detail-empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--text2);font-size:15px}
.detail-header{margin-bottom:20px}
.detail-header h2{font-size:20px;font-weight:700;margin-bottom:8px}
.detail-header .meta{font-size:13px;color:var(--text2);display:flex;gap:16px;flex-wrap:wrap}
.tier-tabs{display:flex;gap:4px;margin-bottom:16px}
.tier-tab{padding:6px 14px;border-radius:6px;border:1px solid var(--border);
background:transparent;color:var(--text2);font-size:13px;cursor:pointer;transition:all .2s}
.tier-tab.active{background:var(--accent);border-color:var(--accent);color:#fff}
.tier-tab:disabled{opacity:.4;cursor:not-allowed}
.content-block{background:var(--bg);border:1px solid var(--border);border-radius:8px;
padding:16px;font-size:14px;line-height:1.7;white-space:pre-wrap;word-break:break-word;
max-height:60vh;overflow-y:auto}
.content-block code{background:var(--surface2);padding:1px 4px;border-radius:3px;font-size:13px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin:16px 0}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:16px;text-align:center}
.stat-card .num{font-size:28px;font-weight:700;color:var(--accent)}
.stat-card .label{font-size:12px;color:var(--text2);margin-top:4px}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;
align-items:center;justify-content:center}
.modal-overlay.show{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:24px;width:90%;max-width:560px;max-height:80vh;overflow-y:auto}
.modal h3{font-size:18px;margin-bottom:16px}
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:13px;color:var(--text2);margin-bottom:4px}
.form-group input,.form-group textarea,.form-group select{width:100%;padding:8px 12px;
border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:14px}
.form-group textarea{min-height:120px;resize:vertical}
.form-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
.form-actions button{padding:8px 18px;border-radius:6px;font-size:14px;cursor:pointer;border:none}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{background:var(--accent2)}
.btn-cancel{background:var(--surface2);color:var(--text);border:1px solid var(--border) !important}
.toast{position:fixed;bottom:24px;right:24px;background:var(--green);color:#fff;
padding:10px 20px;border-radius:8px;font-size:14px;z-index:200;opacity:0;transition:opacity .3s}
.toast.show{opacity:1}
.toast.error{background:var(--red)}
.upload-zone{border:2px dashed var(--border);border-radius:10px;padding:40px;
text-align:center;color:var(--text2);cursor:pointer;transition:all .2s;margin:12px 0}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);color:var(--accent)}
.loading{display:inline-block;width:16px;height:16px;border:2px solid var(--border);
border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:768px){.main{flex-direction:column}.results-panel{width:100%;border-right:none;
border-bottom:1px solid var(--border);max-height:40vh}}
</style>
</head>
<body>
<div class="app" id="app">
  <header>
    <h1><span>KB</span> Three-Tier Memory</h1>
    <div class="tier-pills">
      <button class="tier-pill active" data-tier="0" onclick="setTier(0)">All</button>
      <button class="tier-pill" data-tier="1" onclick="setTier(1)">L1 Hot <span id="cnt-l1"></span></button>
      <button class="tier-pill" data-tier="2" onclick="setTier(2)">L2 Warm <span id="cnt-l2"></span></button>
      <button class="tier-pill" data-tier="3" onclick="setTier(3)">L3 Cold <span id="cnt-l3"></span></button>
    </div>
    <div class="header-actions">
      <button onclick="showUploadModal()">+ Add</button>
      <button onclick="showStatsModal()">Stats</button>
      <button onclick="runMaintenance()">Maintain</button>
    </div>
  </header>
  <div class="search-bar">
    <div class="search-input-wrap">
      <input id="searchInput" type="search" placeholder="Search your knowledge base... (natural language)"
             onkeydown="if(event.key==='Enter')doSearch()">
      <button onclick="doSearch()">Search</button>
    </div>
  </div>
  <div class="main">
    <div class="results-panel panel" id="resultsPanel">
      <div id="resultsList"></div>
    </div>
    <div class="detail-panel panel" id="detailPanel">
      <div class="detail-empty" id="detailEmpty">Select a result to view details</div>
      <div id="detailContent" style="display:none"></div>
    </div>
  </div>
</div>

<!-- Upload Modal -->
<div class="modal-overlay" id="uploadModal">
  <div class="modal">
    <h3>Add to Knowledge Base</h3>
    <div style="display:flex;gap:8px;margin-bottom:16px">
      <button class="tier-tab active" onclick="switchUploadTab(this,'file')">Upload File</button>
      <button class="tier-tab" onclick="switchUploadTab(this,'folder')">Upload Folder</button>
      <button class="tier-tab" onclick="switchUploadTab(this,'text')">Paste Text</button>
    </div>
    <div id="uploadFile">
      <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()"
           ondragover="event.preventDefault();this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="event.preventDefault();this.classList.remove('dragover');handleDrop(event)">
        Drop files here or click to browse
        <input type="file" id="fileInput" hidden multiple onchange="handleFiles(this.files)">
      </div>
      <div id="fileList" style="font-size:13px;color:var(--text2)"></div>
    </div>
    <div id="uploadFolder" style="display:none">
      <div class="upload-zone" onclick="document.getElementById('folderInput').click()"
           ondragover="event.preventDefault();this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="event.preventDefault();this.classList.remove('dragover');handleFolderDrop(event)">
        Drop a folder here or click to browse
        <input type="file" id="folderInput" hidden webkitdirectory mozdirectory directory multiple
               onchange="handleFolderFiles(this.files)">
      </div>
      <div class="form-group" style="margin-top:12px">
        <label style="display:flex;align-items:center;gap:8px">
          <input type="checkbox" id="recursiveCheck" checked style="width:auto">
          Include subdirectories (recursive)
        </label>
      </div>
      <div id="folderList" style="font-size:13px;color:var(--text2);max-height:100px;overflow-y:auto"></div>
    </div>
    <div id="uploadText" style="display:none">
      <div class="form-group"><label>Content</label><textarea id="textContent" placeholder="Paste or type content..."></textarea></div>
    </div>
    <div class="form-group"><label>Title</label><input id="ingestTitle" placeholder="Optional title"></div>
    <div class="form-group"><label>Tags (comma separated)</label><input id="ingestTags" placeholder="e.g. python, machine-learning"></div>
    <div class="form-actions">
      <button class="btn-cancel" onclick="closeModal('uploadModal')">Cancel</button>
      <button class="btn-primary" onclick="doIngest()">Add to KB</button>
    </div>
  </div>
</div>

<!-- Stats Modal -->
<div class="modal-overlay" id="statsModal">
  <div class="modal">
    <h3>Knowledge Base Statistics</h3>
    <div class="stats-grid" id="statsGrid"></div>
    <div class="form-actions"><button class="btn-cancel" onclick="closeModal('statsModal')">Close</button></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let currentTier = 0;
let searchResults = [];
let selectedId = null;
let pendingFiles = [];
let pendingFolderFiles = [];
let currentUploadTab = 'file';
let activeTasks = [];
let pollingIntervals = {};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => t.className = 'toast', 3000);
}

function setTier(t) {
  currentTier = t;
  document.querySelectorAll('.tier-pill').forEach(p => p.classList.toggle('active', parseInt(p.dataset.tier) === t));
  if (searchResults.length) doSearch();
}

async function doSearch() {
  const q = document.getElementById('searchInput').value.trim();
  if (!q) { await loadDocList(); return; }
  const tierParam = currentTier > 0 ? '&tier=' + currentTier : '';
  try {
    const data = await api('/api/search?q=' + encodeURIComponent(q) + tierParam + '&top_k=20');
    searchResults = data.results || [];
    renderResults();
  } catch (e) { toast('Search failed: ' + e.message, true); }
}

async function loadDocList() {
  try {
    const data = await api('/api/documents?limit=50');
    const docs = data.documents || [];
    const list = document.getElementById('resultsList');
    if (!docs.length) { list.innerHTML = '<p style="color:var(--text2);padding:20px;text-align:center">No documents yet. Click "+ Add" to get started.</p>'; return; }
    list.innerHTML = docs.map(d => `
      <div class="card" onclick="loadDocDetail(${d.id})">
        <div class="card-head"><span class="tier-badge t1">${d.file_type || '?'}</span>
          <span style="font-size:11px;color:var(--text2)">#${d.id}</span></div>
        <h4>${esc(d.title || d.file_path)}</h4>
        <p class="preview">${esc(d.summary || d.file_path)}</p>
        <div class="card-meta"><span>${(d.tags||[]).join(', ') || 'no tags'}</span></div>
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}

function renderResults() {
  const list = document.getElementById('resultsList');
  if (!searchResults.length) { list.innerHTML = '<p style="color:var(--text2);padding:20px;text-align:center">No results found.</p>'; return; }
  list.innerHTML = searchResults.map((r, i) => `
    <div class="card ${selectedId===r.id?'active':''}" onclick="showDetail(${i})">
      <div class="card-head">
        <span class="tier-badge t${r.tier}">L${r.tier}</span>
        <span class="score-badge">${(r.score*100).toFixed(1)}%</span>
        ${r.is_duplicate ? '<span class="dup-badge">DUP</span>' : ''}
        <span style="font-size:11px;color:var(--text2)">${r.source}</span>
      </div>
      <h4>${esc(r.title || 'Chunk #' + r.chunk_index)}</h4>
      <p class="preview">${esc(r.content.substring(0,200))}</p>
      <div class="card-meta">
        <span>${r.doc_path ? r.doc_path.split('/').pop() : ''}</span>
        ${r.start_line ? '<span>L' + r.start_line + '-' + r.end_line + '</span>' : ''}
      </div>
    </div>
  `).join('');
}

function showDetail(idx) {
  const r = searchResults[idx];
  if (!r) return;
  selectedId = r.id;
  renderResults();
  document.getElementById('detailEmpty').style.display = 'none';
  const dc = document.getElementById('detailContent');
  dc.style.display = 'block';
  dc.innerHTML = `
    <div class="detail-header">
      <h2>${esc(r.title || 'Chunk #' + r.chunk_index)}</h2>
      <div class="meta">
        <span>Tier: <b>L${r.tier}</b></span>
        <span>Score: <b>${(r.score*100).toFixed(1)}%</b></span>
        <span>Source: ${r.source}</span>
        ${r.doc_path ? '<span>File: ' + esc(r.doc_path.split('/').pop()) + '</span>' : ''}
        ${r.start_line ? '<span>Lines: ' + r.start_line + '-' + r.end_line + '</span>' : ''}
      </div>
    </div>
    <div class="tier-tabs">
      <button class="tier-tab active" onclick="showTierContent(this,'full')">Full Content</button>
      <button class="tier-tab" onclick="showTierContent(this,'summary')" ${r.summary?'':'disabled'}>Summary</button>
      <button class="tier-tab" onclick="showTierContent(this,'overview')" ${r.overview?'':'disabled'}>Overview</button>
    </div>
    <div class="content-block" id="tierContent">${esc(r.content)}</div>
    <input type="hidden" id="hid-summary" value="${esc(r.summary||'')}">
    <input type="hidden" id="hid-overview" value="${esc(r.overview||'')}">
    <input type="hidden" id="hid-full" value="${esc(r.content)}">
  `;
}

async function loadDocDetail(docId) {
  try {
    const data = await api('/api/documents/' + docId);
    const doc = data.document;
    const chunks = data.chunks || [];
    document.getElementById('detailEmpty').style.display = 'none';
    const dc = document.getElementById('detailContent');
    dc.style.display = 'block';
    dc.innerHTML = `
      <div class="detail-header">
        <h2>${esc(doc.title || doc.file_path)}</h2>
        <div class="meta">
          <span>Type: ${doc.file_type}</span>
          <span>Tags: ${(doc.tags||[]).join(', ') || 'none'}</span>
          <span>Chunks: ${chunks.length}</span>
          <button style="margin-left:auto;padding:4px 12px;border-radius:4px;border:1px solid var(--red);
            background:transparent;color:var(--red);cursor:pointer;font-size:12px"
            onclick="deleteDoc(${doc.id})">Delete</button>
        </div>
      </div>
      ${doc.summary ? '<div class="content-block" style="margin-bottom:16px;border-left:3px solid var(--accent);font-style:italic">' + esc(doc.summary) + '</div>' : ''}
      ${chunks.map((c,i) => `
        <details style="margin-bottom:8px">
          <summary style="cursor:pointer;padding:8px;background:var(--surface);border-radius:6px;font-size:13px">
            Chunk ${c.chunk_index} | L${c.memory_tier} | Accessed ${c.access_count}x
            ${c.start_line ? ' | Lines ' + c.start_line + '-' + c.end_line : ''}
          </summary>
          <div class="content-block" style="margin-top:4px;font-size:13px">${esc(c.raw_content)}</div>
        </details>
      `).join('')}
    `;
  } catch (e) { toast('Failed to load document', true); }
}

function showTierContent(btn, tier) {
  document.querySelectorAll('.tier-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  const el = document.getElementById('tierContent');
  const val = document.getElementById('hid-' + tier);
  if (val) el.textContent = val.value || '(not available)';
}

async function deleteDoc(id) {
  if (!confirm('Delete this document and all its chunks?')) return;
  try {
    await api('/api/documents/' + id, {method:'DELETE'});
    toast('Document deleted');
    loadDocList();
    document.getElementById('detailContent').style.display = 'none';
    document.getElementById('detailEmpty').style.display = 'flex';
  } catch (e) { toast('Delete failed', true); }
}

function showUploadModal() { document.getElementById('uploadModal').classList.add('show'); }
function closeModal(id) { document.getElementById(id).classList.remove('show'); }

function switchUploadTab(btn, tab) {
  currentUploadTab = tab;
  document.querySelectorAll('.modal .tier-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('uploadFile').style.display = tab === 'file' ? '' : 'none';
  document.getElementById('uploadFolder').style.display = tab === 'folder' ? '' : 'none';
  document.getElementById('uploadText').style.display = tab === 'text' ? '' : 'none';
}

function handleFolderDrop(e) {
  // Folder drop handling - browsers handle this differently
  if (e.dataTransfer.items) {
    const items = Array.from(e.dataTransfer.items);
    const fileItems = items.filter(item => item.kind === 'file');
    const files = fileItems.map(item => item.getAsFile()).filter(f => f);
    handleFolderFiles(files);
  }
}

function handleFolderFiles(files) {
  pendingFolderFiles = Array.from(files);
  const folderNames = pendingFolderFiles.map(f => {
    // Show relative path if available (webkitRelativePath)
    return f.webkitRelativePath || f.name;
  });
  document.getElementById('folderList').innerHTML = folderNames.map(name =>
    `<div>${esc(name)}</div>`
  ).join('');
}

function handleDrop(e) { handleFiles(e.dataTransfer.files); }
function handleFiles(files) {
  pendingFiles = Array.from(files);
  document.getElementById('fileList').textContent = pendingFiles.map(f => f.name).join(', ');
}

async function doIngest() {
  const title = document.getElementById('ingestTitle').value;
  const tags = document.getElementById('ingestTags').value;
  const textEl = document.getElementById('textContent');

  if (currentUploadTab === 'text' && textEl.value.trim()) {
    // Text mode
    const fd = new FormData();
    fd.append('text', textEl.value);
    fd.append('title', title);
    fd.append('tags', tags);
    try {
      const r = await api('/api/ingest-text', {method:'POST', body:fd});
      toast('Text added: ' + (r.chunks_added || 0) + ' chunks');
      closeModal('uploadModal');
      textEl.value = '';
      loadDocList(); loadStats();
    } catch (e) { toast('Ingest failed: ' + e.message, true); }
  } else if (currentUploadTab === 'folder' && pendingFolderFiles.length) {
    // Folder mode - upload files first (fast), then process in background
    toast('Uploading ' + pendingFolderFiles.length + ' files...');
    let uploaded = 0;
    let failed = 0;
    
    // Upload all files first (fast operation)
    for (const f of pendingFolderFiles) {
      const fd = new FormData();
      fd.append('file', f);
      fd.append('title', '');
      fd.append('tags', tags);
      try {
        await api('/api/upload', {method:'POST', body:fd});
        uploaded++;
      } catch (e) {
        failed++;
        console.error('Failed to upload:', f.name, e);
      }
    }
    
    toast(uploaded + ' files uploaded. Processing in background...');
    
    // Now start background processing
    const recursive = document.getElementById('recursiveCheck').checked;
    const fd2 = new FormData();
    fd2.append('tags', tags);
    fd2.append('recursive', recursive.toString());
    fd2.append('async_mode', 'true');
    
    try {
      const r = await api('/api/ingest-directory', {method:'POST', body:fd2});
      if (r.status === 'queued') {
        toast('Processing started for ' + uploaded + ' files. Check progress in header.');
        pollTaskStatus(r.task_id);
      }
    } catch (e) {
      toast('Failed to start processing: ' + e.message, true);
    }
    
    pendingFolderFiles = [];
    document.getElementById('folderList').innerHTML = '';
    closeModal('uploadModal');
    loadDocList(); loadStats();
  } else if (pendingFiles.length) {
    // Single file mode - use async for multiple files
    if (pendingFiles.length > 1) {
      toast('Uploading ' + pendingFiles.length + ' files...');
      for (const f of pendingFiles) {
        const fd = new FormData();
        fd.append('file', f);
        fd.append('title', title);
        fd.append('tags', tags);
        try {
          await api('/api/upload', {method:'POST', body:fd});
        } catch (e) {
          console.error('Failed to upload:', f.name, e);
        }
      }
      
      // Process each file in background
      for (const f of pendingFiles) {
        const fd = new FormData();
        fd.append('file', f);
        fd.append('title', title);
        fd.append('tags', tags);
        fd.append('async_mode', 'true');
        try {
          const r = await api('/api/ingest', {method:'POST', body:fd});
          if (r.status === 'queued') {
            pollTaskStatus(r.task_id);
          }
        } catch (e) {
          console.error('Failed to queue:', f.name, e);
        }
      }
      
      toast(pendingFiles.length + ' files queued for processing');
    } else {
      // Single file - synchronous
      const f = pendingFiles[0];
      const fd = new FormData();
      fd.append('file', f);
      fd.append('title', title);
      fd.append('tags', tags);
      try {
        const r = await api('/api/ingest', {method:'POST', body:fd});
        toast(f.name + ': ' + (r.chunks_added || 0) + ' chunks');
      } catch (e) { toast(f.name + ' failed', true); }
    }
    
    pendingFiles = [];
    document.getElementById('fileList').textContent = '';
    closeModal('uploadModal');
    loadDocList(); loadStats();
  } else {
    toast('No content to add', true);
  }
}

function pollTaskStatus(taskId) {
  activeTasks.push({id: taskId, startTime: Date.now()});
  updateTaskIndicator();
  
  const interval = setInterval(async () => {
    try {
      const r = await api('/api/task/' + taskId);
      if (r.status === 'completed' || r.status === 'failed') {
        clearInterval(interval);
        delete pollingIntervals[taskId];
        activeTasks = activeTasks.filter(t => t.id !== taskId);
        updateTaskIndicator();
        
        if (r.status === 'completed' && r.result) {
          const chunks = r.result.chunks_added || r.result.success || 0;
          toast('Processing complete: ' + chunks + ' chunks added');
          loadDocList();
          loadStats();
        } else if (r.status === 'failed') {
          toast('Processing failed: ' + (r.error || 'Unknown error'), true);
        }
      }
    } catch (e) {
      console.error('Error polling task:', e);
    }
  }, 1000);
  
  pollingIntervals[taskId] = interval;
}

function formatTime(seconds) {
  if (seconds < 60) return Math.ceil(seconds) + 's';
  if (seconds < 3600) return Math.ceil(seconds / 60) + 'm';
  return Math.ceil(seconds / 3600) + 'h';
}

function updateTaskIndicator() {
  const header = document.querySelector('header');
  let indicator = document.getElementById('taskIndicator');
  
  if (activeTasks.length > 0) {
    if (!indicator) {
      indicator = document.createElement('span');
      indicator.id = 'taskIndicator';
      indicator.style.cssText = 'font-size:12px;color:var(--yellow);margin-left:8px;display:flex;align-items:center;gap:4px;';
      header.appendChild(indicator);
    }
    
    // Get latest progress for all active tasks
    Promise.all(activeTasks.map(t => api('/api/task/' + t.id).catch(() => null)))
      .then(results => {
        const validResults = results.filter(r => r !== null);
        if (validResults.length === 0) return;
        
        // Calculate average progress
        const totalProgress = validResults.reduce((sum, r) => sum + (r.progress || 0), 0);
        const avgProgress = Math.round(totalProgress / validResults.length);
        
        // Estimate time remaining based on current progress and elapsed time
        let etaText = '';
        if (avgProgress > 0 && avgProgress < 100) {
          const elapsed = (Date.now() - activeTasks[0].startTime) / 1000;
          const estimatedTotal = elapsed / (avgProgress / 100);
          const remaining = estimatedTotal - elapsed;
          if (remaining > 0) {
            etaText = ' ~' + formatTime(remaining);
          }
        } else if (avgProgress >= 100) {
          etaText = ' ✓';
        }
        
        // Get current file if available
        const currentFile = validResults[validResults.length - 1].current_file || '';
        const fileText = currentFile ? ' | ' + currentFile.substring(0, 15) + (currentFile.length > 15 ? '...' : '') : '';
        
        indicator.innerHTML = '<span class="loading" style="width:12px;height:12px;flex-shrink:0"></span>' + 
                             avgProgress + '%' + etaText + fileText;
      });
  } else if (indicator) {
    indicator.remove();
  }
}

async function showStatsModal() {
  document.getElementById('statsModal').classList.add('show');
  await loadStats();
}

async function loadStats() {
  try {
    const s = await api('/api/stats');
    document.getElementById('cnt-l1').textContent = s.l1_chunks || 0;
    document.getElementById('cnt-l2').textContent = s.l2_chunks || 0;
    document.getElementById('cnt-l3').textContent = s.l3_chunks || 0;
    document.getElementById('statsGrid').innerHTML = [
      {n: s.documents, l: 'Documents'}, {n: s.l1_chunks, l: 'L1 Hot'},
      {n: s.l2_chunks, l: 'L2 Warm'}, {n: s.l3_chunks, l: 'L3 Cold'},
      {n: s.total_chunks, l: 'Total Chunks'}, {n: s.tags, l: 'Tags'},
      {n: s.searches, l: 'Searches'}, {n: s.ollama_available ? 'Yes' : 'No', l: 'Ollama'}
    ].map(x => `<div class="stat-card"><div class="num">${x.n}</div><div class="label">${x.l}</div></div>`).join('');
  } catch (e) { console.error(e); }
}

async function runMaintenance() {
  try {
    const r = await api('/api/maintain', {method:'POST'});
    toast('Maintenance done. L1->L2: ' + r.demoted_l1_to_l2 + ', L2->L3: ' + r.demoted_l2_to_l3);
    loadStats();
  } catch (e) { toast('Maintenance failed', true); }
}

function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// Init
loadDocList();
loadStats();
</script>
</body>
</html>"""
