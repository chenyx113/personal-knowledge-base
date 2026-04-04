"""
embedder.py - Embedding generation via Ollama with local fallback.
When Ollama is unavailable, uses TF-IDF based vectors for offline operation.
"""

import hashlib
import json
import math
import struct
import re
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class OllamaEmbedder:
    """Generate embeddings via Ollama API. Falls back to TF-IDF if unavailable."""

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "nomic-embed-text", timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._ollama_available: Optional[bool] = None
        self._idf_cache: dict = {}
        self._fallback_dim = 768  # Dimension for fallback vectors

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        if self._ollama_available is not None:
            return self._ollama_available
        if not HAS_REQUESTS:
            self._ollama_available = False
            return False
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            self._ollama_available = r.status_code == 200
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if self.is_available():
            return self._ollama_embed(text)
        return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if self.is_available():
            return [self._ollama_embed(t) for t in texts]
        return [self._fallback_embed(t) for t in texts]

    def _ollama_embed(self, text: str) -> np.ndarray:
        """Call Ollama embedding API."""
        try:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text[:8000]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding", [])
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            # Fallback on error
            self._ollama_available = False
            return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding for offline use.
        Produces consistent vectors that preserve some token overlap similarity.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self._fallback_dim, dtype=np.float32)

        # Use token hashes to fill vector dimensions deterministically
        vec = np.zeros(self._fallback_dim, dtype=np.float32)
        token_counts = Counter(tokens)

        for token, count in token_counts.items():
            # Hash token to get deterministic positions and values
            h = hashlib.md5(token.encode("utf-8")).digest()
            for i in range(0, len(h), 4):
                dim_idx = struct.unpack("<I", h[i:i+4])[0] % self._fallback_dim
                val = struct.unpack("<f", h[i:i+4])[0]
                # Normalize the value
                val = (val % 2.0) - 1.0
                vec[dim_idx] += val * math.log1p(count)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: split on non-alphanumeric, lowercase, filter short."""
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    @property
    def dimensions(self) -> int:
        """Return expected vector dimensions."""
        if self.is_available():
            # Try to get from a test embedding
            try:
                test_vec = self._ollama_embed("test")
                return len(test_vec)
            except Exception:
                pass
        return self._fallback_dim


class OllamaGenerator:
    """Text generation via Ollama API with fallback to extractive summarization."""

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "phi3:mini", timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not HAS_REQUESTS:
            self._available = False
            return False
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt."""
        if self.is_available():
            return self._ollama_generate(prompt, max_tokens)
        return self._fallback_generate(prompt)

    def _ollama_generate(self, prompt: str, max_tokens: int) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            self._available = False
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Extractive summary: return first few meaningful sentences."""
        # Extract the content to summarize from the prompt
        lines = prompt.split("\n")
        content_lines = []
        in_content = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip instruction lines
            if any(kw in stripped for kw in ["请", "要求", "格式", "内容：", "content:"]):
                in_content = True
                continue
            if in_content or len(stripped) > 20:
                content_lines.append(stripped)

        if not content_lines:
            content_lines = [l.strip() for l in lines if l.strip()]

        # Take first 3-5 sentences as summary
        summary_lines = content_lines[:5]
        return " ".join(summary_lines)[:500]

    def generate_summary(self, text: str, max_tokens: int = 256) -> str:
        """Generate a summary for the given text."""
        prompt = (
            "请用1-3句话总结以下内容的核心要点：\n\n"
            f"{text[:3000]}"
        )
        return self.generate(prompt, max_tokens)

    def generate_overview(self, text: str, max_tokens: int = 128) -> str:
        """Generate a brief overview with keywords."""
        prompt = (
            "请用一句话概述以下内容，并列出3-5个关键词：\n\n"
            f"{text[:2000]}"
        )
        return self.generate(prompt, max_tokens)


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Convert numpy vector to bytes for SQLite storage."""
    return vec.astype(np.float32).tobytes()


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert bytes back to numpy vector."""
    return np.frombuffer(data, dtype=np.float32).copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
