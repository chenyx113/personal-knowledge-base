"""
dedup.py - Multi-level deduplication engine.
Uses content hash, SimHash (locality-sensitive), and vector similarity.
"""

import hashlib
import struct
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import cosine_similarity, bytes_to_vector


class SimHash:
    """Locality-Sensitive Hashing for near-duplicate detection."""

    def __init__(self, bits: int = 64):
        self.bits = bits

    def compute(self, text: str) -> int:
        """Compute SimHash for text. Similar texts yield similar hashes."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0

        token_counts = Counter(tokens)
        v = [0] * self.bits

        for token, weight in token_counts.items():
            token_hash = self._hash_token(token)
            for i in range(self.bits):
                if token_hash & (1 << i):
                    v[i] += weight
                else:
                    v[i] -= weight

        fingerprint = 0
        for i in range(self.bits):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Count differing bits between two SimHashes."""
        xor = hash1 ^ hash2
        count = 0
        while xor:
            count += xor & 1
            xor >>= 1
        return count

    def _hash_token(self, token: str) -> int:
        """Hash a token to a bits-length integer."""
        h = hashlib.md5(token.encode("utf-8")).digest()
        result = 0
        for i in range(min(8, len(h))):
            result |= h[i] << (i * 8)
        return result & ((1 << self.bits) - 1)

    def _tokenize(self, text: str) -> List[str]:
        """Extract word-level and bigram tokens."""
        words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        words = [w for w in words if len(w) > 1]
        # Add bigrams for better fingerprinting
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams


class SemanticHasher:
    """Generate semantic fingerprints based on key phrases."""

    def compute(self, text: str) -> str:
        """Extract key phrases and hash them for semantic fingerprinting."""
        phrases = self._extract_key_phrases(text, top_k=10)
        combined = "|".join(sorted(phrases))
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]

    def _extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """Simple TF-based key phrase extraction."""
        words = re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())
        # Filter stopwords (basic set)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "this",
            "that", "these", "those", "it", "its", "my", "your", "our",
            "their", "his", "her", "we", "they", "you", "he", "she",
            "and", "or", "but", "not", "no", "if", "then", "else",
            "for", "with", "from", "to", "of", "in", "on", "at", "by",
            "的", "了", "是", "在", "有", "和", "与", "或", "不", "也",
            "就", "都", "而", "及", "为", "这", "那", "他", "她", "它",
        }
        filtered = [w for w in words if w not in stopwords]
        freq = Counter(filtered)
        return [w for w, _ in freq.most_common(top_k)]


class DeduplicationEngine:
    """Multi-level deduplication: exact hash → SimHash → vector similarity."""

    def __init__(self, simhash_threshold: int = 3,
                 vector_threshold: float = 0.92):
        self.simhash = SimHash(bits=64)
        self.semantic_hasher = SemanticHasher()
        self.simhash_threshold = simhash_threshold
        self.vector_threshold = vector_threshold

    def compute_content_hash(self, text: str) -> str:
        """Exact content hash (SHA256)."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def compute_simhash(self, text: str) -> int:
        """Locality-sensitive hash."""
        return self.simhash.compute(text)

    def compute_semantic_hash(self, text: str) -> str:
        """Semantic key-phrase hash."""
        return self.semantic_hasher.compute(text)

    def check_duplicate(self, new_text: str, new_vector: np.ndarray,
                        existing_records: List[Dict]) -> Dict:
        """Check if new_text is a duplicate of any existing record.

        Args:
            new_text: The new text to check.
            new_vector: Embedding vector of the new text.
            existing_records: List of dicts with keys:
                id, content_hash, semantic_hash, vector (bytes), raw_content

        Returns:
            Dict with is_duplicate, level, match_id, similarity fields.
        """
        new_content_hash = self.compute_content_hash(new_text)
        new_simhash = self.compute_simhash(new_text)
        new_semantic_hash = self.compute_semantic_hash(new_text)

        # Level 1: Exact content hash match
        for rec in existing_records:
            if rec.get("content_hash") == new_content_hash:
                return {
                    "is_duplicate": True,
                    "level": "exact",
                    "match_id": rec["id"],
                    "similarity": 1.0,
                }

        # Level 2: SimHash hamming distance
        for rec in existing_records:
            rec_simhash = self.compute_simhash(rec.get("raw_content", ""))
            distance = self.simhash.hamming_distance(new_simhash, rec_simhash)
            if distance <= self.simhash_threshold:
                # Verify with vector similarity
                if rec.get("vector"):
                    rec_vec = bytes_to_vector(rec["vector"])
                    sim = cosine_similarity(new_vector, rec_vec)
                    if sim > self.vector_threshold:
                        return {
                            "is_duplicate": True,
                            "level": "simhash",
                            "match_id": rec["id"],
                            "similarity": float(sim),
                            "hamming_distance": distance,
                        }

        # Level 3: Vector similarity (most expensive, last resort)
        best_sim = 0.0
        best_id = None
        for rec in existing_records:
            if rec.get("vector"):
                rec_vec = bytes_to_vector(rec["vector"])
                sim = cosine_similarity(new_vector, rec_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_id = rec["id"]

        if best_sim > self.vector_threshold:
            return {
                "is_duplicate": True,
                "level": "vector",
                "match_id": best_id,
                "similarity": float(best_sim),
            }

        return {"is_duplicate": False, "similarity": float(best_sim)}

    def recommend_merge_strategy(self, similarity: float) -> str:
        """Recommend how to handle duplicate content."""
        if similarity > 0.98:
            return "exact_duplicate"
        elif similarity > 0.95:
            return "incremental_update"
        elif similarity > 0.90:
            return "soft_reference"
        else:
            return "distinct"

    def find_similar_pairs(self, records: List[Dict],
                           threshold: float = 0.90) -> List[Tuple[int, int, float]]:
        """Find all pairs of similar records above threshold.
        Returns list of (id1, id2, similarity) tuples.
        """
        pairs = []
        vecs = {}
        for rec in records:
            if rec.get("vector"):
                vecs[rec["id"]] = bytes_to_vector(rec["vector"])

        ids = list(vecs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = cosine_similarity(vecs[ids[i]], vecs[ids[j]])
                if sim >= threshold:
                    pairs.append((ids[i], ids[j], float(sim)))

        return sorted(pairs, key=lambda x: x[2], reverse=True)
