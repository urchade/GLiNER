"""
Hard Negative Sampling for GLiNER training.

Validated by arXiv:2402.16602: using semantically confusable entity types as negatives
forces the model to learn finer-grained distinctions, yielding significantly better F1
than random sampling from the batch.

Example: when the positive type is "Medication", using "Chemical Compound" or "Drug Class"
as negatives is far more informative than using "Country" or "Sports Team".

Two backends are supported:
  1. sentence-transformers (recommended) — embeds type strings via a small encoder
     (all-MiniLM-L6-v2, 22M params) and retrieves nearest neighbours by cosine similarity.
  2. Lightweight string-overlap fallback — uses character n-gram overlap as a proxy for
     semantic similarity. No extra dependencies. Lower quality but zero install cost.

Usage in training (via TrainingArguments):
    args = TrainingArguments(
        hard_negative_ratio=0.5,                        # 50% hard, 50% random
        hard_negative_encoder="all-MiniLM-L6-v2",      # or None for string fallback
        ...
    )
"""

from __future__ import annotations

import random
from typing import Set, List, Optional

from ..utils import is_module_available

IS_SBERT = is_module_available("sentence_transformers")


# ---------------------------------------------------------------------------
# String-overlap similarity (no-dependency fallback)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = text.lower()
    return {text[i: i + n] for i in range(max(1, len(text) - n + 1))}


def _string_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram overlap (Jaccard) as a cheap semantic proxy."""
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    union = ga | gb
    if not union:
        return 0.0
    return len(ga & gb) / len(union)


# ---------------------------------------------------------------------------
# TypeSimilarityIndex
# ---------------------------------------------------------------------------

class TypeSimilarityIndex:
    """
    Semantic similarity index over entity type strings.

    Given a set of all known entity types, pre-computes pairwise similarity scores
    and caches nearest neighbours for efficient hard-negative retrieval at train time.

    Falls back to character n-gram similarity when sentence-transformers is not
    installed.

    Args:
        encoder_name: sentence-transformers model name (e.g. "all-MiniLM-L6-v2").
                      Set to None to force the string-overlap fallback.
        cache_dir:    Optional cache directory for downloaded encoder weights.

    Example:
        idx = TypeSimilarityIndex()
        idx.build(["person", "organization", "location", "medication", "chemical compound"])
        hard_negs = idx.get_hard_negatives(["medication"], n=3, exclude={"medication"})
        # → e.g. ["chemical compound", "drug", "biological substance"]
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.encoder_name = encoder_name
        self.cache_dir = cache_dir
        self._types: List[str] = []
        self._sim_matrix: Optional[List[List[float]]] = None  # (N, N)
        self._built = False
        self._use_sbert = IS_SBERT and encoder_name is not None

    def build(self, all_types: List[str]) -> None:
        """
        Compute pairwise similarity for all_types and cache the result.

        Called once at the start of training after the full type vocabulary is known.

        Args:
            all_types: All entity type strings that appear in the training corpus.
        """
        self._types = sorted(set(all_types))
        n = len(self._types)
        if n == 0:
            self._built = True
            return

        if self._use_sbert:
            self._build_sbert(self._types)
        else:
            self._build_string(self._types)

        self._built = True

    def _build_sbert(self, types: List[str]) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(self.encoder_name, cache_folder=self.cache_dir)
        embeddings = model.encode(types, normalize_embeddings=True, show_progress_bar=False)
        # Cosine similarity matrix: (N, N)
        sim = (embeddings @ embeddings.T).tolist()
        self._sim_matrix = sim

    def _build_string(self, types: List[str]) -> None:
        n = len(types)
        self._sim_matrix = [
            [_string_similarity(types[i], types[j]) for j in range(n)]
            for i in range(n)
        ]

    def get_hard_negatives(
        self,
        positive_types: List[str],
        n: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return up to n entity types semantically closest to positive_types but not in them.

        Args:
            positive_types: The entity types that are actually present in this example.
            n:              Number of hard negatives to return.
            exclude:        Additional types to exclude (e.g. the positive types themselves).

        Returns:
            List of hard-negative type strings, length ≤ n.
        """
        if not self._built or not self._types or n <= 0:
            return []

        excluded = set(positive_types) | (exclude or set())
        candidates = [t for t in self._types if t not in excluded]
        if not candidates:
            return []

        if self._sim_matrix is None:
            return random.sample(candidates, k=min(n, len(candidates)))

        # For each candidate, compute its max similarity to any positive type
        type_to_idx = {t: i for i, t in enumerate(self._types)}
        pos_indices = [type_to_idx[t] for t in positive_types if t in type_to_idx]

        if not pos_indices:
            return random.sample(candidates, k=min(n, len(candidates)))

        scored: List[tuple] = []
        for cand in candidates:
            if cand not in type_to_idx:
                continue
            cand_idx = type_to_idx[cand]
            max_sim = max(self._sim_matrix[cand_idx][pi] for pi in pos_indices)
            scored.append((max_sim, cand))

        scored.sort(reverse=True)
        # Return top-n most similar (confusable) candidates
        return [t for _, t in scored[:n]]

    def is_ready(self) -> bool:
        return self._built and len(self._types) > 0
