from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _normalized(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a rank-2 matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


@dataclass
class SearchResult:
    ids: NDArray[np.int64]
    scores: NDArray[np.float32]


class VectorIndex:
    """Cosine-similarity index backed by FAISS when it is installed."""

    def __init__(self, embeddings: NDArray[np.floating], ids: NDArray[np.integer] | None = None):
        self.embeddings = _normalized(embeddings)
        row_count, dimensions = self.embeddings.shape
        self.ids = (
            np.arange(row_count, dtype=np.int64)
            if ids is None
            else np.asarray(ids, dtype=np.int64)
        )
        if len(self.ids) != row_count:
            raise ValueError("ids must have one entry per embedding")

        self._faiss = None
        try:
            import faiss  # type: ignore[import-not-found]

            index = faiss.IndexFlatIP(dimensions)
            index.add(self.embeddings)
            self._faiss = index
        except ImportError:
            pass

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, queries: NDArray[np.floating], k: int) -> SearchResult:
        if not 0 < k <= len(self.ids):
            raise ValueError("k must be between 1 and the number of indexed vectors")
        normalized_queries = _normalized(queries)
        if normalized_queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query and index dimensions must match")

        if self._faiss is not None:
            scores, positions = self._faiss.search(normalized_queries, k)
        else:
            all_scores = normalized_queries @ self.embeddings.T
            positions = np.argpartition(-all_scores, kth=k - 1, axis=1)[:, :k]
            partial_scores = np.take_along_axis(all_scores, positions, axis=1)
            order = np.argsort(-partial_scores, axis=1)
            positions = np.take_along_axis(positions, order, axis=1)
            scores = np.take_along_axis(all_scores, positions, axis=1)
        return SearchResult(self.ids[positions], np.asarray(scores, dtype=np.float32))


def recall_at_k(retrieved: NDArray[np.integer], relevant: NDArray[np.integer]) -> float:
    """Return mean recall when each query has one relevant item."""
    found = np.any(np.asarray(retrieved) == np.asarray(relevant)[:, None], axis=1)
    return float(found.mean())
