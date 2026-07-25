import numpy as np

from ads_engine.retrieval import VectorIndex, recall_at_k


def test_exact_vectors_retrieve_themselves() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    index = VectorIndex(embeddings, np.array([10, 11, 12, 13]))
    result = index.search(embeddings[[2, 0]], k=1)
    assert result.ids.tolist() == [[12], [10]]
    assert recall_at_k(result.ids, np.array([12, 10])) == 1.0


def test_search_rejects_wrong_dimensions() -> None:
    index = VectorIndex(np.eye(3, dtype=np.float32))
    try:
        index.search(np.ones((1, 2), dtype=np.float32), k=1)
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("expected a dimension error")
