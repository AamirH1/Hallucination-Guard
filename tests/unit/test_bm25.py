from app.retrieval.bm25 import BM25Index


def test_bm25_ranks_exact_keyword_match_highest():
    index = BM25Index()
    index.build(
        [
            ("doc1", "the quick brown fox jumps over the lazy dog", {}),
            ("doc2", "refund policy allows thirty day returns for customers", {}),
            ("doc3", "unrelated content about mountains and rivers", {}),
        ]
    )
    results = index.query("refund policy customers", top_k=3)
    assert results[0][0] == "doc2"


def test_bm25_empty_index_returns_empty():
    index = BM25Index()
    assert index.query("anything", 5) == []


def test_bm25_stopword_only_query_returns_nothing_useful():
    index = BM25Index()
    index.build([("doc1", "some content about pricing", {})])
    results = index.query("what is the of and", 5)
    assert results == []
