import unittest
from unittest.mock import patch

from retrieval import (
    format_retrieved_chunk,
    fuse_ranked_matches,
    load_indexed_chunks,
    reciprocal_rank,
    retrieve_top_chunks,
    search_bm25_chunks,
    search_vector_chunks,
)


class FakeCollection:
    def __init__(self) -> None:
        self.last_query_kwargs = None
        self.indexed_chunks = {
            "ids": ["doc_chunk_0", "doc_chunk_1"],
            "documents": ["First chunk text.", "Second chunk text."],
            "metadatas": [{"chunk_index": 0, "page": 4}, {"chunk_index": 1}],
        }

    def get(self, **kwargs):
        return self.indexed_chunks

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        return {
            "documents": [["First chunk text.", "Second chunk text."]],
            "distances": [[0.12345, 0.67891]],
            "metadatas": [[{"chunk_index": 0, "page": 4}, {"chunk_index": 1}]],
        }


class LoadIndexedChunksTests(unittest.TestCase):
    def test_loads_documents_with_ids_and_metadata(self) -> None:
        collection = FakeCollection()

        chunks = load_indexed_chunks(collection)

        self.assertEqual(chunks[0]["id"], "doc_chunk_0")
        self.assertEqual(chunks[0]["document"], "First chunk text.")
        self.assertEqual(chunks[0]["metadata"]["page"], 4)


class FormatRetrievedChunkTests(unittest.TestCase):
    def test_formats_metadata_and_score_label(self) -> None:
        chunk = format_retrieved_chunk(
            1,
            "Important text.",
            {"chunk_index": 2, "page": 9},
            "bm25=1.2345",
        )

        self.assertIn("Source 1 | chunk=2 | page=9 | bm25=1.2345", chunk)
        self.assertIn("Important text.", chunk)


class FuseRankedMatchesTests(unittest.TestCase):
    def test_reciprocal_rank_decreases_with_rank(self) -> None:
        self.assertGreater(reciprocal_rank(1), reciprocal_rank(2))

    def test_fuses_vector_and_bm25_matches_by_chunk_metadata(self) -> None:
        vector_matches = [
            {
                "id": "vector-a",
                "document": "Shared text.",
                "metadata": {"source": "doc.pdf", "chunk_index": 0},
                "distance": 0.2,
                "vector_rank": 1,
            }
        ]
        bm25_matches = [
            {
                "id": "bm25-a",
                "document": "Shared text.",
                "metadata": {"source": "doc.pdf", "chunk_index": 0},
                "bm25_score": 2.0,
                "bm25_rank": 1,
            },
            {
                "id": "bm25-b",
                "document": "Keyword only.",
                "metadata": {"source": "doc.pdf", "chunk_index": 1},
                "bm25_score": 1.0,
                "bm25_rank": 2,
            },
        ]

        fused = fuse_ranked_matches(vector_matches, bm25_matches, top_k=2)

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["metadata"]["chunk_index"], 0)
        self.assertIn("vector_rank", fused[0])
        self.assertIn("bm25_rank", fused[0])
        self.assertGreater(fused[0]["hybrid_score"], fused[1]["hybrid_score"])


class RetrieveTopChunksTests(unittest.TestCase):
    def test_search_bm25_chunks_returns_keyword_matches(self) -> None:
        collection = FakeCollection()
        collection.indexed_chunks = {
            "ids": ["doc_chunk_0", "doc_chunk_1"],
            "documents": ["alpha beta beta", "gamma delta"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}],
        }

        matches = search_bm25_chunks("beta", collection, top_k=2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["document"], "alpha beta beta")
        self.assertGreater(matches[0]["bm25_score"], 0)
        self.assertEqual(matches[0]["bm25_rank"], 1)

    def test_search_vector_chunks_returns_structured_matches(self) -> None:
        collection = FakeCollection()

        with patch("retrieval.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            matches = search_vector_chunks(
                "What matters?",
                collection,
                embedding_model=object(),
                top_k=2,
            )

        self.assertEqual(matches[0]["document"], "First chunk text.")
        self.assertEqual(matches[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(matches[0]["distance"], 0.12345)
        self.assertEqual(matches[0]["vector_rank"], 1)

    def test_formats_retrieved_chunks_with_metadata(self) -> None:
        collection = FakeCollection()

        with patch("retrieval.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            chunks = retrieve_top_chunks(
                "What matters?",
                collection,
                embedding_model=object(),
                top_k=2,
            )

        self.assertEqual(len(chunks), 2)
        self.assertIn("Source 1 | chunk=0 | page=4 | distance=0.1235", chunks[0])
        self.assertIn("First chunk text.", chunks[0])
        self.assertIn("Source 2 | chunk=1 | distance=0.6789", chunks[1])
        self.assertNotIn("page=", chunks[1])

    def test_queries_collection_with_embedded_query_and_top_k(self) -> None:
        collection = FakeCollection()

        with patch("retrieval.embed_texts", return_value=[[0.1, 0.2, 0.3]]) as embed_texts:
            retrieve_top_chunks(
                "What matters?",
                collection,
                embedding_model=object(),
                top_k=2,
            )

        embed_texts.assert_called_once()
        self.assertEqual(collection.last_query_kwargs["query_embeddings"], [[0.1, 0.2, 0.3]])
        self.assertEqual(collection.last_query_kwargs["n_results"], 2)
        self.assertEqual(
            collection.last_query_kwargs["include"],
            ["documents", "distances", "metadatas"],
        )

    def test_filters_chunks_above_max_distance(self) -> None:
        collection = FakeCollection()

        with patch("retrieval.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            chunks = retrieve_top_chunks(
                "What matters?",
                collection,
                embedding_model=object(),
                top_k=2,
                max_distance=0.2,
            )

        self.assertEqual(len(chunks), 1)
        self.assertIn("First chunk text.", chunks[0])


if __name__ == "__main__":
    unittest.main()
