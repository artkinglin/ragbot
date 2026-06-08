import unittest
from unittest.mock import patch

from retrieval import load_indexed_chunks, retrieve_top_chunks


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


class RetrieveTopChunksTests(unittest.TestCase):
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
