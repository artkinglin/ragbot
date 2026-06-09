import unittest

from reranking import rerank_matches


class FakeReranker:
    def predict(self, pairs):
        return [0.1, 0.9, 0.4]


class RerankMatchesTests(unittest.TestCase):
    def test_returns_empty_list_without_candidates(self) -> None:
        self.assertEqual(rerank_matches("question", [], FakeReranker(), top_k=2), [])

    def test_orders_matches_by_cross_encoder_score(self) -> None:
        matches = [
            {"document": "weak match", "metadata": {"chunk_index": 0}},
            {"document": "best match", "metadata": {"chunk_index": 1}},
            {"document": "middle match", "metadata": {"chunk_index": 2}},
        ]

        reranked = rerank_matches("question", matches, FakeReranker(), top_k=2)

        self.assertEqual([match["document"] for match in reranked], ["best match", "middle match"])
        self.assertEqual(reranked[0]["rerank_rank"], 1)
        self.assertEqual(reranked[0]["rerank_score"], 0.9)

    def test_does_not_mutate_original_matches(self) -> None:
        matches = [{"document": "text", "metadata": {"chunk_index": 0}}]

        rerank_matches("question", matches, FakeReranker(), top_k=1)

        self.assertNotIn("rerank_score", matches[0])


if __name__ == "__main__":
    unittest.main()
