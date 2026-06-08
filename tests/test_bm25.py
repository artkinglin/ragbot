import unittest

from bm25 import BM25Index, tokenize


class TokenizeTests(unittest.TestCase):
    def test_tokenizes_alphanumeric_terms_lowercase(self) -> None:
        self.assertEqual(tokenize("AI-powered RAG, v2."), ["ai", "powered", "rag", "v2"])


class BM25IndexTests(unittest.TestCase):
    def test_scores_keyword_match_above_unmatched_document(self) -> None:
        index = BM25Index(["alpha beta beta", "gamma delta"])

        scores = index.score("beta")

        self.assertGreater(scores[0], scores[1])

    def test_returns_zero_scores_for_empty_query(self) -> None:
        index = BM25Index(["alpha beta", "gamma delta"])

        self.assertEqual(index.score("   "), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
