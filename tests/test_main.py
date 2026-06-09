import argparse
import unittest

from main import normalize_query, validate_cli_options


def make_args(**overrides) -> argparse.Namespace:
    values = {
        "top_k": 3,
        "vector_candidates": 8,
        "bm25_candidates": 8,
        "vector_weight": 1.0,
        "bm25_weight": 1.0,
        "rerank_candidates": 8,
        "max_distance": None,
        "chunk_size": 1200,
        "chunk_overlap": 200,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ValidateCliOptionsTests(unittest.TestCase):
    def test_accepts_default_numeric_options(self) -> None:
        validate_cli_options(make_args())

    def test_rejects_non_positive_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "--top-k"):
            validate_cli_options(make_args(top_k=0))

    def test_rejects_negative_max_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-distance"):
            validate_cli_options(make_args(max_distance=-0.1))

    def test_rejects_non_positive_vector_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "--vector-candidates"):
            validate_cli_options(make_args(vector_candidates=0))

    def test_rejects_non_positive_bm25_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "--bm25-candidates"):
            validate_cli_options(make_args(bm25_candidates=0))

    def test_rejects_negative_vector_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "--vector-weight"):
            validate_cli_options(make_args(vector_weight=-0.1))

    def test_rejects_negative_bm25_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "--bm25-weight"):
            validate_cli_options(make_args(bm25_weight=-0.1))

    def test_rejects_zero_vector_and_bm25_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieval weight"):
            validate_cli_options(make_args(vector_weight=0, bm25_weight=0))

    def test_rejects_non_positive_rerank_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "--rerank-candidates"):
            validate_cli_options(make_args(rerank_candidates=0))

    def test_rejects_tiny_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "--chunk-size"):
            validate_cli_options(make_args(chunk_size=99))

    def test_rejects_negative_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "--chunk-overlap"):
            validate_cli_options(make_args(chunk_overlap=-1))

    def test_rejects_overlap_at_or_above_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "--chunk-overlap"):
            validate_cli_options(make_args(chunk_size=100, chunk_overlap=100))


class NormalizeQueryTests(unittest.TestCase):
    def test_strips_terminal_input(self) -> None:
        self.assertEqual(normalize_query("  question?  "), "question?")


if __name__ == "__main__":
    unittest.main()
