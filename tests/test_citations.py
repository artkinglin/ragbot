import unittest

from citations import (
    extract_available_sources,
    extract_citations,
    validate_citations,
)


RETRIEVED_CHUNKS = [
    "Source 1 | chunk=4 | rerank=0.9000\nFirst evidence.",
    "Source 2 | chunk=8 | rerank=0.7000\nSecond evidence.",
]


class ExtractCitationTests(unittest.TestCase):
    def test_extracts_available_sources_from_chunk_headers(self) -> None:
        self.assertEqual(extract_available_sources(RETRIEVED_CHUNKS), frozenset({1, 2}))

    def test_extracts_unique_canonical_citations(self) -> None:
        answer = "One claim [Source 2]. Another claim [Source 1] [Source 2]."

        self.assertEqual(extract_citations(answer), frozenset({1, 2}))


class ValidateCitationsTests(unittest.TestCase):
    def test_accepts_citations_to_available_sources(self) -> None:
        result = validate_citations("Supported answer [Source 1].", RETRIEVED_CHUNKS)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, ())

    def test_rejects_answer_without_citations(self) -> None:
        result = validate_citations("Unsupported answer.", RETRIEVED_CHUNKS)

        self.assertFalse(result.is_valid)
        self.assertIn("at least one citation", result.errors[0])

    def test_rejects_citation_to_unavailable_source(self) -> None:
        result = validate_citations("Invented support [Source 3].", RETRIEVED_CHUNKS)

        self.assertFalse(result.is_valid)
        self.assertIn("Source 3", result.errors[0])

    def test_rejects_noncanonical_source_mentions(self) -> None:
        result = validate_citations("Supported by Source 1.", RETRIEVED_CHUNKS)

        self.assertFalse(result.is_valid)
        self.assertTrue(any("canonical" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
