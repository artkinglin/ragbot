import unittest
from pathlib import Path

from embeddings import build_chunk_metadata, build_index_settings, extract_page_number


class ExtractPageNumberTests(unittest.TestCase):
    def test_extracts_first_page_marker(self) -> None:
        chunk = "[Page 12]\nFirst page text.\n\n[Page 13]\nNext page text."

        self.assertEqual(extract_page_number(chunk), 12)

    def test_returns_none_when_no_page_marker_exists(self) -> None:
        self.assertIsNone(extract_page_number("plain chunk text"))


class BuildChunkMetadataTests(unittest.TestCase):
    def test_includes_source_chunk_index_and_page(self) -> None:
        metadata = build_chunk_metadata(Path("paper.pdf"), "[Page 7]\ntext", 3)

        self.assertEqual(metadata["source"], "paper.pdf")
        self.assertEqual(metadata["chunk_index"], 3)
        self.assertEqual(metadata["page"], 7)

    def test_omits_page_when_chunk_has_no_marker(self) -> None:
        metadata = build_chunk_metadata(Path("paper.pdf"), "text", 3)

        self.assertEqual(metadata, {"source": "paper.pdf", "chunk_index": 3})

    def test_includes_index_settings_when_provided(self) -> None:
        settings = build_index_settings("model-a", 1200, 200)

        metadata = build_chunk_metadata(Path("paper.pdf"), "text", 3, settings)

        self.assertEqual(metadata["embedding_model"], "model-a")
        self.assertEqual(metadata["chunk_size"], 1200)
        self.assertEqual(metadata["chunk_overlap"], 200)


class BuildIndexSettingsTests(unittest.TestCase):
    def test_records_embedding_and_chunking_settings(self) -> None:
        settings = build_index_settings("model-a", 1200, 200)

        self.assertEqual(
            settings,
            {
                "embedding_model": "model-a",
                "chunk_size": 1200,
                "chunk_overlap": 200,
            },
        )


if __name__ == "__main__":
    unittest.main()
