import unittest

from ingestion import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_chunk_text_returns_non_empty_chunks(self) -> None:
        text = "alpha beta gamma. " * 100

        chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strip() for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
