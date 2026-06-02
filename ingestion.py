"""
PDF loading and text chunking.

This module owns the "document -> chunks" step of RAG.
"""

from pathlib import Path

from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf_text(pdf_path: Path) -> str:
    """Extract text from every page in a PDF file."""
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        # Some PDFs have pages with images or unusual encoding, so pypdf may return None.
        text = page.extract_text() or ""

        # Page markers make retrieved chunks easier to inspect and cite.
        page_texts.append(f"\n\n[Page {page_number}]\n{text}")

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError("No extractable text found. A scanned PDF needs OCR before RAG.")

    return full_text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long document text into overlapping chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Prefer a paragraph break or sentence break so chunks are readable units.
        natural_end = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if natural_end > start + chunk_size // 2:
            end = natural_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step backward by the overlap amount so the next chunk repeats useful context.
        next_start = end - chunk_overlap

        # Force progress if punctuation-based splitting ever creates a bad boundary.
        start = max(next_start, start + 1)

    return chunks
