"""
Embedding and ChromaDB storage.

This module owns the "chunks -> vectors -> vector database" step.
"""

import hashlib
import re
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_PREFIX, EMBEDDING_MODEL_NAME


PAGE_MARKER_PATTERN = re.compile(r"\[Page (\d+)\]")
INDEX_SETTING_KEYS = ("embedding_model", "chunk_size", "chunk_overlap")


def create_document_id(pdf_path: Path) -> str:
    """Create a stable ID from the PDF path and contents."""
    hasher = hashlib.sha256()

    # Include path plus bytes so copied PDFs can be indexed separately if needed.
    hasher.update(str(pdf_path.resolve()).encode("utf-8"))

    with pdf_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()[:16]


def get_collection_name(pdf_path: Path) -> str:
    """Build the Chroma collection name for a PDF."""
    document_id = create_document_id(pdf_path)
    return f"{COLLECTION_PREFIX}_{document_id}"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the local SentenceTransformers embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> list[list[float]]:
    """Turn text strings into vector embeddings."""
    text_list = list(texts)

    embeddings = model.encode(
        text_list,
        normalize_embeddings=True,
        show_progress_bar=len(text_list) > 1,
    )

    # Chroma stores JSON-like values, so convert numpy arrays to plain lists.
    return embeddings.tolist()


def extract_page_number(chunk: str) -> int | None:
    """Read the first page marker from a chunk when one is available."""
    match = PAGE_MARKER_PATTERN.search(chunk)
    if not match:
        return None

    return int(match.group(1))


def build_index_settings(
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, str | int]:
    """Record the settings that determine stored chunk embeddings."""
    return {
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def build_chunk_metadata(
    pdf_path: Path,
    chunk: str,
    chunk_index: int,
    index_settings: dict[str, str | int] | None = None,
) -> dict[str, str | int]:
    """Create Chroma metadata for one stored chunk."""
    metadata: dict[str, str | int] = {
        "source": str(pdf_path),
        "chunk_index": chunk_index,
    }

    if index_settings:
        metadata.update(index_settings)

    page_number = extract_page_number(chunk)
    if page_number is not None:
        metadata["page"] = page_number

    return metadata


def get_collection(pdf_path: Path, chroma_dir: Path = CHROMA_DIR):
    """Open the Chroma collection for one PDF."""
    # PersistentClient writes vectors to disk, so indexing survives restarts.
    client = chromadb.PersistentClient(path=str(chroma_dir))

    return client.get_or_create_collection(name=get_collection_name(pdf_path))


def delete_collection(pdf_path: Path, chroma_dir: Path = CHROMA_DIR) -> None:
    """Delete the Chroma collection for a PDF when it exists."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection_name = get_collection_name(pdf_path)

    try:
        client.delete_collection(name=collection_name)
    except ValueError:
        # Chroma raises ValueError when the collection does not exist, which is safe to ignore.
        return


def collection_matches_index_settings(collection, index_settings: dict[str, str | int]) -> bool:
    """Return whether an existing collection was built with the requested settings."""
    if collection.count() == 0:
        return False

    result = collection.get(limit=1, include=["metadatas"])
    metadatas = result.get("metadatas", [])
    if not metadatas:
        return False

    stored_metadata = metadatas[0] or {}
    return all(stored_metadata.get(key) == value for key, value in index_settings.items())


def index_chunks(
    pdf_path: Path,
    chunks: list[str],
    model: SentenceTransformer,
    chroma_dir: Path = CHROMA_DIR,
    reindex: bool = False,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Embed chunks and store them in Chroma if the PDF is not already indexed."""
    index_settings = build_index_settings(embedding_model_name, chunk_size, chunk_overlap)

    if reindex:
        delete_collection(pdf_path, chroma_dir)

    collection = get_collection(pdf_path, chroma_dir)

    # Reuse existing vectors so repeated runs start fast and do not duplicate data.
    if collection.count() > 0:
        if collection_matches_index_settings(collection, index_settings):
            return collection

        delete_collection(pdf_path, chroma_dir)
        collection = get_collection(pdf_path, chroma_dir)

    document_id = create_document_id(pdf_path)
    embeddings = embed_texts(model, chunks)

    ids = [f"{document_id}_chunk_{index}" for index in range(len(chunks))]
    metadatas = [
        build_chunk_metadata(pdf_path, chunk, index, index_settings)
        for index, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection
